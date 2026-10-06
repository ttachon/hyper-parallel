# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""find parallelization"""

from contextlib import nullcontext
import time
import copy
import multiprocessing as proc
import json
import os
import logging
from typing import Any, Dict, Optional, Tuple

from hyper_parallel.auto_parallel._exec_spec import ExecSpec
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.estimate_v2 import EvaluatorV2
from hyper_parallel.auto_parallel.sapp_nd.perf_estimation.estimate import LayerTimes, estimate_performance
from hyper_parallel.auto_parallel.sapp_nd.recompute.candidate import (
    MODES,
    RecomputeChoice,
    choose_recompute,
    describe,
    mode_ranges,
    trainer_plan,
    whole_modes,
)

from hyper_parallel.auto_parallel.sapp_nd.nd.global_config import GlobalConfig
from hyper_parallel.auto_parallel.sapp_nd.nd.logger import logger
import hyper_parallel.auto_parallel.sapp_nd.nd.dimensions as Dim
import hyper_parallel.auto_parallel.sapp_nd.nd.common.hardware as Hard
import hyper_parallel.auto_parallel.sapp_nd.nd.debug as Debug
from hyper_parallel.auto_parallel.sapp_nd.nd.dimensions import validate_cp_constraints
from hyper_parallel.auto_parallel.sapp_nd.nd.common.apply_exec import apply_exec
from hyper_parallel.auto_parallel.sapp_nd.nd.common.cost_model_preprocess import (
    CostModelConfig,
    arm_strategy_guard,
    detect_attention_type,
)

# logger = proc.log_to_stderr()
# logger.setLevel(proc.SUBDEBUG)


class ParallelizeLayer:
    """Parallelize one layer type"""

    def __init__(
        self,
        evaluator: Any,
        machine: Any,
        global_batch_size: Any = None,
        dimensions: Any = None,
        **extra_config: Any,
    ) -> None:
        """Build the search driver for one evaluator and machine."""
        self.enable_debug = logger.level < logging.CRITICAL
        self.machine = machine
        if "mppb" in extra_config:
            manual_ppb = extra_config.pop("mppb")
        else:
            manual_ppb = False
        self._take_recompute_options(extra_config, manual_ppb)

        self.mem_eval = evaluator
        # The options chosen for each configuration the ordering scored.
        self.recompute_choices = {}

        self.model_name = self.mem_eval._ccfg.model_name
        logger.debug("model is %s", self.model_name)

        # The cap replaces the device's capacity, and the reserve comes out
        # of whichever capacity holds.
        if "max_mem" in extra_config:
            max_mem = extra_config.pop("max_mem")
            if max_mem is not None:
                self.mem_eval._ccfg.device_capacity.set(max_mem)

        if "mem_for_ppb" in extra_config:
            reserve_mem = extra_config.pop("mem_for_ppb")
            self.mem_eval._ccfg.device_capacity.decrease(reserve_mem)

        logger.debug("before global config init")

        if "sub_model" in extra_config:
            sub_model = extra_config.pop("sub_model")
            if sub_model is not None:
                self.config = GlobalConfig(
                    self.mem_eval._ccfg.mm_ccfgs[sub_model],
                    dimensions,
                    mppb=manual_ppb,
                    parent=self.mem_eval._ccfg,
                )
            else:
                self.config = GlobalConfig(
                    self.mem_eval._ccfg, dimensions, mppb=manual_ppb
                )
        else:
            self.config = GlobalConfig(
                self.mem_eval._ccfg, dimensions, mppb=manual_ppb
            )

        self.mem_eval.set_passes(**extra_config)

        self.machine.update_num_if_none(
            self.config.ccfg.strategy_num_devices()
        )

        if global_batch_size:
            self.global_batch_size = global_batch_size
        else:
            self.global_batch_size = self.config.ccfg.gbs

        self.bound_space()
        # From here on the configs this search owns take a strategy only
        # through set_strategy; an estimator's copy of one starts unarmed.
        arm_strategy_guard(self.mem_eval.ccfg)

    def _take_recompute_options(self, extra_config: Dict[str, Any], manual_ppb: bool) -> None:
        """Take the recompute and offload options out of *extra_config*, refusing those that do not go together."""
        auto_recompute = extra_config.pop("auto_recompute", False)
        if auto_recompute and manual_ppb:
            raise ValueError(
                "auto_recompute chooses every layer's recompute, so it cannot also take it from the config (mppb)"
            )
        # Choose every layer's recompute option for each candidate, rather
        # than score it fully recomputed. A multimodal model, whose
        # candidates are priced on every submodule (F2), gets one mode for
        # every layer, priced whole, and no option per layer: the options are
        # built for one submodule's layers, whose budgets would leave the
        # others out.
        self.auto_recompute = bool(auto_recompute)
        # With auto_recompute, the modes a runtime that runs every layer one
        # way offers, of recompute.candidate.MODES; without them, each layer
        # gets its own option.
        self.recompute_modes = extra_config.pop("recompute_modes", None)
        unknown = sorted(set(self.recompute_modes or ()) - set(MODES))
        if unknown:
            raise ValueError(f"unknown recompute modes {unknown}; expected some of {', '.join(MODES)}")
        # The switches such a runtime's selective mode sets, where they are
        # not the config's; see recompute.candidate.choose_recompute.
        self.recompute_selective = extra_config.pop("recompute_selective", None)
        # With recompute_modes, each layer gets the fastest of them that fits
        # rather than one for every layer: a runtime that runs a mode per
        # layer, as HyperParallel's trainer does with activation_checkpoint.layers.
        self.recompute_mode_per_layer = bool(extra_config.pop("recompute_mode_per_layer", False))
        if self.recompute_mode_per_layer and not (auto_recompute and self.recompute_modes):
            raise ValueError("recompute_mode_per_layer gives each layer one of the recompute_modes auto_recompute "
                             "chooses among; give both")
        # With auto_offload, a choice per layer may offload each stage's first
        # layers over the host link: host_link, else the device's own.
        auto_offload = extra_config.pop("auto_offload", False)
        host_link = extra_config.pop("host_link", None)
        self.offload_link = None
        if auto_offload:
            if not auto_recompute:
                raise ValueError("auto_offload offloads in the choice per layer that auto_recompute makes")
            self.offload_link = host_link or self.machine.device.host_link
            if self.offload_link is None:
                raise ValueError(f"device {self.machine.device} states no host link to offload over; give host_link")

    def bound_space(self) -> None:
        """Set bounds for parallel dimensions"""
        # Bounds live on the module-level dimensions: start each search from
        # none, or it keeps the tightest bound any earlier search set.
        for dim in Dim.ALL_DIMS:
            dim.reset_bound()
        vpp = (
            1
            if Dim.VPP in self.config.dimensions
            else Dim.VPP.from_config(self.config.ccfg)
        )
        pp_bound = min(
            self.machine.pipeline_bound(),
            self.config.total_layer_num() // vpp,
            self.global_batch_size,
        )
        Dim.PP.set_bound(pp_bound)
        logger.info(
            "PP bound is %d, machine bound = %d, L = %d, VPP = %d, B = %d",
            pp_bound,
            self.machine.pipeline_bound(),
            self.config.total_layer_num(),
            vpp,
            self.global_batch_size,
        )
        Dim.EP.set_bound(self.config.ccfg.n_exp)
        # if (
        #     self.config.dimensions.count(Dim.EP) > 0
        #     and Dim.EP.from_config(self.config.ccfg) <= 1
        # ):
        #     Dim.EP.set_bound(1)
        #     self.config.dimensions.remove(Dim.EP)
        kv_heads = self.config.ccfg.n_kv
        if kv_heads:
            Dim.TP.set_bound(kv_heads)
            logger.warning(
                "Because of n_kv_heads, MP will be limited to %s",
                str(kv_heads),
            )
        else:
            # num_head % (TP * UP) == 0. Add UP later
            Dim.TP.set_bound(
                Hard.highest_power_of_2_divisor(self.config.ccfg.a)
            )

    @staticmethod
    def filtered_out(_: Any) -> bool:
        """Manual conditions to remove config patterns"""
        # if parallel_config.has_dim(Dim.EP):
        #     if self.config.dim_val(Dim.EP, parallel_config) < 8:
        #         return True
        return False

    def is_valid(self, parallel_config: Any) -> bool:
        """Check configuration validity"""
        if not parallel_config.is_valid():
            logger.warning("configuration %s not valid", str(parallel_config))
            return False
        if not self.config.moe_valid(parallel_config):
            logger.warning("expert parallel is higher than expert number")
            return False
        if hasattr(self.config, 'ep_constraints_valid') and not self.config.ep_constraints_valid(parallel_config):
            logger.warning("EP divisibility constraints not satisfied")
            return False
        if self.filtered_out(parallel_config):
            logger.warning("Config manually filtered out")
            return False

        if hasattr(parallel_config, 'dims_val') and Dim.CP in parallel_config.dims_val:
            cp_degree = parallel_config.dims_val[Dim.CP]
            if cp_degree > 1:
                seq_len = self.config.ccfg.s
                tp_degree = parallel_config.dims_val.get(Dim.TP, 1)
                pp_degree = parallel_config.dims_val.get(Dim.PP, 1)
                device_per_node = self.machine.device.intra_node_num()
                total_devices = self.machine.number

                attention_type = detect_attention_type(self.config.ccfg).name.lower()

                bw_intra = self.config.ccfg.bw_intra
                bw_inter = self.config.ccfg.bw_inter

                sp_enabled = bool(parallel_config.dims_val.get(Dim.SP, False))

                cp_result = validate_cp_constraints(
                    seq_len=seq_len,
                    cp_degree=cp_degree,
                    tp_degree=tp_degree,
                    pp_degree=pp_degree,
                    device_per_node=device_per_node,
                    attention_type_str=attention_type,
                    bw_intra=bw_intra,
                    bw_inter=bw_inter,
                    total_devices=total_devices,
                    sp_enabled=sp_enabled,
                    cp_algo=getattr(self.config.ccfg, 'cp_algo', 'colossalai_cp'),
                    attention_heads=self.config.ccfg.a,
                    num_kv_heads=getattr(self.config.ccfg, 'n_kv', 0),
                )

                if not cp_result.is_valid:
                    logger.warning("CP constraints violated: %s", cp_result.error_message)
                    return False

                if cp_result.warning_message:
                    logger.info("CP warning: %s", cp_result.warning_message)

        gbs = self.config.global_batch_size(parallel_config)
        if not gbs == self.global_batch_size:
            logger.error(
                "wrong global batch size: ccfg is %d, instead of %d",
                gbs,
                self.global_batch_size,
            )
            return False
        return True

    def priced(self) -> Any:
        """The config a candidate is priced on: the whole model, every submodule of it.

        The search drives one submodule's strategy (:class:`GlobalConfig`
        gives the others the same degrees), and a candidate is priced on the
        model the run trains: a vision tower's parameters, activations and
        compute count toward the stage that holds them (F2).
        """
        return self.mem_eval.ccfg

    def _priced_on_submodules(self) -> bool:
        """Whether a candidate is priced on several submodules, the one the search drives among them."""
        return bool(getattr(self.priced(), "multimodal", False))

    def memory_estim(self, debugger: Any = None) -> Any:
        """Whether the config fits memory"""
        logger.debug("estimate_peak")
        verbose = logger.level < logging.INFO
        self.mem_eval.set_config(self.priced())
        # self.mem_eval = EvaluatorV2(self.config)
        logger.debug("ccfg = %s", str(self.config.ccfg))
        peak = self.mem_eval.estimate_peak(
            verbose=verbose
        )  # (logger.level>2))
        logger.debug("peak memory = %d", peak)
        if debugger and debugger.is_enabled():
            debugger.info[Debug.MemParts.TOTAL] = peak
        return peak

    def generate_search_space(self, folder: Any, threads_num: Any) -> Any:
        """Return a search space computed with memory estimation"""
        # With a pool the accumulator holds AsyncResult, which the direct
        # branch's float values hide from static inference.
        # pylint: disable=no-member
        space = ({}, 0)
        configs = []
        results = {}
        if threads_num:
            with proc.Pool(processes=threads_num) as pool:
                logger.debug("before loops")
                results, size = self.device_loops(space, pool)
                logger.debug("%d results", len(results))
                for config, result in results.items():
                    logger.debug("result = %s", str(result))
                    logger.debug(
                        "before get: is ready ? %s", str(result.ready())
                    )
                    peak_mem = result.get()
                    logger.debug(
                        "after get: is ready ? %s", str(result.ready())
                    )
                    logger.debug(
                        "after get: is successful ? %s",
                        str(result.successful()),
                    )
                    logger.debug("peak_mem = %s", str(peak_mem))
                    if self.mem_eval.mem_fit(peak_mem):
                        configs.append((config, peak_mem))
                pool.close()
                pool.join()
        else:
            results, size = self.device_loops(space, None)
            for config, peak_mem in results.items():
                if self.mem_eval.mem_fit(peak_mem):
                    configs.append((config, peak_mem))
                    if folder:
                        self.config.write(folder, config)
        logger.output("%d valid configurations generated", size)
        logger.output("%d configuration fitting memory to order", len(configs))

        return configs

    def device_loops(self, space: Any, pool: Any) -> Tuple[dict, int]:
        """Exploration loop nest level 0: parallel dimensions dividing devices"""
        for tp in self.config.space(Dim.TP, self.machine.number):
            for pp in self.config.space(Dim.PP, self.machine.number // tp):
                for cp in self.config.space(
                    Dim.CP, self.machine.number // tp // pp
                ):
                    logger.debug(
                        "dp = %d / %d / %d / %d",
                        self.machine.number,
                        tp,
                        cp,
                        pp,
                    )
                    dp = self.machine.number // tp // cp // pp
                    if dp < 1:
                        break
                    space = self.batch_loops(space, pool, (dp, tp, pp, cp))
        return space

    def batch_loops(self, space: Any, pool: Any, dtpc_p: Any) -> Tuple[dict, int]:
        """Exploration loop nest level 1: dimensions dividing batch (except already processed DP)"""
        dp, _, pp, _ = dtpc_p
        # if pp > 1:
        for mbs in self.config.space(
            Dim.MBS, self.global_batch_size // pp // dp
        ):
            logger.debug("mbn= %d / %d / %d", self.global_batch_size, dp, mbs)
            mbn = self.global_batch_size // dp // mbs
            space = self.parallel_loops(space, pool, (dtpc_p, (mbs, mbn)))
        # else:
        #     logger.debug("no pipeline so mbn = 1")
        #     mbs = self.global_batch_size // dp
        #     space = self.parallel_loops(space, pool, (dtpc_p, (mbs, 1)))
        return space

    def parallel_loops(self, space: Any, pool: Any, dims: Any) -> Tuple[dict, int]:
        """Exploration loop nest level 2: dimensions dependent on others"""
        dtpc_p, mbsn = dims
        dp, tp, pp, _ = dtpc_p
        for ep in self.config.space(Dim.EP, dp * tp):
            for vpp in self.config.range_space(
                Dim.VPP, min(4, pp, self.config.total_layer_num() // pp)
            ):
                for op in self.config.space(
                    Dim.OP, self.config.max_op(dp, tp, ep)
                ):
                    for sp in self.config.bool_space(Dim.SP):
                        space = self.inside_loop_nest(
                            space,
                            pool,
                            (dtpc_p, mbsn, (ep, vpp, op, sp)),
                        )
        return space

    def inside_loop_nest(self, space: Any, pool: Any, dims: Any) -> Tuple[dict, int]:
        """Exploration loop nest statements"""
        dtpc_p, mbsn, evos_p = dims
        configs, size = space
        parallel_config = self.config.make_parallel_config(
            dtpc_p, mbsn, evos_p
        )
        logger.info("test config %d : %s", size, str(parallel_config))
        size += 1

        if self.is_valid(parallel_config) and self.config.set_parallel_config(
            parallel_config
        ):
            if pool is None:
                if self.enable_debug:
                    mem_debugger = Debug.Debug(
                        parallel_config,
                        info_type=Debug.MemParts,
                        enable=self.enable_debug,
                        output_file="debug_mem.csv",
                    )
                    # try:
                    peak = self.memory_estim(mem_debugger)
                    mem_debugger.write()
                else:
                    peak = self.memory_estim()
                # except:
                # logger.error()
                # return (configs, size)
            else:
                # logger.debug("before evaluator copy")
                # evaluator = copy.deepcopy(self.mem_eval)
                logger.debug("before apply_async")
                peak = pool.apply_async(
                    pool_estimate_memory,
                    args=(copy.deepcopy(self.priced()),),
                    # args=(evaluator,),
                    # self.memory_estim,
                )
                logger.debug("after apply_async")
            configs[parallel_config] = peak

        return (configs, size)

    def order_search_space(self, space: Any, threads_num: Any, cache_file: Any) -> Any:
        """Sort the search space computed with performance estimation"""
        if not space:
            return ([], [])
        multiproc = False
        if threads_num and threads_num > 5 * len(space):
            multiproc = True
        scored_space = []
        debug_parts = []
        self.recompute_choices = {}
        with (
            proc.Pool(processes=threads_num)
            if multiproc
            else nullcontext()
        ) as pool:
            for config, peak in space:
                self.config.set_parallel_config(config)
                values = []
                mem, savings = peak, None
                choice = self.choose_recompute(config)
                priced = self.priced()
                if choice is not None:
                    mem, savings = int(round(choice.memory)), choice.stage_savings
                    if choice.score is not None:
                        # The whole model, with every submodule running the mode.
                        priced, savings = self.priced_with_mode(choice.mode), None
                if multiproc:
                    score = pool.apply_async(
                        pool_estimate_performance,
                        args=(
                            copy.deepcopy(priced),
                            self.machine.device,
                            mem,
                            cache_file,
                            savings,
                        ),
                    )
                else:
                    if self.enable_debug:
                        debugger = Debug.Debug(
                            config,
                            info_type=Debug.PerfParts,
                            enable=self.enable_debug,
                        )
                        score = estimate_performance(
                            priced,
                            debugger=debugger,
                            device_type=self.machine.device,
                            memory=mem,
                            cache_file=cache_file,
                            stage_savings=savings,
                        )
                        debugger.write()
                        debug_parts = list(debugger.info.keys())
                        values = list(debugger.info.values())
                        del values[-2:]
                        del debug_parts[-2:]
                    else:
                        score = estimate_performance(
                            priced,
                            device_type=self.machine.device,
                            memory=mem,
                            stage_savings=savings,
                        )
                scored_space.append((config, mem, score, values))

                if not multiproc:
                    logger.info("config %s has score %f", str(config), score)

            if multiproc:
                new_scored_space = []
                for config, mem, score, values in scored_space:
                    score_value = score.get()
                    logger.info(
                        "config %s has score %f", str(config), score_value
                    )
                    new_scored_space.append(
                        (config, mem, score_value, values)
                    )
            else:
                new_scored_space = scored_space
        return (sorted(new_scored_space, key=lambda x: x[2]), debug_parts)

    def choose_recompute(self, parallel_config: Any) -> Optional[RecomputeChoice]:
        """Every layer's recompute option for the configuration just set, with auto_recompute.

        The options are the fastest that fit the device, and are kept in
        ``recompute_choices`` under *parallel_config*.

        Args:
            parallel_config: The configuration the config was just set to.

        Returns:
            The choice; ``None`` without auto_recompute, or when there is
            none to make and the configuration keeps its own recompute.
        """
        if not self.auto_recompute:
            return None
        if self._priced_on_submodules():
            choice = self._one_mode_whole()
        else:
            self.mem_eval.set_config(self.config.ccfg)
            choice = choose_recompute(self.mem_eval, self.machine.device, modes=self.recompute_modes,
                                      link=self.offload_link, selective=self.recompute_selective,
                                      per_layer=self.recompute_mode_per_layer)
        if choice is not None:
            self.recompute_choices[parallel_config] = choice
        return choice

    def _one_mode_whole(self) -> Optional[RecomputeChoice]:
        """The fastest mode that fits, for every layer, each mode priced on the whole model.

        A model priced on several submodules runs its runtime's mode on every
        submodule's layers, a vision tower's as well as its language
        model's, and a choice per layer, whose budgets are built for one
        submodule's layers, would leave the others out. So each mode is
        stated on every submodule's config, and the whole model priced under
        it: its stages' memory by the memory model, its score by the
        performance estimate. Without a runtime's modes, every one of
        :data:`MODES` is weighed, selective with each submodule's switches.

        Returns:
            The fastest mode that fits, with the stage memory and the score
            the whole model has under it; ``None`` when none fits.
        """
        whole = self.priced()
        modes = whole_modes([whole.mm_ccfgs[name] for name in whole.mm_order], self.recompute_modes or MODES,
                            self.recompute_selective)
        best = None
        try:
            for mode in modes:
                # Each mode on a copy of the whole model, which the evaluator holds between them.
                self.mem_eval.set_config(whole)
                priced = self.priced_with_mode(mode)
                self.mem_eval.set_config(priced)
                stage_memory = tuple(insight["Static"] + insight["Dynamic"]
                                     for insight in self.mem_eval.estimate_peak_insight())
                if not self.mem_eval.mem_fit(max(stage_memory)):
                    continue
                score = estimate_performance(priced, device_type=self.machine.device,
                                             memory=int(round(max(stage_memory))))
                if best is None or score < best.score:
                    best = RecomputeChoice((), stage_memory, (), mode=mode, score=score)
        finally:
            self.mem_eval.set_config(whole)
        return best

    def priced_with_mode(self, mode: str) -> Any:
        """A copy of the config a candidate is priced on, every submodule's layers running *mode*.

        Args:
            mode: One of the runtime's modes; its selective mode sets the
                runtime's own switches, where it states them, else each
                submodule's.
        """
        priced = copy.deepcopy(self.priced())
        for name in priced.mm_order:
            config = priced.mm_ccfgs[name]
            rec_op = getattr(config, "rec_op", None)
            switches = self.recompute_selective or (vars(rec_op) if rec_op is not None else {})
            apply_exec(config, ExecSpec(recompute=mode_ranges(mode, switches)))
        return priced

    def recompute_per_layer(self, parallel_config: Any) -> Tuple[Optional[RecomputeChoice], Optional[float]]:
        """Each layer's own fastest option for one configuration, and the score it gives.

        What the configuration gains when every layer may run its own way,
        for a search that chose one mode for all of them.

        Args:
            parallel_config: The configuration.

        Returns:
            ``(choice, score)``, or ``(None, None)`` when there is no choice
            to make, a multimodal model's among them.
        """
        if self._priced_on_submodules():
            return None, None
        self.config.set_parallel_config(parallel_config)
        self.mem_eval.set_config(self.config.ccfg)
        choice = choose_recompute(self.mem_eval, self.machine.device, link=self.offload_link)
        if choice is None:
            return None, None
        score = estimate_performance(
            self.config.ccfg,
            device_type=self.machine.device,
            memory=int(round(choice.memory)),
            stage_savings=choice.stage_savings,
        )
        return choice, score

    def order_space_test_comm_classified(self, space: Any, order_by: Any = 2) -> Any:
        """Order the given space with performance estimation"""
        scored_space = []
        debug_parts = []
        for config, real_time, real_comm_wait in space:
            debugger = Debug.Debug(
                config, info_type=Debug.PerfParts, enable=self.enable_debug
            )
            self.config.set_parallel_config(config)
            peak_mem = self.memory_estim()
            score = estimate_performance(
                self.priced(),
                debugger=debugger,
                device_type=self.machine.device,
                stage_focused=0,
            )  # , memory = mem)
            debugger.write()
            debug_parts = list(debugger.info.keys())
            values = list(debugger.info.values())
            del values[-2:]
            scored_space.append(
                (config, peak_mem, real_time, score, values, real_comm_wait)
            )

            logger.info("config %s has score %f", str(config), score)
        del debug_parts[-2:]
        return (sorted(scored_space, key=lambda x: x[order_by]), debug_parts)

    def order_space_test(self, space: Any, order_by: Any = 2) -> Any:
        """Order the given space with performance estimation"""
        scored_space = []
        debug_parts = []
        for config, real_time in space:
            debugger = Debug.Debug(
                config, info_type=Debug.PerfParts, enable=self.enable_debug
            )
            logger.info("Test config %s", str(config))
            self.config.set_parallel_config(config)
            logger.debug(self.mem_eval.get_strategy())
            peak_mem = self.memory_estim()
            score = estimate_performance(
                self.priced(),
                debugger=debugger,
                device_type=self.machine.device,
            )  # , memory = mem)
            debugger.write()
            debug_parts = list(debugger.info.keys())
            values = list(debugger.info.values())
            del values[-2:]
            scored_space.append((config, peak_mem, real_time, score, values))

            logger.info("config %s has score %f", str(config), score)
        del debug_parts[-2:]
        return (sorted(scored_space, key=lambda x: x[order_by]), debug_parts)

    def plot_title(self) -> str:
        """Generate plot title"""
        return (
            f"{self.model_name} on {self.machine.number}"
            + f" {self.machine.device} with {self.global_batch_size} GBS"
        )

    def run_generation_to_ordering(
        self,
        yaml_folder: Any,
        threads_num: Any = None,
        top_num: Any = None,
        cache_file: Any = None,
    ) -> Any:
        """Test some functions"""
        start = time.time()
        space = self.generate_search_space(yaml_folder, threads_num)
        generation = time.time()
        scored_space, dbg = self.order_search_space(
            space, threads_num, cache_file=cache_file
        )
        ordering = time.time()
        logger.output(
            space_to_string(scored_space, max_num=top_num, debug_parts=dbg)
        )
        logger.output(
            "Space generation took %.2fs and ordering took %.2fs",
            generation - start,
            ordering - generation,
        )
        is_not = " NOT" if not self.config.balancing.from_config else ""
        logger.output(
            "Offset & Recompute were%s computed from config info", is_not
        )
        if self.auto_recompute:
            self._log_recompute(scored_space)
        logger.output(
            "Device number is %d, global batch size is %d, dimensions are %s",
            self.machine.number,
            self.global_batch_size,
            str(self.config.dimensions),
        )
        if self.enable_debug:
            output_path = Debug.output_dir()
            if scored_space:
                Debug.plot_nd(
                    scored_space,
                    output_path,
                    dbg,
                    title=self.plot_title(),
                    max_num=top_num,
                )
        return scored_space

    def _log_recompute(self, scored_space: Any) -> None:
        """Log how the recompute was chosen, and the best configuration's, as the trainer states it where it can."""
        if self.recompute_modes is None:
            how = "per layer"
        else:
            how = ("per layer among " if self.recompute_mode_per_layer else "among ") + ", ".join(self.recompute_modes)
        logger.output("Recompute was chosen %s for %d of %d configurations", how, len(self.recompute_choices),
                      len(scored_space))
        best = self.recompute_choices.get(scored_space[0][0]) if scored_space else None
        if best is None:
            return
        logger.output("Recompute of the best configuration:\n%s", describe(best))
        if self.recompute_mode_per_layer and best.mode is None:
            mode, layers = trainer_plan(best)
            logger.output("As the trainer runs it: activation_checkpoint mode %s, layers %s", mode, layers)

    def to_ppb(self, scored_space: Any, k: Any, cfg_name: Any, folder: Optional[str] = None) -> str:
        """Write the pipeline balancer's layer description of the k-th configuration.

        Every layer carries its forward time and the backward time of each of
        its recompute options, priced by the estimate the search scores with,
        in units of the forward time of a plain layer of the first body.

        Args:
            scored_space: The ordered search space.
            k: Rank of the configuration to describe.
            cfg_name: Prefix of the model name the balancer is given.
            folder: Where to write; ND's output directory by default.

        Returns:
            The path of the file written.
        """
        parallel_config = scored_space[k][0]
        self.config.set_parallel_config(parallel_config)
        self.mem_eval.set_config(self.config.ccfg)
        name = f"{cfg_name}_nd_to_ppb_{k}"
        folder = os.path.abspath(folder or Debug.output_dir())
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, name + ".json")
        description = self.mem_eval.estimate_layer_memory(
            device_type=self.machine.device,
            layer_times=LayerTimes(self.machine.device),
        )
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(description, fp, indent=4)
        logger.output(
            "To run pipeline balancing on configuration %s:"
            "\npython run_pipeline_balance.py -lf %s -m %s -s %s -mb %s -i %s -mem %s",
            parallel_config,
            folder,
            name,
            self.config.dim_val(Dim.PP, parallel_config),
            self.config.dim_val(Dim.MBN, parallel_config),
            self.config.dim_val(Dim.VPP, parallel_config),
            int(self.config.ccfg.device_capacity.to_mb().size),
        )
        return path

    def test_from_csv(self, csv_f, output_path=None):
        """Run estimation tests against a real run profiling in csv format"""
        configs, row_num = Debug.get_real_data(csv_f)
        configs_estimated, debug_parts = self.order_space_test(
            configs, order_by=2
        )
        if output_path is not None:
            Debug.plot_vs_real(
                configs_estimated,
                csv_f,
                output_path,
                debug_parts,
                title=self.plot_title(),
            )
        correl, topk = Debug.correlation_topk(configs_estimated, csv_f)
        return correl, topk, row_num

    def test_from_csv_comm_classified(
        self, csv_f, output_path=None, plot_idle=False
    ):
        """Run test to compare estimation with detailed profiling"""
        configs = Debug.get_comm_classified_data(csv_f, plot_idle=plot_idle)
        configs_estimated, debug_parts = self.order_space_test_comm_classified(
            configs, order_by=2
        )

        if output_path is not None:
            Debug.plot_vs_real_comm_classified(
                configs_estimated,
                csv_f,
                output_path,
                debug_parts,
                title=self.plot_title(),
                plot_idle=plot_idle,
            )

        return Debug.correlation_with_classified_comms(configs_estimated)


class ParallelizeMultiModal(ParallelizeLayer):
    """Parallelize a MultiModel"""

    def __init__(
        self,
        evaluator: Any,
        machine: Any,
        global_batch_size: Any = None,
        dimensions: Any = None,
        **extra_config: Any,
    ) -> None:
        """Drive the search from the submodule the parser marked as main."""
        super().__init__(
            evaluator,
            machine,
            global_batch_size=global_batch_size,
            dimensions=dimensions,
            sub_model=getattr(
                evaluator.ccfg, "mm_main", None
            ) or evaluator.ccfg.mm_order[-1],
            **extra_config,
        )


class Parallelize:  # pylint: disable=R0903
    """Main class instantiated by one of the above two"""

    def __init__(self, framework: Any, config: Any, machine: Any, **extra_config: Any) -> None:
        """Dispatch to the unimodal or multimodal search driver."""
        logger.debug("before evaluator init")
        if "model" in extra_config:
            model_name = extra_config.pop("model")
            mem_eval = EvaluatorV2(
                config, framework=framework, hook_cls=model_name, machine=machine
            )
        else:
            mem_eval = EvaluatorV2(config, framework=framework, machine=machine)

        if "global_batch_size" in extra_config:
            global_batch_size = extra_config.pop("global_batch_size")
        else:
            global_batch_size = None

        if "dimensions" in extra_config:
            dimensions = extra_config.pop("dimensions")
        else:
            dimensions = None

        if mem_eval.ccfg.multimodal:
            logger.debug("MultiModal is triggered")
            self.instance = ParallelizeMultiModal(
                mem_eval,
                machine,
                global_batch_size=global_batch_size,
                dimensions=dimensions,
                **extra_config,
            )
        else:
            self.instance = ParallelizeLayer(
                mem_eval,
                machine,
                global_batch_size=global_batch_size,
                dimensions=dimensions,
                sub_model=None,
                **extra_config,
            )

    def __getattr__(self, name):
        return self.instance.__getattribute__(name)


def space_to_string(space: Any, max_num: Any = None, debug_parts: Any = None) -> str:
    """Space printer"""
    i = 0
    s = ""
    if max_num is not None:
        s += "Top " + str(max_num) + " configurations:\n"
    else:
        s += "\n"
    if len(space) == 0:
        return s
    s += "\t"
    for d in space[0][0].all_dims:
        s += str(d) + " " * (6 - len(str(d)))
    s += "Memory    Performance score  "
    if debug_parts is not None:
        for dbg_part in debug_parts:
            s += "\t" + dbg_part.short_name()
    s += "\n"
    for config in space:
        if max_num is not None and max_num == i:
            break
        s += "\t"
        for v in config[0].values():
            s += v + " " * (6 - len(v))
        s += str(config[1]) + " MB  "  # + str(config[2])
        s += f"{(config[2]):16.12e}"
        for v in config[3]:
            s += f"\t{(100*v/config[2]):.2f}%"
        s += "\n"
        i += 1
    return s


def pool_estimate_memory(config: CostModelConfig) -> float:
    """Calls memory estimation for multiprocessing"""
    logger.debug("estimate_peak")
    # print("estimate_peak")
    e = EvaluatorV2(None, ccfg=config)
    return e.estimate_peak()


# def pool_estimate_memory(evaluator):
#     """Calls memory estimation for multiprocessing"""
#     logger.debug("estimate_peak")
#     return evaluator.estimate_peak()


def pool_estimate_performance(
    config: CostModelConfig,
    device: Hard.Type,
    memory: Optional[float] = None,
    cache_file: Optional[str] = None,
    stage_savings: Optional[Tuple[float, ...]] = None,
) -> float:
    """Calls performance estimation for multiprocessing"""
    return estimate_performance(
        config,
        device_type=device,
        memory=memory,
        cache_file=cache_file,
        stage_savings=stage_savings,
    )
