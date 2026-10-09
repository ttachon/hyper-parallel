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

from collections import Counter
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
    unset_reads_report,
)
from hyper_parallel.auto_parallel.sapp_nd.nd.recompute_dimension import (
    HYPER_FRAMEWORKS,
    RECOMPUTE_MODES,
    parsed_recompute,
    restore_recompute,
    state_recompute_mode,
)

# logger = proc.log_to_stderr()
# logger.setLevel(proc.SUBDEBUG)


class ParallelizeLayer:
    """Parallelize one layer type"""

    # No recompute dimension, no mode stated and no recompute chosen per
    # candidate, also for an instance a test builds without __init__.
    recompute_modes = None
    recompute_dimension = None
    _stated_mode = None
    auto_recompute = False

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
        # A recompute dimension: the activation checkpoint modes every
        # candidate is priced under, each pair ranked on its own; None prices
        # each candidate with the recompute it derives, as without one.
        self.recompute_dimension = extra_config.pop("recompute_dimension", None)
        if self.recompute_dimension is not None:
            unknown = sorted(set(self.recompute_dimension) - set(RECOMPUTE_MODES))
            if unknown or not self.recompute_dimension:
                raise ValueError(f"recompute_dimension {list(self.recompute_dimension)}: expected some of "
                                 f"{', '.join(RECOMPUTE_MODES)}")
            if manual_ppb:
                raise ValueError("recompute_dimension prices every candidate under each mode, and mppb takes "
                                 "the recompute from the config: give one of them")
            if self.auto_recompute:
                raise ValueError("recompute_dimension ranks every candidate under each mode, and "
                                 "auto_recompute chooses one for each: give one of them")

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
        # What a mode of the recompute dimension replaces, for when no mode is
        # stated, and the mode stated now: None until one is.
        self._parsed_recompute = parsed_recompute(self.mem_eval.ccfg)
        self._stated_mode = None

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
        self._report_bounds(vpp)

    def _report_bounds(self, vpp: int) -> None:
        """Name the bound each searched dimension takes, and why.

        A degree over its bound is never generated, so no check refuses it
        and no estimate drops it: this line is the only trace it leaves.
        """
        heads = "KV heads" if self.config.ccfg.n_kv else "attention heads' largest power of two"
        bounds = (
            (Dim.PP, f"the machine's {self.machine.pipeline_bound()}, {self.config.total_layer_num()} layers "
                     f"over VPP {vpp}, a batch of {self.global_batch_size}"),
            (Dim.EP, "the model's experts"),
            (Dim.TP, f"the model's {heads}"),
        )
        named = [f"{dim} up to {dim.get_bound()} ({why})" for dim, why in bounds if dim in self.config.dimensions]
        if named:
            logger.output("The search bounds %s", "; ".join(named))

    @staticmethod
    def filtered_out(_: Any) -> bool:
        """Manual conditions to remove config patterns"""
        # if parallel_config.has_dim(Dim.EP):
        #     if self.config.dim_val(Dim.EP, parallel_config) < 8:
        #         return True
        return False

    def _refuse(self, reason: str) -> bool:
        """Count a candidate refused for *reason*, for the search's summary, and return False."""
        self.__dict__.setdefault("refused", Counter())[reason] += 1
        return False

    def is_valid(self, parallel_config: Any) -> bool:
        """Check configuration validity"""
        if not parallel_config.is_valid():
            logger.warning("configuration %s not valid", str(parallel_config))
            return self._refuse("a degree out of bounds")
        if not self.config.moe_valid(parallel_config):
            logger.warning("expert parallel is higher than expert number")
            return self._refuse("EP over the experts or DP x TP")
        if hasattr(self.config, 'ep_constraints_valid') and not self.config.ep_constraints_valid(parallel_config):
            logger.warning("EP divisibility constraints not satisfied")
            return self._refuse("EP divisibility")
        if self.filtered_out(parallel_config):
            logger.warning("Config manually filtered out")
            return self._refuse("filtered out")

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
                    return self._refuse("CP constraints")

                if cp_result.warning_message:
                    logger.info("CP warning: %s", cp_result.warning_message)

        gbs = self.config.global_batch_size(parallel_config)
        if not gbs == self.global_batch_size:
            logger.error(
                "wrong global batch size: ccfg is %d, instead of %d",
                gbs,
                self.global_batch_size,
            )
            return self._refuse("global batch")
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

    def _drop(self, config: Any, peak: float) -> None:
        """Say that the memory budget drops *config*, and by how much its peak misses it."""
        budget = self.mem_eval.get_max_device_memory()
        logger.output(
            "%s dropped: its peak of %.0f MB is over the %.0f MB budget by %.0f MB",
            config,
            peak,
            budget,
            peak - budget,
        )

    def _summarise_search(self, size: int, priced: int, fitting: int) -> None:
        """The search's two counts: what was tested, refused and priced, and what fits."""
        refused = self.__dict__.get("refused", Counter())
        reasons = ", ".join(f"{reason} {count}" for reason, count in refused.most_common())
        logger.output(
            "%d configurations tested: %d refused%s, %d priced for memory",
            size,
            sum(refused.values()),
            f" ({reasons})" if reasons else "",
            priced,
        )
        logger.output(
            "%d configuration fitting memory to order, %d dropped over the memory budget",
            fitting,
            priced - fitting,
        )

    def generate_search_space(self, folder: Any, threads_num: Any) -> Any:
        """Return a search space computed with memory estimation.

        A candidate the memory budget drops is named with its peak, and the
        summary counts the candidates the checks refused apart from those the
        budget dropped, each check by name.
        """
        # With a pool the accumulator holds AsyncResult, which the direct
        # branch's float values hide from static inference.
        # pylint: disable=no-member
        space = ({}, 0)
        configs = []
        results = {}
        self.refused = Counter()
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
                    else:
                        self._drop(config, peak_mem)
                pool.close()
                pool.join()
        else:
            results, size = self.device_loops(space, None)
            for config, peak_mem in results.items():
                if self.mem_eval.mem_fit(peak_mem):
                    configs.append((config, peak_mem))
                    if folder:
                        self.config.write(folder, config)
                else:
                    self._drop(config, peak_mem)
        self._summarise_search(size, len(results), len(configs))

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

    def batch_reachable(self) -> bool:
        """Whether any candidate the loops build makes the global batch; one line says so where none does.

        Each candidate's batch is its DP x MB x MBS, MB the batch over DP and
        MBS, so a batch that is no whole number of micro-batches of any DP x
        MBS the loops reach is refused in every candidate, each with an error.
        The search stops before building one, with this line instead.
        """
        number = self.machine.number
        reached = set()
        for tp in self.config.space(Dim.TP, number):
            for pp in self.config.space(Dim.PP, number // tp):
                for cp in self.config.space(Dim.CP, number // tp // pp):
                    dp = number // tp // cp // pp
                    if dp < 1:
                        break
                    for mbs in self.config.space(Dim.MBS, self.global_batch_size // pp // dp):
                        mbn = self.global_batch_size // dp // mbs
                        candidate = self.config.make_parallel_config((dp, tp, pp, cp), (mbs, mbn), (1, 1, 1, False))
                        if self.config.global_batch_size(candidate) == self.global_batch_size:
                            return True
                        reached.add(self.config.dim_val(Dim.DP, candidate) * mbs)
        logger.error(
            "No candidate makes a global batch of %d (the yaml states %d): the candidates the search builds "
            "have DP x MBS of %s, and %d is no whole number of micro-batches of any. State -b as a multiple "
            "of one of them, or let DP shrink by naming PP, CP or MP in -l.",
            self.global_batch_size,
            self.config.ccfg.gbs,
            ", ".join(str(batch) for batch in sorted(reached)),
            self.global_batch_size,
        )
        return False

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
        """Order the given space with performance estimation.

        A measured configuration the cost model cannot represent, such as
        expert parallelism wider than DP x TP, is left out and named, so that
        it does not end the comparison of every other one.
        """
        scored_space = []
        debug_parts = []
        for config, real_time, real_comm_wait in space:
            # The comparison needs the per-part split at any verbosity; debug.csv stays opt-in.
            debugger = Debug.Debug(config, info_type=Debug.PerfParts, enable=True)
            try:
                self.set_recompute_mode(self._measured_mode(config))
                self.config.set_parallel_config(config)
                peak_mem = self.memory_estim()
                score = estimate_performance(
                    self.priced(),
                    debugger=debugger,
                    device_type=self.machine.device,
                    stage_focused=0,
                )  # , memory = mem)
            except (TypeError, ValueError, ZeroDivisionError) as exc:
                logger.output("ND cannot cost %s, left out of the comparison: %s", config, exc)
                continue
            if self.enable_debug:
                debugger.write()
            debug_parts = list(debugger.info.keys())
            values = list(debugger.info.values())
            del values[-2:]
            scored_space.append(
                (config, peak_mem, real_time, score, values, real_comm_wait)
            )

            logger.info("config %s has score %f", str(config), score)
        self.set_recompute_mode(None)
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
            self.set_recompute_mode(self._measured_mode(config))
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
        self.set_recompute_mode(None)
        del debug_parts[-2:]
        return (sorted(scored_space, key=lambda x: x[order_by]), debug_parts)

    def plot_title(self) -> str:
        """Generate plot title"""
        return (
            f"{self.model_name} on {self.machine.number}"
            + f" {self.machine.device} with {self.global_batch_size} GBS"
        )

    def set_recompute_mode(self, mode: Optional[str]) -> None:
        """Price what follows under one activation checkpoint mode, or None for the config's own recompute.

        Nothing changes when the mode is the one already stated, so a search
        without a recompute dimension never touches the config's recompute.

        Args:
            mode: One of the recompute dimension's modes, or None.
        """
        if mode == self._stated_mode:
            return
        self._stated_mode = mode
        if mode is None:
            restore_recompute(self._parsed_recompute)
            self.config.state_recompute(None)
            return
        state_recompute_mode(self.mem_eval.ccfg, mode)
        self.config.state_recompute(mode == "full")

    def _measured_mode(self, config: Any) -> Optional[str]:
        """The mode a measured configuration ran: its own, else the dimension's only one, else None."""
        mode = getattr(config, "recompute", None)
        if mode is None and self.recompute_dimension is not None and len(self.recompute_dimension) == 1:
            mode = self.recompute_dimension[0]
        return mode

    def _search_and_order(self, yaml_folder: Any, threads_num: Any, cache_file: Any) -> Tuple[list, list, float, float]:
        """Generate and order the space once, or once per mode of a recompute dimension.

        Each mode's pass prices and checks every candidate under that mode and
        tags it with the mode; the passes are then ranked together, so a mode
        that does not fit a candidate only removes that pair.

        Returns:
            The ordered entries, the parts of their scores, and the seconds
            generation and ordering took.
        """
        scored_space, dbg, generation, ordering = [], [], 0.0, 0.0
        if not self.batch_reachable():
            return scored_space, dbg, generation, ordering
        for mode in self.recompute_dimension or (None,):
            if mode is not None:
                logger.output("Search with recompute %s", mode)
                self.set_recompute_mode(mode)
            start = time.time()
            space = self.generate_search_space(yaml_folder, threads_num)
            generated = time.time()
            scored, parts = self.order_search_space(space, threads_num, cache_file=cache_file)
            generation += generated - start
            ordering += time.time() - generated
            dbg = parts or dbg
            for entry in scored:
                entry[0].recompute = mode
            scored_space += scored
        if self.recompute_dimension is not None:
            self.set_recompute_mode(None)
            scored_space.sort(key=lambda entry: entry[2])
        return scored_space, dbg, generation, ordering

    def run_generation_to_ordering(
        self,
        yaml_folder: Any,
        threads_num: Any = None,
        top_num: Any = None,
        cache_file: Any = None,
        ranking_csv: Optional[str] = None,
        exhaustive: int = 0,
        force_exhaustive: int = 0,
        fforce_exhaustive: int = 0,
        dimensions: Optional[list] = None,
    ) -> Any:
        """Search, order and print the configurations that fit memory.

        ``ranking_csv``, when given, receives every one of them in ND's order.
        It is written before anything is plotted, so a plot that fails cannot
        take the ranking with it. ``exhaustive`` sets the minimum number of
        plotted results with each requested dimension above degree one.
        When ``force_exhaustive >= exhaustive``, append that many qualifying
        results per dimension. Otherwise, append at least ``force_exhaustive``
        and count those additions toward the ``exhaustive`` minimum.
        ``fforce_exhaustive`` takes precedence and reserves distinct additions
        for each requested dimension.
        """
        scored_space, dbg, generation, ordering = self._search_and_order(
            yaml_folder, threads_num, cache_file
        )
        for rank, entry in enumerate(scored_space, start=1):
            entry[0].rank = rank
        if ranking_csv:
            Debug.write_ranking_csv(scored_space, ranking_csv)
            logger.output(
                "ND's order of %d configuration(s) written to %s",
                len(scored_space),
                ranking_csv,
            )

        plot_space = scored_space
        exhaustive_additions = []
        has_exhaustive = (
            exhaustive > 0 or force_exhaustive > 0 or fforce_exhaustive > 0
        )
        if fforce_exhaustive > 0:
            plot_space, exhaustive_additions = _fforce_exhaustive_plot_space(
                scored_space,
                dimensions if dimensions is not None else self.config.dimensions,
                top_num,
                fforce_exhaustive,
            )
        elif exhaustive > 0 or force_exhaustive > 0:
            plot_space, exhaustive_additions = _exhaustive_plot_space(
                scored_space,
                dimensions if dimensions is not None else self.config.dimensions,
                top_num,
                exhaustive,
                force_exhaustive,
            )

        logger.output(
            space_to_string(scored_space, max_num=top_num, debug_parts=dbg)
        )
        if exhaustive_additions:
            if top_num is not None:
                printed_ids = {
                    config[0].unique_name()
                    for config in scored_space[:max(0, top_num)]
                }
                additional_to_print = [
                    config
                    for config in exhaustive_additions
                    if config[0].unique_name() not in printed_ids
                ]
                if additional_to_print:
                    logger.output(
                        space_to_string(
                            additional_to_print,
                            max_num=len(additional_to_print),
                            debug_parts=dbg,
                            heading="Additional exhaustive configurations",
                        )
                    )
            logger.output(
                "Exhaustive plot additions: %d unique configuration(s)",
                len(exhaustive_additions),
            )
        priced = self.priced()
        if not getattr(priced, "seq_len_stated", True):
            # A warning at parse time scrolls past a search's lines; this one
            # sits under the table it qualifies (I11).
            logger.output(
                "Every number above is priced at %d tokens, the model's context limit: the config states no "
                "training sequence length (dataset.data_transform.max_seq_len or dataset.data_config.seq_length)",
                priced.s,
            )
        for line in unset_reads_report():
            logger.output(line)
        logger.output(
            "Space generation took %.2fs and ordering took %.2fs",
            generation,
            ordering,
        )
        if self.recompute_dimension is None:
            is_not = " NOT" if not self.config.balancing.from_config else ""
            logger.output(
                "Offset & Recompute were%s computed from config info", is_not
            )
        else:
            logger.output(
                "Offset was NOT computed from config info; recompute was searched over %s",
                ", ".join(self.recompute_dimension),
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
            if plot_space:
                Debug.plot_nd(
                    plot_space,
                    output_path,
                    dbg,
                    title=self.plot_title(),
                    max_num=(
                        len(plot_space) if has_exhaustive else top_num
                    ),
                    include_all=has_exhaustive,
                    top_result_count=(
                        len(plot_space) - len(exhaustive_additions)
                        if exhaustive_additions else None
                    ),
                    recompute_plans=(
                        {
                            entry[0]: trainer_plan(self.recompute_choices[entry[0]])
                            for entry in plot_space
                            if entry[0] in self.recompute_choices
                        }
                        if getattr(self, "recompute_mode_per_layer", False) else None
                    ),
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
        self.set_recompute_mode(getattr(parallel_config, "recompute", None))
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
        return self.compare_with_csv(csv_f, output_path=output_path, plot_idle=plot_idle)[1]

    def compare_with_csv(
        self, csv_f: str, output_path: Optional[str] = None, plot_idle: bool = False
    ) -> Tuple[list, Any]:
        """Estimate every measured configuration of a classified profiling CSV.

        With ``output_path``, three files named after the CSV go there: the
        real-versus-estimate plot, ``<stem>.pdf``; with ``plot_idle``, the same
        plot without idle, ``<stem>_no_idle.pdf``, ordered by the measured step
        less idle; and ND's estimate of each configuration, memory included,
        ``<stem>_estimates.csv``. Idle is half the step or more on a short
        one and differs from run to run, while ND estimates none of it.

        Args:
            csv_f: CSV read by ``Debug.get_comm_classified_data``, e.g. written by
                ``nd.trace_classify``.
            output_path: Directory for ND's plots and estimates; none are written when None.
            plot_idle: Whether the measured bars include the idle remainder.

        Returns:
            ``(configs_estimated, metrics)``: one ``(config, peak_mem, real_time, score,
            values, real_parts)`` entry per measured configuration, ordered by measured
            time, and ``Debug.correlation_with_classified_comms`` over them.
        """
        configs = Debug.get_comm_classified_data(csv_f, plot_idle=plot_idle)
        configs_estimated, debug_parts = self.order_space_test_comm_classified(
            configs, order_by=2
        )
        if not configs_estimated:
            raise ValueError(f"ND cannot cost any configuration of {csv_f}")

        if output_path is not None:
            title = self.plot_title()
            Debug.plot_vs_real_comm_classified(
                configs_estimated,
                csv_f,
                output_path,
                debug_parts,
                title=title,
                plot_idle=plot_idle,
            )
            if plot_idle:
                Debug.plot_vs_real_comm_classified(
                    sorted(configs_estimated, key=Debug.busy_time),
                    csv_f,
                    output_path,
                    debug_parts,
                    title=title,
                    plot_idle=False,
                    suffix="_no_idle",
                )
            stem = os.path.splitext(os.path.basename(csv_f))[0]
            estimates = os.path.join(output_path, f"{stem}_estimates.csv")
            Debug.write_estimates_csv(configs_estimated, estimates)
            logger.output("ND's estimate of every configuration written to %s", estimates)

        return configs_estimated, Debug.correlation_with_classified_comms(configs_estimated)


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
        if extra_config.get("recompute_dimension") is not None and framework not in HYPER_FRAMEWORKS:
            raise ValueError(
                f"recompute_dimension states HyperParallel's activation checkpoint modes, and the {framework} "
                f"parser states its own recompute: use one of {', '.join(HYPER_FRAMEWORKS)}"
            )
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


def space_to_string(
    space: Any,
    max_num: Any = None,
    debug_parts: Any = None,
    heading: Optional[str] = None,
) -> str:
    """Format ranked configurations for CLI output.

    Args:
        space: Configurations with memory, performance score and score components.
            TOP uses the rank stored on each configuration, falling back to its
            position in the supplied list when it has not been ranked.
        max_num: Maximum number of configurations to print.
        debug_parts: Performance components to name in the header.
        heading: Optional heading replacing the default top-results heading.
    """
    s = ""
    if heading is not None:
        s += heading + ":\n"
    elif max_num is not None:
        s += "Top " + str(max_num) + " configurations:\n"
    else:
        s += "\n"
    if len(space) == 0:
        return s
    # A search with a recompute dimension names each entry's mode after its degrees.
    moded = any(getattr(entry[0], "recompute", None) for entry in space)
    s += "\tTOP   "
    for d in space[0][0].all_dims:
        s += str(d) + " " * (6 - len(str(d)))
    if moded:
        s += "Recompute  "
    s += "Memory    Performance score  "
    if debug_parts is not None:
        for dbg_part in debug_parts:
            s += "\t" + dbg_part.short_name()
    s += "\n"
    for i, config in enumerate(space):
        if max_num is not None and max_num == i:
            break
        s += f"\t{config[0].rank or i + 1:<6}"
        for v in config[0].values():
            s += v + " " * (6 - len(v))
        if moded:
            s += f"{config[0].recompute:<11s}"
        s += str(config[1]) + " MB  "  # + str(config[2])
        s += f"{(config[2]):16.12e}"
        for v in config[3]:
            s += f"\t{(100*v/config[2]):.2f}%"
        s += "\n"
    return s


def _exhaustive_plot_space(
    scored_space: list,
    dimensions: list,
    top_num: Optional[int],
    exhaustive: int,
    force_exhaustive: int,
) -> tuple:
    """Add ranked configs to the normal plot to meet per-dimension counts."""
    normal_plot = Debug.top_plot_configs(scored_space, top_num)
    normal_ids = {entry[0].unique_name() for entry in normal_plot}
    selected_ids = set(normal_ids)

    for dimension in dimensions:
        present = sum(
            1
            for entry in normal_plot
            if entry[0].has_dim(dimension) and entry[0].val(dimension) > 1
        )
        if force_exhaustive >= exhaustive:
            needed = force_exhaustive
        else:
            needed = max(force_exhaustive, exhaustive - present)
        added = 0
        if needed == 0:
            continue
        for entry in scored_space:
            config_id = entry[0].unique_name()
            if config_id in normal_ids:
                continue
            if entry[0].has_dim(dimension) and entry[0].val(dimension) > 1:
                selected_ids.add(config_id)
                added += 1
                if added >= needed:
                    break

    plot_space = [
        entry for entry in scored_space if entry[0].unique_name() in selected_ids
    ]
    additions = [
        entry for entry in plot_space if entry[0].unique_name() not in normal_ids
    ]
    return plot_space, additions


def _fforce_exhaustive_plot_space(
    scored_space: list,
    dimensions: list,
    top_num: Optional[int],
    fforce_exhaustive: int,
) -> tuple:
    """Add distinct ranked configs for each dimension to the normal plot."""
    normal_plot = Debug.top_plot_configs(scored_space, top_num)
    selected_ids = {entry[0].unique_name() for entry in normal_plot}
    added_by_dimension = {}

    for dimension in dimensions:
        added = 0
        for entry in scored_space:
            config_id = entry[0].unique_name()
            if config_id in selected_ids:
                continue
            if entry[0].has_dim(dimension) and entry[0].val(dimension) > 1:
                selected_ids.add(config_id)
                added += 1
                if added >= fforce_exhaustive:
                    break
        added_by_dimension[dimension] = added

    for dimension, added in added_by_dimension.items():
        if added < fforce_exhaustive:
            logger.warning(
                "Only %d of %d unique exhaustive result(s) found for dimension %s",
                added,
                fforce_exhaustive,
                dimension,
            )

    plot_space = [
        entry for entry in scored_space if entry[0].unique_name() in selected_ids
    ]
    normal_ids = {entry[0].unique_name() for entry in normal_plot}
    additions = [
        entry for entry in plot_space if entry[0].unique_name() not in normal_ids
    ]
    return plot_space, additions


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
