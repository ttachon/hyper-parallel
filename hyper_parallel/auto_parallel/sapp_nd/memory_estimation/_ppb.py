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
"""PPB input module"""
from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Tuple, TYPE_CHECKING

from hyper_parallel.auto_parallel._model_spec import KindActivations
from hyper_parallel.auto_parallel.sapp_nd.nd.common.config import Config
from hyper_parallel.auto_parallel.sapp_nd.nd.common.derive import HYPER_SELECTIVE_REC_OP
from hyper_parallel.auto_parallel.sapp_nd.nd.common.layer_type import LayerType
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation._context import Context
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.body import EvalBody
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.utils import EvalUtils
from hyper_parallel.auto_parallel.sapp_nd.recompute.profile import OPTIONAL, SWITCHES, Cost, SwitchProfile

if TYPE_CHECKING:
    from hyper_parallel.auto_parallel._op_profiles import LayerKind
    from hyper_parallel.auto_parallel.sapp_nd.nd.common.cost_model_preprocess import CostModelConfig

# The balancer's recompute options, in the order that settles a tie between two.
_OPTIONS = ("NONE", "SLCT", "COMM", "BOTH", "FULL")
# Where the balancer reads each option's memory per micro-batch and its backward time.
_MEMORY_KEY = {
    "NONE": "memory_activation",
    "SLCT": "memory_select_rec",
    "COMM": "memory_select_comm",
    "BOTH": "memory_both_comm_select",
    "FULL": "memory_recompute",
}
_TIME_KEY = {
    "NONE": "backward_time",
    "SLCT": "select_rec_time",
    "COMM": "select_comm_time",
    "BOTH": "both_comm_select_time",
    "FULL": "recompute_time",
}


def _layer_switches(ccfg: CostModelConfig, ctx: Optional[Context]) -> Dict[str, Any]:
    """The switches of the layer *ctx* evaluates, its own where the context carries them, else the config's."""
    stated = EvalUtils.switches(ccfg, ctx)
    return dict(stated) if isinstance(stated, Mapping) else dict(vars(stated))


class _PPB:
    """Pipeline balance payload builder."""

    def __init__(self, eval_cfg: Config, inner_dyn_fun: Callable) -> None:
        """Initialize _PPB with evaluation config and dynamic memory function.

        Args:
            eval_cfg: Evaluation configuration object.
            inner_dyn_fun: Function to compute inner dynamic memory.
        """
        self.eval_cfg = eval_cfg
        self._inner_dynamic_mem = inner_dyn_fun
        self.mb = EvalUtils.mb
        # Prices a layer as ``(forward, backward)`` from its config, kind,
        # layer type and recompute switches; set while a description with
        # times is being built.
        self.layer_times: Optional[Callable] = None
        # Each model's and layer kind's switch profile; set while profiles
        # are being measured.
        self.profiles: Optional[Dict[tuple, SwitchProfile]] = None
        # The most micro-batches any stage keeps in flight, where profiles
        # split the buffers when set.
        self.profile_in_flight: Optional[int] = None
        # The counts of micro-batches in flight the stages keep, at which
        # profiles state what the split charges beyond the buffers.
        self.profile_counts: Tuple[int, ...] = ()
        # Whether profiles measure each switch alone, or only the plain and
        # the fully recomputed layer.
        self.profile_each_switch = True

    @staticmethod
    def add_to_ppb_list(ppb_lay_desc: list, desc: dict) -> None:
        """layer description list preparation"""
        if desc:
            already_comp = False
            body_idx = 0
            for d in ppb_lay_desc:
                if all(v == d[k] for k, v in desc.items()):
                    # already exist desc
                    d["nb_layer"] += 1
                    already_comp = True
                if d["type"] == "BODY":
                    body_idx += 1
            if desc and not already_comp:
                desc["nb_layer"] = 1
                if desc["type"] == "BODY":
                    desc["name"] = f"BODY_{body_idx}"
                else:
                    desc["name"] = desc["type"]
                ppb_lay_desc += [desc]

    @staticmethod
    def selective_switches(ccfg: CostModelConfig, ctx: Optional[Context] = None) -> Dict[str, Dict[str, int]]:
        """The recompute switches of each selective option.

        SLCT is the selective recompute the layer runs, its own where *ctx*
        carries them, else the config's; COMM recomputes the tensor-parallel
        gathers alone, and BOTH does both.
        """
        rec_op = _layer_switches(ccfg, ctx)
        keep = dict.fromkeys(SWITCHES, 1)
        configured = {name: int(bool(rec_op.get(name, 1))) for name in SWITCHES}
        return {"SLCT": configured, "COMM": dict(keep, gather=0), "BOTH": dict(configured, gather=0)}

    def lay_ppb(
        self, ccfg: CostModelConfig, ctx: Context, res_stat: float, kind: Optional[LayerKind] = None
    ) -> dict:
        """layer description preparation

        With ``layer_times`` set, a body also offers COMM and BOTH, and the
        description carries the layer's times, priced on *kind*, the
        layer's kind. While ``profiles`` are measured, a body is profiled
        instead and no layer is described.
        """
        desc = {"model_name": ccfg.model_name}
        timed = self.layer_times is not None
        original_enable_node_log = ctx.enable_node_log
        ctx.enable_node_log = False
        try:
            if self.profiles is not None:
                if ctx.current_node == ctx.head_node:
                    # The pricer copies a config where it first sees it, which
                    # must come before the walk gives any layer its kind.
                    self.layer_times(ccfg, None, LayerType.EMBEDDING_LAYER)
                elif ctx.current_node != ctx.tail_node:
                    self._profile(ccfg, ctx, kind)
                return {}
            if ctx.current_node == ctx.head_node:
                d_emb = self.mb(sum(self._inner_dynamic_mem(ppb=True)))
                desc["type"] = "HEAD"
                desc["memory_parameter"] = self.mb(res_stat) + d_emb
                desc["time"] = 1
            elif ctx.current_node == ctx.tail_node:
                d_out = self.mb(sum(self._inner_dynamic_mem(ppb=True)))
                desc["type"] = "TAIL"
                desc["memory_parameter"] = self.mb(res_stat) + d_out
                desc["time"] = 1
            else:
                self._body_memory(desc, ccfg, ctx, res_stat, timed)
        finally:
            ctx.enable_node_log = original_enable_node_log
        if timed:
            self._time_ppb(desc, ccfg, ctx, kind)
        return desc

    def _body_memory(self, desc: dict, ccfg: CostModelConfig, ctx: Context, res_stat: float, timed: bool) -> None:
        """Describe a body layer's memory under each option.

        The balancer charges an option's memory once per micro-batch in
        flight, and the layer's parameter memory once. Activations are kept
        per micro-batch. Of the communication buffers, what grows with the
        micro-batches in flight, such as the tensor-parallel gathers COMM
        recomputes, is charged per micro-batch, and the rest, the largest
        any option keeps, once.
        """
        many = self._many(ccfg)
        ctx.current_node = LayerType.NOT_REC_LAYER
        dyn = {"NONE": self._dynamic_mem(many)}
        ctx.current_node = LayerType.SEL_REC_LAYER
        dyn["SLCT"] = self._dynamic_mem(many)
        if timed:
            switches = self.selective_switches(ccfg, ctx)
            for name in ("COMM", "BOTH"):
                dyn[name] = self._selective(ccfg, ctx, switches[name], lambda _: self._dynamic_mem(many))
        ctx.current_node = LayerType.FULL_REC_LAYER
        dyn["FULL"] = self._dynamic_mem(many)
        desc["type"] = "BODY"
        desc["memory_parameter"] = self.mb(res_stat) + self.mb(max(once for _, _, once, _ in dyn.values()))
        for name in _OPTIONS:
            if name in dyn:
                activation, per_micro_batch, _, _ = dyn[name]
                desc[_MEMORY_KEY[name]] = self.mb(activation) + self.mb(per_micro_batch)
        desc["time"] = 1

    @staticmethod
    def _many(ccfg: CostModelConfig) -> int:
        """The most micro-batches a stage keeps in flight under 1F1B, and at least 2."""
        return max(2, min(getattr(ccfg, "p", 1), getattr(ccfg, "m", 1)))

    def _profile(self, ccfg: CostModelConfig, ctx: Context, kind: Optional[LayerKind]) -> None:
        """Measure the layer plain, with each op alone recomputed and fully recomputed, once per model and kind.

        Where a census prices the kind, it prices the plain layer and
        HyperParallel's selective policy, and its records per op, or else
        the formulas, every other setting: the layer selective with every
        op kept is measured too, as the base the settings add up from, and
        the policy's setting whole.
        """
        key = (ccfg.model_name, kind)
        if key in self.profiles:
            return
        many = max(2, self.profile_in_flight) if self.profile_in_flight else self._many(ccfg)
        # The split is exact at one micro-batch in flight and at *many*.
        counts = tuple(count for count in self.profile_counts if 1 < count < many)
        keep = dict.fromkeys(SWITCHES, 1)
        census = isinstance(getattr(ccfg, "kind_activations", None), KindActivations)

        def _measure(at: Context) -> Tuple[Any, ...]:
            """The current layer's memory, the working sets of its backward at both points, and their census terms."""
            return self._dynamic_mem(many, counts) + self._workings(at) + (EvalBody.census_working_terms(ccfg, at),)

        ctx.current_node = LayerType.NOT_REC_LAYER
        plain = _measure(ctx)
        # A fully recomputed layer's backward runs the plain layer, on
        # activations the stage does not keep; without a census, the working
        # set is the same either way.
        full_working = self._working_extra(ctx, 2, on_saved=False) if census else plain[4]
        base = self._selective(ccfg, ctx, keep, _measure) if census else plain
        alone = self._alone(ccfg, ctx, base, _measure, many) if self.profile_each_switch else {}
        policy = dict(HYPER_SELECTIVE_REC_OP)
        whole = {frozenset(name for name, state in policy.items() if not state): self._selective(
            ccfg, ctx, policy, _measure)} if census else {}
        ctx.current_node = LayerType.FULL_REC_LAYER
        full = self._dynamic_mem(many, counts) + (full_working, plain[5], None)

        def _cost(memory: Tuple[Any, ...], backward: float) -> Cost:
            """A measurement as a cost: activations and growing buffers per micro-batch, the rest once.

            Where a census's records per op price the layer, its working set
            as warm-up ends before the clamp, and what it keeps of them.
            """
            activation, per_micro_batch, once, excess, working, first_working, terms = memory
            kept = 0.0
            if terms is not None:
                kept, held = terms
                working -= max(0.0, kept - held)
            return Cost(activation + per_micro_batch, once, backward, excess, working, first_working, kept)

        forward, backward = self.layer_times(ccfg, kind, LayerType.NOT_REC_LAYER)
        self.profiles[key] = SwitchProfile(
            forward_time=forward,
            plain=_cost(plain, backward),
            alone=self._alone_costs(ccfg, kind, alone, _cost, _cost(base, backward)),
            full=_cost(full, self.layer_times(ccfg, kind, LayerType.FULL_REC_LAYER)[1]),
            counts=counts,
            # A selective layer that keeps every op recomputes nothing.
            selective_base=_cost(base, backward) if census else None,
            census_held=self._census_held(base),
            whole={
                recompute: _cost(memory, self.layer_times(ccfg, kind, LayerType.SEL_REC_LAYER, policy)[1])
                for recompute, memory in whole.items()
            },
        )

    def _alone_costs(
        self, ccfg: CostModelConfig, kind: Optional[LayerKind], alone: Dict[str, Tuple[Any, ...]],
        cost: Callable, base: Cost,
    ) -> Dict[str, Cost]:
        """Each switch's cost, its op alone recomputed; an optional one the layer keeps nothing under costs *base*."""
        keep = dict.fromkeys(SWITCHES, 1)
        costs = {
            name: cost(memory, self.layer_times(ccfg, kind, LayerType.SEL_REC_LAYER, dict(keep, **{name: 0}))[1])
            for name, memory in alone.items()
        }
        if self.profile_each_switch:
            costs.update({name: base for name in OPTIONAL if name not in costs})
        return costs

    @staticmethod
    def _census_held(measured: Tuple[Any, ...]) -> Optional[float]:
        """What the backward of a layer :meth:`_profile` measured holds, where its census's records per op price it."""
        terms = measured[6]
        return None if terms is None else terms[1]

    def _alone(
        self, ccfg: CostModelConfig, ctx: Context, base: Tuple[Any, ...], measure: Callable, many: int
    ) -> Dict[str, Tuple[Any, ...]]:
        """The current layer measured with each op alone recomputed, by *measure* as *base* was.

        Only gather acts on the buffers, so any other op recomputed alone
        leaves the split, and the working sets, as the base has them; but
        where a census prices the layer, what the layer keeps sets its
        working set as warm-up ends, and each op is measured whole.
        """
        keep = dict.fromkeys(SWITCHES, 1)
        census = isinstance(getattr(ccfg, "kind_activations", None), KindActivations)
        alone = {}
        for name in SWITCHES:
            if name in OPTIONAL and not self._drops_any(ccfg, name):
                continue
            switches = dict(keep, **{name: 0})
            if name == "gather" or census:
                alone[name] = self._selective(ccfg, ctx, switches, measure)
            else:
                measured = self._selective(ccfg, ctx, switches, lambda _: self._dynamic_mem(many))
                alone[name] = measured[:3] + base[3:]
        return alone

    @staticmethod
    def _drops_any(ccfg: CostModelConfig, switch: str) -> bool:
        """Whether the current layer keeps anything an optional *switch* drops.

        attUp, the one optional switch, drops the heads an MLA layer's
        up-projections build, as the records' part of it states: a layer
        that compresses its keys and values has them.
        """
        return switch != "attUp" or bool(getattr(ccfg, "dc_kv", 0))

    def _workings(self, ctx: Context) -> Tuple[float, float]:
        """What the current layer's backward holds beyond what it keeps at one micro-batch, at both points.

        As warm-up ends, where the stage keeps what the layer keeps for this
        backward already, and on a stage's first layer as a micro-batch's
        backward ends, where it keeps none of it.
        """
        return self._working_extra(ctx, 2, on_saved=True), self._working_extra(ctx, 1, on_saved=False)

    def _working_extra(self, ctx: Context, gathered: int, on_saved: bool) -> float:
        """What the working set of the current layer's backward holds beyond what it keeps at one micro-batch.

        Under FSDP that reshards, the backward holds *gathered* layers'
        parameters: two, its own and the next's, in the working set that
        ends warm-up, and one in the first layer's as the backward ends.
        *on_saved* says the stage counts what the layer keeps for this
        backward already, which a census's working set then leaves out.
        """
        kept = sum(self._inner_dynamic_mem(ppb=True))
        ctx.working_set = gathered
        ctx.working_on_saved = on_saved
        try:
            return sum(self._inner_dynamic_mem(default_micro_factor=1)) - kept
        finally:
            ctx.working_set = 0
            ctx.working_on_saved = False

    def _dynamic_mem(
        self, many: int, counts: Tuple[int, ...] = ()
    ) -> Tuple[float, float, float, Tuple[float, ...]]:
        """``(activation, buffers per micro-batch, buffers once, excess)`` of the current layer.

        The buffers are split by how they grow from one micro-batch in
        flight to *many*. At each of *counts*, the excess is what the split
        charges beyond the buffers kept with that many micro-batches in
        flight.
        """
        at = [self._inner_dynamic_mem(default_micro_factor=count)[1] for count in counts]
        _, more = self._inner_dynamic_mem(default_micro_factor=many)
        activation, one = self._inner_dynamic_mem(ppb=True)
        per_micro_batch = (more - one) / (many - 1)
        excess = tuple(one + (count - 1) * per_micro_batch - kept for count, kept in zip(counts, at))
        return activation, per_micro_batch, one - per_micro_batch, excess

    @staticmethod
    def _selective(ccfg: CostModelConfig, ctx: Context, switches: Dict[str, int], measure: Callable) -> Any:
        """*measure* of *ctx*, the current layer selective with *switches*.

        The context carries them to the evaluators, over the layer's own,
        and the config's are left as they are.
        """
        before = ctx.switches
        stated = _layer_switches(ccfg, ctx)
        ctx.current_node = LayerType.SEL_REC_LAYER
        ctx.switches = {**stated, **switches}
        try:
            return measure(ctx)
        finally:
            ctx.switches = before

    def _time_ppb(self, desc: dict, ccfg: CostModelConfig, ctx: Context, kind: Optional[LayerKind]) -> None:
        """Add the layer's forward time and the backward time of each option, where the balancer reads them."""
        end = {"HEAD": LayerType.EMBEDDING_LAYER, "TAIL": LayerType.OUTPUT_LAYER}.get(desc["type"])
        if end is not None:
            desc["forward_time"], desc["backward_time"] = self.layer_times(ccfg, None, end)
        else:
            desc["forward_time"], desc["backward_time"] = self.layer_times(
                ccfg, kind, LayerType.NOT_REC_LAYER
            )
            for name, switches in self.selective_switches(ccfg, ctx).items():
                desc[_TIME_KEY[name]] = self.layer_times(ccfg, kind, LayerType.SEL_REC_LAYER, switches)[1]
            desc["recompute_time"] = self.layer_times(ccfg, kind, LayerType.FULL_REC_LAYER)[1]
        desc["time"] = desc["forward_time"]

    @staticmethod
    def _beaten(desc: dict, name: str) -> bool:
        """Whether another option of *desc* needs no more memory and no more backward time than *name*.

        Of two options that tie, the one first in :data:`_OPTIONS` wins.
        """
        def _cost(option: str) -> tuple:
            """The option's memory and backward time."""
            return desc[_MEMORY_KEY[option]], desc[_TIME_KEY[option]]

        mine = _cost(name)
        return any(
            other != name
            and all(theirs <= own for theirs, own in zip(_cost(other), mine))
            and (_cost(other) != mine or _OPTIONS.index(other) < _OPTIONS.index(name))
            for other in _OPTIONS
            if _MEMORY_KEY[other] in desc
        )

    def ppb_withdraw_dominated(self, ppb_lay_desc: list) -> None:
        """Withdraw the selective options no body is better off with.

        An option is withdrawn from every body when, in each of them, another
        option needs no more memory and no more backward time. The balancer
        takes its options from the first body, so every body keeps the same
        ones. Descriptions without times are left as they are.
        """
        bodies = [d for d in ppb_lay_desc if d["type"] == "BODY"]
        if not bodies or any("backward_time" not in d for d in bodies):
            return
        withdrawn = [name for name in ("SLCT", "COMM", "BOTH") if all(self._beaten(d, name) for d in bodies)]
        for d in bodies:
            for name in withdrawn:
                del d[_MEMORY_KEY[name]], d[_TIME_KEY[name]]

    @staticmethod
    def ppb_scale_times(ppb_lay_desc: list) -> None:
        """Express the times in units of the first body's forward time.

        The balancer only compares times with one another, and its solver
        finds feasible problems infeasible at the estimate's own magnitudes,
        around 1e13. Descriptions without times are left as they are.
        """
        unit = next((d["forward_time"] for d in ppb_lay_desc if d["type"] == "BODY" and d.get("forward_time")), None)
        if not unit:
            return
        for d in ppb_lay_desc:
            for key in d:
                if key == "time" or key.endswith("_time"):
                    d[key] /= unit

    def ppb_combine_bodies(self, ppb_lay_desc: list) -> None:
        """combine descriptions into a new body

        Memories add up over the combined bodies, and so do the times of a
        timed description.
        """
        if not self.eval_cfg.ppb_combined:
            return
        for new_body in self.eval_cfg.ppb_combined:
            desc = {
                "model_name": "combined",
                "type": "BODY",
                "memory_parameter": 0,
                "memory_activation": 0,
                "memory_recompute": 0,
                "time": 1,
                "nb_layer": 1,
                "name": "COMBINED",
            }
            idx = -1
            for mod, t in new_body:
                target = next(
                    (
                        d
                        for d in ppb_lay_desc
                        if d["model_name"] == mod and d["type"] == t.upper()
                    ),
                    None,
                )
                if target:
                    desc["model_name"] += "_" + mod
                    desc["name"] += "_" + target["name"]
                    for key, value in target.items():
                        if key.startswith("memory") or key.endswith("_time"):
                            desc[key] = desc.get(key, 0) + value
                    target_idx = ppb_lay_desc.index(target)
                    idx = target_idx if idx < 0 else min(idx, target_idx)
                    del ppb_lay_desc[target_idx]
            if "forward_time" in desc:
                desc["time"] = desc["forward_time"]
            idx = max(idx, 0)
            ppb_lay_desc.insert(idx, desc)
