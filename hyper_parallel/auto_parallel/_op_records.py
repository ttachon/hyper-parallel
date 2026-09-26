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
"""Op records: what each op of a layer keeps for its backward, as data (shared decision S1).

``op_records.yaml`` states, for every op of the op vector, the recompute
switch that drops it, if any, the ops whose outputs recomputing it takes,
and whether it is communication; and, for each of the memory model's
activation formulas, a *slot*, the formula and the elements each of its ops
keeps.  The memory model prices a layer's activations from the slots, so
the records are the one source of what a layer keeps, op by op: a caller
prices any setting of the switches from them without setting the switches
on a config.

The expressions are Python arithmetic, parsed once and refused unless every
node is one of a closed set: numbers, names, the arithmetic and comparison
operators, ``and``, ``or``, ``not``, conditional expressions, and calls of
``max`` and ``min``.  A name is a field of the layer config, or one of the
names the caller binds.
"""
import ast
import functools
import os
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Any, Callable, Dict, FrozenSet, Mapping, Optional, Tuple

import yaml

from hyper_parallel.auto_parallel._model_spec import ModelSpecError, OpCounts

RECORDS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "op_records.yaml")

# The parts of a layer a slot belongs to, as the memory logs name them.
PARTS = ("attention", "ffn", "norm")

# The recompute switches: a selective layer drops what the ops they name keep.
SWITCHES = ("attBMM", "headCast", "dropout", "softmax", "normOp", "gather", "ffAct")

_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp, ast.Call, ast.Name,
    ast.Constant, ast.Load, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd, ast.Not, ast.And, ast.Or, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
)
_CALLS = MappingProxyType({"max": max, "min": min})
# What an expression sees beside its names: max and min, and no builtins.
_GLOBALS = {"__builtins__": {}, **_CALLS}


class _Names(dict):
    """The names an expression reads, each looked up the first time it is read.

    A conditional expression reads only its branch's names, as the code it
    stands for read only its branch's fields.
    """

    def __init__(self, value: Callable[[str], Any]) -> None:
        super().__init__()
        self._value = value

    def __missing__(self, name: str) -> Any:
        if name in _CALLS:
            # Not a name to look up: the call the expression makes.
            raise KeyError(name)
        found = self._value(name)
        self[name] = found
        return found


@dataclass(frozen=True)
class Expression:
    """One record expression, checked and compiled.

    Attributes:
        source: The expression as written.
        names: The names it reads, the calls it makes apart.
    """

    source: str
    names: FrozenSet[str]
    code: Any

    @classmethod
    def parse(cls, source: Any, where: str) -> "Expression":
        """Parse *source*, refusing a node outside the closed set or a call of anything but max and min."""
        if not isinstance(source, (str, int, float)) or isinstance(source, bool):
            raise ModelSpecError(f"{where} must be an expression, got {source!r}")
        text = str(source).strip()
        try:
            tree = ast.parse(text, mode="eval")
        except SyntaxError as exc:
            raise ModelSpecError(f"{where}: {text!r} is not an expression: {exc.msg}") from exc
        names = set()
        for node in ast.walk(tree):
            if not isinstance(node, _NODES):
                raise ModelSpecError(f"{where}: {text!r} uses {type(node).__name__}, which a record may not")
            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name) or node.func.id not in _CALLS or node.keywords:
                    raise ModelSpecError(f"{where}: {text!r} calls something other than max or min")
            elif isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
                raise ModelSpecError(f"{where}: {text!r} holds {node.value!r}, which is not a number")
        return cls(text, frozenset(names - set(_CALLS)), compile(tree, where, "eval"))

    def __call__(self, names: Dict[str, Any]) -> Any:
        """Evaluate the expression with *names* bound."""
        return eval(self.code, _GLOBALS, names)  # pylint: disable=eval-used


@dataclass(frozen=True)
class OpRecord:
    """What the records state of one op, beside what it keeps in each slot.

    Attributes:
        name: The op, one of the op vector's.
        switch: The recompute switch that drops what it keeps, or ``None``
            for an op every selective layer keeps.
        needs: The ops whose outputs recomputing it takes.
        comm: Whether it is communication rather than compute.
    """

    name: str
    switch: Optional[str]
    needs: Tuple[str, ...]
    comm: bool


@dataclass(frozen=True)
class Slot:
    """One of the memory model's activation formulas, over the elements its ops keep.

    Attributes:
        name: The slot.
        part: The part of the layer its bytes count toward.
        formula: The bytes a layer keeps in it, over ``total``.
        ops: Each op's elements in it, in the order the formula sums them.
        reads: Every name its expressions read but ``total``.
        when: Of which layers it is part, or ``None`` for every layer.
    """

    name: str
    part: str
    formula: Expression
    ops: Tuple[Tuple[str, Expression], ...]
    reads: FrozenSet[str]
    when: Optional[Expression] = None

    def holds(self, value: Callable[[str], Any]) -> bool:
        """Whether the slot is part of a layer whose names *value* gives."""
        return self.when is None or bool(self.when(_Names(value)))


@dataclass(frozen=True)
class OpRecords:
    """Every op's record and every slot.

    Attributes:
        ops: The record of each op of the op vector, by name.
        slots: The slots, by name.
        defaults: The value of a field a layer config may not hold.
    """

    ops: Mapping[str, OpRecord]
    slots: Mapping[str, Slot]
    defaults: Mapping[str, float]

    def keeps(self, op: str) -> Dict[str, Expression]:
        """The elements *op* keeps in each slot it keeps anything in, by slot, whichever layers they hold in."""
        if op not in self.ops:
            raise ModelSpecError(f"no op {op!r}; the ops are {sorted(self.ops)}")
        return {slot.name: elements for slot in self.slots.values() for name, elements in slot.ops if name == op}

    def evaluate(self, slot: str, value: Callable[[str], Any], keep: Callable[[str], Any],
                 only: Optional[str] = None) -> float:
        """The bytes a layer keeps in *slot*.

        Args:
            slot: The slot.
            value: The value of each name the slot reads.
            keep: For each op, 1 where the layer keeps what it keeps, 0
                where it drops it.
            only: An op to price alone: the slot's formula over its
                elements, whatever *keep* says.

        Returns:
            The bytes, one rank's share of a micro-batch.
        """
        record = self.slots[slot]
        names = _Names(value)
        total = 0
        for op, elements in record.ops:
            kept = int(op == only) if only is not None else keep(op)
            if kept:
                total += kept * elements(names)
        names["total"] = total
        return record.formula(names)


def _op_names() -> Tuple[str, ...]:
    """The ops of the op vector, in its order."""
    return tuple(field.name for field in fields(OpCounts))


def _op_record(name: str, data: Any) -> OpRecord:
    """Build one op's record from its ``{switch, needs, comm}`` mapping."""
    where = f"op_records.ops.{name}"
    if not isinstance(data, Mapping):
        raise ModelSpecError(f"{where} must map switch, needs and comm, got {data!r}")
    unknown = sorted(set(data) - {"switch", "needs", "comm"})
    if unknown:
        raise ModelSpecError(f"{where} has unknown keys {unknown}; a record has switch, needs and comm")
    switch = data.get("switch")
    if switch is not None and switch not in SWITCHES:
        raise ModelSpecError(f"{where}.switch must be one of {list(SWITCHES)}, got {switch!r}")
    needs = data.get("needs", [])
    ops = _op_names()
    if not isinstance(needs, list) or any(need not in ops for need in needs):
        raise ModelSpecError(f"{where}.needs must list ops of {ops}, got {needs!r}")
    comm = data.get("comm", False)
    if not isinstance(comm, bool):
        raise ModelSpecError(f"{where}.comm must be true or false, got {comm!r}")
    return OpRecord(str(name), switch, tuple(needs), comm)


def _slot(name: str, data: Any, ops: Mapping[str, OpRecord]) -> Slot:
    """Build one slot from its ``{part, formula, ops}`` mapping."""
    where = f"op_records.slots.{name}"
    if not isinstance(data, Mapping):
        raise ModelSpecError(f"{where} must map part, formula and ops, got {data!r}")
    unknown = sorted(set(data) - {"part", "formula", "ops", "when"})
    if unknown:
        raise ModelSpecError(f"{where} has unknown keys {unknown}; a slot has part, formula, ops and when")
    part = data.get("part")
    if part not in PARTS:
        raise ModelSpecError(f"{where}.part must be one of {list(PARTS)}, got {part!r}")
    formula = Expression.parse(data.get("formula"), f"{where}.formula")
    if "total" not in formula.names:
        raise ModelSpecError(f"{where}.formula must read total, what the slot's ops keep")
    members = data.get("ops")
    if not isinstance(members, Mapping) or not members:
        raise ModelSpecError(f"{where}.ops must map ops to the elements they keep")
    unknown_ops = sorted(set(members) - set(ops))
    if unknown_ops:
        raise ModelSpecError(f"{where}.ops names {unknown_ops}, which have no record")
    members = tuple((str(op), Expression.parse(elements, f"{where}.ops.{op}")) for op, elements in members.items())
    reads = set(formula.names)
    for _, elements in members:
        reads |= elements.names
    when = None if data.get("when") is None else Expression.parse(data["when"], f"{where}.when")
    return Slot(str(name), part, formula, members, frozenset(reads - {"total"}), when)


def _defaults(data: Any) -> Mapping[str, float]:
    """The values the records give a field a layer config may not hold."""
    defaults = data or {}
    if not isinstance(defaults, Mapping) or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) for value in defaults.values()):
        raise ModelSpecError(f"op records' defaults must map fields to numbers, got {defaults!r}")
    return MappingProxyType(dict(defaults))


@functools.lru_cache(maxsize=None)
def load_op_records(path: str = RECORDS_PATH) -> OpRecords:
    """Read and check the op records.

    Raises:
        ModelSpecError: If the file misses an op of the op vector, holds an
            unknown key, op or switch, or an expression outside the closed
            set of nodes.
    """
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    unknown = sorted(set(data) - {"defaults", "ops", "slots"})
    if unknown:
        raise ModelSpecError(f"op records have unknown keys {unknown}")
    stated = data.get("ops") or {}
    names = _op_names()
    missing = [name for name in names if name not in stated]
    extra = sorted(set(stated) - set(names))
    if missing or extra:
        raise ModelSpecError(f"op records must state every op of {list(names)}: missing {missing}, unknown {extra}")
    ops = {name: _op_record(name, stated[name]) for name in names}
    slots = {str(name): _slot(name, slot, ops) for name, slot in (data.get("slots") or {}).items()}
    return OpRecords(MappingProxyType(ops), MappingProxyType(slots), _defaults(data.get("defaults")))
