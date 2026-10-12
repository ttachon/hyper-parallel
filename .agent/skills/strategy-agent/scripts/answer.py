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
"""Turn an ND ranking into the strategy agent's answer: the layout, the next two, and why.

Every figure the answer gives is computed here from the ranking ND wrote with
``--ranking_csv``, so the language model only quotes it.

    python answer.py <nd_rank.csv> [--calibrated] [--spread 0.79] [--budget 58GB]
    python answer.py <nd_rank.csv> --compare "EP=2,OP=32;EP=1,OP=64"
"""
import argparse
import csv
import sys
from typing import Dict, List, Optional, Sequence, Tuple

# Degrees named as people say them; ND's CSV calls tensor parallelism MP.
_NAMES = {"MP": "TP"}
_SHOWN = ("DP", "MP", "CP", "PP", "VPP", "EP", "OP", "MB", "MBS")
_RECOMPUTE = "recompute"
_COMPUTE_PARTS = ("FW_COMPUTE", "BW_COMPUTE")


class Ranking:
    """ND's ranking CSV: its degree columns, its part columns and its rows, best first."""

    def __init__(self, path: str) -> None:
        """Read the ranking.

        Args:
            path: the CSV ND wrote with ``--ranking_csv``.

        Raises:
            ValueError: the file has no ``memory_mb`` or ``score`` column, so ND did not write it.
        """
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            self.columns: List[str] = list(reader.fieldnames or [])
            self.rows: List[Dict[str, str]] = list(reader)
        if "memory_mb" not in self.columns or "score" not in self.columns:
            raise ValueError(f"{path} is not an ND ranking: no memory_mb or score column")
        start, stop = self.columns.index("rank") + 1, self.columns.index("memory_mb")
        self.degrees = [name for name in self.columns[start:stop] if name != _RECOMPUTE]
        self.moded = _RECOMPUTE in self.columns
        self.parts = self.columns[self.columns.index("score") + 1:]

    def distinct(self) -> List[Dict[str, str]]:
        """The rows once each: without TP, sequence parallel changes nothing, so SP twins count once."""
        seen = set()
        kept = []
        for row in self.rows:
            key = tuple(row[name] for name in self.degrees if not (name == "SP" and row.get("MP") == "1"))
            key += (row.get(_RECOMPUTE, ""),)
            if key not in seen:
                seen.add(key)
                kept.append(row)
        return kept


def layout(row: Dict[str, str], degrees: Sequence[str]) -> str:
    """A row's degrees as people write them, for example ``DP64 TP1 CP1 PP1 EP1 OP64 MB1 MBS1``."""
    words = [f"{_NAMES.get(name, name)}{row[name]}" for name in _SHOWN if name in degrees]
    if "VPP" in degrees and row["VPP"] == "1":
        words = [word for word in words if not word.startswith("VPP")]
    if row.get("MP", "1") != "1" and "SP" in degrees:
        words.append("SP on" if row["SP"] == "True" else "SP off")
    if _RECOMPUTE in row:
        words.append(f"recompute {row[_RECOMPUTE]}")
    return " ".join(words)


def _gib(row: Dict[str, str]) -> float:
    """The row's peak memory per die in GiB (ND's MB are MiB)."""
    return float(row["memory_mb"]) / 1024.0


def _versus(row: Dict[str, str], pick: Dict[str, str]) -> float:
    """How much slower than the pick the row's step is, in percent."""
    return 100.0 * (float(row["score"]) / float(pick["score"]) - 1.0)


def _part_moves(row: Dict[str, str], pick: Dict[str, str], parts: Sequence[str], count: int = 3) -> str:
    """The parts that move most from the pick to the row, forward and backward compute as one."""
    if not parts or not row.get(parts[0]):
        return "parts not recorded"
    moves: Dict[str, float] = {}
    for part in parts:
        delta = float(row[part]) - float(pick[part])
        label = "compute" if part in _COMPUTE_PARTS else part.lower()
        moves[label] = moves.get(label, 0.0) + delta
    largest = sorted(moves.items(), key=lambda item: -abs(item[1]))[:count]
    return ", ".join(f"{label} {delta:+.1f}" for label, delta in largest if abs(delta) >= 0.05) or "none"


def _shared(rows: Sequence[Dict[str, str]], degrees: Sequence[str]) -> str:
    """The degrees every row of a group agrees on."""
    names = [name for name in _SHOWN if name in degrees] + ([_RECOMPUTE] if _RECOMPUTE in rows[0] else [])
    same = [f"{_NAMES.get(name, name)}{rows[0][name]}" if name != _RECOMPUTE else f"recompute {rows[0][name]}"
            for name in names if all(row[name] == rows[0][name] for row in rows)
            and not (name == "VPP" and rows[0][name] == "1")]
    return " ".join(same) or "nothing"


def card(ranking: Ranking, calibrated: bool, spread: Optional[float], budget: Optional[str]) -> str:
    """The answer in Markdown: the pick, the next two, what a near tie leaves firm, and why."""
    rows = ranking.distinct()
    if not rows:
        return "ND kept no layout: none fits the budget, or the request leaves no valid layout.\n"
    pick, unit = rows[0], "ms" if calibrated else "score"
    lines = [f"Layouts that fit, each counted once: {len(rows)} (ND's ranking has {len(ranking.rows)} rows).", ""]
    head = f"| | Layout | Step ({unit}) | vs the pick | Peak per die (GiB) |"
    lines += [head, "|---|---|---|---|---|"]
    for place, row in zip(("Pick", "2nd", "3rd"), rows[:3]):
        versus = "" if row is pick else f"{_versus(row, pick):+.2f}%"
        score = f"{float(row['score']):.1f}" if calibrated else f"{float(row['score']):.4g}"
        lines.append(f"| {place} | {layout(row, ranking.degrees)} | {score} | {versus} | {_gib(row):.1f} |")
    if budget:
        lines += ["", f"Budget: {budget} a die. The pick's peak is {_gib(pick):.1f} GiB."]
    if spread is not None and calibrated:
        close = [row for row in rows if _versus(row, pick) < spread]
        if len(close) > 1:
            lines += ["", f"{len(close)} layouts are within {spread}% of the pick, the spread one layout's step "
                      "showed between two rounds on this cluster: one round cannot order them. "
                      f"What they all share is the firm part of the answer: {_shared(close, ranking.degrees)}."]
    lines += ["", "Why, from ND's parts: the best layout with another value of each degree that varies.", ""]
    lines += _why(ranking, rows, calibrated)
    return "\n".join(lines) + "\n"


def _why(ranking: Ranking, rows: Sequence[Dict[str, str]], calibrated: bool) -> List[str]:
    """One table line per varying degree: its best other layout and the parts that move most."""
    pick = rows[0]
    unit = "ms" if calibrated else "score"
    lines = [f"| Degree | The pick | Best other | That layout | vs the pick | Largest part moves ({unit}) |",
             "|---|---|---|---|---|---|"]
    varying = [name for name in _SHOWN if name in ranking.degrees and len({row[name] for row in rows}) > 1]
    if ranking.moded and len({row[_RECOMPUTE] for row in rows}) > 1:
        varying.append(_RECOMPUTE)
    # Degrees that move together (CP takes ranks from DP) share their best other layout: one line each.
    groups: Dict[int, List[str]] = {}
    for name in varying:
        place = next(index for index, row in enumerate(rows) if row[name] != pick[name])
        groups.setdefault(place, []).append(name)
    for place, names in groups.items():
        other = rows[place]
        versus = f"{_versus(other, pick):+.2f}%" if calibrated else "n/a"
        lines.append(f"| {', '.join(_NAMES.get(name, name) for name in names)} "
                     f"| {' '.join(pick[name] for name in names)} | {' '.join(other[name] for name in names)} "
                     f"| {layout(other, ranking.degrees)} | {versus} | {_part_moves(other, pick, ranking.parts)} |")
    return lines


def _wanted(spec: str) -> List[Tuple[str, str]]:
    """``EP=2,OP=32`` as ``[("EP", "2"), ("OP", "32")]``, TP accepted for MP."""
    pairs = []
    for item in spec.split(","):
        name, _, value = item.partition("=")
        name = name.strip().upper()
        pairs.append(("MP" if name == "TP" else name if name != "RECOMPUTE" else _RECOMPUTE, value.strip()))
    return pairs


def compare(ranking: Ranking, specs: str, calibrated: bool) -> str:
    """Each named candidate's place, step and peak, or why it has no row."""
    rows = ranking.distinct()
    pick = rows[0] if rows else None
    lines = ["| Candidate | Place | Layout | Step | vs ND's pick | Peak per die (GiB) |", "|---|---|---|---|---|---|"]
    for spec in [part for part in specs.split(";") if part.strip()]:
        wanted = _wanted(spec)
        unknown = [name for name, _ in wanted if name not in ranking.degrees + [_RECOMPUTE]]
        if unknown:
            lines.append(f"| {spec} | | unknown degree {unknown} | | | |")
            continue
        found = [(place, row) for place, row in enumerate(rows, start=1)
                 if all(row.get(name) == value for name, value in wanted)]
        if not found:
            lines.append(f"| {spec} | none | no row: it does not fit the budget or is outside ND's search | | | |")
            continue
        place, row = found[0]
        step = f"{float(row['score']):.1f} ms" if calibrated else f"{float(row['score']):.4g}"
        lines.append(f"| {spec} | {place} of {len(rows)} | {layout(row, ranking.degrees)} | {step} "
                     f"| {_versus(row, pick):+.2f}% | {_gib(row):.1f} |")
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Print the answer, or the comparison, for one ranking."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ranking", help="the CSV ND wrote with --ranking_csv")
    parser.add_argument("--calibrated", action="store_true", help="the search ran with a cluster's ratios (-c)")
    parser.add_argument("--spread", type=float, default=None, help="the cluster's re-measure spread, in percent")
    parser.add_argument("--budget", default=None, help="the memory budget the search ran with, as given to -M")
    parser.add_argument("--compare", default=None, help='named candidates, for example "EP=2,OP=32;EP=1,OP=64"')
    args = parser.parse_args(argv)
    ranking = Ranking(args.ranking)
    if args.compare:
        sys.stdout.write(compare(ranking, args.compare, args.calibrated))
    else:
        sys.stdout.write(card(ranking, args.calibrated, args.spread, args.budget))
    return 0


if __name__ == "__main__":
    sys.exit(main())
