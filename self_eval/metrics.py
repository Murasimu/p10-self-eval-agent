"""Improvement metrics: make 'the system improves over time' a measured
claim, not a vibe.

Per-run: score trajectory, delta, cycles used, regression detection.
Across runs: average improvement, convergence rate (passing within N
cycles), pass rate, and trend over time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .cycle import CycleResult


@dataclass
class RunSummary:
    task: str
    scores: list[float]
    passed: bool
    cycles: int
    improved: bool

    @property
    def delta(self) -> float:
        return self.scores[-1] - self.scores[0] if len(self.scores) > 1 else 0.0


@dataclass
class ImprovementLog:
    runs: list[RunSummary] = field(default_factory=list)

    def add_run(self, task: str, results: list[CycleResult],
                pass_bar: float = 0.8) -> RunSummary:
        scores = [r.verdict.score for r in results]
        summary = RunSummary(
            task=task,
            scores=scores,
            passed=bool(scores) and max(scores) >= pass_bar,
            cycles=len(results),
            improved=len(scores) > 1 and scores[-1] > scores[0],
        )
        self.runs.append(summary)
        return summary

    # ------------------------------------------------------------------
    def pass_rate(self) -> float:
        return sum(1 for r in self.runs if r.passed) / len(self.runs) if self.runs else 0.0

    def avg_improvement(self) -> float:
        deltas = [r.delta for r in self.runs if r.cycles > 1]
        return sum(deltas) / len(deltas) if deltas else 0.0

    def convergence_rate(self) -> float:
        """Runs that ended better than they started."""
        multi = [r for r in self.runs if r.cycles > 1]
        return sum(1 for r in multi if r.improved) / len(multi) if multi else 0.0

    def regressions(self) -> list[RunSummary]:
        return [r for r in self.runs if r.cycles > 1 and r.delta < 0]

    def report(self) -> dict[str, Any]:
        return {
            "runs": len(self.runs),
            "pass_rate": round(self.pass_rate(), 3),
            "avg_improvement_per_run": round(self.avg_improvement(), 3),
            "convergence_rate": round(self.convergence_rate(), 3),
            "regressions": len(self.regressions()),
            "score_trajectories": [r.scores for r in self.runs],
        }
