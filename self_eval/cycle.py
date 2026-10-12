"""The reflective cycle: execute -> judge -> regenerate with constraints.

Each cycle:
1. Generator produces an artifact for the task.
2. Judge scores it against the rubric.
3. If below the pass bar and cycles remain, the judge's constraints are
   injected into the generator's context and it regenerates.
4. Every cycle's verdict is logged for improvement metrics.

The loop stops on: pass, cycle budget exhaustion, or regression guard
(N consecutive non-improving cycles stop early instead of burning calls).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from .judge import Judge, Verdict

Generator = Callable[[str, list[str]], str]
# (task, constraints) -> artifact


@dataclass
class CycleResult:
    cycle: int
    artifact: str
    verdict: Verdict
    seconds: float


@dataclass
class ReflectiveAgent:
    generator: Generator
    judge: Judge
    max_cycles: int = 3
    pass_bar: float = 0.8
    stop_after_flat: int = 2   # consecutive non-improving cycles

    results: list[CycleResult] = field(default_factory=list)

    def run(self, task: str) -> list[CycleResult]:
        constraints: list[str] = []
        flat = 0
        best = -1.0

        for cycle in range(1, self.max_cycles + 1):
            t0 = time.time()
            artifact = self.generator(task, list(constraints))
            verdict = self.judge.evaluate(task, artifact)
            self.results.append(CycleResult(cycle, artifact, verdict,
                                            time.time() - t0))

            if verdict.score >= self.pass_bar:
                break

            if verdict.score > best + 1e-9:
                best = verdict.score
                flat = 0
            else:
                flat += 1
                if flat >= self.stop_after_flat:
                    break  # not converging; stop burning cycles

            # carry constraints forward (bounded to avoid prompt bloat)
            constraints = (constraints + verdict.constraints)[-8:]

        return self.results

    @property
    def best(self) -> CycleResult:
        return max(self.results, key=lambda r: r.verdict.score)

    def improved(self) -> bool:
        """Did reflection measurably help? (final > first)"""
        if len(self.results) < 2:
            return False
        return self.results[-1].verdict.score > self.results[0].verdict.score
