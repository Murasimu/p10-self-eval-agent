"""LLM-as-judge: rubric scoring with parseable verdicts.

The judge scores an artifact against explicit criteria and MUST return
its verdict in a fixed header format:

    SCORE: <0-100>
    ISSUES: <semicolon-separated issues>
    CONSTRAINTS: <imperative instructions for the next attempt>

Robust parsing clamps and defaults; a judge that rambles still yields a
usable verdict (score 50, issues unparsed) rather than crashing the
loop.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

LLM = Callable[[str], str]

JUDGE_PROMPT = """You are a strict evaluator. Score the artifact against the task and rubric.

Task: {task}
Rubric: {rubric}

Artifact:
{artifact}

Respond in EXACTLY this format:
SCORE: <integer 0-100>
ISSUES: <semicolon-separated list of concrete problems>
CONSTRAINTS: <imperative instructions the next attempt MUST follow to fix them>"""


@dataclass
class Verdict:
    score: float          # 0..1
    issues: list[str]
    constraints: list[str]

    @property
    def passed(self) -> bool:
        return self.score >= 0.8  # default pass bar, tunable per use


@dataclass
class Judge:
    llm: LLM
    rubric: str = "correctness, completeness, clarity, and absence of hallucination"

    def evaluate(self, task: str, artifact: str) -> Verdict:
        raw = self.llm(JUDGE_PROMPT.format(
            task=task, rubric=self.rubric, artifact=artifact))
        return self.parse(raw)

    @staticmethod
    def parse(raw: str) -> Verdict:
        score_m = re.search(r"SCORE:\s*(\d{1,3})", raw)
        issues_m = re.search(r"ISSUES:\s*(.+?)(?:\nCONSTRAINTS:|\Z)", raw, re.S)
        cons_m = re.search(r"CONSTRAINTS:\s*(.+)", raw, re.S)

        score = min(1.0, int(score_m.group(1)) / 100.0) if score_m else 0.5

        def split(field: re.Match | None) -> list[str]:
            if not field:
                return []
            parts = [p.strip(" -\t") for p in re.split(r"[;\n]", field.group(1))]
            return [p for p in parts if p][:6]

        return Verdict(score=score, issues=split(issues_m),
                       constraints=split(cons_m))
