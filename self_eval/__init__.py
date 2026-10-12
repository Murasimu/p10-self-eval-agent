"""Self-reflective agent with auto-eval.

Execute -> LLM-as-judge critique -> regenerate under constraints ->
log improvement metrics. The loop is bounded, every cycle's scores are
kept, and stopping is driven by measured improvement (or its absence),
not hope.

Modules:
- judge.py     LLM-as-judge: rubric scoring with parseable verdicts
- cycle.py     execute/critique/regenerate cycle with constraint feedback
- metrics.py   improvement metrics: deltas, convergence, regression
"""

from .cycle import ReflectiveAgent, CycleResult
from .judge import Judge, Verdict
from .metrics import ImprovementLog

__all__ = ["ReflectiveAgent", "CycleResult", "Judge", "Verdict", "ImprovementLog"]
__version__ = "1.0.0"
