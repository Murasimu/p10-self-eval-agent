# Self-Reflective Agent with Auto-Eval

**Project 10 / 12 — Agentic AI portfolio.**
*Shows: you can build systems that improve over time.*

Execute → LLM-as-judge critique → regenerate under explicit constraints →
log improvement metrics. The loop is bounded, constraints are carried
forward, and "improvement" is a measured delta, not a claim.

## Why this matters

Generation without evaluation is vibes; evaluation without regeneration
is a report nobody acts on. Closed-loop self-improvement needs four
pieces working together:

1. **LLM-as-judge with a strict verdict format** — `SCORE / ISSUES /
   CONSTRAINTS` parsed robustly (clamped scores, defaults on rambles).
   The constraints line is the critical part: it converts critique into
   *imperative instructions* the next attempt must follow.
2. **Constraint-driven regeneration** — each cycle's constraints are
   injected into the generator's context (bounded to 8 to prevent prompt
   bloat), so the loop compounds rather than restarts.
3. **Bounded, convergent loop** — stops on pass, cycle budget, or a
   flat-line guard (N consecutive non-improving cycles stop early
   instead of burning calls on a hopeless task).
4. **Improvement metrics** — per-run trajectories and deltas; across
   runs: pass rate, average improvement, convergence rate, and
   regression detection (runs that got *worse*).

## Architecture

```
task ──> generator(task, constraints) ──> artifact
                                             │
                              Judge.evaluate (rubric, strict format)
                                             │
                  verdict {score, issues, constraints}
                                             │
        score >= pass_bar? ──yes──> STOP (passed)
                                             │no
        flat-line guard?  ──yes──> STOP (not converging)
                                             │no
        constraints = (constraints + verdict.constraints)[-8:]
                                             │
                                   next cycle ◄┘
        every cycle -> CycleResult -> ImprovementLog.add_run()
        report(): pass_rate, avg improvement, convergence, regressions
```

- `self_eval/judge.py` — rubric judge with robust verdict parsing
- `self_eval/cycle.py` — the reflective loop with guards
- `self_eval/metrics.py` — run summaries and cross-run improvement stats

## Quickstart

```bash
uv pip install -e ".[demo]"
python demo.py            # scripted: watch a weak answer climb 0.35 -> 0.88
python demo.py --live     # live generator + judge on two tasks
```

The scripted demo shows the money shot: attempt 1 is hand-wavy
("indexes use trees or something", score 35), the judge issues concrete
constraints (explain the data structure, state complexity, state write
cost), and attempt 2 lands at 88. The live demo runs the same loop with
a real model and logs the aggregate metrics.

## Tests

```bash
pytest -q
```

16 tests cover: verdict parsing (full format, clamped overflow, garbage
default, multiline issues), pass-bar boundary; cycle behavior (pass on
first attempt, multi-cycle improvement with constraint propagation,
flat-line early stop, budget enforcement, best-result selection,
constraint bounding, single-cycle non-improved); metrics (run summary
delta, aggregates with a partial pass rate, regression detection, empty
log).

## Design decisions

- **Constraints, not vibes** — telling the generator "do better" is
  noise; the judge must emit imperative, checkable instructions, and
  they must reach the next attempt.
- **Flat-line guard** — self-reflection has a real failure mode: the
  model thrashing between equally flawed attempts. Two flat cycles stop
  the loop and record it.
- **Judge robustness beats judge eloquence** — a judge that fails to
  parse yields score 0.5 and empty constraints; the loop survives and
  the next cycle still gets a fair shot.
- **Regressions are tracked, not hidden** — an improvement log that only
  counts wins is marketing; `regressions()` is the honest counterweight.
- **Pass bar is explicit** — 0.8 default, tunable per domain; hidden
  thresholds make "improved" unfalsifiable.

## Portfolio context

Project 10 of a 12-project agentic-AI portfolio (structured output, RAG
grounding, ReAct planning, tool orchestration, memory, HITL approval,
cost routing, event automation, debate, self-eval, observability, OSS
contribution). See the profile README for the full map.
