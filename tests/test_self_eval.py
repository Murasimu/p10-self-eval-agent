"""Tests for the self-reflective agent. Offline, deterministic."""

import pytest

from self_eval import ImprovementLog, Judge, ReflectiveAgent, Verdict
from self_eval.cycle import CycleResult


# ---------------------------------------------------------------- judge
class TestJudge:
    def _judge(self, replies):
        stack = list(replies)
        return Judge(llm=lambda p: stack.pop(0))

    def test_parse_full_verdict(self):
        j = self._judge([
            "SCORE: 85\nISSUES: minor typo; vague conclusion\n"
            "CONSTRAINTS: fix the typo; quantify the conclusion"
        ])
        v = j.evaluate("task", "artifact")
        assert v.score == 0.85
        assert v.issues == ["minor typo", "vague conclusion"]
        assert v.constraints == ["fix the typo", "quantify the conclusion"]
        assert v.passed

    def test_parse_clamps_overflow(self):
        v = Judge.parse("SCORE: 250\nISSUES: x\nCONSTRAINTS: y")
        assert v.score == 1.0

    def test_parse_defaults_on_garbage(self):
        v = Judge.parse("I liked it, no structure at all")
        assert v.score == 0.5 and not v.issues and not v.constraints

    def test_parse_multiline_issues(self):
        raw = "SCORE: 60\nISSUES: a\nb\nc\nCONSTRAINTS: do a\n do b"
        v = Judge.parse(raw)
        assert v.score == 0.6
        assert set(v.issues) == {"a", "b", "c"}

    def test_pass_bar_boundary(self):
        assert Verdict(0.79, [], []).passed is False
        assert Verdict(0.80, [], []).passed is True


# ---------------------------------------------------------------- cycle
def gen_seq(artifacts):
    """Generator that returns queued artifacts regardless of constraints."""
    queue = list(artifacts)
    seen_constraints: list[list[str]] = []
    def gen(task, constraints):
        seen_constraints.append(list(constraints))
        return queue.pop(0)
    gen.seen = seen_constraints
    return gen


def judge_seq(scores):
    """Judge whose scores advance per call; constraints grow each time."""
    calls = {"n": 0}
    def llm(prompt):
        i = calls["n"]
        calls["n"] += 1
        return (f"SCORE: {scores[i]}\n"
                f"ISSUES: issue {i}\n"
                f"CONSTRAINTS: always satisfy requirement {i}")
    return llm


class TestReflectiveAgent:
    def test_pass_first_cycle(self):
        gen = gen_seq(["perfect answer"])
        agent = ReflectiveAgent(gen, Judge(judge_seq([95])))
        results = agent.run("task")
        assert len(results) == 1
        assert results[0].verdict.passed
        assert gen.seen[0] == []  # no constraints on first attempt

    def test_improves_over_cycles(self):
        scores = [40, 70, 90]
        gen = gen_seq(["bad", "better", "good"])
        agent = ReflectiveAgent(gen, Judge(judge_seq(scores)), max_cycles=3)
        results = agent.run("task")
        assert [r.verdict.score for r in results] == [0.4, 0.7, 0.9]
        assert agent.improved()
        # constraints from cycle 1 reached cycle 2's generator call
        assert gen.seen[1] == ["always satisfy requirement 0"]
        assert gen.seen[2] == ["always satisfy requirement 0",
                               "always satisfy requirement 1"]

    def test_flat_stops_early(self):
        # never improves: 50, 49, 48 -> flat guard stops after 2 flat cycles
        gen = gen_seq(["a", "b", "c", "d"])
        agent = ReflectiveAgent(gen, Judge(judge_seq([50, 49, 48, 47])),
                                max_cycles=5, stop_after_flat=2)
        results = agent.run("task")
        assert len(results) <= 3
        assert not agent.improved()

    def test_cycle_budget_respected(self):
        gen = gen_seq(["x"] * 10)
        agent = ReflectiveAgent(gen, Judge(judge_seq([50] * 10)), max_cycles=3)
        assert len(agent.run("task")) == 3

    def test_best_picks_max(self):
        gen = gen_seq(["bad", "mid", "good"])
        agent = ReflectiveAgent(gen, Judge(judge_seq([30, 95, 40])), max_cycles=3)
        agent.run("task")
        assert agent.best.verdict.score == 0.95

    def test_constraints_bounded(self):
        gen = gen_seq([f"a{i}" for i in range(10)])
        agent = ReflectiveAgent(gen, Judge(judge_seq([10] * 10)), max_cycles=6)
        agent.run("task")
        # constraints list must never exceed 8 entries (prompt bloat guard)
        for seen in gen.seen:
            assert len(seen) <= 8

    def test_single_cycle_not_improved(self):
        gen = gen_seq(["ok"])
        agent = ReflectiveAgent(gen, Judge(judge_seq([90])))
        agent.run("task")
        assert agent.improved() is False  # nothing to compare


# ---------------------------------------------------------------- metrics
def make_result(i, score):
    v = Verdict(score=score, issues=[], constraints=[])
    return CycleResult(cycle=i, artifact="x", verdict=v, seconds=0.1)


class TestImprovementLog:
    def test_single_run_summary(self):
        log = ImprovementLog()
        s = log.add_run("t", [make_result(1, 0.4), make_result(2, 0.9)])
        assert s.scores == [0.4, 0.9] and s.passed and s.improved
        assert s.delta == pytest.approx(0.5)

    def test_aggregates(self):
        log = ImprovementLog()
        log.add_run("t1", [make_result(1, 0.4), make_result(2, 0.9)])
        log.add_run("t2", [make_result(1, 0.3), make_result(2, 0.7)])
        log.add_run("t3", [make_result(1, 0.85)])  # passed first try
        rep = log.report()
        assert rep["runs"] == 3
        assert rep["pass_rate"] == pytest.approx(2 / 3, abs=0.001)  # t2 never hit 0.8
        assert rep["avg_improvement_per_run"] == pytest.approx(0.45)
        assert rep["convergence_rate"] == 1.0
        assert rep["regressions"] == 0

    def test_regression_detected(self):
        log = ImprovementLog()
        log.add_run("t", [make_result(1, 0.8), make_result(2, 0.5)])
        assert len(log.regressions()) == 1

    def test_empty_log(self):
        rep = ImprovementLog().report()
        assert rep["runs"] == 0 and rep["pass_rate"] == 0.0
