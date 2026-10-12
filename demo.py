"""Demo: self-improving answers via judge-and-regenerate.

Task type: short technical Q&A where the first attempt is deliberately
weakened (in the scripted demo) so you can watch the loop climb. The
live demo uses a real generator + judge.

Usage:
  python demo.py            # scripted, no network
  python demo.py --live     # live LLM generator + judge
"""

from __future__ import annotations

import sys

from self_eval import ImprovementLog, Judge, ReflectiveAgent


def scripted(replies):
    stack = list(replies)
    def llm(p):
        return stack.pop(0)
    return llm


def main():
    log = ImprovementLog()

    if "--live" in sys.argv:
        from self_eval.llm import make_generator, make_judge_llm
        try:
            generator = make_generator()
            judge_llm = make_judge_llm()
        except RuntimeError as e:
            print(f"skipping live demo: {e}")
            return
        tasks = [
            "Explain why database indexes speed up reads but slow down writes, in <=80 words.",
            "A train leaves at 3pm going 60 km/h; another at 4pm going 90 km/h the same way. "
            "When does the second catch the first? Show the math briefly.",
        ]
    else:
        def generator(task, constraints):
            # first attempt ignores constraints; later attempts "comply"
            if not constraints:
                return "Indexes make queries faster somehow. They use trees or something."
            return ("Indexes are sorted lookup structures (usually B-trees): reads become "
                    "O(log n) seeks instead of full scans. Writes must also update every "
                    "index, so INSERT/UPDATE/DELETE do extra work. That is the trade-off.")
        judge_llm = scripted([
            "SCORE: 35\nISSUES: no mechanism; no complexity terms; hand-wavy\n"
            "CONSTRAINTS: explain the data structure; state the complexity effect on reads; state the write cost",
            "SCORE: 88\nISSUES: none material\nCONSTRAINTS: none",
        ])
        tasks = ["Explain why indexes speed reads but slow writes."]

    agent = ReflectiveAgent(generator, Judge(judge_llm), max_cycles=3)
    for task in tasks:
        print("=" * 62)
        print(f"TASK: {task[:70]}...")
        print("=" * 62)
        results = agent.run(task)
        for r in results:
            v = r.verdict
            print(f"\ncycle {r.cycle}  score={v.score}  ({r.seconds:.2f}s)")
            if v.issues:
                print(f"  issues: {'; '.join(v.issues)}")
            if r.cycle > 1:
                print(f"  constraints applied: {len(results[r.cycle-2].verdict.constraints)}")
            print(f"  artifact: {r.artifact[:150]}")
        s = log.add_run(task, results, pass_bar=agent.pass_bar)
        print(f"\nrun summary: scores={s.scores} passed={s.passed} improved={s.improved}")

    print()
    print("=" * 62)
    print("IMPROVEMENT LOG")
    print("=" * 62)
    for k, v in log.report().items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
