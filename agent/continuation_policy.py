"""Checkpoint-continue policy for the conversation loop.

Feature 2 of the "don't spin, don't dead-stop" pair (2026-07-20, requested by
Brian). Today, when a turn exhausts its ``IterationBudget`` the loop just stops
mid-task and a human has to say "continue" (Brian's "stopped 80% done, had to
tell it to continue" complaint). This policy replaces the dead-stop with a
bounded, guarded auto-continue: on cap-hit the agent checkpoints (what's done /
what's left / next action) and either resumes on a fresh grace budget or hands
off with that checkpoint — it never silently dies mid-build.

The safety property that makes auto-continue sane lives here and is unit-tested:
    stuck ⇒ ESCALATE   (a real loop, flagged by StuckDetector, can NEVER get a
                        continuation — so continuation can't ride a loop forever)
    auto_continues_used >= max ⇒ ESCALATE   (hard ceiling on total continues)
    otherwise ⇒ CONTINUE

Pure and dependency-free so it unit-tests in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ContinuationDecision(str, Enum):
    CONTINUE = "continue"    # grant a grace budget extension, checkpoint, keep working
    ESCALATE = "escalate"    # checkpoint + hand off (pill Claude / ping Brian) — never silent-stop


@dataclass
class ContinuationPolicy:
    """Decides what happens when a turn hits the iteration cap.

    ``max_auto_continues`` — hard ceiling on how many times ONE task may
                             auto-resume past the cap before it must hand off.
    ``grace_iterations``   — how much extra budget each auto-continue grants.
    """

    max_auto_continues: int = 3
    grace_iterations: int = 30

    def decide(self, *, auto_continues_used: int, stuck: bool) -> ContinuationDecision:
        # Guard 1 (the safety property): a loop flagged by StuckDetector never
        # earns a continuation, no matter how much budget remains.
        if stuck:
            return ContinuationDecision.ESCALATE
        # Guard 2: hard ceiling on total auto-continues for this task.
        if auto_continues_used >= self.max_auto_continues:
            return ContinuationDecision.ESCALATE
        return ContinuationDecision.CONTINUE

    def next_budget(self, current_max_total: int) -> int:
        """New IterationBudget cap after granting one grace window."""
        return current_max_total + self.grace_iterations


__all__ = ["ContinuationPolicy", "ContinuationDecision"]


# -- Self-test (run: `python3 agent/continuation_policy.py`) -----------------
if __name__ == "__main__":
    def _eq(got, want, msg):
        assert got == want, f"FAIL {msg}: got {got!r}, want {want!r}"

    p = ContinuationPolicy(max_auto_continues=3, grace_iterations=30)

    # normal path: resume while under the ceiling and not stuck
    _eq(p.decide(auto_continues_used=0, stuck=False), ContinuationDecision.CONTINUE, "first cap-hit continues")
    _eq(p.decide(auto_continues_used=2, stuck=False), ContinuationDecision.CONTINUE, "under ceiling continues")

    # SAFETY PROPERTY: stuck always escalates, even with continues to spare
    _eq(p.decide(auto_continues_used=0, stuck=True), ContinuationDecision.ESCALATE, "stuck never continues (0 used)")
    _eq(p.decide(auto_continues_used=2, stuck=True), ContinuationDecision.ESCALATE, "stuck never continues (2 used)")

    # hard ceiling: at/над the max, hand off
    _eq(p.decide(auto_continues_used=3, stuck=False), ContinuationDecision.ESCALATE, "ceiling reached escalates")
    _eq(p.decide(auto_continues_used=9, stuck=False), ContinuationDecision.ESCALATE, "over ceiling escalates")

    # budget math
    _eq(p.next_budget(90), 120, "grace extends 90 -> 120")

    print("continuation_policy self-test: ALL PASS")
