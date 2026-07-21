"""Cross-turn stuck detection for the conversation loop.

Feature 1 of the "don't spin, don't dead-stop" pair (2026-07-20, requested by
Brian). The iteration cap (``IterationBudget``) stops runaway loops by
exhausting a budget — but that lets a genuinely stuck agent burn all 90 turns
repeating the *same failing action* before it gives up (e.g. the auto-injected
skill-housekeeping prompt that hammered a non-existent skill for 60 turns on
2026-07-20). This module catches the *pattern* — same tool + same failure, N
times in a row — early, so the loop can force a new approach or escalate
instead of grinding to the cap.

Pairs with checkpoint-continue (Feature 2): auto-continuing past the cap is only
safe *because* this detector guarantees a real loop can't ride the continuation
forever — an escalate verdict breaks out before that can happen.

Dependency-free and pure (no I/O, no agent refs) so it unit-tests in isolation
and imports into the turn loop without an import cycle.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class StuckVerdict(str, Enum):
    PROCEED = "proceed"    # nothing repeating; carry on
    NUDGE = "nudge"        # same failure `threshold` times — inject a "change approach" turn
    ESCALATE = "escalate"  # nudged and STILL repeating — checkpoint and hand off


def result_error_signature(tool_name: str, result: Any) -> Optional[str]:
    """Derive a stable failure signature from a tool result, or ``None`` if the
    result is not a recognizable error.

    Recognizes the shapes Hermes tool results actually take:
      * a dict with a truthy ``"error"`` key
      * a JSON-object string with a truthy ``"error"`` key
      * a plain string that starts with an error marker / carries a traceback

    The signature is coarse (tool name + first line of the error, lowercased,
    digits collapsed to ``#``) so semantically-identical failures that differ
    only by id/path/line/timestamp still collapse to ONE signature and count as
    the same repeat.
    """
    text: Optional[str] = None
    if isinstance(result, dict):
        err = result.get("error")
        text = err if isinstance(err, str) else (json.dumps(err) if err else None)
    elif isinstance(result, str):
        s = result.strip()
        if s.startswith("{"):
            try:
                data = json.loads(s)
            except Exception:
                data = None
            if isinstance(data, dict) and data.get("error"):
                e = data["error"]
                text = e if isinstance(e, str) else json.dumps(e)
        if text is None:
            low = s.lower()
            if low.startswith("error") or "traceback (most recent call last)" in low:
                text = s
    if not text:
        return None
    first_line = text.strip().splitlines()[0].lower()
    normalized = "".join("#" if c.isdigit() else c for c in first_line)[:160]
    return f"{tool_name}:{normalized}"


@dataclass
class StuckDetector:
    """Tracks *consecutive* identical tool failures across turns.

    ``threshold``      — identical (tool, error) count that triggers a NUDGE.
    ``escalate_after`` — additional identical failures *after* a nudge that
                         trigger ESCALATE. Total repeats to escalate =
                         ``threshold + escalate_after``.

    Keys on consecutive repeats of the same signature: any turn that produces a
    different signature (or a success / non-error result) resets the run. This
    is deliberately conservative — it fires on a true stuck loop (hammering one
    broken action), never on an agent making varied progress.
    """

    threshold: int = 3
    escalate_after: int = 2
    _last_sig: Optional[str] = field(default=None, repr=False)
    _run: int = field(default=0, repr=False)
    _nudged: bool = field(default=False, repr=False)

    def record(self, tool_name: str, result: Any) -> StuckVerdict:
        sig = result_error_signature(tool_name, result)
        if sig is None:
            self._reset()               # success / non-error breaks the streak
            return StuckVerdict.PROCEED
        if sig != self._last_sig:
            self._last_sig = sig
            self._run = 1
            self._nudged = False
            return StuckVerdict.PROCEED
        self._run += 1
        if self._nudged and self._run >= self.threshold + self.escalate_after:
            return StuckVerdict.ESCALATE
        if not self._nudged and self._run >= self.threshold:
            self._nudged = True
            return StuckVerdict.NUDGE
        return StuckVerdict.PROCEED

    def note_new_approach(self) -> None:
        """Call when the agent switches tool/args on its own — resets the streak
        so a later, unrelated repeat starts counting fresh."""
        self._reset()

    def _reset(self) -> None:
        self._last_sig = None
        self._run = 0
        self._nudged = False

    @property
    def current_signature(self) -> Optional[str]:
        return self._last_sig

    @property
    def run_length(self) -> int:
        return self._run


__all__ = ["StuckDetector", "StuckVerdict", "result_error_signature"]


# -- Self-test (run: `python3 agent/stuck_detector.py`) ----------------------
if __name__ == "__main__":
    def _eq(got, want, msg):
        assert got == want, f"FAIL {msg}: got {got!r}, want {want!r}"

    _eq(result_error_signature("read_file", '{"bytes_written": 12}'), None, "success has no sig")
    _eq(result_error_signature("web_search", "here are 5 results"), None, "non-error string no sig")
    s1 = result_error_signature("skill_view", '{"error": "skill \'foo\' not found (id 4821)"}')
    s2 = result_error_signature("skill_view", '{"error": "skill \'foo\' not found (id 9999)"}')
    assert s1 and s1.startswith("skill_view:"), "json error yields sig"
    _eq(s1, s2, "digit-varying ids collapse to same sig")
    assert result_error_signature("t", "Error: boom\nline 2").startswith("t:error"), "plain error marker"
    assert result_error_signature("t", "Traceback (most recent call last):\n ...") is not None, "traceback detected"

    d = StuckDetector(threshold=3, escalate_after=2)
    err = '{"error": "skill \'x\' not found"}'
    _eq(d.record("skill_view", err), StuckVerdict.PROCEED, "hit 1")
    _eq(d.record("skill_view", err), StuckVerdict.PROCEED, "hit 2")
    _eq(d.record("skill_view", err), StuckVerdict.NUDGE, "hit 3 -> nudge")
    _eq(d.record("skill_view", err), StuckVerdict.PROCEED, "hit 4")
    _eq(d.record("skill_view", err), StuckVerdict.ESCALATE, "hit 5 -> escalate")

    d = StuckDetector(threshold=3)
    _eq(d.record("read_file", '{"error": "no such file a"}'), StuckVerdict.PROCEED, "varied 1")
    _eq(d.record("read_file", '{"error": "no such file b"}'), StuckVerdict.PROCEED, "varied 2")
    _eq(d.record("web_search", "ok results"), StuckVerdict.PROCEED, "varied 3 (success)")
    _eq(d.record("read_file", '{"error": "no such file a"}'), StuckVerdict.PROCEED, "varied 4")

    d = StuckDetector(threshold=3)
    err = '{"error": "boom"}'
    d.record("t", err); d.record("t", err)
    _eq(d.record("t", "success!"), StuckVerdict.PROCEED, "success resets")
    _eq(d.record("t", err), StuckVerdict.PROCEED, "post-reset 1")
    _eq(d.record("t", err), StuckVerdict.PROCEED, "post-reset 2")
    _eq(d.record("t", err), StuckVerdict.NUDGE, "post-reset 3 -> nudge")

    d = StuckDetector(threshold=3)
    _eq(d.record("a", '{"error": "x"}'), StuckVerdict.PROCEED, "toolA 1")
    _eq(d.record("b", '{"error": "x"}'), StuckVerdict.PROCEED, "toolB resets")
    _eq(d.record("b", '{"error": "x"}'), StuckVerdict.PROCEED, "toolB 2")

    print("stuck_detector self-test: ALL PASS")
