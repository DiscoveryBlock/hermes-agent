"""Loop guards: integration glue for stuck-detection + checkpoint-continue.

Wires the two pure policy modules (stuck_detector, continuation_policy) into the
conversation loop behind a SINGLE, config-gated entry object (LoopGuardState) so
conversation_loop.py needs only a few anchored call sites. Everything here is
defensively guarded — a bug in a guard must NEVER break a turn; on any internal
error the helpers degrade to original loop behavior.

Feature 1 (stuck) + Feature 2 (checkpoint-continue), requested by Brian
2026-07-20. See stuck_detector.py / continuation_policy.py for the pure policies
and their unit tests. This module is the ONLY place that touches agent state.

GATE: reads ~/.hermes/state/loop_guards.json {"enabled": bool, ...}. Defaults to
DISABLED, so a rebase-restart loads this code INERT; flip the JSON to activate on
the next turn (no restart needed), flip back to roll back instantly.

Survival: idempotent re-apply via ~/.hermes/routing/patch_loop_guards.py, wired
into hermes-auto-update.sh REAPPLY_LOCAL_PATCHES. conversation_loop.py is NOT
committed (churny upstream file); the patch re-applies its call sites (anchored
by string) after every nightly rebase.
"""
from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:  # normal in-package import
    from agent.stuck_detector import StuckDetector, StuckVerdict
    from agent.continuation_policy import ContinuationPolicy, ContinuationDecision
except ImportError:  # standalone self-test (python3 agent/loop_guards.py)
    from stuck_detector import StuckDetector, StuckVerdict
    from continuation_policy import ContinuationPolicy, ContinuationDecision

_CONFIG_PATH = Path.home() / ".hermes" / "state" / "loop_guards.json"
_EVENTS_PATH = Path.home() / ".hermes" / "state" / "loop-guard-events.jsonl"
_NOTIFY_SCRIPT = Path.home() / ".hermes" / "scripts" / "notify-alert.py"

_DEFAULTS = {
    "enabled": False,
    "stuck_threshold": 3,
    "stuck_escalate_after": 2,
    "max_auto_continues": 3,
    "grace_iterations": 30,
}


def load_config() -> dict:
    cfg = dict(_DEFAULTS)
    try:
        if _CONFIG_PATH.exists():
            data = json.loads(_CONFIG_PATH.read_text())
            if isinstance(data, dict):
                cfg.update({k: data[k] for k in _DEFAULTS if k in data})
    except Exception:
        pass
    return cfg


@dataclass
class LoopGuardState:
    enabled: bool
    detector: StuckDetector
    policy: ContinuationPolicy
    auto_continues_used: int = 0
    granted_grace: int = 0
    stuck: bool = False
    pending_handoff: Optional[str] = None
    _notified: bool = field(default=False, repr=False)


def init_state(agent) -> Optional[LoopGuardState]:
    """Build per-turn guard state, or None when disabled/unavailable.

    None => every call site short-circuits to original loop behavior.
    """
    try:
        cfg = load_config()
        if not cfg.get("enabled"):
            return None
        return LoopGuardState(
            enabled=True,
            detector=StuckDetector(
                threshold=int(cfg["stuck_threshold"]),
                escalate_after=int(cfg["stuck_escalate_after"]),
            ),
            policy=ContinuationPolicy(
                max_auto_continues=int(cfg["max_auto_continues"]),
                grace_iterations=int(cfg["grace_iterations"]),
            ),
        )
    except Exception:
        return None


def _raw_condition(agent, api_call_count: int, grace: int) -> bool:
    """The ORIGINAL loop condition, with an optional turn-local grace added to
    the iteration cap. grace=0 reproduces the un-patched condition exactly."""
    return (
        (api_call_count < agent.max_iterations + grace
         and agent.iteration_budget.remaining > 0)
        or agent._budget_grace_call
    )


def should_continue(agent, api_call_count: int, messages: list,
                    state: Optional[LoopGuardState]) -> bool:
    """While-loop condition replacement. Returns True to keep iterating.

    Disabled/None => exact original condition (grace=0). Enabled => on cap-hit,
    consult ContinuationPolicy: CONTINUE extends the (turn-local) budget and
    injects a checkpoint; ESCALATE stashes a handoff and stops.
    """
    grace = state.granted_grace if state is not None else 0
    if _raw_condition(agent, api_call_count, grace):
        return True
    if state is None or not state.enabled:
        return False
    try:
        decision = state.policy.decide(
            auto_continues_used=state.auto_continues_used,
            stuck=state.stuck,
        )
        if decision == ContinuationDecision.CONTINUE:
            state.auto_continues_used += 1
            state.granted_grace += state.policy.grace_iterations
            # turn-local: iteration_budget is rebuilt every turn (turn_context),
            # so extending max_total here does NOT leak into later turns. We do
            # NOT touch agent.max_iterations (persistent) — grace lives in state.
            try:
                agent.iteration_budget.max_total += state.policy.grace_iterations
            except Exception:
                pass
            _inject_checkpoint(messages, state)
            _emit_event("continue", agent, state, api_call_count)
            return True
        # ESCALATE at ceiling (stuck is handled earlier in record_tools)
        state.pending_handoff = _handoff_text(agent, state, reason="ceiling")
        _emit_event("escalate_ceiling", agent, state, api_call_count)
        _notify_once(agent, state, "hit its auto-continue ceiling and is escalating")
        return False
    except Exception:
        # never strand the loop on a guard bug — stop exactly as the un-patched
        # loop would on cap-hit.
        return False


def record_tools(agent, messages: list, pre_len: int, assistant_message,
                 state: Optional[LoopGuardState]) -> str:
    """Feed newly-appended tool results to the StuckDetector.

    Returns "escalate" | "nudge" | "proceed". On NUDGE, injects a change-approach
    message. On ESCALATE, sets state.stuck + pending_handoff so the caller breaks
    with a handoff (and continuation can then never resume a real loop).
    """
    if state is None or not state.enabled:
        return "proceed"
    try:
        id_to_name = {}
        for tc in (getattr(assistant_message, "tool_calls", None) or []):
            try:
                id_to_name[tc.id] = tc.function.name
            except Exception:
                pass
        worst = StuckVerdict.PROCEED
        worst_name = None
        for msg in messages[pre_len:]:
            if not isinstance(msg, dict) or msg.get("role") != "tool":
                continue
            name = id_to_name.get(msg.get("tool_call_id"), "tool")
            verdict = state.detector.record(name, msg.get("content"))
            if verdict == StuckVerdict.ESCALATE:
                worst, worst_name = StuckVerdict.ESCALATE, name
                break
            if verdict == StuckVerdict.NUDGE and worst == StuckVerdict.PROCEED:
                worst, worst_name = StuckVerdict.NUDGE, name
        if worst == StuckVerdict.ESCALATE:
            state.stuck = True
            state.pending_handoff = _handoff_text(agent, state, reason="stuck", tool=worst_name)
            _emit_event("escalate_stuck", agent, state, None, tool=worst_name)
            _notify_once(agent, state, f"is stuck repeating {worst_name} and is escalating")
            return "escalate"
        if worst == StuckVerdict.NUDGE:
            _inject_nudge(messages, state, worst_name)
            _emit_event("nudge", agent, state, None, tool=worst_name)
            return "nudge"
        return "proceed"
    except Exception:
        return "proceed"


def _inject_checkpoint(messages: list, state: LoopGuardState) -> None:
    try:
        messages.append({
            "role": "user",
            "content": (
                "[loop-guard] You've reached your iteration budget but this task "
                "isn't finished. In one short line note what's done and what's "
                "left, then keep going and finish it — you have a fresh budget. "
                f"(auto-continue {state.auto_continues_used}/"
                f"{state.policy.max_auto_continues})"
            ),
        })
    except Exception:
        pass


def _inject_nudge(messages: list, state: LoopGuardState, tool: Optional[str]) -> None:
    try:
        messages.append({
            "role": "user",
            "content": (
                f"[loop-guard] You've called {tool or 'the same tool'} and gotten "
                "the same error several times in a row. Stop repeating it — change "
                "your approach, try a different tool, or if you truly can't "
                "proceed, say so and summarize where you're stuck."
            ),
        })
    except Exception:
        pass


def _handoff_text(agent, state: LoopGuardState, reason: str,
                  tool: Optional[str] = None) -> str:
    if reason == "stuck":
        return (
            f"I've hit the same failure repeatedly ({tool or 'a tool'}) and can't "
            "make progress on my own, so I'm stopping rather than spinning. Here's "
            "where I got stuck — flagging this for a second set of eyes."
        )
    return (
        "I've worked well past my normal iteration budget and hit the "
        "auto-continue ceiling without finishing. I'm checkpointing here rather "
        "than looping indefinitely — this likely needs a hand-off or a smaller "
        "next step. Flagging it."
    )


def _emit_event(kind: str, agent, state: LoopGuardState,
                api_call_count: Optional[int], tool: Optional[str] = None) -> None:
    try:
        _EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        rec = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "kind": kind,
            "session": getattr(agent, "session_id", None) or "none",
            "model": getattr(agent, "model", None),
            "auto_continues_used": state.auto_continues_used,
            "api_call_count": api_call_count,
            "tool": tool,
        }
        with open(_EVENTS_PATH, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    except Exception:
        pass


def _notify_once(agent, state: LoopGuardState, what: str) -> None:
    """One Telegram ping per turn max; fire-and-forget so the loop never blocks."""
    if state._notified:
        return
    state._notified = True
    try:
        if not _NOTIFY_SCRIPT.exists():
            return
        summary = f"Kira {what} (session {getattr(agent, 'session_id', '?')})."
        subprocess.Popen(
            ["/usr/bin/python3", str(_NOTIFY_SCRIPT),
             "--key", "kira-loop-guard", "--severity", "warning",
             "--summary", summary],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


__all__ = ["LoopGuardState", "init_state", "should_continue", "record_tools", "load_config"]


# -- Self-test (run: python3 agent/loop_guards.py) --------------------------
if __name__ == "__main__":
    def _eq(got, want, msg):
        assert got == want, f"FAIL {msg}: got {got!r}, want {want!r}"

    class _Budget:
        def __init__(self, max_total):
            self.max_total = max_total
            self._used = 0
        @property
        def remaining(self):
            return max(0, self.max_total - self._used)
        def consume(self):
            if self._used >= self.max_total:
                return False
            self._used += 1
            return True

    class _Agent:
        def __init__(self, max_iter):
            self.max_iterations = max_iter
            self.iteration_budget = _Budget(max_iter)
            self._budget_grace_call = False
            self.session_id = "test"
            self.model = "test-model"

    def _mk_state(**kw):
        return LoopGuardState(
            enabled=True,
            detector=StuckDetector(threshold=3, escalate_after=2),
            policy=ContinuationPolicy(max_auto_continues=3, grace_iterations=30),
            **kw,
        )

    # disabled/None => raw condition, grace 0
    a = _Agent(5)
    _eq(should_continue(a, 4, [], None), True, "None: under cap continues")
    _eq(should_continue(a, 5, [], None), False, "None: at cap stops (no continuation)")

    # enabled: at cap -> CONTINUE extends turn-local budget, does NOT touch max_iterations
    a = _Agent(5)
    st = _mk_state()
    a.iteration_budget._used = 5  # exhaust budget too
    msgs = []
    _eq(should_continue(a, 5, msgs, st), True, "enabled: first cap-hit continues")
    _eq(a.max_iterations, 5, "max_iterations NOT mutated (persistent-safe)")
    _eq(a.iteration_budget.max_total, 35, "iteration_budget extended by grace (turn-local)")
    _eq(st.granted_grace, 30, "granted_grace tracked in state")
    _eq(st.auto_continues_used, 1, "one continue used")
    assert msgs and msgs[-1]["role"] == "user", "checkpoint injected"

    # ceiling: after max_auto_continues, ESCALATE (stop + handoff)
    a = _Agent(5); st = _mk_state(auto_continues_used=3)
    a.iteration_budget._used = a.iteration_budget.max_total
    _eq(should_continue(a, 999, [], st), False, "ceiling reached -> stop")
    assert st.pending_handoff, "handoff text set on ceiling escalate"

    # stuck: repeated identical tool error -> nudge then escalate
    class _TC:
        def __init__(self, id, name):
            self.id = id
            self.function = type("F", (), {"name": name})()
    class _AM:
        tool_calls = [_TC("c1", "skill_view")]
    a = _Agent(90); st = _mk_state()
    err = '{"error": "skill \'x\' not found"}'
    def _run_once(msgs_before_len, content):
        msgs = [None] * msgs_before_len
        msgs.append({"role": "tool", "tool_call_id": "c1", "content": content})
        return record_tools(a, msgs, msgs_before_len, _AM(), st)
    _eq(_run_once(0, err), "proceed", "stuck hit 1")
    _eq(_run_once(0, err), "proceed", "stuck hit 2")
    _eq(_run_once(0, err), "nudge", "stuck hit 3 -> nudge")
    _eq(_run_once(0, err), "proceed", "stuck hit 4")
    _eq(_run_once(0, err), "escalate", "stuck hit 5 -> escalate")
    _eq(st.stuck, True, "stuck flag set")
    assert st.pending_handoff, "handoff set on stuck escalate"

    # SAFETY: a stuck turn can NEVER earn a continuation
    a = _Agent(5); st = _mk_state(stuck=True)
    a.iteration_budget._used = a.iteration_budget.max_total
    _eq(should_continue(a, 999, [], st), False, "stuck => never continue (safety property)")

    print("loop_guards self-test: ALL PASS")
