"""Trace pass-macro commands through the real input queue without changing physics.

Call ``before_step`` after MacroState.apply, then ``after_step`` after env.step.
An attempt ID travels alongside act_hist, so cancelling the controller does not
erase a kick still waiting for latency. Delays and frame_skip use exactly the
index calculation in kernel.decision_step, including non-multiple delays.

The environment ORs kicks/contacts across a decision. We count a kick only when
its attempt is identifiable; ambiguous observations have separate counters.
``_s_kick`` identifies a kick on the LAST simulated tick. ``_s_kinfo`` has no
timestamp and cannot identify earlier kicks, so it is deliberately not used.
Reception means a later, exclusive contact by the selected receiver. Retention
means observed team possession (no rival contact/out/reset for retention_ticks),
not demonstrated ball control. These diagnostics are unsuitable as success
rewards for drills that require physical control.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


COUNTERS = (
    "macro_starts", "macro_kick_emitted", "macro_applied_action_ticks",
    "macro_kicks", "macro_kicks_ambiguous", "untracked_kicks",
    "macro_receives", "macro_outcome_ambiguous", "macro_interceptions",
    "macro_other_teammate", "macro_sender_retouch", "macro_out",
    "macro_retained", "macro_retention_lost", "macro_goal_after_receive",
    "macro_timeout", "macro_reset_pending", "macro_cancelled_before_kick",
)


@dataclass
class _Attempt:
    row: int
    sender: int
    receiver: int
    kicked: bool = False
    kick_ambiguous: bool = False
    terminal: bool = False
    kick_end: int = 0
    received_end: int | None = None


class MacroTrace:
    """Optional per-env diagnostic; all outcome counters belong to the sender.

    ``stats(mask)`` accepts a (N, 8) sender mask (e.g. arena.learner). Counters
    are cumulative and JSON serializable; ``reset_stats`` preserves provenance.
    ``after_step`` returns independent (N, 8) integer arrays for that decision,
    including starts/emissions recorded by before_step. Reset explicit rows
    after externally replacing a state/episode; goal/end resets are automatic.
    Attempts already queued when attaching/resetting the trace are untracked.
    """

    def __init__(self, env, *, retention_ticks=120, max_flight_ticks=180):
        if retention_ticks < 1 or max_flight_ticks < 1:
            raise ValueError("retention_ticks and max_flight_ticks must be positive")
        self.shape = tuple(env.active.shape)
        self.retention_ticks = int(retention_ticks)
        self.max_flight_ticks = int(max_flight_ticks)
        self.origin_hist = np.full(env.act_hist.shape, -1, np.int64)
        self._hist_actions = env.act_hist.copy()
        self._episode = np.asarray(env.episode).copy()
        self._resync = np.zeros(env.N, bool)
        self._clock = np.zeros(env.N, np.int64)
        self._last_k = np.full(self.shape, -1, np.int64)
        self._current = np.zeros(self.shape, np.int64)
        self._attempts: dict[int, _Attempt] = {}
        self._flights: dict[int, _Attempt] = {}
        self._next_flight = 1
        self._next_id = 1
        self._pending = None
        self._counts = {key: np.zeros(self.shape, np.int64) for key in COUNTERS}
        self._delta = {key: np.zeros(self.shape, np.int64) for key in COUNTERS}

    def _add(self, key, row, player, amount=1):
        self._counts[key][row, player] += amount
        self._delta[key][row, player] += amount

    def stats(self, mask=None):
        """Return cumulative integer counters, optionally restricted to senders."""
        if mask is None:
            return {key: int(value.sum()) for key, value in self._counts.items()}
        mask = np.asarray(mask, bool)
        if mask.shape != self.shape:
            raise ValueError("sender mask must have shape (N, 8)")
        return {key: int(value[mask].sum()) for key, value in self._counts.items()}

    def reset_stats(self):
        """Start another logging window without discarding delayed actions."""
        for value in self._counts.values():
            value.fill(0)

    def reset(self, rows):
        """Forget provenance/outcomes of replaced episodes, retaining counters."""
        if self._pending is not None:
            raise RuntimeError("reset must happen outside a before_step/after_step pair")
        rows = np.asarray(rows)
        if rows.dtype == bool:
            rows = np.flatnonzero(rows)
        rows = np.asarray(rows, np.int64).reshape(-1)
        if not len(rows):
            return
        chosen = set(rows.tolist())
        for ident in list(self._flights):
            attempt = self._flights[ident]
            if attempt.row in chosen:
                self._add("macro_reset_pending", attempt.row, attempt.sender)
                del self._flights[ident]
        self._attempts = {i: a for i, a in self._attempts.items() if a.row not in chosen}
        self.origin_hist[rows] = -1
        self._current[rows] = 0
        self._last_k[rows] = -1
        self._resync[rows] = True
        self._clock[rows] = 0

    def before_step(self, env, macro_state, primitive_actions, sampled=None, starts=None):
        """Label the actions about to enter act_hist; does not mutate env.

        ``starts`` is an optional accepted-start boolean (N, 8). Without it,
        MacroState.steps == 0 identifies new starts, including restarting the
        same receiver on consecutive decisions. ``sampled`` can be supplied for
        integration symmetry, but a rejected sampled macro is never counted.
        The state must be the one AFTER applying macros to primitive_actions.
        """
        if self._pending is not None:
            raise RuntimeError("after_step is required before another before_step")
        for value in self._delta.values():
            value.fill(0)
        changed = np.flatnonzero(np.asarray(env.episode) != self._episode)
        self.reset(changed)
        self._episode[:] = env.episode
        self._hist_actions[self._resync] = env.act_hist[self._resync]
        self._resync.fill(False)
        if not np.array_equal(self._hist_actions, env.act_hist):
            raise RuntimeError("env was advanced/replaced without tracing; call reset(rows)")
        primitive = np.asarray(primitive_actions, np.int64)
        if primitive.shape != self.shape or np.any((primitive < 0) | (primitive >= 18)):
            raise ValueError("primitive_actions must have shape (N, 8) and values 0..17")
        k = np.asarray(macro_state.k, np.int64)
        if k.shape != self.shape:
            raise ValueError("MacroState must have shape (N, 8)")
        running = (k >= 0) & env.active
        if starts is None:
            steps = getattr(macro_state, "steps", np.ones(self.shape, np.int64))
            new = running & ((k != self._last_k) | (steps == 0) | (self._current == 0))
        else:
            new = np.asarray(starts, bool)
            if new.shape != self.shape:
                raise ValueError("starts must have shape (N, 8)")
            new = new & running
            # Attaching mid-macro still needs an ID, but is not a new start.
        for n, p in np.argwhere(running & (new | (self._current == 0))):
            mates = np.flatnonzero(env.active[n] & (env.team == env.team[p]))
            mates = mates[mates != p]
            if k[n, p] >= len(mates):
                raise ValueError("macro receiver index is not an active teammate")
            ident = self._next_id
            self._next_id += 1
            self._attempts[ident] = _Attempt(int(n), int(p), int(mates[k[n, p]]))
            self._current[n, p] = ident
            if new[n, p]:
                self._add("macro_starts", n, p)
        self._current[~running] = 0
        self._last_k[:] = k
        emitted = running & (primitive >= 9)
        self._counts["macro_kick_emitted"] += emitted
        self._delta["macro_kick_emitted"] += emitted
        world = primitive.copy()
        from env.rs4z.core import MIRROR_ACTION
        world[:, 4:] = MIRROR_ACTION[world[:, 4:]]
        world[~env.active] = 0
        if self.origin_hist.shape[2] > 1:
            self.origin_hist[:, :, 1:] = self.origin_hist[:, :, :-1]
            self._hist_actions[:, :, 1:] = self._hist_actions[:, :, :-1]
        self.origin_hist[:, :, 0] = np.where(running, self._current, 0)
        self._hist_actions[:, :, 0] = world
        self._pending = dict(cancel=env.kick_cancel.copy(), active=env.active.copy(),
                             delay=env.delay.copy(), frame_skip=int(env.frame_skip))

    def _finish(self, ident, key):
        attempt = self._flights[ident]
        self._add(key, attempt.row, attempt.sender)
        attempt.terminal = True
        del self._flights[ident]

    def after_step(self, env, events):
        """Observe aggregated events and return per-sender counter increments."""
        if self._pending is None:
            raise RuntimeError("before_step is required before after_step")
        pending = self._pending
        if not np.array_equal(self._hist_actions, env.act_hist):
            raise RuntimeError("env.step actions differ from traced primitive_actions")
        fs = pending["frame_skip"]
        ticks = np.asarray(events["ticks"], np.int64)
        if ticks.shape != (env.N,) or np.any((ticks < 0) | (ticks > fs)):
            raise ValueError("events.ticks must contain actual simulated ticks for each row")
        kicked = np.asarray(events["kicked"], bool)
        touched = np.asarray(events["touched"], bool) | kicked
        goal = np.asarray(events.get("goal", np.zeros(env.N)), np.int64)
        out = np.asarray(events.get("out", np.zeros(env.N)), bool)
        end = np.asarray(events.get("match_end", np.zeros(env.N)), bool)
        safety = np.asarray(events.get("safety", np.zeros(env.N)), np.int64)
        restart = np.asarray(events.get("restart_start", np.zeros(env.N)), np.int64)
        disrupted = out | (restart != 0) | (safety != 0)
        reset_rows = np.flatnonzero((goal != 0) | end | (safety == 2))
        self._clock += ticks
        origins = np.zeros((env.N, self.shape[1], fs), np.int64)
        actions = np.zeros_like(origins)
        for tick in range(fs):
            lag = pending["delay"] - tick
            h = np.minimum(np.maximum((lag + fs - 1) // fs, 0), self.origin_hist.shape[2] - 1)
            origins[..., tick] = np.take_along_axis(self.origin_hist, h[..., None], 2)[..., 0]
            actions[..., tick] = np.take_along_axis(self._hist_actions, h[..., None], 2)[..., 0]
            applied = (origins[..., tick] > 0) & pending["active"] & (ticks[:, None] > tick)
            self._counts["macro_applied_action_ticks"] += applied
            self._delta["macro_applied_action_ticks"] += applied

        # Previously launched passes can be resolved from this whole decision.
        for ident in list(self._flights):
            attempt = self._flights[ident]
            n, p = attempt.row, attempt.sender
            contacts = np.flatnonzero(touched[n])
            if attempt.received_end is not None:
                if out[n] or (contacts.size and np.any(env.team[contacts] != env.team[p])):
                    self._finish(ident, "macro_retention_lost")
                elif goal[n] != 0:
                    key = "macro_goal_after_receive" if (0 if goal[n] > 0 else 1) == env.team[p] else "macro_retention_lost"
                    self._finish(ident, key)
                elif n in reset_rows or disrupted[n]:
                    self._finish(ident, "macro_reset_pending")
                elif self._clock[n] - attempt.received_end >= self.retention_ticks:
                    self._finish(ident, "macro_retained")
            elif out[n]:
                if contacts.size:
                    self._finish(ident, "macro_outcome_ambiguous")
                else:
                    self._finish(ident, "macro_out")
            elif goal[n] != 0 or end[n] or disrupted[n]:
                self._finish(ident, "macro_outcome_ambiguous" if contacts.size else "macro_reset_pending")
            elif contacts.size:
                if np.all(env.team[contacts] != env.team[p]):
                    self._finish(ident, "macro_interceptions")
                elif len(contacts) != 1:
                    self._finish(ident, "macro_outcome_ambiguous")
                elif contacts[0] == attempt.receiver:
                    self._add("macro_receives", n, p)
                    attempt.received_end = int(self._clock[n])
                elif contacts[0] == p:
                    self._finish(ident, "macro_sender_retouch")
                else:
                    self._finish(ident, "macro_other_teammate")
            elif self._clock[n] - attempt.kick_end >= self.max_flight_ticks:
                self._finish(ident, "macro_timeout")

        for n, p in np.argwhere(kicked):
            candidates = []
            can_kick = not pending["cancel"][n, p]
            for tick in range(int(ticks[n])):
                if actions[n, p, tick] < 9:
                    can_kick = True
                elif can_kick or disrupted[n]:
                    candidates.append(tick)
            unique = {int(origins[n, p, tick]) for tick in candidates}
            final_kick = bool(env._s_kick[n, p]) and ticks[n] > 0
            # Restart may reset the kick latch during the same held action, so
            # even the per-player aggregate multiplicity becomes uncertain.
            if final_kick and not disrupted[n]:
                unique = {int(origins[n, p, ticks[n] - 1])}
            ident = next(iter(unique)) if len(unique) == 1 and not disrupted[n] else -1
            if ident > 0 and ident in self._attempts:
                self._add("macro_kicks", n, p)
                attempt = self._attempts[ident]
                attempt.kicked = True
                # A macro may emit release/kick commands before its first
                # delayed kick is observed. Trace every resulting ball launch.
                for previous, flight in list(self._flights.items()):
                    if flight.row == n and flight.received_end is None:
                        self._finish(previous, "macro_outcome_ambiguous")
                others = np.flatnonzero(touched[n] & (np.arange(self.shape[1]) != p))
                if others.size or out[n] or restart[n] or goal[n] or end[n]:
                    self._add("macro_outcome_ambiguous", n, p)
                else:
                    self._flights[self._next_flight] = _Attempt(
                        attempt.row, attempt.sender, attempt.receiver, kicked=True,
                        kick_end=int(self._clock[n]),
                    )
                    self._next_flight += 1
            elif any(origin > 0 for origin in unique):
                self._add("macro_kicks_ambiguous", n, p)
                for origin in unique:
                    if origin in self._attempts:
                        self._attempts[origin].kick_ambiguous = True
            elif -1 in unique or not unique:
                self._add("untracked_kicks", n, p)

        self._pending = None
        self.reset(reset_rows)
        # Reclaim cancelled unlaunched attempts only after their queue drained.
        queued = set(self.origin_hist[self.origin_hist > 0].tolist())
        live = queued | set(self._current[self._current > 0].tolist())
        for ident in list(self._attempts):
            if ident not in live:
                attempt = self._attempts.pop(ident)
                if not attempt.kicked and not attempt.kick_ambiguous:
                    self._add("macro_cancelled_before_kick", attempt.row, attempt.sender)
        return {key: value.copy() for key, value in self._delta.items()}
