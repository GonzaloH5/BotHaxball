"""Naturally pressured human 4v4 starts, without changing players or rewards.

Construct the bank only from ``x4_data.load(split_names(..., 'train'))``.
X4Data does not retain split labels, so the caller owns that restriction. The
bank selects recorded states; it never replays future kick-event labels, moves
players, or turns a successful human continuation into an agent reward.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import math

import numpy as np

from env.rs4z import contract as C, obs_v3
from learn.x4_coop_curriculum import CoopBank, TEAM
from learn.x4_data import move_of_input

PRESSURE_RADIUS = 75.0
RETENTION_TICKS = 120
SOURCE_LEAD_TICKS = 12
FRAME_SKIP = 3


def _stable_window(data, lo, hi):
    """Complete, finite, consecutive X4Data ticks with one unchanged roster.

    ``valid`` is sorted by the X4Data loader. Its construction already checks
    original replay frame continuity and the full 4v4 count. Searching the
    small interval avoids creating a dataset-sized mask on every reset.
    """
    if lo < 0 or hi <= lo or hi > len(data.state):
        return False
    valid = np.asarray(data.valid)
    start, end = np.searchsorted(valid, (lo, hi))
    if not np.array_equal(valid[start:end], np.arange(lo, hi)):
        return False
    pid = np.asarray(data.pid[lo:hi])
    if pid.shape != (hi - lo, 8) or (pid < 0).any() or len(np.unique(pid[0])) != 8:
        return False
    if not np.all(pid == pid[0]):
        return False
    recording = np.asarray(data.rec_of_tick[lo:hi])
    if (recording < 0).any() or not np.all(recording == recording[0]):
        return False
    maps = np.asarray(data.map_idx[lo:hi])
    if not np.all(maps == maps[0]) or not 0 <= int(maps[0]) < len(obs_v3.MAP_NAMES):
        return False
    if not np.isin(data.state[lo:hi], (0, 1)).all():
        return False
    for name, shape in (("ball", (hi - lo, 4)), ("pos", (hi - lo, 8, 2)),
                        ("vel", (hi - lo, 8, 2)), ("inp", (hi - lo, 8)),
                        ("kicking", (hi - lo, 8))):
        values = np.asarray(getattr(data, name)[lo:hi])
        if values.shape != shape or not np.isfinite(values).all():
            return False
    for name in ("ball_r", "mass", "kstr"):
        values = np.asarray(getattr(data, name)[lo:hi])
        if not np.isfinite(values).all() or (values <= 0).any():
            return False
    return True


def valid_source_window(data, t, delays, history_len):
    """Whether raw pending inputs can be restored faithfully at source tick t.

    This helper deliberately accepts open play, kickoffs and restarts. Both
    experimental arms can therefore use identical restoration for ordinary
    human starts. The pressure bank separately requires uninterrupted open
    play. ``delays`` is either an integer or the eight assigned robot delays.
    """
    try:
        if int(t) != t or int(history_len) != history_len:
            return False
        t, history_len = int(t), int(history_len)
        raw = np.broadcast_to(np.asarray(delays), (8,))
        if not np.isfinite(raw).all():
            return False
        delay = raw.astype(np.int64)
    except (TypeError, ValueError, OverflowError):
        return False
    if history_len < 1 or not np.array_equal(raw, delay) or (delay < 0).any():
        return False
    lo = min(t, t + int(delay.min()) + 1 - FRAME_SKIP * history_len)
    hi = t + int(delay.max()) + 2
    return _stable_window(data, lo, hi)


def restore_source_history(env, row, data, t):
    """Restore world-frame held inputs, never future imitation kick labels.

    A decision issued at t-3(h+1) applies at t+D+1-3(h+1), exactly as in
    CoopCurriculum._source_actions. The caller restores physics/restart state
    first. This function only changes action history and stale contact events.
    Invalid windows fail before any history is modified.
    """
    row, t = int(row), int(t)
    if int(env.frame_skip) != FRAME_SKIP:
        raise ValueError("source history restoration requires frame_skip=3")
    delay = env.delay[row]
    if not valid_source_window(data, t, delay, env.H):
        raise ValueError(f"invalid source history window at tick {t}")
    roster = env.active[row]
    for h in range(env.H):
        idx = t + delay + 1 - FRAME_SKIP * (h + 1)
        raw = data.inp[idx, np.arange(8)]
        world = move_of_input(raw) + 9 * ((raw & 16) != 0)
        env.act_hist[row, :, h] = np.where(roster, world, 0)
    raw = data.inp[t]
    env._s_act[row] = np.where(roster, move_of_input(raw) + 9 * ((raw & 16) != 0), 0)
    env.ev_kicked[row] = env.ev_touch[row] = False
    return t


class PressureStartBank:
    """Select clean human passes released under pressure and retained for 2 s.

    Pressure is measured from the passer to the nearest recorded opponent at
    release. After the initial reception, repasses are allowed: no rival touch,
    out or discontinuity may occur for 120 ticks, and some teammate must be in
    physical contact range at the endpoint. Starts precede the recorded release
    by max_delay+12 ticks, keeping that release outside the inherited queue.
    """

    def __init__(self, data, seed=0, max_delay=15, min_examples=200, min_recordings=10):
        if (int(max_delay) != max_delay or max_delay < 0 or min_examples < 1 or min_recordings < 1
                or int(min_examples) != min_examples or int(min_recordings) != min_recordings):
            raise ValueError("invalid pressure-start delay or evidence requirements")
        if getattr(data, "split", "train") != "train":
            raise ValueError("pressure starts must be mined from the train split only")
        self.d, self.rng = data, np.random.default_rng(seed)
        self.max_delay = int(max_delay)
        history_len = max(3, 1 + math.ceil(self.max_delay / FRAME_SKIP))
        # These are fixed density/scan bounds, rather than an unbounded search
        # through later source ticks for every release.
        human = CoopBank(data, max_records=20000, max_candidates=100000)
        accepted, indices, rejected = [], [], Counter()
        for record in human.records:
            release, receive, passer, _, team = map(int, record)
            foe = TEAM != team
            pressure = np.linalg.norm(data.pos[release, foe] - data.pos[release, passer], axis=1).min()
            if pressure > PRESSURE_RADIUS:
                rejected["not_pressured"] += 1
                continue
            endpoint = receive + RETENTION_TICKS
            t = release - self.max_delay - SOURCE_LEAD_TICKS
            # Validate the full possible delay range, including the forward
            # inputs used by the largest assigned delay and the oldest D=0
            # history. A single mixed delay vector spans both boundaries.
            delays = np.full(8, self.max_delay, np.int64)
            delays[0] = 0
            if not valid_source_window(data, t, delays, history_len):
                rejected["source_history"] += 1
                continue
            if not _stable_window(data, t, endpoint + 1):
                rejected["continuity"] += 1
                continue
            span = slice(t, endpoint + 1)
            if not ((data.state[span] == 1) & (data.rkind[span] == 0)).all():
                rejected["not_open"] += 1
                continue
            ball = data.ball[span, :2]
            if (np.abs(ball[:, 0]) > C.LINE_W).any() or (np.abs(ball[:, 1]) > C.LINE_H).any():
                rejected["out"] += 1
                continue
            span = slice(release, endpoint + 1)
            distances = np.linalg.norm(data.pos[span, foe] - data.ball[span, None, :2], axis=2)
            contacts = distances <= C.PLAYER_RADIUS + data.ball_r[span, None] + .05
            if contacts.any() or data.kick_ev[span, foe].any():
                rejected["rival_touch"] += 1
                continue
            endpoint_dist = np.linalg.norm(data.pos[endpoint, TEAM == team] - data.ball[endpoint, :2], axis=1)
            if endpoint_dist.min() > C.PLAYER_RADIUS + data.ball_r[endpoint] + .05:
                rejected["no_control_at_2s"] += 1
                continue
            accepted.append(record)
            indices.append(t)
        self.records = np.asarray(accepted, np.int64).reshape(-1, 5)
        self.indices = np.asarray(indices, np.int64)
        rec_ids = np.asarray(data.rec_of_tick[self.indices], np.int64)
        counts = Counter(map(int, rec_ids))
        names = getattr(data, "names", None)
        source_names = [str(names[r]) if names is not None and r < len(names) else str(r) for r in sorted(counts)]
        self.manifest = dict(version=1, split="train", seed=int(seed), max_delay=self.max_delay,
                             pressure_radius_px=PRESSURE_RADIUS, pressure_reference="passer",
                             retention_ticks=RETENTION_TICKS, endpoint_control="any_teammate_physical",
                             source_lead_ticks=SOURCE_LEAD_TICKS, history_len=history_len,
                             cooperative_candidates=len(human.records), examples=len(self.indices),
                             recordings=len(counts), source_recording_names=source_names,
                             recording_counts={str(r): n for r, n in sorted(counts.items())},
                             rejections=dict(rejected), min_examples=int(min_examples),
                             min_recordings=int(min_recordings),
                             indices_sha256=hashlib.sha256(self.indices.tobytes()).hexdigest())
        if len(self.indices) < min_examples or len(counts) < min_recordings:
            raise ValueError(f"insufficient pressure starts: {len(self.indices)} examples from {len(counts)} recordings; "
                             f"need >= {min_examples} examples and >= {min_recordings} recordings; "
                             f"rejections={dict(rejected)}")

    def sample(self, rng=None):
        rng = self.rng if rng is None else rng
        return int(self.indices[int(rng.integers(len(self.indices)))])
