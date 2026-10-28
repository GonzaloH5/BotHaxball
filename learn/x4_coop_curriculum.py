"""Short cooperative exercises mined from training recordings, using the real RS4Z physics.

The caller must construct ``CoopBank`` from the *training* split only. All rewards
are terminal team rewards, and the normal 4v4 rows receive no curriculum reward.
Stage advancement measures mastery of these exercises, not competitive strength.
The actor receives the normal public observation; ``descriptor`` is critic-only.
"""
from __future__ import annotations

from collections import deque
import math

import numpy as np

from env.rs4z import contract as C, kernel as K, obs_v3
from learn.x4_data import move_of_input

TEAM = np.array([0, 0, 0, 0, 1, 1, 1, 1])
MIN_PASS = 40.0
STAGES = ("human_reception", "pass_and_retain", "escape_pressure", "finish_advantage", "transfer_4v4")


class CoopBank:
    """Mine actual kicked passes, rejecting interceptions, discontinuities and bounces.

    ``records`` contains ``(release_t, receive_t, passer, receiver, team)``.
    Each candidate scans at most ``max_flight`` ticks. Contact is checked at every
    source tick: striding it can miss a short defensive touch. ``tick_stride``
    bounds candidate density without dropping effective kick events.
    """

    def __init__(self, data, *, limit=None, max_records=20000, max_candidates=100000,
                 max_flight=180, min_flight=18, tick_stride=3):
        if limit is not None:
            max_records = int(limit)
        if max_records <= 0 or max_candidates <= 0:
            raise ValueError("cooperative bank limits must be positive")
        self.d = data
        self.max_flight = int(max_flight)
        self.min_flight = int(min_flight)
        valid = np.zeros(len(data.state), bool)
        valid[data.valid] = True
        self.valid = valid
        live = valid & (data.state == 1) & (data.rkind == 0)
        if hasattr(data, "map_idx"):
            live &= data.map_idx == obs_v3.MAP_NAMES.index("sanguchito_rs_x4")
        self.live = live
        releases = np.flatnonzero(live & np.asarray(data.kick_ev).any(1))
        # Evenly bound the number of candidates; do not silently take only the
        # beginning of the first recording when a large bank is supplied.
        if len(releases) > max_candidates:
            releases = releases[np.linspace(0, len(releases) - 1, max_candidates, dtype=int)]
        records = []
        last_candidate = -10**9
        for t in releases:
            t = int(t)
            if t - last_candidate < tick_stride:
                continue
            last_candidate = t
            # Leave enough stable history for the longest backplay reset and
            # enough source input after each reset to rebuild delayed commands.
            if t < 60 or not live[t - 45:t + 1].all():
                continue
            if data.rec_of_tick[t - 45] != data.rec_of_tick[t]:
                continue
            kickers = np.flatnonzero(data.kick_ev[t])
            if len(kickers) != 1:
                continue
            p = int(kickers[0])
            team = int(TEAM[p])
            release = np.asarray(data.ball[t, :2])
            velocity = np.asarray(data.ball[t, 2:4])
            if np.linalg.norm(velocity) < 0.5:
                continue
            end = min(t + self.max_flight + 1, len(live))
            for u in range(t + 1, end):
                if (not live[u] or data.rec_of_tick[u] != data.rec_of_tick[t]
                        or not np.array_equal(data.pid[u], data.pid[t])):
                    break
                # No wall-bounce examples: their travel direction reverses or
                # reaches the exterior of the ordinary playing rectangle.
                ball = data.ball[u, :2]
                if abs(ball[0]) > C.LINE_W or abs(ball[1]) > C.LINE_H:
                    break
                if np.dot(data.ball[u, 2:4], velocity) < -0.1:
                    break
                dist = np.linalg.norm(data.pos[u] - ball, axis=1)
                contacts = (dist <= C.PLAYER_RADIUS + data.ball_r[u] + 0.05) | data.kick_ev[u]
                if (contacts & (TEAM != team)).any():
                    break
                others = np.flatnonzero(contacts & (TEAM == team) & (np.arange(8) != p))
                if not len(others):
                    continue
                q = int(others[np.argmin(dist[others])])
                travel = float(np.linalg.norm(ball - release))
                target = data.pos[t, q] - release
                aim = np.dot(velocity, target) / max(np.linalg.norm(velocity) * np.linalg.norm(target), 1e-9)
                if u - t >= self.min_flight and travel >= MIN_PASS and aim >= 0.8:
                    # Delay reconstruction can refer up to 15 ticks ahead.
                    tail = min(u + 16, len(live))
                    if (tail == u + 16 and live[u:tail].all()
                            and np.all(data.rec_of_tick[u:tail] == data.rec_of_tick[t])
                            and np.all(data.pid[u:tail] == data.pid[t])):
                        records.append((t, u, p, q, team))
                break
            if len(records) >= max_records:
                break
        self.records = np.asarray(records, dtype=np.int64).reshape(-1, 5)
        if not len(self.records):
            raise ValueError("no clean human passes with sufficient flight/history in the training bank")

    def __len__(self):
        return len(self.records)

    def sample(self, rng):
        return self.records[int(rng.integers(len(self.records)))]


def wilson_lower(successes, total, z=1.645):
    """One-sided 95% lower bound (equivalently two-sided 90% Wilson interval)."""
    if total <= 0:
        return 0.0
    p = successes / total
    den = 1.0 + z * z / total
    return (p + z * z / (2.0 * total) - z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total**2))) / den


def pressure_position(player_pos, active, defender, passer, ball_pos, ball_radius):
    """Bring a distant defender into pressure only at an unoccupied legal point.

    Search around the carrier, starting in the source defender's direction. All
    other active discs matter, including another defender; moving a disc into a
    teammate or the ball would create artificial impulses at the exercise start.
    Preserve the human source position if the small search has no legal result.
    """
    pp = np.asarray(player_pos)
    original = pp[defender].copy()
    delta = original - pp[passer]
    if np.linalg.norm(delta) <= 65.0:
        return original
    angle = math.atan2(float(delta[1]), float(delta[0]))
    others = np.asarray(active, bool).copy()
    others[defender] = False
    # Half a pixel prevents floating point contacts at the placement boundary.
    player_clearance = 2.0 * C.PLAYER_RADIUS + 0.5
    ball_clearance = C.PLAYER_RADIUS + float(ball_radius) + 0.5
    offsets = (0, 15, -15, 30, -30, 60, -60, 90, -90, 120, -120, 150, -150, 180)
    for radius in (65.0, 70.0, 60.0):
        for degrees in offsets:
            a = angle + math.radians(degrees)
            point = pp[passer] + radius * np.array([math.cos(a), math.sin(a)])
            if abs(point[0]) > C.LINE_W - C.PLAYER_RADIUS or abs(point[1]) > C.LINE_H - C.PLAYER_RADIUS:
                continue
            if np.linalg.norm(point - ball_pos) < ball_clearance:
                continue
            if (np.linalg.norm(pp[others] - point, axis=1) < player_clearance).any():
                continue
            return point
    return original


class CoopCurriculum:
    """Reset/score selected Arena rows; defenders remain frozen and never learn.

    Call ``reset`` *after* Arena.new_match. For exercise rows ``opp == -2``;
    the caller overrides the active nonlearner actions using ``defender_kind``
    (0 BC, 1 frozen starting policy, 2 scripted). Call ``before_step`` before
    env.step, then replace normal team rewards on ``active`` rows with ``step``
    rewards. Terminal masks must use match_end | task_done before resetting.
    """

    def __init__(self, bank, N, rng, *, frac=0.5, min_episodes=128, threshold=0.55,
                 window=512, timeout_s=12.0, retain_s=2.0, start_stage=0,
                 transfer_frac=None, replay_frac=0.0, mastery_windows=1,
                 min_stage_updates=0, possession_retention=False,
                 finish_frac=0.0, reward_cooperative_goals=False):
        if not 0.0 <= frac <= 0.5:
            raise ValueError("curriculum frac must be between 0 and 0.5")
        if min_episodes <= 0 or window < min_episodes or not 0 < threshold < 1:
            raise ValueError("invalid mastery window/threshold")
        if not 0 <= start_stage <= 4 or not 0 <= replay_frac <= 1 or mastery_windows < 1 or min_stage_updates < 0:
            raise ValueError("invalid persistent curriculum configuration")
        if transfer_frac is not None and not 0 <= transfer_frac <= frac:
            raise ValueError("transfer fraction must preserve the full-match allocation")
        if not 0 <= finish_frac <= 1 or finish_frac + replay_frac > 1:
            raise ValueError("replay and finishing allocations must fit inside exercise rows")
        self.bank, self.N, self.rng = bank, int(N), rng
        self.frac, self.min_episodes, self.threshold = float(frac), int(min_episodes), float(threshold)
        self.window_size, self.timeout_s, self.retain_s = int(window), float(timeout_s), float(retain_s)
        self.stage = int(start_stage)
        self.transfer_frac, self.replay_frac = transfer_frac, float(replay_frac)
        self.mastery_windows, self.min_stage_updates = int(mastery_windows), int(min_stage_updates)
        self.possession_retention = bool(possession_retention)
        self.finish_frac = float(finish_frac)
        self.reward_cooperative_goals = bool(reward_cooperative_goals)
        records = bank.records
        sign = np.where(records[:, 4] == 0, 1.0, -1.0)
        receiving_ball = bank.d.ball[records[:, 1] + 1, :2]
        finishing = (sign * receiving_ball[:, 0] >= C.LINE_W * .35)
        self.finishing_records = records[finishing]
        if finish_frac and not len(self.finishing_records):
            raise ValueError("no human receptions in the attacking third for finishing practice")
        self.mastery_streak = 0
        self.stage_started_learning_update = 0
        # Reserve rows rather than resampling at every episode boundary. Short
        # exercises would otherwise disappear from the rollout as soon as their
        # rows draw a much longer full match (duration-biased mixture).
        self._exercise_order = self.rng.permutation(self.N)
        self.exercise_rows = np.zeros(self.N, bool)
        self._reserve_rows()
        self.window = deque(maxlen=self.window_size)
        self.total_episodes = np.zeros(5, np.int64)
        self.total_successes = np.zeros(5, np.int64)
        self.active = np.zeros(self.N, bool)
        self.kind = np.full(self.N, -1, np.int64)
        self.attack_team = np.full(self.N, -1, np.int64)
        self.defender_kind = np.zeros(self.N, np.int64)
        self.age = np.zeros(self.N, np.int64)  # physical ticks; independent of frozen match clock
        self.received = np.zeros(self.N, bool)
        self.retained = np.zeros(self.N, np.int64)
        self.finish_kicked = np.zeros(self.N, bool)
        self.passer = np.full(self.N, -1, np.int64)
        self.receiver = np.full(self.N, -1, np.int64)
        self.pending = np.zeros(self.N, bool)
        self.finished = np.zeros(self.N, bool)
        self.release_ball = np.zeros((self.N, 2))
        self.release_dir = np.zeros((self.N, 2))
        self.start_ball = np.zeros((self.N, 2))
        self.escape_x = np.zeros(self.N)
        self.delay_max = np.zeros(self.N, np.int64)
        self.prev_ball = np.zeros((self.N, 2))
        self.prev_player = np.zeros((self.N, 8, 2))

    @property
    def exercise_fraction(self):
        if self.stage >= 4 and self.transfer_frac is not None:
            return self.transfer_frac
        return min(0.15, self.frac * 0.25) if self.stage >= 4 else self.frac

    def row_stage(self, row):
        # Fixed rows retain earlier pass skills; short replay episodes must not
        # disappear behind longer finishing episodes through reset sampling.
        rank = int(np.flatnonzero(self._exercise_order == row)[0])
        finish_count = int(self.exercise_rows.sum() * self.finish_frac) if self.stage < 3 else 0
        if rank < finish_count:
            return 3
        if self.stage <= 1 or not self.replay_frac:
            return self.stage
        replay_count = int(self.exercise_rows.sum() * self.replay_frac)
        rank -= finish_count
        return 1 + rank % min(2, self.stage - 1) if rank < replay_count else self.stage

    def _reserve_rows(self):
        self.exercise_rows[:] = False
        self.exercise_rows[self._exercise_order[:int(self.N * self.exercise_fraction)]] = True

    def reset(self, arena, rows):
        rows = np.atleast_1d(np.asarray(rows, np.int64))
        self.active[rows] = False
        self.kind[rows] = -1
        self.attack_team[rows] = -1
        self.age[rows] = self.retained[rows] = 0
        self.finish_kicked[rows] = False
        self.received[rows] = self.pending[rows] = False
        self.finished[rows] = False
        env, d = arena.env, self.bank.d
        for r in rows:
            if not self.exercise_rows[r]:
                continue
            stage = self.row_stage(r)
            if stage >= 3 and self.reward_cooperative_goals:
                if not len(self.finishing_records):
                    raise ValueError("finishing requires attacking-third reception sources")
                record = self.finishing_records[int(self.rng.integers(len(self.finishing_records)))]
            else:
                record = self.bank.sample(self.rng)
            release_t, receive_t, passer, receiver, team = map(int, record)
            delay = env.delay[r].copy()
            max_delay = int(delay.max())
            if stage >= 3 and self.reward_cooperative_goals:
                # Start after an actual human reception in attack, rather than
                # rewarding a delayed shot inherited from a pre-pass reset.
                t = receive_t + 1
            elif stage == 0:
                # Incoming human ball; allow a full delay before the reception.
                t = max(release_t + 1, receive_t - max_delay - 6)
                if receive_t - t <= max_delay + 1:
                    continue
            else:
                # Pending source commands cannot contain the recorded release.
                t = release_t - max_delay - int(self.rng.integers(6, 13))
            if not self._source_window_valid(t, delay, env.H):
                continue
            attacks = [passer, receiver]
            defenders = []
            pp = np.asarray(d.pos[t]).copy()
            if stage >= 2:
                mates = np.flatnonzero((TEAM == team) & ~np.isin(np.arange(8), attacks))
                if stage >= 3 or self.rng.random() < 0.5:
                    attacks.append(int(mates[np.argmin(np.linalg.norm(pp[mates] - d.ball[t, :2], axis=1))]))
                foes = np.flatnonzero(TEAM != team)
                order = foes[np.argsort(np.linalg.norm(pp[foes] - pp[passer], axis=1))]
                defenders = list(map(int, order[:len(attacks) - 1]))
                if stage >= 3 and self.reward_cooperative_goals:
                    nearest = int(foes[np.argmin(np.linalg.norm(pp[foes] - pp[receiver], axis=1))])
                    remaining = foes[foes != nearest]
                    target_goal = np.array([(1 if team == 0 else -1) * C.LINE_W, 0.0])
                    cover = int(remaining[np.argmin(np.linalg.norm(pp[remaining] - target_goal, axis=1))])
                    defenders = [nearest, cover]
                if stage == 2:
                    # Select naturally pressured source states. Move the nearest
                    # defender closer only if necessary, along its original
                    # direction; positions remain separated and physically legal.
                    q = defenders[0]
                    roster_for_pressure = np.zeros(8, bool)
                    roster_for_pressure[attacks + defenders] = True
                    pp[q] = pressure_position(pp, roster_for_pressure, q, passer,
                                              d.ball[t, :2], d.ball_r[t])
            roster = np.zeros(8, bool)
            roster[attacks + defenders] = True
            env.start_match([r], active=roster[None], delay=delay[None], match_ticks=10**9)
            env.radius[r, 0] = d.ball_r[t]
            env.rf[r, K.RF_KSTR] = d.kstr[t]
            held = ((d.inp[t] & 16) != 0) & ~d.kicking[t]
            env.place(r, ball_pos=d.ball[t, :2], ball_vel=d.ball[t, 2:4], player_pos=pp,
                      player_vel=d.vel[t], kick_held=held, last_touch=team,
                      mass_phase=0 if abs(d.mass[t] - 0.5) < 1e-6 else 1)
            self._source_actions(env, r, t, delay)
            arena.learner[r] = TEAM == team
            arena.opp[r] = -2
            if hasattr(arena, "macro"):
                arena.macro.k[r] = -1
            if hasattr(arena, "passes"):
                arena.passes.reset([r])
            self.active[r] = True
            self.kind[r], self.attack_team[r], self.defender_kind[r] = stage, team, int(self.rng.integers(3))
            self.passer[r], self.receiver[r], self.delay_max[r] = passer, receiver, max_delay
            self.start_ball[r] = d.ball[t, :2]
            sign = 1.0 if team == 0 else -1.0
            # Escape means controlled forward progress out of the starting
            # pressure area, not merely lofting the ball past a checkpoint.
            self.escape_x[r] = min(sign * d.ball[t, 0] + 100.0, C.LINE_W - 40.0)
            if stage == 0:
                self.pending[r] = True
                self.release_ball[r] = d.ball[release_t, :2]
                self.release_dir[r] = d.ball[release_t, 2:4]
        self.prev_ball[rows] = env.ball_pos[rows]
        self.prev_player[rows] = env.player_pos[rows]

    def _source_window_valid(self, t, delay, history_len):
        d = self.bank.d
        lo = t + int(delay.min()) + 1 - 3 * history_len
        hi = t + int(delay.max()) + 2
        if lo < 0 or hi > len(self.bank.live) or not self.bank.live[lo:hi].all():
            return False
        return (np.all(d.rec_of_tick[lo:hi] == d.rec_of_tick[t])
                and np.all(d.pid[lo:hi] == d.pid[t]))

    def _source_actions(self, env, row, t, delay):
        """World-frame issued decisions for the assigned synthetic robot delay.

        A command issued at t-3(h+1) applies at t+D+1-3(h+1).
        Use raw held inputs; BC labels include future kick events and cannot be
        used to reconstruct the pending command queue.
        """
        d = self.bank.d
        for h in range(env.H):
            idx = t + delay + 1 - 3 * (h + 1)
            raw = d.inp[idx, np.arange(8)]
            world = move_of_input(raw) + 9 * ((raw & 16) != 0)
            env.act_hist[row, :, h] = np.where(env.active[row], world, 0)
        raw = d.inp[t]
        env._s_act[row] = np.where(env.active[row], move_of_input(raw) + 9 * ((raw & 16) != 0), 0)
        env.ev_kicked[row] = env.ev_touch[row] = False

    def before_step(self, env):
        self.prev_ball[...] = env.ball_pos
        self.prev_player[...] = env.player_pos

    def descriptor(self):
        out = np.zeros((self.N, 4), np.float32)
        rows = self.active
        out[rows, 0] = 1.0
        out[rows, 1] = self.kind[rows] / 4.0
        out[rows, 2] = self.received[rows]
        out[rows, 3] = np.minimum(1.0, self.retained[rows] / max(self.retain_s * 60.0, 1.0))
        return out

    def step(self, env, ev):
        reward = np.zeros((self.N, 2), np.float32)
        done = np.zeros(self.N, bool)
        stats = dict(episodes=0, successes=0, failures=0, losses=0, safe_receives=0,
                     offball_samples=0, human_receptions=0, policy_passes=0,
                     cooperative_goals=0, neutral_own_goals=0)
        fs = int(env.frame_skip)
        ticks = np.asarray(ev.get("ticks", np.full(self.N, fs)))
        for r in np.flatnonzero(self.active):
            if self.finished[r]:
                done[r] = True
                continue
            stage, team = int(self.kind[r]), int(self.attack_team[r])
            self.age[r] += max(1, int(ticks[r]))
            own = env.active[r] & (TEAM == team)
            foe = env.active[r] & (TEAM != team)
            touched = np.asarray(ev["touched"][r]) | np.asarray(ev["kicked"][r])
            loss = bool((touched & foe).any())
            restart = bool(ev["out"][r] or ev["restart_start"][r] or env.kickoff[r]
                           or env.restart_team[r] >= 0 or env.ri[r, K.RI_PEND])
            goal = int(ev["goal"][r])
            our_goal = goal == (1 if team == 0 else -1)
            if (stage >= 3 and np.any(ev["kicked"][r] & own)
                    and self.age[r] > self.delay_max[r] + fs):
                self.finish_kicked[r] = True
            timeout = self.age[r] >= int(self.timeout_s * 60)
            dist = np.linalg.norm(env.player_pos[r] - env.ball_pos[r], axis=1)
            control_r = C.PLAYER_RADIUS + env.radius[r, 0] + 15.0
            carrier = np.flatnonzero(own & (dist <= control_r))
            stats["offball_samples"] += max(0, int(own.sum()) - int(bool(len(carrier))))
            safe_new_receive = False
            if not loss and not restart and not goal:
                kickers = np.flatnonzero(ev["kicked"][r] & own)
                if (stage > 0 and not self.received[r] and len(kickers) == 1
                        and self.age[r] > self.delay_max[r] + fs):
                    p = int(kickers[0])
                    # Aim at a currently active teammate: a random ricochet is
                    # not an intentional pass for the skill stages.
                    direction = env.ball_vel[r]
                    targets = np.flatnonzero(own & (np.arange(8) != p))
                    rays = self.prev_player[r, targets] - self.prev_ball[r]
                    norms = np.linalg.norm(rays, axis=1) * max(np.linalg.norm(direction), 1e-9)
                    aims = rays @ direction / np.maximum(norms, 1e-9)
                    if len(targets) and aims.max() >= 0.8:
                        self.pending[r] = True
                        self.passer[r] = p
                        self.receiver[r] = int(targets[np.argmax(aims)])
                        self.release_ball[r] = self.prev_ball[r]
                        self.release_dir[r] = direction
                if self.pending[r] and not self.received[r]:
                    # A flight that reverses relative to its initial direction
                    # probably bounced. Do not reward the eventual other touch.
                    reversed_flight = np.dot(env.ball_vel[r], self.release_dir[r]) < -0.1
                    if reversed_flight:
                        self.pending[r] = False
                    q = int(self.receiver[r])
                    travel = np.linalg.norm(env.ball_pos[r] - self.release_ball[r])
                    if (self.pending[r] and q >= 0 and touched[q] and own[q]
                            and q != self.passer[r] and travel >= MIN_PASS and dist[q] <= control_r):
                        self.received[r] = True
                        safe_new_receive = True
                        stats["safe_receives"] += 1
                        stats["human_receptions" if stage == 0 else "policy_passes"] += 1
                if self.received[r]:
                    self.retained[r] = self.retained[r] + max(1, int(ticks[r])) \
                        if (len(carrier) or self.possession_retention) and not safe_new_receive else 0
            # Team possession can include a subsequent pass in flight, but a
            # loose ball alone cannot finish the exercise: own control is
            # required at the success instant; foe touches/out still fail.
            stable = (self.received[r] and self.retained[r] >= int(self.retain_s * 60)
                      and (not self.possession_retention or bool(len(carrier))))
            success = bool(stable and stage <= 1 and not (loss or restart or goal))
            if stage == 2 and stable and len(carrier) and not (loss or restart or goal):
                sign = 1.0 if team == 0 else -1.0
                pressure = np.linalg.norm(env.player_pos[r, foe] - env.ball_pos[r], axis=1)
                success = bool(sign * env.ball_pos[r, 0] >= self.escape_x[r]
                               and (not len(pressure) or pressure.min() > 75.0))
            if stage >= 3:
                # A goal is the objective here; requiring a fixed number of
                # passes would punish a legitimate open shot.
                success = our_goal and not loss
                if self.reward_cooperative_goals:
                    success = success and self.finish_kicked[r]
            if self.reward_cooperative_goals and 0 < stage < 3 and our_goal:
                success = bool(self.received[r] and not loss
                               and self.age[r] > self.delay_max[r] + fs)
                stats["cooperative_goals"] += int(success)
            failure = bool(loss or restart or goal or timeout or ev["match_end"][r])
            if success or failure:
                done[r] = True
                self.finished[r] = True
                neutral_goal = self.reward_cooperative_goals and our_goal and not success
                reward[r, team] = 1.0 if success else (0.0 if neutral_goal else -1.0)
                stats["neutral_own_goals"] += int(neutral_goal)
                reward[r, 1 - team] = -reward[r, team]
                stats["episodes"] += 1
                stats["successes" if success else "failures"] += 1
                stats["losses"] += int(loss)
                stats[f"stage{stage}_episodes"] = stats.get(f"stage{stage}_episodes", 0) + 1
                stats[f"stage{stage}_successes"] = stats.get(f"stage{stage}_successes", 0) + int(success)
                stats[f"stage{stage}_losses"] = stats.get(f"stage{stage}_losses", 0) + int(loss)
                self.total_episodes[stage] += 1
                self.total_successes[stage] += int(success)
                if stage == self.stage:
                    self.window.append(int(success))
        # ``active`` remains true through reward substitution. The caller resets
        # terminal rows immediately, which also clears all episode bookkeeping.
        self.prev_ball[...] = env.ball_pos
        self.prev_player[...] = env.player_pos
        return reward, done, stats

    def advance(self, learning_updates=0):
        if self.stage >= 4 or len(self.window) < self.min_episodes:
            return None
        if learning_updates - self.stage_started_learning_update < self.min_stage_updates:
            return None
        n, wins = len(self.window), sum(self.window)
        lower = wilson_lower(wins, n)
        if self.mastery_windows > 1:
            self.mastery_streak = self.mastery_streak + 1 if lower >= self.threshold else 0
            self.window.clear()  # repeated checks cannot reuse the same episodes
            if self.mastery_streak < self.mastery_windows:
                return None
        if lower < self.threshold:
            return None
        old = self.stage
        self.stage += 1
        self.stage_started_learning_update = int(learning_updates)
        self.mastery_streak = 0
        self._reserve_rows()
        self.window.clear()
        return dict(from_stage=old, to_stage=self.stage, episodes=n, success_rate=wins / n,
                    wilson_lower=lower, threshold=self.threshold, fraction=self.exercise_fraction)

    def state_dict(self):
        # Arena physics is not resumed by the existing trainer. Persist mastery,
        # not partial exercises which would then describe unrelated fresh states.
        return dict(version=1, stage=self.stage, window=list(self.window),
                    mastery_streak=self.mastery_streak,
                    stage_started_learning_update=self.stage_started_learning_update,
                    total_episodes=self.total_episodes.tolist(), total_successes=self.total_successes.tolist(),
                    exercise_order=self._exercise_order.tolist(),
                    rng_state=self.rng.bit_generator.state)

    def load_state_dict(self, state):
        if state.get("version") != 1 or not 0 <= int(state["stage"]) <= 4:
            raise ValueError("unsupported cooperative curriculum state")
        self.stage = int(state["stage"])
        self.mastery_streak = int(state.get("mastery_streak", 0))
        self.stage_started_learning_update = int(state.get("stage_started_learning_update", 0))
        order = np.asarray(state.get("exercise_order", self._exercise_order), np.int64)
        if len(order) == self.N and np.array_equal(np.sort(order), np.arange(self.N)):
            self._exercise_order = order
        self._reserve_rows()
        self.window = deque(map(int, state["window"]), maxlen=self.window_size)
        self.total_episodes[:] = state["total_episodes"]
        self.total_successes[:] = state["total_successes"]
        if "rng_state" in state:
            self.rng.bit_generator.state = state["rng_state"]
        self.active[:] = self.received[:] = self.pending[:] = False
        self.finished[:] = False
        self.kind[:] = self.attack_team[:] = -1
        self.age[:] = self.retained[:] = 0
        self.finish_kicked[:] = False
