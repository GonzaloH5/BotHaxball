"""RS4 v3 practice, public-geometry guidance and resolved restart cohorts.

Scenario labels and the simplified referee are exclusively scorer/trainer state.
They are never appended to observations. All scenarios retain the RS4 4v4 sim.
"""
from __future__ import annotations

from collections import Counter
import numpy as np
from numba import njit
from .rs4_tactics import ASSIGNMENTS

# Keep the original indices stable for saved configurations and legacy callers.
SCENARIOS = ("corner", "lateral", "goal_kick", "exit", "attack", "defense", "transition",
             "defensive_transition", "offensive_transition")
RESTART_NAMES = {1: "lateral", 2: "corner", 3: "goal_kick", 4: "kickoff"}


@njit(cache=True)
def _restart_geometry(positions, teams, owner, ball, gx, fh, coefficient):
    result = np.zeros((len(ball), len(teams)), dtype=np.float64)
    if coefficient == 0:
        return result
    for row in range(len(ball)):
        team = owner[row]
        if team < 0:
            continue
        sign = 1.0 if team == 0 else -1.0
        bx, by = sign * ball[row, 0], ball[row, 1]
        inward = -1.0 if by >= 0 else 1.0
        own = np.empty((4, 2))
        count = 0
        for p in range(len(teams)):
            if teams[p] == team:
                own[count, 0], own[count, 1] = sign * positions[row, p, 0], positions[row, p, 1]
                count += 1
        targets = np.empty((3, 2))
        x_offsets, y_offsets = (-.28, .18, -.48), (.32, .60, .85)
        for role in range(3):
            targets[role, 0] = min(.85 * gx, max(-.88 * gx, bx + gx * x_offsets[role]))
            targets[role, 1] = min(.8 * fh, max(-.8 * fh, by + inward * fh * y_offsets[role]))
        best = 0.0
        # Joint maximum over four viable executors and 3! unique supports. A
        # tied distance can never select an identity-dependent argmin player.
        for assignment in ASSIGNMENTS:
            taker = assignment[0]
            distance = np.hypot(own[taker, 0] - bx, own[taker, 1] - by)
            executor = np.exp(-distance / (.18 * gx))
            cost = 0.0
            for role in range(3):
                player = assignment[role + 1]
                cost += np.hypot(own[player, 0] - targets[role, 0], own[player, 1] - targets[role, 1]) / 3
            support = np.exp(-cost / (.28 * gx))
            best = max(best, .7 * executor + .3 * support)
        for p in range(len(teams)):
            if teams[p] == team:
                result[row, p] = coefficient * best
    return result


def restart_potential(env):
    """One viable executor plus three distinct supports; bounded shared potential.

    A crowd near the ball cannot fill support locations: each candidate executor
    is excluded and the remaining players are assigned uniquely to three targets.
    The potential is based on geometry, not a hidden taker identity.
    """
    owner = np.where(env.setpiece_team >= 0, env.setpiece_team,
                     np.where(env.sim.kickoff, env.sim.kickoff_team, -1))
    return _restart_geometry(env.sim.player_pos, env.sim.player_team, owner,
                             env.sim.ball_pos, env.goal_x, env.field_h, env.rcfg.rs4_restart_approach)


def configure_rs4_v3(env, settings):
    """Apply a phase at a rollout boundary and rebase potentials, not rewards."""
    if isinstance(env, RS4ScenarioEnv):
        env.configure(settings)
        return
    base = getattr(env, "base", env)
    if base.T != 4 or base.rules is not None or base.goal_kick_speed != 10.5:
        raise ValueError("RS4 v3 requires rs_one 4v4 with simplified restarts")
    rc = base.rcfg
    old_coefficients = (rc.rs4_reward_version, rc.rs4_tactical_coef, rc.rs4_restart_approach)
    rc.rs4_reward_version = 3
    rc.goal = 1.0
    rc.gamma = .998
    rc.kickoff_approach = 0.0
    rc.kickoff_stall = .25
    rc.rs4_restart_stall = .25
    guide_coef = float(settings.get("tactical_coef", settings.get("guide_coef", .08)))
    rc.rs4_restart_approach = float(settings.get("restart_potential_coef", .025 * min(1., max(0., guide_coef / .08))))
    rc.corner_execute = 0.0  # the cohort scorer pays all useful restart kinds once
    rc.rs4_tactical_coef = guide_coef
    rc.team_spread_floor = 0.0
    rc.shaping_coef = 0.0
    rc.rs4_pass_participant = float(settings.get("pass_participant_credit", .0005))
    base.rs4_formation_version = 3
    base.corner_reset_prob = 0.0  # handled by full-match-aware scenario assignment
    base.random_reset_prob = 0.0
    if old_coefficients != (rc.rs4_reward_version, rc.rs4_tactical_coef, rc.rs4_restart_approach):
        base._rs4_phi = base._rs4_potential() if rc.rs4_tactical_coef else None
        base._phi = base._potentials()


class RestartCohorts:
    """Follow each opportunity to one outcome, including delayed confirmation.

    Rates divide resolved successes by resolved opportunities from the same
    cohort. Open opportunities are separately reported, never put in that rate.
    """
    def __init__(self, env):
        self.env = env
        n = env.N
        self.owner = np.full(n, -1, dtype=np.int64)
        self.kind = np.zeros(n, dtype=np.int64)
        self.age = np.zeros(n, dtype=np.int64)
        self.executed = np.zeros(n, dtype=bool)
        self.execute_age = np.zeros(n, dtype=np.int64)
        self.origin = np.zeros((n, 2))
        self.totals = Counter()

    def begin(self):
        env = self.env
        owner = np.where(env.setpiece_team >= 0, env.setpiece_team,
                         np.where(env.sim.kickoff, env.sim.kickoff_team, -1))
        kind = np.where(env.setpiece_team >= 0, env.setpiece_kind, 4)
        start = (self.owner < 0) & (owner >= 0)
        self.owner[start] = owner[start]
        self.kind[start] = kind[start]
        self.origin[start] = env.sim.ball_pos[start]
        self.age[start] = 0
        self.executed[start] = False
        self.execute_age[start] = 0
        for row in np.flatnonzero(start):
            self.totals[(int(owner[row]), int(kind[row]), "opportunities")] += 1

    def update(self, info, done):
        env = self.env
        active = self.owner >= 0
        self.age[active] += env.frame_skip
        kicked = np.asarray(info["kicked"])
        own_kicked = (kicked & (env.sim.player_team[None] == self.owner[:, None])).any(axis=1)
        released = ~env.sim.kickoff & (env.setpiece_team < 0)
        attempted = active & ~self.executed & own_kicked
        self.executed[attempted] = True
        self.execute_age[self.executed] += env.frame_skip
        for row in np.flatnonzero(attempted):
            self.totals[(int(self.owner[row]), int(self.kind[row]), "attempts")] += 1
            self.totals[(int(self.owner[row]), int(self.kind[row]), "time_to_execute_ticks")] += int(self.age[row])
        # Use pre-reset final_obs for the ball when a goal or episode ends.
        ball = env.sim.ball_pos.copy()
        if env.obs_layout == "universal" and np.any(done):
            ball[done] = info["final_obs"][done, 0, 4:6] * [env.field_w, env.field_h]
        travel = np.linalg.norm(ball - self.origin, axis=1)
        goal_for = ((info["goal"] == 1) & (self.owner == 0)) | ((info["goal"] == -1) & (self.owner == 1))
        success = active & self.executed & ((released & (self.execute_age >= 12)
                   & (travel >= .04 * env.field_w) & (env.last_touch == self.owner)) | goal_for)
        timeout = np.asarray(info["events"].get("restart_timeouts", np.zeros((env.N, 2))))
        timed_out = active & timeout[np.arange(env.N), np.maximum(self.owner, 0)].astype(bool)
        lost = active & self.executed & (env.last_touch >= 0) & (env.last_touch != self.owner)
        # A kick can go out before the retention window, with last_touch still
        # belonging to the taker. It is nevertheless a resolved failed restart,
        # not an open cohort that hides the newly awarded opposing opportunity.
        next_owner = np.where(env.setpiece_team >= 0, env.setpiece_team,
                              np.where(env.sim.kickoff, env.sim.kickoff_team, -1))
        next_kind = np.where(env.setpiece_team >= 0, env.setpiece_kind, 4)
        next_origin = np.where((env.setpiece_team >= 0)[:, None], env.setpiece_pos, env.sim.ball_pos)
        changed_restart = ((next_owner != self.owner) | (next_kind != self.kind)
                           | (np.linalg.norm(next_origin - self.origin, axis=1) > 1e-6))
        replaced = active & (next_owner >= 0) & (self.executed | changed_restart)
        external_cut = done & np.asarray(info.get("truncated", np.zeros(env.N, dtype=bool)))
        failure = active & ~success & ~external_cut & (timed_out | lost | replaced | (self.execute_age >= 180) | done)
        outcomes = []
        for row in np.flatnonzero(success | failure):
            outcome = {"row": int(row), "team": int(self.owner[row]), "kind": int(self.kind[row]),
                       "success": bool(success[row]), "attempted": bool(self.executed[row]),
                       "ticks": int(self.age[row]), "timeout": bool(timed_out[row])}
            outcomes.append(outcome)
            key = (outcome["team"], outcome["kind"])
            self.totals[(*key, "resolved")] += 1
            self.totals[(*key, "successes")] += int(outcome["success"])
            self.totals[(*key, "timeouts")] += int(outcome["timeout"])
        rows = np.flatnonzero(success | failure)
        self.owner[rows] = -1
        self.executed[rows] = False
        self.cut(np.flatnonzero(external_cut & active & ~success & ~failure))
        return outcomes

    def cut(self, rows):
        """Censor open opportunities at an external cut; never invent failure."""
        for row in rows:
            if self.owner[row] >= 0:
                self.totals[(int(self.owner[row]), int(self.kind[row]), "censored")] += 1
        self.owner[rows] = -1
        self.executed[rows] = False

    def report(self):
        report = {}
        for team in (0, 1):
            for kind, name in RESTART_NAMES.items():
                values = {key: int(self.totals[(team, kind, key)]) for key in
                          ("opportunities", "resolved", "successes", "attempts", "timeouts", "censored")}
                values["open"] = int(((self.owner == team) & (self.kind == kind)).sum())
                values["success_rate"] = values["successes"] / values["resolved"] if values["resolved"] else None
                values["mean_execute_seconds"] = (self.totals[(team, kind, "time_to_execute_ticks")]
                                                     / (60 * values["attempts"]) if values["attempts"] else None)
                report[f"{name}_{'red' if team == 0 else 'blue'}"] = values
        return report


class RS4ScenarioEnv:
    """Delegate vectorized physics while assigning drills only at match boundaries."""
    def __init__(self, base, settings=None):
        if base.T != 4 or base.rules is not None or base.goal_kick_speed != 10.5:
            raise ValueError("Practice must remain RS4 4v4")
        self.base = base
        self.is_drill = np.zeros(base.N, dtype=bool)
        self.scenario = np.full(base.N, -1, dtype=np.int64)
        self.drill_team = np.zeros(base.N, dtype=np.int64)
        self.drill_ticks = np.zeros(base.N, dtype=np.int64)
        self.drill_limit = np.full(base.N, 900, dtype=np.int64)
        self.continuation = np.full(base.N, -1, dtype=np.int64)
        self.scenario_success = np.zeros(base.N, dtype=bool)
        self.transition_attacking = np.zeros(base.N, dtype=bool)
        self.metrics = {name: np.zeros(base.N, dtype=np.int64) for name in
                        ("passes", "progressive_passes", "shots", "goals", "danger_ticks", "safe_ticks",
                         "control_ticks", "exit_ticks", "creation_ticks", "created", "created_shots")}
        self.learner_team = None  # optional trainer setup; never an actor feature
        self.cohorts = RestartCohorts(base)
        self.settings = {}
        self.configure(settings or {})

    def __getattr__(self, name):
        return getattr(self.base, name)

    def __setattr__(self, name, value):
        base = self.__dict__.get("base")
        if base is not None and name not in self.__dict__ and hasattr(base, name):
            setattr(base, name, value)
        else:
            object.__setattr__(self, name, value)

    def configure(self, settings):
        self.settings.update(settings)
        configure_rs4_v3(self.base, self.settings)
        fraction = float(self.settings.get("drill_fraction", self.settings.get("exercise_fraction", .4)))
        if not 0 <= fraction <= .4:
            raise ValueError("Full matches must retain at least 60% of rows")
        self.drill_fraction = fraction
        self.difficulty = float(self.settings.get("scenario_difficulty", 0.))
        if not 0 <= self.difficulty <= 1:
            raise ValueError("scenario_difficulty must be in [0,1]")
        self.restart_bonus = float(self.settings.get("restart_bonus", self.settings.get("restart_execute_bonus", .02)))
        if not 0 <= self.restart_bonus <= .02:
            raise ValueError("Restart bonus must be in [0,.02]")
        weights = dict(self.settings.get("scenario_weights", self.settings.get("exercise_weights", {name: 1 for name in SCENARIOS})))
        for alias, canonical in (("throw_in", "lateral"), ("build_up", "exit")):
            if alias in weights:
                weights[canonical] = weights.get(canonical, 0) + weights.pop(alias)
        self.weights = np.asarray([weights.get(name, 0) for name in SCENARIOS], dtype=float)
        if (self.weights < 0).any() or not np.isfinite(self.weights).all() or self.weights.sum() <= 0:
            raise ValueError("Invalid scenario weights")
        self.weights /= self.weights.sum()

    def reset(self):
        self.base.reset()
        self.cohorts.cut(np.arange(self.N))
        self.assign(np.arange(self.N))
        return self.base.observe()

    def assign(self, rows, *, scenario=None, team=None):
        rows = np.asarray(rows, dtype=np.int64)
        if scenario is None:
            outside = self.is_drill.copy()
            outside[rows] = False
            quota = max(0, int(self.N * self.drill_fraction) - int(outside.sum()))
            self.is_drill[rows] = False
            if quota:
                self.is_drill[self.rng.choice(rows, min(len(rows), quota), replace=False)] = True
        else:
            self.is_drill[rows] = True  # explicit scorer-only functional evaluation
        self.scenario[rows] = -1
        selected = rows[self.is_drill[rows]]
        if not len(selected):
            return
        if scenario is None:
            self.scenario[selected] = self.rng.choice(len(SCENARIOS), len(selected), p=self.weights)
        else:
            self.scenario[selected] = SCENARIOS.index(scenario)
        if team is None and self.learner_team is not None:
            learner = np.asarray(self.learner_team, dtype=np.int64)
            if learner.shape != (self.N,) or not np.isin(learner, (0, 1)).all():
                raise ValueError("learner_team must contain one color per match")
            self.drill_team[selected] = learner[selected]
        else:
            self.drill_team[selected] = self.rng.integers(0, 2, len(selected)) if team is None else team
        self.drill_ticks[selected] = 0
        self.continuation[selected] = -1
        self.scenario_success[selected] = False
        self.transition_attacking[selected] = False
        self.base.match_ticks[selected] = 0
        self.base.match_score[selected] = 0
        self.cohorts.cut(selected)
        self._place(selected)

    def retarget(self, rows, teams):
        """Realign a newly assigned drill with its actual learner controllers.

        Call only at a reset/controller boundary before the first decision.
        Scenario kind and the reserved practice/full-match quota stay unchanged.
        """
        rows = np.asarray(rows, dtype=np.int64)
        teams = np.asarray(teams, dtype=np.int64)
        if teams.shape != rows.shape or not np.isin(teams, (0, 1)).all():
            raise ValueError("retarget requires one valid color per row")
        if len(rows) and (not self.is_drill[rows].all()):
            raise ValueError("Only newly assigned practice rows can be retargeted")
        self.drill_team[rows] = teams
        self.drill_ticks[rows] = 0
        self.continuation[rows] = -1
        self.scenario_success[rows] = False
        self.transition_attacking[rows] = False
        self.cohorts.cut(rows)
        self.base.match_ticks[rows] = 0
        self.base.match_score[rows] = 0
        self._place(rows)

    def _place(self, rows):
        for value in self.metrics.values():
            value[rows] = 0
        if self.base._public_signals is not None:
            self.base._public_signals.reset(rows)
        env, sim = self.base, self.sim
        env._reset_envs(rows)
        W, H = self.field_w, self.field_h
        radius = sim.st.ball["radius"]
        for row in rows:
            kind = SCENARIOS[self.scenario[row]]
            team = int(self.drill_team[row])
            sign = 1.0 if team == 0 else -1.0
            wing = float(self.rng.choice([-1, 1]))
            own_idx = np.flatnonzero(sim.player_team == team)
            opp_idx = np.flatnonzero(sim.player_team != team)
            own_idx = self.rng.permutation(own_idx)
            opp_idx = self.rng.permutation(opp_idx)
            own = np.array([[-.75 * W, 0], [-.2 * W, -.32 * H], [-.05 * W, .32 * H], [.4 * W, 0]])
            opp = np.array([[-.15 * W, -.05 * H], [.15 * W, -.4 * H], [.35 * W, .3 * H], [.88 * W, 0]])
            ball = np.array([-.05 * W, .1 * wing * H])
            velocity = self.rng.uniform(-1, 1, 2)
            restart_kind = {"corner": 2, "lateral": 1, "goal_kick": 3}.get(kind, 0)
            if restart_kind:
                if kind == "corner":
                    ball = np.array([W - radius - 1, wing * (H - radius - 1)])
                elif kind == "lateral":
                    ball = np.array([self.rng.uniform(-.6, .6) * W, wing * (H - radius - 1)])
                else:
                    ball = np.array([-.85 * W, wing * .14 * H])
                distance = self.rng.uniform(35, .25 * W)
                own[0] = ball + np.array([-distance, -.1 * wing * H])
                own[1] = ball + np.array([-.28 * W, -.35 * wing * H])
                own[2] = ball + np.array([-.05 * W, -.60 * wing * H])
                own[3] = ball + np.array([-.48 * W, -.85 * wing * H])
                velocity[:] = 0
            elif kind == "exit":
                ball[0] = -.65 * W
                own[1] = ball + [0, 24]
                opp[0] = ball + [70, 0]
                opp[1] = ball + [110, .2 * H]
            elif kind == "attack":
                ball[0] = .20 * W
                own[1] = ball + [-25, 0]
                own[2, 0], own[3, 0] = .40 * W, .65 * W
                opp[0] = ball + [90 - 55 * self.difficulty, 0]
                opp[1] = ball + [140 - 50 * self.difficulty, .2 * wing * H]
            else:  # defensive access and loss with teammates initially advanced
                ball[0] = (-.40 - .20 * self.difficulty) * W
                velocity[0] = -self.rng.uniform(1 + self.difficulty, 4 + self.difficulty)
                opp[0] = ball + [25, 0]
                if kind in ("transition", "defensive_transition", "offensive_transition"):
                    self.transition_attacking[row] = (kind == "offensive_transition" or
                        (kind == "transition" and bool(self.rng.integers(0, 2))))
                    if self.transition_attacking[row]:
                        ball = np.array([-.15 * W, .15 * wing * H])
                        velocity[0] = self.rng.uniform(1, 3)
                        own[1] = ball + [-24, 0]
                        own[2:, 0] = [.15 * W, .40 * W]
                        opp[0] = ball + [85, .2 * wing * H]
                    else:
                        own[1:, 0] = [.05 * W, .3 * W, .5 * W]
            own += self.rng.uniform(-15, 15, own.shape)
            opp += self.rng.uniform(-25, 25, opp.shape)
            own[..., 0] = np.clip(own[..., 0], -.95 * W, .95 * W) * sign
            opp[..., 0] = np.clip(opp[..., 0], -.95 * W, .95 * W) * sign
            own[..., 1] = np.clip(own[..., 1], -.88 * H, .88 * H)
            opp[..., 1] = np.clip(opp[..., 1], -.88 * H, .88 * H)
            sim.pos[row, sim.first_player + own_idx] = own
            sim.pos[row, sim.first_player + opp_idx] = opp
            sim.player_vel[row] = 0
            sim.pos[row, 0] = ball * [sign, 1]
            sim.vel[row, 0] = velocity * [sign, 1]
            sim._reset_ball_state([row])
            sim.kickoff[row] = False
            sim.mask[row] = sim.base_mask
            sim.kick_cancel[row] = False
            sim.kickoff_team[row] = team
            env.setpiece_team[row] = team if restart_kind else -1
            env.setpiece_kind[row] = restart_kind
            env.setpiece_pos[row] = sim.ball_pos[row]
            env.setpiece_ticks[row] = 0
            env.setpiece_limit[row] = max(env.setpiece_timeout, env._restart_travel_ticks(row, team) + 180)
            env._corner_from_curriculum[row] = restart_kind == 2
            self.drill_limit[row] = (env.setpiece_limit[row] + 90) if restart_kind else 900
        env._protect_setpieces()
        env._phi = env._potentials()
        env._rs4_phi = env._rs4_potential() if env.rcfg.rs4_tactical_coef else None

    def step(self, actions):
        drill = self.is_drill.copy()
        scenario = self.scenario.copy()
        focus = self.drill_team.copy()
        self.cohorts.begin()
        obs, reward, done, info = self.base.step(actions)
        outcomes = self.cohorts.update(info, done)
        for outcome in outcomes:
            row = outcome["row"]
            if outcome["success"]:
                reward[row, self.sim.player_team == outcome["team"]] += self.restart_bonus
            if drill[row] and scenario[row] < 3 and outcome["team"] == focus[row]:
                self.scenario_success[row] = outcome["success"]
                self.continuation[row] = 60
        self.drill_ticks[drill] += self.frame_skip
        ongoing = drill & (self.continuation >= 0)
        self.continuation[ongoing] -= self.frame_skip
        scored_for = ((info["goal"] == 1) & (focus == 0)) | ((info["goal"] == -1) & (focus == 1))
        scored_against = (info["goal"] != 0) & ~scored_for
        if self.obs_layout == "universal":
            ball_own_x = info["final_obs"][:, 0, 4] * np.where(focus == 0, 1, -1)
            ball_y = info["final_obs"][:, 0, 5]
        else:
            ball_own_x = self.sim.ball_pos[:, 0] / self.field_w * np.where(focus == 0, 1, -1)
            ball_y = self.sim.ball_pos[:, 1] / self.field_h
        transition = np.isin(scenario, (6, 7, 8))
        attacking = drill & ((scenario == 3) | (scenario == 4) | (transition & self.transition_attacking))
        # Positions/velocity from final_obs precede terminal resets. Otherwise a
        # goal can be scored against the geometry of the newly spawned players.
        ball_world = info["final_obs"][:, 0, 4:6] * [self.field_w, self.field_h]
        positions = info["final_obs"][:, :, :2] * [self.field_w, self.field_h]
        positions[..., 0] *= self.sign[None]
        velocity = info["final_obs"][:, 0, 6:8] * 5
        dist = np.linalg.norm(positions - ball_world[:, None], axis=-1)
        own = self.sim.player_team[None] == focus[:, None]
        near = own & (dist < .4 * self.field_w)
        support = near.sum(axis=1) >= 2
        own_distance = np.where(own, dist, np.inf).min(axis=1)
        opp_distance = np.where(~own, dist, np.inf).min(axis=1)
        # Last touch alone does not mean present control of a long drifting ball.
        control = (own_distance <= 60) & (own_distance <= opp_distance + 10)
        open_play = ~done & ~self.sim.kickoff & (self.setpiece_team < 0)
        control &= open_play
        danger = (ball_own_x > .6) & (np.abs(ball_y) < .55) & support & control
        escape = (ball_own_x > -.15) & (np.abs(ball_y) < .9) & support & control
        defensive = drill & ((scenario == 5) | (transition & ~self.transition_attacking))
        events = info["events"]
        rows_all = np.arange(self.N)
        self.metrics["passes"] += events["passes"][rows_all, focus] * drill
        self.metrics["progressive_passes"] += events["progressive_passes"][rows_all, focus] * drill
        self.metrics["goals"] += scored_for & drill
        own_vx = velocity[:, 0] * np.where(focus == 0, 1, -1)
        remaining = self.goal_x - ball_own_x * self.field_w
        projected_y = ball_world[:, 1] + velocity[:, 1] * remaining / np.maximum(own_vx, 1e-6)
        shot = ((info["kicked"] & own).any(axis=1) & (own_vx > 1)
                & (ball_own_x > .45) & (remaining > 0)
                & (np.abs(projected_y) <= self.sim.st.goal_half_height) & ~done)
        self.metrics["shots"] += shot & drill
        threatened = (ball_own_x < -.6) & (np.abs(ball_y) < .55) & (opp_distance < own_distance)
        self.metrics["danger_ticks"] += threatened * drill * self.frame_skip
        self.metrics["safe_ticks"] = np.where(escape & defensive,
                                              self.metrics["safe_ticks"] + self.frame_skip, 0)
        self.metrics["control_ticks"] += control * drill * self.frame_skip
        self.metrics["exit_ticks"] = np.where(escape & attacking,
                                              self.metrics["exit_ticks"] + self.frame_skip, 0)
        self.metrics["creation_ticks"] = np.where(danger & attacking,
                                                  self.metrics["creation_ticks"] + self.frame_skip, 0)
        exit_success = (self.metrics["exit_ticks"] >= 60) & (self.metrics["passes"] >= 1)
        creation = (self.metrics["creation_ticks"] >= 30) & (self.metrics["progressive_passes"] >= 1)
        self.metrics["created"] |= creation.astype(np.int64)
        self.metrics["created_shots"] += shot & drill & self.metrics["created"].astype(bool)
        finish_attack = self.metrics["created_shots"] >= 1
        self.scenario_success[attacking] |= (np.where(scenario == 3, exit_success, finish_attack)
                                               | scored_for)[attacking]
        self.scenario_success[defensive] |= (self.metrics["safe_ticks"] >= 60)[defensive]
        cut = drill & ((self.drill_ticks >= self.drill_limit) | (ongoing & (self.continuation <= 0)))
        finish = drill & (done | cut)
        results = [{"row": int(row), "scenario": SCENARIOS[scenario[row]], "team": int(focus[row]),
                    "success": bool(self.scenario_success[row] and not scored_against[row]),
                    "conceded": int(scored_against[row]), "ticks": int(self.drill_ticks[row]),
                    "transition_attacking": bool(self.transition_attacking[row]),
                    "metrics": {name: int(value[row]) for name, value in self.metrics.items()},
                    "truncated": bool(cut[row] and not done[row])} for row in np.flatnonzero(finish)]
        practice_cut = cut & ~done
        if practice_cut.any():
            done[practice_cut] = True
            info["truncated"][practice_cut] = True
        # These are not full football results, regardless of goal/timeout/reset.
        info["match_done"][drill] = False
        info["final_score"][drill] = -1
        info["is_drill"] = drill
        info["scenario_result"] = results
        info["restart_outcomes"] = outcomes
        rows = np.flatnonzero(finish | info["match_done"])
        if len(rows):
            self.cohorts.cut(rows)
            self.base._reset_envs(rows, kickoff_team=self.rng.integers(0, 2, len(rows)))
            self.base.match_ticks[rows] = 0
            self.base.match_score[rows] = 0
            self.assign(rows)
            # Returning a fresh start is essential; final_obs remains pre-cut.
            obs = obs.copy()  # base may alias obs and final_obs when no real terminal occurred
            obs[rows] = self.base.observe(rows)
        return obs, reward, done, info
