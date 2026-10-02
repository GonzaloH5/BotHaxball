"""Behavioral regressions from the 2B objective audit, with no production PPO."""
import copy
import json
from itertools import product

import numpy as np
import pytest
import torch
import yaml

from env.haxball_env import HaxballEnv
from env.rs4_v3 import RS4ScenarioEnv, SCENARIOS
from eval import rs4_v3 as evaluator
from eval.rs4_objective import CONTRACT, gates, eligibility, selection_score
from train.rs4_program import ProgramState, default_program
from tools import run_rs4_v3 as runner


def report():
    cells = {str(count): dict(learner_count=count, games=96, points=.6) for count in (1, 2, 3, 4)}
    skills = dict(restart_success=.9, corner_success_red=.9, corner_success_blue=.9,
                  defense_conceded=0., defense_recovery=.8, defense_danger_fraction=.05,
                  exit_success=.7, attack_success=.6, integrated_success=.6, teammate_success=.6)
    base = dict(evaluation_contract=CONTRACT, source_fingerprint={"sha256": "same"},
                suite=dict(games=16, functional_games=16, minutes=2., action_mode="greedy"),
                seeds=[51, 73, 91], full_games=dict(cells=cells, balanced_points=.6),
                skills=skills, functional=dict(mean_success=.7))
    candidate = copy.deepcopy(base)
    candidate.update(reference=copy.deepcopy(base), baseline=copy.deepcopy(skills), checkpoint_sha256="test")
    return candidate


def test_zero_conceded_baseline_can_approve_competent_recovery_but_not_inactivity():
    r = report()
    assert gates(r)["defense"]
    r["skills"]["defense_recovery"] = 0
    assert not gates(r)["defense"]
    r["skills"].pop("defense_recovery")
    assert not gates(r)["defense"]


def test_regressed_single_instance_blocks_global_improvement_and_champion():
    r = report()
    r["full_games"]["balanced_points"] = .9
    r["full_games"]["cells"]["1"]["points"] = .4
    assert not gates(r)["teammates"]
    assert not eligibility(r)["passed"]
    assert selection_score(r) < selection_score(r["reference"])


def test_incomplete_cells_and_other_source_or_action_contract_are_ineligible():
    r = report()
    r["full_games"]["cells"].pop("1")
    assert not eligibility(r)["passed"]
    r = report()
    r["reference"]["suite"]["action_mode"] = "sampled"
    assert not eligibility(r)["passed"]
    r = report()
    r["reference"]["source_fingerprint"]["sha256"] = "old"
    assert not eligibility(r)["passed"]


@pytest.mark.parametrize("name,attacking", [("offensive_transition", True), ("defensive_transition", False)])
def test_transition_direction_survives_sampling_and_retargeting(name, attacking):
    env = RS4ScenarioEnv(HaxballEnv(10, 4, "rs_one", out_of_bounds=True, obs_layout="universal", seed=1),
                         dict(exercise_fraction=.4, exercise_weights={name: 1}))
    env.reset()
    selected = np.flatnonzero(env.is_drill)
    assert np.all(env.scenario[selected] == SCENARIOS.index(name))
    assert np.all(env.transition_attacking[selected] == attacking)
    env.retarget(selected, 1 - env.drill_team[selected])
    assert np.all(env.transition_attacking[selected] == attacking)


@pytest.mark.parametrize("passes,control", [(0, True), (1, False), (1, True)])
def test_exit_requires_received_pass_and_one_second_of_continuous_control(monkeypatch, passes, control):
    env = RS4ScenarioEnv(HaxballEnv(1, 4, "rs_one", out_of_bounds=True, obs_layout="universal", seed=1),
                         dict(guide_coef=0, restart_bonus=0))
    env.reset()
    env.assign([0], scenario="exit", team=0)
    env.drill_limit[:] = 63
    env.sim.ball_pos[:] = [-.1 * env.field_w, 0]
    env.sim.player_pos[0, :4] = env.sim.ball_pos[0] + [[-20, 0], [-100, 100], [-150, -100], [-400, 0]]
    env.sim.player_pos[0, 4:] = env.sim.ball_pos[0] + [[150, 0], [200, 100], [300, -100], [500, 0]]
    if not control:
        env.sim.player_pos[0, :4, 0] -= 200
    calls = 0
    def step(actions):
        nonlocal calls
        calls += 1
        obs = env.observe().copy()
        events = {k: np.zeros((1, 2), dtype=int) for k in ("passes", "progressive_passes", "restart_timeouts")}
        events["passes"][0, 0] = passes if calls == 1 else 0
        return obs, np.zeros((1, 8)), np.zeros(1, bool), dict(final_obs=obs, events=events,
            kicked=np.zeros((1, 8), bool), goal=np.zeros(1, int), truncated=np.zeros(1, bool),
            match_done=np.zeros(1, bool), final_score=np.full((1, 2), -1))
    monkeypatch.setattr(env.base, "step", step)
    for _ in range(21):
        _, _, _, info = env.step(np.zeros((1, 8), int))
    assert info["scenario_result"][0]["success"] == bool(passes and control)


def test_factorial_matrix_has_every_seed_composition_style_color_and_proxy(monkeypatch):
    calls = []
    class Agent:
        pass
    monkeypatch.setattr(evaluator, "make_agent", lambda *a, **kw: Agent())
    def full(agent, opponent, games, minutes, seed, color):
        calls.append((seed, color, len(agent.learner_slots) if isinstance(agent, evaluator.MixedTeamAgent) else 4))
        return dict(games=games, color=color, seed=seed, wins=games, draws=0, losses=0, points=1.,
                    goals_for=games, goals_against=0, behavior={})
    def functional(agent, scenario, games, seed, color, **kwargs):
        return dict(scenario=scenario, games=games, seed=seed, color=color, success=.5, conceded=0,
                    danger_fraction=0., goal_rate=.1, restarts=dict(resolved=games, successes=games))
    monkeypatch.setattr(evaluator, "full_game", full)
    monkeypatch.setattr(evaluator, "functional_trial", functional)
    r = evaluator.evaluate("weights.pt", teammate_reference="teacher.pt", seeds=(51, 73, 91))
    rows = r["full_games"]["rows"]
    mixed = {(row["seed"], row["learner_count"], row["opponent"], row["teammate"], row["color"])
             for row in rows if row["team_mode"] == "mixed"}
    assert mixed == set(product((51, 73, 91), (1, 2, 3),
        ("scripted:r3:0", "scripted:r3:1", "scripted:r3:2"),
        ("scripted:r3:0", "scripted:r3:2", "teacher.pt"), (0, 1)))
    assert r["suite"]["action_mode"] == "greedy"
    assert r["skills"]["integrated_success"] == .5


def test_reconciliation_keeps_steps_adam_weights_history_and_never_advances(tmp_path):
    p = ProgramState(default_program(4_500_000_000))
    p.advance_steps(1_942_311_810)
    p.skill_debts = ["defense", "attack"]
    p.pass_streak = 1
    p.evaluations = [dict(steps=1_900_000_000, phase="C", report={"old": True}, gates={})]
    p.last_evaluation_steps = 1_900_000_000
    directory = tmp_path / "runs/program"
    for branch in ("control", "memory"):
        sub = directory / branch
        sub.mkdir(parents=True)
        (sub / "config.yaml").write_text(yaml.safe_dump({"rs4_program": p.config}))
        torch.save(dict(steps=p.config["start_steps"] + p.relative_steps,
                        model={"tensor": torch.tensor([3.])}, opt={"adam": torch.tensor([7.])},
                        rs4_program_state=p.state_dict()), sub / "latest.pt")
    ledger = dict(selected="control", total_budget_steps=6_000_000_000,
                  candidates={b: dict(useful_steps=0) for b in ("control", "memory")}, assessments=[])
    torch.save(dict(model={"tensor": torch.tensor([2.])}), directory / "parent.pt")
    r = report()
    r["checkpoint_sha256"] = runner.file_hash(directory / "control/latest.pt")
    r["full_games"]["cells"]["1"]["points"] = .4  # no champion mutation; genuine debt assessment still occurs
    output = directory / "evaluation.json"
    output.write_text(json.dumps(r))
    runner._persist_evaluation(directory, "control", output, r, ledger)
    ck = torch.load(directory / "control/latest.pt", weights_only=False)
    after = ck["rs4_program_state"]
    assert after["relative_steps"] == p.relative_steps
    assert after["phase_index"] == p.phase_index
    assert after["pass_streak"] == 0
    assert "defense" not in after["skill_debts"]
    assert after["evaluations"][0]["report"] == {"old": True}
    assert torch.equal(ck["model"]["tensor"], torch.tensor([3.]))
    assert torch.equal(ck["opt"]["adam"], torch.tensor([7.]))
    assert (directory / "objective_before_v4.pt").is_file()
    restored = ProgramState(p.config, after)
    assert restored.settings()["frozen_teammates_fraction"] == .35
    restored.reconcile_objective(r)
    assert len(restored.evaluations) == 2


def test_old_checkpoint_missing_objective_fields_loads_without_progress_reset():
    p = ProgramState(default_program())
    p.advance_steps(1_942_311_810)
    old = p.state_dict()
    old.pop("objective_signature")
    old.pop("objective_contract")
    restored = ProgramState(p.config, old)
    assert restored.relative_steps == p.relative_steps
    assert restored.objective_signature is None


def test_reserved_report_never_updates_curriculum_or_streak():
    p = ProgramState(default_program())
    p.advance_steps(100_000_000)
    r = report()
    r["suite"]["holdout"] = True
    r["reference"]["suite"]["holdout"] = True
    before = p.state_dict()
    with pytest.raises(ValueError, match="reservada"):
        p.reconcile_objective(r)
    with pytest.raises(ValueError, match="reservada"):
        p.record_evaluation(r)
    assert p.state_dict() == before


def test_champion_is_remeasured_and_kept_when_single_bot_cell_regresses(tmp_path, monkeypatch):
    directory = tmp_path
    champion = directory / "champion.pt"
    champion.write_bytes(b"original-checkpoint")
    ledger = dict(champion=dict(sha256=runner.file_hash(champion), score=[.99, .99]))
    candidate = report()
    candidate["full_games"]["cells"]["1"]["points"] = .75
    candidate["full_games"]["balanced_points"] = .8
    # Current parent .6 permits it, but the incumbent's same-suite result .85
    # exposes a regression that the old global score could have hidden.
    incumbent = report()
    incumbent["full_games"]["cells"]["1"]["points"] = .85
    calls = []
    def evaluate(root, branch, **options):
        calls.append(options)
        return directory / "incumbent.json", incumbent
    monkeypatch.setattr(runner, "_evaluate", evaluate)
    payload = dict(rs4_program_state=dict(relative_steps=2_000_000_000))
    runner._consider_champion(directory, "control", payload, directory / "candidate.json",
                              candidate, ledger, closing_phase=2)
    assert calls[0]["checkpoint"] == champion
    assert calls[0]["minutes"] == 2
    assert calls[0]["games"] == 16
    assert champion.read_bytes() == b"original-checkpoint"
    assert not (directory / "phase_C_champion.pt").exists()
    assert ledger["champion"]["score"][0] == .85


def test_checkpoint_hash_mismatch_is_rejected_before_any_mutation(tmp_path, monkeypatch):
    path = tmp_path / "latest.pt"
    path.write_bytes(b"current-checkpoint")
    p = ProgramState(default_program())
    monkeypatch.setattr(runner, "_checkpoint", lambda *a: (path, {}, {}, p))
    before = p.state_dict()
    with pytest.raises(ValueError, match="checkpoint actual"):
        runner._persist_evaluation(tmp_path, "control", tmp_path / "stale.json", report(), {})
    assert p.state_dict() == before
    assert path.read_bytes() == b"current-checkpoint"
    assert not (tmp_path / "objective_before_v4.pt").exists()
