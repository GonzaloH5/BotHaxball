"""Bloque RS4-b1 (PLAN_RS4.md sección 3): configuración efectiva, recompensa, paridad A/B y lanzador."""
import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest
import torch

from env.haxball_env import HaxballEnv
from env.rewards import RewardConfig
from env.rs4_v3 import RS4ScenarioEnv
from tools import rs4_b1_launch
from tools.evaluate_rs4_b1 import EVAL_RIVALS, EVAL_TEAMMATES
from train.model import build_model
from train.rs4_program_v5 import ProgramStateV5
from train.rs4_trainer import SituationWindow, block_settings
from train.runtime import load_config

ROOT = Path(__file__).resolve().parent.parent
CONFIG_A = ROOT / "train" / "config_rs4_b1.yaml"
CONFIG_B = ROOT / "train" / "config_rs4_b1_b.yaml"


def test_effective_config_is_the_planned_experiment():
    cfg = load_config(CONFIG_A)
    program = ProgramStateV5.from_config(cfg)
    settings = program.settings()
    assert settings["exercise_fraction"] == pytest.approx(.20)
    assert settings["exercise_weights"] == {"corner": 0., "lateral": 0., "goal_kick": 0., "recorded": 1.}
    assert settings["opponent_mix"]["selfplay"] == 0
    assert settings["bc_coef"] == 0 and cfg["ppo"]["bc_kl_coef"] == 0
    for key in ("ball_progress", "near_ball", "spread", "pass_possession_cap", "possession_change",
                "no_goal_penalty", "no_goal_step_penalty", "restart_potential_coef", "restart_bonus", "guide_coef"):
        assert settings[key] == 0, key
    assert cfg["rs4_program"]["total_budget_steps"] == cfg["ppo"]["total_steps"] == 100_000_000
    assert cfg["env"]["referee"] == "rs_one_v1" and cfg["env"]["max_ticks"] == 3 * 3600
    assert cfg["model"]["critic_features"] == 12
    block = cfg["rs4_b1"]
    weights = np.asarray(block["learner_weights"])
    assert weights.sum() == pytest.approx(1) and weights[0] == pytest.approx(.7)
    assert block["scripted_style"] == 0 and block["pass_participant_credit"] == 0
    assert block["restart_deadline_ticks"] == 600 and cfg["reward"]["rs4_restart_stall"] == .25
    # Familias separadas: nada de lo reservado para evaluar se usa para entrenar.
    reserved = {Path(p).name if not p.startswith("scripted") else p for p in EVAL_TEAMMATES + EVAL_RIVALS}
    reserved_runs = {Path(p).parent.name for p in EVAL_TEAMMATES + EVAL_RIVALS if not p.startswith("scripted")}
    for seed in cfg["rs4_v5"]["league_seed"]:
        path = Path(seed)
        assert path.parent.name not in reserved_runs - {"rs4_v3_public"}, seed
        assert path.name not in {"champion.pt", "phase_B_champion.pt", "teacher.pt"}, seed
    assert f"scripted:r3:{block['scripted_style']}" not in reserved
    assert cfg["rs4_v5"]["add_initial"] is False


def test_candidate_b_changes_only_the_architecture():
    a, b = load_config(CONFIG_A), load_config(CONFIG_B)
    assert b["model"]["type"] == "recurrent_set" and b["model"]["memory_size"] == 64
    for cfg in (a, b):
        cfg.pop("run_name")
        cfg["model"].pop("type")
        cfg["model"].pop("memory_size", None)
    assert a == b  # mismos hiperparámetros, recompensa, compañeros, presupuesto e inicialización


def test_final_config_only_adds_variety_and_budget():
    final, a = load_config(ROOT / "train" / "config_rs4_b1_final.yaml"), load_config(CONFIG_A)
    seeds = final["rs4_v5"]["league_seed"]
    assert len(seeds) == len(set(seeds)) == 16 and set(a["rs4_v5"]["league_seed"]) <= set(seeds)
    for seed in seeds:  # la familia reservada para evaluar sigue afuera
        assert "v5d" not in seed and Path(seed).name not in {"champion.pt", "phase_B_champion.pt", "teacher.pt"}
        assert "bc_rs4_v2" not in seed
    assert final["rs4_b1"]["scripted_style"] == -1 and final["rs4_program"]["opponents"]["scripted"] == .25
    assert final["rs4_program"]["total_budget_steps"] == final["ppo"]["total_steps"] == 1_600_000_000
    assert final["rs4_b1"]["learner_weights"] == [.4, .2, .2, .2] and final["env"]["agents"] == 4608
    assert final["rs4_program"]["drills"] == .35 and final["rs4_program"]["entropy"] == {"initial": .004, "final": .004}
    ProgramStateV5.from_config(final)  # el programa valida (drills <= 0.4, calendarios)
    for cfg in (final, a):
        for key in ("run_name", "runtime", "log", "league"):
            cfg.pop(key)
        cfg["ppo"].pop("total_steps")
        cfg["env"].pop("agents")
        cfg["rs4_v5"].pop("league_seed")
        for key in ("scripted_style", "learner_weights"):
            cfg["rs4_b1"].pop(key)
        for key in ("total_budget_steps", "snapshot_every_steps", "opponents", "drills", "entropy"):
            cfg["rs4_program"].pop(key)
    assert final == a  # misma recompensa, plazo de saques, inicialización, LR y arquitectura que A


def test_block_settings_resolve_the_bank_and_remove_pass_credit():
    settings = {"lr": 1e-4}
    assert block_settings(settings, None) == settings
    merged = block_settings(settings, {"recorded": {"path": "data\\rs4_states\\entrenamiento.npz", "ticks": 720},
                                       "pass_participant_credit": 0, "restart_deadline_ticks": 600})
    assert Path(merged["recorded"]["path"]) == ROOT / "data" / "rs4_states" / "entrenamiento.npz"
    assert merged["recorded"]["ticks"] == 720 and merged["pass_participant_credit"] == 0
    assert merged["restart_deadline_ticks"] == 600
    assert "recorded" not in settings  # no modifica el calendario del programa


def test_situation_window_counts_goals_by_kind_and_attacker():
    window = SituationWindow()
    window.add([{"scenario": "recorded", "pool": "attack", "goal_team": 0, "attacker": 0},
                {"scenario": "recorded", "pool": "attack", "goal_team": 1, "attacker": 0},
                {"scenario": "recorded", "pool": "attack", "goal_team": -1, "attacker": 0},
                {"scenario": "recorded", "pool": "attack", "goal_team": -1, "attacker": 1},
                {"scenario": "recorded", "pool": "restart", "goal_team": -1, "attacker": 1},
                {"scenario": "corner", "goal_team": 0}])
    report = window.report()
    assert report["attack"] == (4, 50.0, 25.0)
    assert report["restart"] == (1, 0.0, 0.0) and "open" not in report
    assert window.report() == {}


def _bank(tmp_path):
    from tests.test_rs4_recorded import bank_file, state
    from env.rs4_states import KIND_LATERAL
    return bank_file(tmp_path, state(ball=(700.0, 50.0), last=0), state(ball=(-200.0, 0.0), last=1),
                     state(KIND_LATERAL, ball=(300.0, 688.0), last=0, taker=1, spot=(300.0, 688.0)))


def test_reward_is_only_goals_plus_the_restart_safety_limits(tmp_path):
    """Entorno armado como el entrenador: recompensa del config, programa v5 y bloque rs4_b1.

    Además del gol sólo existen las multas de 0,25 al equipo que traba el saque inicial (terminal)
    o deja vencer el plazo de entrenamiento de un saque (no terminal); ambas se cuentan en
    events["restart_timeouts"] del equipo multado."""
    cfg = load_config(CONFIG_A)
    names = {field.name for field in fields(RewardConfig)}
    reward = RewardConfig(**{k: v for k, v in cfg["reward"].items() if k in names}, gamma=cfg["ppo"]["gamma"])
    e = cfg["env"]
    base = HaxballEnv(8, 4, "rs_one", frame_skip=e["frame_skip"], max_ticks=900, random_reset_prob=0,
                      out_of_bounds=True, obs_layout="universal", max_entities=7, seed=5,
                      kickoff_timeout=e["kickoff_timeout"], reward=reward, referee=e["referee"])
    env = RS4ScenarioEnv(base)
    block = dict(cfg["rs4_b1"], recorded=dict(cfg["rs4_b1"]["recorded"], path=str(_bank(tmp_path))))
    env.configure(block_settings(ProgramStateV5.from_config(cfg).settings(), block))
    assert env.rcfg.rs4_pass_participant == 0 and env.rcfg.team_pass_possession_cap == 0
    assert env.base._rs1.deadline == cfg["rs4_b1"]["restart_deadline_ticks"] == 600
    env.reset()
    rng = np.random.default_rng(0)
    team = np.where(env.sim.player_team == 0, 1.0, -1.0)
    fines = goals = 0
    for _ in range(600):
        _, rew, done, info = env.step(rng.integers(0, 18, size=(env.N, env.P)))
        extra = rew - info["goal"][:, None] * team[None]
        fined = info["events"]["restart_timeouts"][:, env.sim.player_team]
        np.testing.assert_allclose(extra, -.25 * fined, atol=1e-12)
        fines += int(fined.sum())
        goals += int((info["goal"] != 0).sum())
        for name, values in info.get("reward_terms", {}).items():
            if name != "goal":
                assert np.allclose(values, 0), name
    assert fines + goals > 0  # se ejercitaron multas o goles reales


def _initial_checkpoint():
    cfg = load_config(CONFIG_A)["model"]
    source = build_model(dict(type="set", self_dim=71, ent_dim=8, hidden=cfg["hidden"], layers=cfg["layers"],
                              ent_hidden=cfg["ent_hidden"], pooling=cfg["pooling"], ent_layers=cfg["ent_layers"],
                              rule_observation=cfg["rule_observation"],
                              public_signals_version=cfg["public_signals_version"]))
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.normal_(0, .2)
    return {"model": source.state_dict(), "model_config": source.config()}, cfg


def test_candidate_b_starts_with_the_actions_and_values_of_a():
    ck, cfg = _initial_checkpoint()
    common = dict(self_dim=71, ent_dim=8, hidden=cfg["hidden"], layers=cfg["layers"], ent_hidden=cfg["ent_hidden"],
                  pooling=cfg["pooling"], ent_layers=cfg["ent_layers"], rule_observation=cfg["rule_observation"],
                  public_signals_version=cfg["public_signals_version"], critic_features=cfg["critic_features"])
    a = build_model(dict(type="set", **common))
    b = build_model(dict(type="recurrent_set", memory_size=64, **common))
    a.initialize_from(ck)
    b.initialize_from(ck)
    torch.manual_seed(3)
    length, batch = 12, 6
    obs = torch.randn(length, batch, 127)
    critic = torch.randn(length, batch, cfg["critic_features"])
    previous = torch.randint(0, 18, (length, batch))
    start = torch.zeros(length, batch, dtype=torch.bool)
    start[0] = True
    start[5, 2] = True
    with torch.no_grad():
        logits_b, values_b, _ = b.sequence(obs, b.initial_state(batch), previous, start, critic)
        logits_a, values_a = a(obs.reshape(-1, 127), critic.reshape(-1, cfg["critic_features"]))
    torch.testing.assert_close(logits_b.reshape(-1, 18), logits_a)
    torch.testing.assert_close(values_b.reshape(-1), values_a)
    assert a.config()["critic_features"] == b.config()["critic_features"] == 12


def test_launcher_refuses_changed_code_or_a_wrong_initial_checkpoint(tmp_path, monkeypatch):
    init = tmp_path / "init.pt"
    init.write_bytes(b"pesos")
    bank = _bank(tmp_path)
    good = rs4_b1_launch.sha256(init)
    config = tmp_path / "config.yaml"

    def write(sha):
        config.write_text(json.dumps({
            "run_name": "rs4_b1_prueba_tmp", "rs4_program": {"total_budget_steps": 100},
            "runtime": {"max_wall_seconds": 60}, "rs4_v5": {"league_seed": []},
            "rs4_b1": {"init_from": str(init), "init_sha256": sha, "recorded": {"path": str(bank)}}}), encoding="utf-8")

    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({"id": "abc", "sha256": "f" * 64, "files": {}}), encoding="utf-8")
    monkeypatch.setattr(rs4_b1_launch, "CANDIDATES", {"A": str(config)})
    monkeypatch.setattr(rs4_b1_launch, "verify", lambda path: [])
    write("0" * 64)
    with pytest.raises(SystemExit, match="no el esperado"):
        rs4_b1_launch.plan("A", 1, freeze)
    write(good)
    command, manifest = rs4_b1_launch.plan("A", 2, freeze)
    assert command[command.index("--init-from") + 1] == str(init) and command[-1] == "seed=2"
    assert manifest["run"] == "rs4_b1_prueba_tmp_s2" and manifest["inputs"]["init"]["sha256"] == good
    assert manifest["freeze_id"] == "abc" and manifest["budget_samples"] == 100
    assert set(manifest["inputs"]) == {"init", "states", "contract"}
    _, tagged = rs4_b1_launch.plan("A", 3, freeze, "plazo")
    assert tagged["run"] == "rs4_b1_prueba_tmp_plazo_s3" and tagged["tag"] == "plazo"
    monkeypatch.setattr(rs4_b1_launch, "verify", lambda path: ["cambiado: env/rs4_v3.py"])
    with pytest.raises(SystemExit, match="difiere"):
        rs4_b1_launch.plan("A", 2, freeze)
