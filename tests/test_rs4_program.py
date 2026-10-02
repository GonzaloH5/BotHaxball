"""Programa largo, ledger exacto y selección conservadora sin entrenar el Pod."""
import copy
import json

import numpy as np
import pytest
import torch
import yaml

from tools import prepare_rs4_v3 as preparer
from tools.run_rs4_v3 import normalized_evaluation, refresh_ledger, select_candidate
from train.league import League
from train.model import build_model
from train.rs4_program import BRANCH_BUDGET, ProgramState, default_program
from train.runtime import load_config


def state(config=None):
    return ProgramState.from_config({"rs4_program": config or default_program(4_500_000_000)})


def good_report():
    return dict(skills=dict(restart_success=.9, corner_success_red=.8, corner_success_blue=.75,
                            defense_conceded=.15, attack_success=.7, integrated_success=.7, teammate_success=.7),
                baseline=dict(defense_conceded=.3, attack_success=.5, integrated_success=.6, teammate_success=.6),
                matches=dict(points=.6, baseline_points=.6))


def test_template_matches_defaults_and_exact_budget():
    profile = load_config(preparer.PROFILE)
    assert profile["rs4_program"] == default_program()
    program = ProgramState.from_config(profile)
    assert sum(p["steps"] for p in program.config["phases"]) == 5_800_000_000
    assert program.config["pilot_steps"] + BRANCH_BUDGET == 6_000_000_000
    assert program.lr == 1e-4
    assert program.entropy_coef == .003
    assert program.bc_coef == .005


def test_early_advance_needs_two_distinct_passes_and_half_budget():
    program = state()
    program.advance_steps(100_000_000)
    assert not program.record_evaluation(good_report())["advanced"]
    with pytest.raises(ValueError, match="repetida"):
        program.record_evaluation(good_report())
    program.advance_steps(100_000_000)
    assert not program.record_evaluation(good_report())["advanced"]
    program.advance_steps(200_000_000)
    assert program.record_evaluation(good_report())["advanced"]
    assert program.phase_index == 1
    assert program.phase_start_steps == 400_000_000
    assert program.remaining_steps == 5_400_000_000  # ahorro va a consolidación, no desaparece


def test_missing_report_evidence_fails_not_success():
    program = state()
    program.advance_steps(100_000_000)
    result = program.record_evaluation({})
    assert not any(result["gates"].values())
    assert program.skill_debts == ["restarts"]


def test_nominal_transitions_mark_unmastered_skills_and_recovery_is_bounded():
    program = state()
    program.advance_steps(5_200_000_000)
    assert program.phase_index == 5
    assert len(program.skill_debts) == 5
    assert program.settings()["recovery"]
    for _ in range(4):
        program.advance_steps(100_000_000)
    assert program.recovery_consumed == 400_000_000
    assert not program.settings()["recovery"]
    assert program.settings()["guide_coef"] == 0
    assert program.settings()["restart_execute_bonus"] == 0
    assert program.settings()["restart_potential_coef"] == 0
    program.advance_steps(200_000_000)
    assert program.complete and program.relative_steps == BRANCH_BUDGET
    assert program.entropy_coef == .001
    assert program.bc_coef == .001


def test_budget_overshoot_is_recorded_not_silently_hidden():
    program = state()
    program.advance_steps(BRANCH_BUDGET + 3)
    assert program.relative_steps == BRANCH_BUDGET + 3
    assert program.remaining_steps == 0


def test_settings_are_bounded_and_use_canonical_scenarios():
    program = state()
    for phase in program.config["phases"]:
        settings = program.settings()
        assert settings["exercise_fraction"] <= .4
        assert sum(settings["exercise_weights"].values()) == pytest.approx(1.)
        assert set(settings["exercise_weights"]) <= {"corner", "lateral", "goal_kick", "exit", "attack", "defense", "transition",
                                                    "defensive_transition", "offensive_transition"}
        assert sum(settings["opponent_mix"].values()) == pytest.approx(1.)
        assert settings["freeze_normalizers"]
        program.advance_steps(phase["steps"])


def test_lr_endpoint_controller_only_every_25_and_clamps():
    program = state()
    assert program.update_lr(0, 24) == 1e-4
    assert program.update_lr(.0003, 25) == pytest.approx(1.1e-4)
    assert program.update_lr(0, 25) == pytest.approx(1.1e-4)
    for iteration in range(50, 350, 25):
        program.update_lr(0, iteration)
    assert program.lr == .0002
    for iteration in range(350, 1000, 25):
        program.update_lr(.02, iteration)
    assert program.lr == .00005
    with pytest.raises(ValueError):
        program.update_lr(float("nan"), 1000)


def test_state_restores_anchors_debts_lr_and_schedule_without_reset():
    program = state()
    program.advance_steps(900_000_000)
    program.update_lr(.0001, 25)
    program.record_evaluation({})
    program.mark_snapshot()
    saved = program.state_dict()
    restored = ProgramState(program.config, saved)
    assert restored.state_dict() == saved
    restored.advance_steps(20_000_000)
    assert restored.relative_steps == 920_000_000
    altered = copy.deepcopy(program.config)
    altered["start_steps"] += 1
    with pytest.raises(ValueError, match="anclas"):
        ProgramState(altered, saved)


@pytest.mark.parametrize("field,value", [("branch_budget_steps", 7_000_000_000), ("start_steps", -1),
                                         ("pilot_steps", True), ("version", 2)])
def test_invalid_program_rejected(field, value):
    cfg = default_program()
    cfg[field] = value
    with pytest.raises(ValueError):
        state(cfg)


def benchmark(speed=100, **extra):
    return dict(steps_per_second=speed, device="cuda:0", rs4_profile=dict(phase_id="A", teammates=.1),
                comparison_contract=dict(rollout_len=128, companions=.1),
                **{"memory_gate_passed": True, **extra})


def selection_report(skill=.5, points=.6):
    return dict(functional=dict(mean_success=skill, by_seed={str(s): dict(mean_success=skill) for s in (51, 73, 91)}),
                full_games=dict(mean_points=points))


def test_memory_selection_requires_all_four_thresholds():
    control, memory = selection_report(), selection_report(.56, .59)
    assert select_candidate(control, memory, benchmark(), benchmark(75))["selected"] == "memory"
    assert select_candidate(control, memory, benchmark(), benchmark(69))["selected"] == "control"
    assert select_candidate(control, selection_report(.54), benchmark(), benchmark(80))["selected"] == "control"
    assert select_candidate(control, selection_report(.6, .56), benchmark(), benchmark(80))["selected"] == "control"
    memory["functional"]["by_seed"]["73"]["mean_success"] = .49
    memory["functional"]["by_seed"]["91"]["mean_success"] = .49
    assert select_candidate(control, memory, benchmark(), benchmark(80))["selected"] == "control"
    assert select_candidate(control, selection_report(.6), benchmark(), benchmark(80, memory_gate_passed=False))["selected"] == "control"


def test_unknown_benchmark_or_unequal_profiles_do_not_select_memory():
    a, b = selection_report(), selection_report(.6)
    assert select_candidate(a, b, {}, {})["selected"] == "control"
    modified = benchmark(80)
    modified["rs4_profile"] = {"phase_id": "B"}
    assert select_candidate(a, b, benchmark(), modified)["selected"] == "control"


def test_evaluation_normalization_preserves_original_counts():
    original = dict(functional=dict(skills={"restart_success": .9}), full_games=dict(mean_points=.7, wins=6))
    baseline = dict(functional=dict(skills={"restart_success": .4}), full_games=dict(mean_points=.6))
    converted = normalized_evaluation(original, baseline)
    assert converted["skills"]["restart_success"] == .9
    assert converted["baseline"]["restart_success"] == .4
    assert converted["matches"] == dict(points=.7, baseline_points=.6)
    assert converted["full_games"]["wins"] == 6
    assert "skills" not in original


def test_match_pfsp_counts_draws_but_goal_record_remains_legacy():
    model = build_model(dict(obs_dim=2, hidden=4, layers=1))
    league = League(match_pfsp=True)
    league.add_snapshot(model, "parent", protected=True)
    member = league.members[0]
    before = member.winrate
    league.record_match(0, .5)
    assert member.matches == 3 and member.match_points == 1.5
    assert member.winrate == before
    league.record(0, 3, 0)
    assert member.games == 5 and member.wins == 4
    assert member.match_rate == .5
    with pytest.raises(ValueError):
        league.record_match(0, .8)


def test_legacy_match_record_is_noop_and_legacy_sampling_is_unchanged():
    model = build_model(dict(obs_dim=2, hidden=4, layers=1))
    league = League(recent_weight=.25)
    for i in range(3):
        league.add_snapshot(model, str(i))
        league.members[i].wins += i
        league.members[i].games += i
    league.record_match(0, 1)
    assert league.members[0].matches == 2
    expected = np.array([(1 - m.winrate) ** 2 + 1e-3 for m in league.members])
    expected[-1] += expected.sum() * .25
    rng = np.random.default_rng(51)
    assert league.sample(20, rng) == list(np.random.default_rng(51).choice(3, size=20, p=expected / expected.sum()))


def test_match_mixture_preserves_protected_snapshots_and_intermediates():
    model = build_model(dict(obs_dim=2, hidden=4, layers=1))
    league = League(max_size=3, match_pfsp=True)
    league.add_snapshot(model, "parent", protected=True)
    league.add_snapshot(model, "easy", steps=50)
    league.members[-1].match_points, league.members[-1].matches = 99, 100
    league.add_snapshot(model, "medium", steps=100)
    assert league.sampling_probabilities()[2] > league.sampling_probabilities()[1]
    league.add_snapshot(model, "new", steps=150)
    assert [m.name for m in league.members] == ["parent", "medium", "new"]
    assert league.members[0].protected
    assert sum(league.sampling_probabilities()) == pytest.approx(1.)


@pytest.fixture
def source(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    cfg = load_config(preparer.PROFILE)
    cfg.pop("rs4_program")
    cfg.pop("bc_reference", None)
    cfg["model"].update(hidden=16, layers=1, ent_hidden=8, ent_layers=1, pooling="meanmax")
    from env.haxball_env import U_SELF_DIM
    model_cfg = {**cfg["model"], "self_dim": U_SELF_DIM, "ent_dim": 8, "n_actions": 18}
    model = build_model(model_cfg)
    opt = torch.optim.Adam(model.parameters(), lr=.00005, eps=1e-5)
    obs = torch.randn(8, U_SELF_DIM + 7 * 8)
    logits, values = model(obs)
    (logits.square().mean() + values.square().mean()).backward()
    opt.step()
    ck = dict(model=model.state_dict(), model_config=model.config(), opt=opt.state_dict(), league=[],
              steps=4_540_765_440, iteration=44115, stage=0, stage_steps=1234,
              env=dict(obs_layout="universal"), task_state={"rs4_4v4": {}}, learner_elo=2300., scripted_elo=1500.)
    directory = tmp_path / "runs" / "rs4_adapt"
    directory.mkdir(parents=True)
    path = directory / "latest.pt"
    torch.save(ck, path)
    (directory / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    monkeypatch.setattr(preparer, "ROOT", tmp_path)
    return path, ck, cfg


def test_prepare_dry_run_is_read_only_and_migrates_both_in_memory(source):
    path, ck, cfg = source
    original = path.read_bytes()
    result = preparer.prepare(path, dry_run=True)
    assert not (path.parents[1] / "rs4_v3").exists()
    assert path.read_bytes() == original
    assert result["candidates"]["control"]["ppo"]["total_steps"] == ck["steps"] + 200_000_000
    assert result["candidates"]["memory"]["model"]["memory_size"] == 32


def test_prepare_protects_source_and_ledger_and_never_overwrites(source):
    path, ck, cfg = source
    before = path.read_bytes()
    result = preparer.prepare(path)
    assert (result / "parent.pt").read_bytes() == before == path.read_bytes()
    assert (result / "parent_config.yaml").read_bytes() == path.with_name("config.yaml").read_bytes()
    for branch in ("control", "memory"):
        candidate = torch.load(result / branch / "latest.pt", weights_only=False)
        assert candidate["steps"] == ck["steps"]
        assert candidate["iteration"] == ck["iteration"]
        assert candidate["rs4_program_state"]["relative_steps"] == 0
        assert candidate["league"][-1]["protected"]
        assert candidate["league"][-1]["name"] == "v3_initial_reference"
        assert candidate["opt"]["state"]
    ledger = json.loads((result / "ledger.json").read_text())
    assert refresh_ledger(result, ledger)["consumed_steps"] == 0
    assert ledger["total_budget_steps"] == 6_000_000_000
    with pytest.raises(FileExistsError):
        preparer.prepare(path)


@pytest.mark.parametrize("branch", ["control", "memory"])
def test_runner_uses_selected_directory_not_inherited_run_name(source, monkeypatch, branch):
    from tools import run_rs4_v3 as runner
    path, _, _ = source
    source_bytes = path.read_bytes()
    directory = preparer.prepare(path, run="rs4_v3_public")
    config_path = directory / branch / "config.yaml"
    config = load_config(config_path)
    config["run_name"] = f"rs4_v3/{branch}"  # reproduce the already-migrated Pod YAML
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    checkpoint_path = directory / branch / "latest.pt"
    initial = torch.load(checkpoint_path, weights_only=False)
    frozen_parent = (directory / "parent.pt").read_bytes()
    other_path = directory / ("memory" if branch == "control" else "control") / "latest.pt"
    other_bytes = other_path.read_bytes()
    commands = []

    def simulated_child(command):
        commands.append(command)
        assert command[command.index("--run") + 1] == f"rs4_v3_public/{branch}"
        assert command[command.index("--config") + 1] == str(config_path)
        assert command[-1] == f"ppo.total_steps={initial['steps'] + 13}"
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        program = ProgramState.from_config(config, checkpoint["rs4_program_state"])
        program.advance_steps(13)
        checkpoint["rs4_program_state"] = program.state_dict()
        checkpoint["steps"] += 13
        torch.save(checkpoint, checkpoint_path)

    monkeypatch.setattr(runner, "_invoke", simulated_child)
    runner._train_chunk(directory, branch, 13)
    assert len(commands) == 1
    assert path.read_bytes() == source_bytes
    assert (directory / "parent.pt").read_bytes() == frozen_parent
    assert other_path.read_bytes() == other_bytes


def test_preparer_refuses_missing_teacher_before_creating_destination(source):
    path, ck, cfg = source
    cfg["bc_reference"] = "runs/missing/bc.pt"
    path.with_name("config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="referencia"):
        preparer.prepare(path)
    assert not (path.parents[1] / "rs4_v3").exists()


def test_ledger_rejects_steps_inconsistent_with_program(source):
    path, _, _ = source
    result = preparer.prepare(path)
    candidate = result / "control/latest.pt"
    checkpoint = torch.load(candidate, weights_only=False)
    checkpoint["steps"] += 1
    torch.save(checkpoint, candidate)
    ledger = json.loads((result / "ledger.json").read_text())
    with pytest.raises(ValueError, match="no coinciden"):
        refresh_ledger(result, ledger)


def test_runner_end_to_end_exact_budget_selection_resume_and_champions(source, monkeypatch):
    from tools import run_rs4_v3 as runner
    path, _, _ = source
    directory = preparer.prepare(path)
    monkeypatch.setattr(runner, "ROOT", directory.parents[1])
    train_calls = []
    def train_chunk(root, branch, target):
        checkpoint_path, ck, cfg, program = runner._checkpoint(root, branch)
        train_calls.append((branch, target))
        program.advance_steps(target - program.relative_steps)
        ck["steps"] = cfg["rs4_program"]["start_steps"] + program.relative_steps
        ck["rs4_program_state"] = program.state_dict()
        torch.save(ck, checkpoint_path)
    monkeypatch.setattr(runner, "_train_chunk", train_chunk)
    def evaluate(root, branch, **options):
        report = selection_report(.5 if branch == "control" else .56, .6)
        report.update(good_report())
        report["reference"] = selection_report(.4, .5)
        for metric in ("full_games",):
            report[metric]["by_seed"] = {str(s): dict(mean_points=.6) for s in (51, 73, 91)}
            report["reference"][metric]["by_seed"] = {str(s): dict(mean_points=.5) for s in (51, 73, 91)}
        reports = root / "evaluations"
        reports.mkdir(exist_ok=True)
        output = reports / f"{branch}_{options['label']}.json"
        output.write_text(json.dumps(report), encoding="utf-8")
        return output, report
    monkeypatch.setattr(runner, "_evaluate", evaluate)
    monkeypatch.setattr(runner, "_benchmarks", lambda *a: dict(
        control=benchmark(100, diagnostic_ppo_samples=3_300_000),
        memory=benchmark(80, diagnostic_ppo_samples=3_300_000)))
    monkeypatch.setattr(runner, "_render_phase", lambda *a: None)
    outcome = runner.run(resume=True)
    assert outcome["selected"] == "memory"
    assert outcome["consumed_steps"] == 6_000_000_000
    assert outcome["candidates"]["control"]["useful_steps"] == 200_000_000
    assert outcome["candidates"]["memory"]["useful_steps"] == BRANCH_BUDGET - 6_600_000
    assert outcome["diagnostic_ppo_steps"] == 6_600_000
    assert outcome["retained_steps"] == 6_000_000_000 - 6_600_000
    assert outcome["complete"] and outcome["sporting_success"]
    assert outcome["pending_skills"] == []
    assert (directory / "champion.pt").is_file()
    assert set(outcome["phase_champions"]) == set("ABCDEF")
    before = len(train_calls)
    assert runner.run(resume=True)["consumed_steps"] == 6_000_000_000
    assert len(train_calls) == before


def test_lr_events_and_sustained_low_kl_survive_resume():
    program = state()
    for iteration in range(1, 26):
        program.update_lr(.003 if iteration == 1 else .0001, iteration)
    assert program.lr == 1e-4  # una ventana mixta no se llama KL persistentemente baja
    assert program.lr_events[-1]["decision"] == "hold"
    restored = ProgramState(program.config, program.state_dict())
    for iteration in range(26, 51):
        restored.update_lr(.0001, iteration)
    assert restored.lr == pytest.approx(1.1e-4)
    assert restored.lr_events[-1]["decision"] == "increase"
    assert len(restored.lr_kl_window) == 25


def test_current_bc_teacher_is_frozen_byte_exact_and_source_config_untouched(source):
    path, ck, cfg = source
    teacher = path.parent / "bc.pt"
    torch.save(dict(model=ck["model"], model_config=ck["model_config"]), teacher)
    cfg["bc_reference"] = str(teacher)
    path.with_name("config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    before = path.with_name("config.yaml").read_bytes()
    destination = preparer.prepare(path)
    assert (destination / "teacher.pt").read_bytes() == teacher.read_bytes()
    assert path.with_name("config.yaml").read_bytes() == before
    for branch in ("control", "memory"):
        config = load_config(destination / branch / "config.yaml")
        assert config["bc_reference"] == "runs/rs4_v3/teacher.pt"


def test_new_teacher_requires_approved_sha_bound_validation_against_current_teacher(source):
    path, ck, cfg = source
    old = path.parent / "old_bc.pt"
    new = path.parent / "new_bc.pt"
    for teacher in (old, new):
        torch.save(dict(model=ck["model"], model_config=ck["model_config"]), teacher)
    cfg["bc_reference"] = str(old)
    path.with_name("config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    validation = path.parent / "validation.json"
    evidence = dict(candidate_sha256=preparer.file_hash(new), source_sha256=preparer.file_hash(old),
                    gate=dict(approved_as_weak_reference=False))
    validation.write_text(json.dumps(evidence), encoding="utf-8")
    with pytest.raises(ValueError, match="validación aprobada"):
        preparer.prepare(path, bc_reference=new, bc_validation=validation)
    evidence["gate"]["approved_as_weak_reference"] = True
    evidence["source_sha256"] = "incorrect"
    validation.write_text(json.dumps(evidence), encoding="utf-8")
    with pytest.raises(ValueError, match="teacher actual"):
        preparer.prepare(path, bc_reference=new, bc_validation=validation)
    evidence["source_sha256"] = preparer.file_hash(old)
    validation.write_text(json.dumps(evidence), encoding="utf-8")
    destination = preparer.prepare(path, bc_reference=new, bc_validation=validation)
    assert (destination / "teacher.pt").read_bytes() == new.read_bytes()
    assert (destination / "teacher_validation.json").read_bytes() == validation.read_bytes()
    assert json.loads((destination / "specialization.json").read_text())["bc_teacher"]["sha256"] == preparer.file_hash(new)


def test_diagnostics_are_charged_once_inside_6b_preserving_final_200m():
    program = state()
    program.advance_steps(200_000_000)
    program.charge_diagnostics(6_600_000)
    assert program.effective_branch_limit == 5_793_400_000
    assert program.relative_steps == 200_000_000  # nunca mover anclas ni fingir entrenamiento retenido
    assert program.config["start_steps"] == 4_500_000_000
    restored = ProgramState(program.config, program.state_dict())
    restored.charge_diagnostics(6_600_000)
    with pytest.raises(ValueError, match="fijado"):
        restored.charge_diagnostics(6_700_000)
    restored.advance_steps(restored.remaining_steps - 200_000_000)
    assert restored.settings()["guide_coef"] == 0
    assert restored.settings()["restart_potential_coef"] == 0
    restored.advance_steps(200_000_000)
    assert restored.complete
    assert restored.relative_steps + restored.diagnostic_steps + 200_000_000 == 6_000_000_000
    assert restored.entropy_coef == .001
    assert restored.bc_coef == .001


def test_runner_interrupt_forwards_to_child_and_does_not_treat_save_as_completion(monkeypatch):
    from tools import run_rs4_v3 as runner
    import signal
    class Child:
        def __init__(self):
            self.waits = 0
            self.returncode = None
            self.signals = []
        def wait(self, timeout=None):
            self.waits += 1
            if self.waits == 1:
                raise KeyboardInterrupt
            assert timeout == 30
            self.returncode = 0
            return 0
        def poll(self):
            return self.returncode
        def send_signal(self, sig):
            self.signals.append(sig)
    child = Child()
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: child)
    with pytest.raises(KeyboardInterrupt):
        runner._invoke(["python", "-m", "train.multitask"])
    expected = signal.CTRL_BREAK_EVENT if runner.os.name == "nt" else signal.SIGINT
    assert child.signals == [expected]
    assert child.returncode == 0


def test_recovery_never_starts_a_100m_block_after_350m_already_consumed():
    program = state()
    program.phase_index = 5
    program.phase_start_steps = 5_200_000_000
    program.relative_steps = 5_250_000_000
    program.recovery_consumed = 350_000_000
    program.skill_debts = ["defense"]
    program.advance_steps(0)
    assert program.recovery_block_start is None
    assert not program.settings()["recovery"]


def test_interrupted_benchmark_reserves_cost_and_resume_charges_all_attempts(source, monkeypatch):
    from tools import run_rs4_v3 as runner
    from pathlib import Path
    path, _, _ = source
    directory = preparer.prepare(path)
    _, _, cfg, _ = runner._checkpoint(directory, "control")
    maximum = 2 * cfg["ppo"]["rollout_len"] * cfg["env"]["agents"]
    monkeypatch.setattr(runner, "_invoke", lambda *args: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        runner._benchmarks(directory, warmup=1, iters=1)
    work_path = directory / "benchmarks/work_ledger.json"
    pending = json.loads(work_path.read_text())
    assert len(pending["attempts"]) == 1
    assert pending["attempts"][0]["charged_steps"] == maximum
    assert pending["attempts"][0]["status"] == "reserved"
    ledger = json.loads((directory / "ledger.json").read_text())
    assert runner.refresh_ledger(directory, ledger)["consumed_steps"] == maximum
    def succeed(command):
        output = Path(command[command.index("--json-output") + 1])
        report = benchmark(100, diagnostic_ppo_samples=123, samples=100, warmup_samples=23)
        output.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(runner, "_invoke", succeed)
    summaries = runner._benchmarks(directory, warmup=1, iters=1)
    assert summaries["control"]["diagnostic_work_total"] == maximum + 6 * 123
    work = json.loads(work_path.read_text())
    assert len(work["attempts"]) == 7
    assert sum(row["status"] == "reconciled" for row in work["attempts"]) == 6
    again = runner._benchmarks(directory, warmup=1, iters=1)
    assert again["control"]["diagnostic_work_total"] == maximum + 6 * 123
    assert len(json.loads(work_path.read_text())["attempts"]) == 7


def test_replay_is_bounded_uses_frozen_champion_and_valid_file_is_not_regenerated(tmp_path, monkeypatch):
    from tools import run_rs4_v3 as runner
    from pathlib import Path
    checkpoint = tmp_path / "phase_A_champion.pt"
    checkpoint.write_bytes(b"frozen-test-policy")
    (tmp_path / "champion.pt").write_bytes(b"final-selected-policy")
    calls = []
    def invoke(command, **kwargs):
        calls.append((command, kwargs))
        Path(command[command.index("--out") + 1]).write_text("<!doctype html><html>test</html>", encoding="utf-8")
    monkeypatch.setattr(runner, "_invoke", invoke)
    ledger = {}
    runner._render_phase(tmp_path, "A", ledger)
    runner._render_phase(tmp_path, "A", ledger)
    assert len(calls) == 1
    assert str(checkpoint) in calls[0][0]
    assert calls[0][1]["env"]["OMP_NUM_THREADS"] == "2"
    assert ledger["replays"]["A"]["path"] == "replays/phase_A_vs_r3_style_0.html"
    runner._render_phase(tmp_path, "F", ledger)
    assert str(tmp_path / "champion.pt") in calls[-1][0]
    assert len(ledger["replays"]) == 2
