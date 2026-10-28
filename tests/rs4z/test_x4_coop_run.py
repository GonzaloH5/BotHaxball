"""Experiment budget, held-out separation, and conservative verdict contracts."""
import json
from types import SimpleNamespace

from learn.x4_coop_run import branch_command, evaluation_command, final_verdict, pilot_gate


def arguments(tmp_path):
    return SimpleNamespace(bc="bc.pt", init="original.pt", device="cpu", envs=512,
        rollout=128, threads=5, recordings=80, eval_every=100, seed=7, root=str(tmp_path),
        matches=32, probe_states=128, eval_recordings=40)


def test_budget_and_training_settings_only_differ_in_curriculum(tmp_path):
    a = arguments(tmp_path)
    control, candidate = [branch_command(a, b, 1500) for b in ("control", "curriculum")]
    prefix_control = control[:control.index("--out")]
    prefix_candidate = candidate[:candidate.index("--out")]
    assert prefix_control == prefix_candidate
    assert control[control.index("--updates") + 1] == "1500"
    assert candidate[candidate.index("--updates") + 1] == "1500"
    assert "--resume" in candidate
    assert prefix_control[prefix_control.index("--pass-ref-key") + 1] == "sanguchito_train"
    assert "--pass-ready" in prefix_control
    assert "--macro-trace" in prefix_control


def test_pilot_uses_dev_and_final_uses_test(tmp_path):
    a = arguments(tmp_path)
    pilot, final = [evaluation_command(a, split, "out.json") for split in ("dev", "test")]
    assert pilot[pilot.index("--split") + 1] == "dev"
    assert final[final.index("--split") + 1] == "test"
    assert final[final.index("--matches") + 1] == "32"
    assert pilot[pilot.index("--matches") + 1] == "4"


def write_pilot(tmp_path, *, incomplete=False, skipped=0):
    for branch in ("control", "curriculum"):
        path = tmp_path / branch
        path.mkdir()
        row = dict(update=100, nonfinite_skipped=skipped, coop_actor_samples=128,
                   coop=dict(episodes=40, successes=1))
        (path / "log.jsonl").write_text(json.dumps(row) + "\n")
        evs = [dict(update=u, vs_bc=dict(score=.8, valid_for_strength=not incomplete)) for u in (0, 100)]
        (path / "eval.jsonl").write_text("\n".join(map(json.dumps, evs)))


def test_pilot_cannot_accept_partial_matches_or_nonfinite_updates(tmp_path):
    write_pilot(tmp_path, incomplete=True)
    assert not pilot_gate(tmp_path, 100)["continue_training"]
    for b in ("control", "curriculum"):
        p = tmp_path / b / "eval.jsonl"
        p.write_text(p.read_text().replace("false", "true"))
    assert pilot_gate(tmp_path, 100)["continue_training"]
    p = tmp_path / "control/log.jsonl"
    row = json.loads(p.read_text())
    row["nonfinite_skipped"] = 1
    p.write_text(json.dumps(row))
    assert not pilot_gate(tmp_path, 100)["continue_training"]


def test_pilot_cannot_reuse_a_longer_preexisting_run(tmp_path):
    write_pilot(tmp_path)
    for branch in ("control", "curriculum"):
        path = tmp_path / branch
        (path / "log.jsonl").write_text((path / "log.jsonl").read_text().replace('"update": 100', '"update": 500'))
    assert not pilot_gate(tmp_path, 100)["continue_training"]


def test_strength_verdict_requires_uncertainty_and_complete_matches():
    match = dict(valid_for_strength=True, point_fraction=dict(valor=.6, ic90=[.49, .72]))
    report = dict(matches={"new": {"control": match,
        "original": dict(valid_for_strength=True, point_fraction=dict(valor=.51, ic90=[.4, .6]))}})
    assert final_verdict(report, "new", "control", "original")["verdict"] == "no_confirmed_improvement"
    match["point_fraction"]["ic90"] = [.51, .7]
    assert final_verdict(report, "new", "control", "original")["verdict"] == "improved_simulated_strength"
    match.update(valid_for_strength=False, point_fraction=dict(valor=None, ic90=[None, None]))
    verdict = final_verdict(report, "new", "control", "original")
    assert verdict["verdict"] == "no_confirmed_improvement"
    assert verdict["human_competitiveness"] == "unverified"
