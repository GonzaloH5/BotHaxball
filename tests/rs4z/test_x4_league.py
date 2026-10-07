"""Liga barata (learn/x4_league.py) con los procesos simulados, y las piezas del trainer que usa."""
import argparse
import json
from pathlib import Path

from learn import x4_league as L
from learn import x4_queue as Q


def _league(tmp_path, **kw):
    a = argparse.Namespace(root=str(tmp_path), bc="bc.pt", device="cpu", init=str(tmp_path / "init.pt"), envs=1024,
                           rollout=64, segments=2, segment_updates=3000, exploiter_updates=600, exploiter_envs=1024,
                           accept=0.55, h2h_matches=64, h2h_minutes=3.0, main_pool_frac=0.5, robust_margin=0.05,
                           min_bc=0.8, refs="runs/a/best.pt", rl_extra="--lambda-dist 0.05 --lambda-decay 1",
                           export_to=str(tmp_path / "x.onnx"), confirm_seed=20261007, cert_extra="", preflight_extra="")
    for k, v in kw.items():
        setattr(a, k, v)
    a.updates = a.extend_to = a.segments * a.segment_updates
    (tmp_path / "init.pt").write_text("init")
    return L.League(a)


def _fake(tmp_path, scores):
    """Procesos simulados: los entrenamientos dejan log y last.pt; los enfrentamientos, su puntaje."""
    calls = []

    def run(name, cmd, retries=5, ok_codes=(0,)):
        calls.append((name, [str(x) for x in cmd]))
        cmd = [str(x) for x in cmd]
        if cmd[0] == "learn.x4_ppo":
            out = Path(cmd[cmd.index("--out") + 1])
            out.mkdir(parents=True, exist_ok=True)
            upd = int(cmd[cmd.index("--updates") + 1])
            (out / "log.jsonl").write_text("\n".join(json.dumps(dict(update=u)) for u in range(1, upd + 1)))
            (out / "last.pt").write_text(f"{out.name}@{upd}")
            if out.name == "principal":
                (out / "best.pt").write_text("best")
        elif cmd[0] == "learn.x4_league":
            out = Path(cmd[cmd.index("--out") + 1])
            out.write_text(json.dumps(dict(score=scores(name), wins=1, draws=0, losses=0)))
        elif cmd[0] == "learn.x4_certify":
            Path(cmd[cmd.index("--out") + 1]).write_text(json.dumps(dict(aprobado=False)))
            return Q.CERT_NOT_APPROVED
        return 0
    return run, calls


def test_league_cycle_exploiters_join_the_pool_and_the_verdict_uses_exploitability(tmp_path):
    q = _league(tmp_path)
    table = {"h2h_explotador_0": 0.70, "h2h_explotador_1": 0.62, "h2h_explotador_2": 0.58,
             "final_vs_bc": 0.88, "final_vs_explotador_0": 0.60, "final_vs_explotador_1": 0.55}
    run, calls = _fake(tmp_path, lambda n: table.get(n, 0.6))
    q.run = run
    q.main()
    names = [n for n, _ in calls]
    order = ["preflight", "explotador_0", "h2h_explotador_0", "principal_1", "explotador_1", "h2h_explotador_1",
             "principal_2", "explotador_2", "h2h_explotador_2"]
    assert names[:len(order)] == order
    cmd = dict(calls)
    ex0 = cmd["explotador_0"]
    assert ex0[-1] == "--no-pool-bc" and ex0[ex0.index("--pool-frac") + 1] == "1.0"
    assert ex0[ex0.index("--pool") + 1].endswith("objetivo_00.pt") and ex0[ex0.index("--init") + 1].endswith("objetivo_00.pt")
    assert ex0[ex0.index("--lambda-dist") + 1] == "0.05" and ex0[ex0.index("--updates") + 1] == "600"
    # el primer objetivo es el checkpoint de arranque; el segundo, el principal al final del tramo 1
    assert (tmp_path / "pool" / "objetivo_00.pt").read_text() == "init"
    assert (tmp_path / "pool" / "objetivo_01.pt").read_text() == "principal@3000"
    p1, p2 = cmd["principal_1"], cmd["principal_2"]
    assert p1[p1.index("--pool") + 1].endswith("explotador_00.pt") and p1[p1.index("--updates") + 1] == "3000"
    assert p2[p2.index("--pool") + 1].count("explotador_0") == 2 and p2[p2.index("--updates") + 1] == "6000"
    assert p2[p2.index("--pool-frac") + 1] == "0.5"
    assert q.state["explotabilidad"] == [0.70, 0.62, 0.58]
    v = q.state["veredicto_liga"]
    assert v["veredicto"] == "mas_dificil_de_explotar" and v["explotabilidad_baja"] and v["le_gana_a_explotadores_viejos"]
    assert set(q.state["evaluacion_final"]) >= {"bc", "scripted", "inicio", "a_best", "explotador_0", "explotador_2"}
    assert "export" in names


def test_a_weak_exploiter_stays_out_and_no_drop_means_no_robustness_claim(tmp_path):
    q = _league(tmp_path, segments=1)
    table = {"h2h_explotador_0": 0.50, "h2h_explotador_1": 0.49, "final_vs_bc": 0.9}
    run, calls = _fake(tmp_path, lambda n: table.get(n, 0.6))
    q.run = run
    q.main()
    p1 = dict(calls)["principal_1"]
    assert p1[p1.index("--pool") + 1] == ""                        # 0,50 < 0,55: no entra al pool
    assert q.state["explotadores"]["0"]["aceptado"] is False
    assert q.state["veredicto_liga"]["veredicto"] == "sin_mejora_de_robustez"   # 0,49 no es 0,05 menos que 0,50


def test_trainer_without_bc_in_the_pool_and_pool_records_by_name(tmp_path):
    import torch
    from learn.x4_ppo import Trainer, parse_args
    bc = Path(__file__).resolve().parents[2] / "runs" / "x4_bc" / "final_sangu_rsone" / "best.pt"
    if not bc.exists():
        import pytest
        pytest.skip("falta el checkpoint de la imitación")
    a_pt, b_pt = tmp_path / "rival_a.pt", tmp_path / "rival_b.pt"
    for p in (a_pt, b_pt):
        p.write_bytes(bc.read_bytes())
    common = ["--bc", str(bc), "--out", str(tmp_path / "run"), "--device", "cpu", "--envs", "4", "--rollout", "4",
              "--critic-warmup", "0", "--bank-recordings", "0", "--eval-every", "0", "--threads", "1",
              "--epochs", "1", "--minibatches", "1", "--snapshot-every", "0", "--human-starts", "0"]
    t = Trainer(parse_args(common + ["--updates", "1", "--no-pool-bc", "--pool", str(a_pt), "--pool-frac", "1.0"]))
    assert [n for n, _ in t.pool] == ["rival_a"] and t.bc_slot == -1
    assert (t.arenas[0].opp == 0).all()                      # todos los partidos contra el único rival
    t.pool_wins[:] = 0.2
    t.train()
    # al reanudar con otro pool fijo (la liga agrega rivales), cada récord sigue con su rival
    t2 = Trainer(parse_args(common + ["--updates", "2", "--resume", "--pool", f"{b_pt},{a_pt}", "--pool-frac", "0.5"]))
    names = [n for n, _ in t2.pool]
    assert names == ["bc", "rival_b", "rival_a"]
    w = dict(zip(names, t2.pool_wins))
    assert abs(w["rival_a"] - t.pool_wins[0]) < 1e-9 and w["rival_b"] == 0.5 and w["bc"] == 0.5
    assert torch.load(tmp_path / "run" / "last.pt", map_location="cpu")["pool_wins_by_name"]["rival_a"] == w["rival_a"]
