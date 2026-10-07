"""Reglas de la cola pre-registrada del pod (learn/x4_queue.py) con los procesos simulados."""
import argparse
import json

from learn import x4_queue as Q


def _queue(tmp_path, **kw):
    a = argparse.Namespace(root=str(tmp_path), bc="bc.pt", device="cpu", envs=1024, rollout=64, updates=3000,
                           extend_to=6000, ab_updates=300, export_to=str(tmp_path / "x.onnx"), rl_extra="",
                           cert_extra="", preflight_extra="", confirm_seed=20261007)
    for k, v in kw.items():
        setattr(a, k, v)
    return Q.Queue(a)


def _evals(path, rows):
    path.mkdir(parents=True, exist_ok=True)
    (path / "eval.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def _row(u, idx, **kw):
    return dict(update=u, baseline=u == 0, vs_bc=dict(score=0.5), drift=[], cadena_pase=dict(indice=idx), **kw)


def _log(path, last_update):
    path.mkdir(parents=True, exist_ok=True)
    (path / "log.jsonl").write_text("\n".join(json.dumps(dict(update=u)) for u in range(1, last_update + 1)))


def test_crashed_main_run_is_a_technical_failure(tmp_path):
    """El RL principal se cae en todos los reintentos a mitad de camino: no se extiende y, si nada aprueba, el veredicto
    es "revisar" (antes seguía igual y terminaba en "no_competitivo", como si se hubiera medido)."""
    q = _queue(tmp_path)
    _evals(tmp_path / "rl_epv", [_row(u, 0.5 + u / 10000) for u in range(0, 1201, 50)])
    _log(tmp_path / "rl_epv", 1200)
    (tmp_path / "rl_epv" / "best_pase.pt").write_text("x")
    q.state["done"].update(preflight=0, rl_epv=0, rl_checkpoint=0)
    calls = []

    def fake_run(name, cmd, retries=5, ok_codes=(0,)):
        calls.append(name)
        if name == "rl_epv_full":
            return 1
        if name.startswith("cert_"):
            out = cmd[cmd.index("--out") + 1]
            out.write_text(json.dumps(dict(aprobado=False, gate_sanguchito=dict(indice_cadena=0.7))))
            return Q.CERT_NOT_APPROVED
        return 0
    q.run = fake_run
    assert q.main() == 2
    assert "rl_epv_full" in q.state["fallo_tecnico"] and q.state["fallo_tecnico"]["rl_epv_full"]["actualizacion"] == 1200
    assert "extension" not in q.state and not any(c.endswith("_ext") for c in calls)
    assert q.state["veredicto"] == "revisar"
    assert "export" in calls                            # igual se exporta el mejor para pruebas


def test_finished_main_run_without_approval_is_not_competitive(tmp_path):
    q = _queue(tmp_path)
    _evals(tmp_path / "rl_epv", [_row(u, 0.8 - u / 100000) for u in range(0, 3001, 50)])   # índice bajando: sin extensión
    _log(tmp_path / "rl_epv", 3000)
    (tmp_path / "rl_epv" / "best.pt").write_text("x")
    q.state["done"].update(preflight=0, rl_epv=0, rl_checkpoint=0)

    def fake_run(name, cmd, retries=5, ok_codes=(0,)):
        if name.startswith("cert_"):
            cmd[cmd.index("--out") + 1].write_text(json.dumps(dict(aprobado=False)))
            return Q.CERT_NOT_APPROVED
        return 0
    q.run = fake_run
    assert q.main() == 2
    assert "fallo_tecnico" not in q.state and q.state["extension"]["decide"] is False
    assert q.state["veredicto"] == "no_competitivo"


def test_certification_needs_a_confirmation_with_other_seeds(tmp_path):
    """El primer candidato aprueba una vez pero no se confirma; el segundo aprueba las dos: se elige el segundo."""
    q = _queue(tmp_path)
    d = tmp_path / "rl_epv"
    d.mkdir()
    for f in ("pase_aprobado_02000.pt", "best_pase.pt", "best.pt"):
        (d / f).write_text("x")
    codes = {"cert_rl_epv_pase_aprobado_02000": 0, "cert_rl_epv_pase_aprobado_02000_confirmacion": Q.CERT_NOT_APPROVED,
             "cert_rl_epv_best_pase": 0, "cert_rl_epv_best_pase_confirmacion": 0}
    seeds = {}

    def fake_run(name, cmd, retries=5, ok_codes=(0,)):
        if name.startswith("cert_"):
            assert ok_codes == (0, Q.CERT_NOT_APPROVED)
            seeds[name] = cmd[cmd.index("--seed") + 1] if "--seed" in cmd else None
            cmd[cmd.index("--out") + 1].write_text(json.dumps(dict(aprobado=codes[name] == 0)))
            return codes[name]
        return 0
    q.run = fake_run
    assert q.certify(["rl_epv"]) == 0
    assert q.state["veredicto"] == "competitivo_en_pases" and q.state["elegido"].endswith("best_pase.pt")
    assert seeds["cert_rl_epv_best_pase"] is None and seeds["cert_rl_epv_best_pase_confirmacion"] == 20261007
    assert "cert_rl_epv_best" not in seeds              # el tercero ya no hace falta


def test_crashed_certification_is_not_a_rejection(tmp_path):
    q = _queue(tmp_path)
    d = tmp_path / "rl_epv"
    d.mkdir()
    (d / "best.pt").write_text("x")

    def fake_run(name, cmd, retries=5, ok_codes=(0,)):
        return 1 if name.startswith("cert_") else 0       # Python cae con 1 y no deja reporte
    q.run = fake_run
    assert q.certify(["rl_epv"]) == 2
    assert q.state["veredicto"] == "revisar" and q.state["certificacion_con_error"]


def test_ab_uses_the_mean_of_the_last_evaluations(tmp_path):
    """Con la última evaluación sola ganaba franjas (0,70 > 0,60 + 0,03); con la media de las últimas 3 la diferencia
    (0,62 contra 0,60) queda dentro del margen y gana el EPV."""
    q = _queue(tmp_path)
    _evals(tmp_path / "rl_epv", [_row(0, 0.5)] + [_row(u, 0.60) for u in (150, 200, 250, 300)])
    _evals(tmp_path / "rl_checkpoint", [_row(0, 0.5)] + [_row(u, v) for u, v in ((150, 0.9), (200, 0.58),
                                                                                (250, 0.58), (300, 0.70))])
    assert q.choose_shaping() == "epv"
    assert q.state["ab_temprano"]["checkpoint"]["indice"] == 0.62
    assert q.state["ab_temprano"]["checkpoint"]["evaluaciones"] == Q.AB_LAST


def test_evaluations_are_deduplicated_by_update(tmp_path):
    """Un corte entre escribir una evaluación y guardar last.pt la repite al reanudar: queda una por actualización."""
    q = _queue(tmp_path)
    _evals(tmp_path / "r", [_row(0, 0.5), _row(50, 0.6), _row(100, 0.61), _row(100, 0.62), _row(150, 0.7)])
    with open(tmp_path / "r" / "eval.jsonl", "a", encoding="utf-8") as fh:
        fh.write('\n{"update": 200, "cadena')           # línea cortada
    ev = q.evals("r")
    assert [e["update"] for e in ev] == [0, 50, 100, 150] and ev[2]["cadena_pase"]["indice"] == 0.62


def test_recovery_keeps_the_pass_arm_of_the_main_run(tmp_path):
    q = _queue(tmp_path)
    d = tmp_path / "rl_epv"
    _evals(d, [_row(0, 0.5), _row(1000, 0.7, brazo_pases_activado=dict(indice=0.7, umbral=0.85, pass_bonus=0.05))])
    (d / "stopped.json").write_text("{}")
    (d / "best_pase.pt").write_text("x")
    q.state["done"].update(preflight=0, rl_epv=0, rl_checkpoint=0)
    q.state["shaping_elegido"] = "epv"
    cmds = {}

    def fake_run(name, cmd, retries=5, ok_codes=(0,)):
        cmds[name] = [str(x) for x in cmd]
        if name == "rl_recuperacion":
            _log(tmp_path / "rl_recuperacion", 3000)
        if name.startswith("cert_"):
            cmd[cmd.index("--out") + 1].write_text(json.dumps(dict(aprobado=False)))
            return Q.CERT_NOT_APPROVED
        return 0
    q.run = fake_run
    q.main()
    rec = cmds["rl_recuperacion"]
    assert rec[rec.index("--pass-bonus") + 1] == "0.05" and rec[rec.index("--lambda-dist") + 1] == "0.4"
    assert "--init" in rec and "fallo_tecnico" not in q.state


STYLE_ARMS = "control=--style-coef 0;estilo=--style-coef 0.002;estilo3=--style-coef 0.006"


def _style_queue(tmp_path, **kw):
    return _queue(tmp_path, ab_arms=STYLE_ARMS, ab_default="control", ab_margin=0.03, ab_min_score=0.6,
                  ab_guards=True, ab_max_useless=0.15, **kw)


def _arm(path, indices, score=0.85, log=None):
    rows = [dict(update=0, baseline=True, vs_bc=dict(score=0.5), drift=[], cadena_pase=dict(indice=0.6))]
    rows += [dict(update=u, vs_bc=dict(score=score), drift=[], cadena_pase=dict(indice=v))
             for u, v in zip((500, 550, 600), indices)]
    _evals(path, rows)
    if log is not None:
        (path / "log.jsonl").write_text("\n".join(json.dumps(r) for r in log), encoding="utf-8")


def test_style_arms_need_a_margin_over_the_control(tmp_path):
    q = _style_queue(tmp_path / "a")
    _arm(tmp_path / "a" / "rl_control", (0.70, 0.70, 0.70))
    _arm(tmp_path / "a" / "rl_estilo", (0.72, 0.72, 0.72))
    _arm(tmp_path / "a" / "rl_estilo3", (0.75, 0.75, 0.75))
    assert q.choose_arm() == "estilo3"                   # 0,75 > 0,70 + 0,03
    q = _style_queue(tmp_path / "b")
    _arm(tmp_path / "b" / "rl_control", (0.70, 0.70, 0.70))
    _arm(tmp_path / "b" / "rl_estilo", (0.71, 0.71, 0.71))
    _arm(tmp_path / "b" / "rl_estilo3", (0.72, 0.73, 0.72))
    assert q.choose_arm() == "control"                   # el estilo no le saca 0,03 al control
    q = _style_queue(tmp_path / "c")
    _arm(tmp_path / "c" / "rl_control", (0.70, 0.70, 0.70))
    _arm(tmp_path / "c" / "rl_estilo", (0.80, 0.80, 0.80), score=0.55)
    _arm(tmp_path / "c" / "rl_estilo3", (0.71, 0.71, 0.71))
    assert q.choose_arm() == "control"                   # el mejor índice perdió fuerza contra la BC (< 0,6)


def test_guards_drop_arms_that_fool_the_discriminator(tmp_path):
    base = [dict(update=u, still_frac=0.12, dist_ball_1=150.0) for u in range(1, 51)]
    q = _style_queue(tmp_path)
    _arm(tmp_path / "rl_control", (0.70, 0.70, 0.70), log=base)
    collapsed = [dict(r, style_d_loss=1.97) for r in base]
    _arm(tmp_path / "rl_estilo", (0.80, 0.80, 0.80), log=collapsed)
    still = [dict(r, still_frac=0.25, style_d_loss=1.2) for r in base]
    _arm(tmp_path / "rl_estilo3", (0.78, 0.78, 0.78), log=still)
    assert q.choose_arm() == "control"
    rows = q.state["ab_temprano"]
    assert "discriminador colapsado" in rows["estilo"]["revisar"][0]
    assert "quietos" in rows["estilo3"]["revisar"][0]


def test_queue_runs_the_arms_with_their_own_args(tmp_path):
    q = _style_queue(tmp_path, rl_extra="--init runs/x/best.pt --lambda-dist 0.05 --lambda-decay 1")
    jobs, cmds = [], {}
    q.run_parallel = lambda js: jobs.extend(js)

    def fake_run(name, cmd, retries=5, ok_codes=(0,)):
        cmds[name] = [str(x) for x in cmd]
        if name.startswith("cert_"):
            cmd[cmd.index("--out") + 1].write_text(json.dumps(dict(aprobado=False)))
            return Q.CERT_NOT_APPROVED
        return 0
    q.run = fake_run
    _arm(tmp_path / "rl_control", (0.70, 0.70, 0.70))
    _arm(tmp_path / "rl_estilo", (0.76, 0.76, 0.76))
    _arm(tmp_path / "rl_estilo3", (0.71, 0.71, 0.71))
    _log(tmp_path / "rl_estilo", 3000)
    q.main()
    by = {n: [str(x) for x in c] for n, c in jobs}
    assert set(by) == {"rl_control", "rl_estilo", "rl_estilo3"}
    est = by["rl_estilo"]
    assert est[est.index("--style-coef") + 1] == "0.002" and est[est.index("--envs") + 1] == "341"
    assert est[est.index("--lambda-dist") + 1] == "0.05" and "--init" in est
    main = cmds["rl_estilo_full"]
    assert main[main.index("--style-coef") + 1] == "0.002" and main[main.index("--envs") + 1] == "1024"
    pre = cmds["preflight"]
    assert pre[pre.index("--style-bank") + 1].endswith("x4_style_bank.npz") and pre[pre.index("--init") + 1] == "runs/x/best.pt"
