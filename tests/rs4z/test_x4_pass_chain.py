"""Cadena de pase (tools/x4_pass_chain.py) y valor de posesión (learn/x4_epv.py) con tramos sintéticos."""
from pathlib import Path

import numpy as np
import pytest

from tools import x4_metrics as XM
from tools import x4_pass_chain as PC

ROOT = Path(__file__).resolve().parents[2]
BASE = np.array([[-300.0, 0], [-600, 200], [-600, -200], [-900, 0], [300, 300], [600, 200], [600, -200], [900, 0]])


def _episode(T):
    pos = np.repeat(BASE[None], T, 0).copy()
    return XM.Episode(stride=3, ball=np.zeros((T, 4)), ball_r=8.325, pos=pos, vel=np.zeros((T, 8, 2)),
                      move=np.zeros((T, 8), np.int64), kick_key=np.zeros((T, 8), bool), kicked=np.zeros((T, 8), bool),
                      open_play=np.ones(T, bool), restart_kind=np.zeros(T, np.int64), restart_age=np.zeros(T, np.int64),
                      kickoff=np.zeros(T, bool), segment=np.zeros(T, np.int64), restart_team=np.full(T, -1),
                      goal_ev=np.zeros(T, np.int8))


def _move_ball(ep, t0, t1, a, b):
    """La pelota viaja en línea recta de `a` a `b` entre los muestreos t0 y t1 (inclusive)."""
    for t in range(t0, t1 + 1):
        u = (t - t0) / max(1, t1 - t0)
        ep.ball[t, :2] = (1 - u) * np.asarray(a) + u * np.asarray(b)
        ep.ball[t, 2:4] = (np.asarray(b) - np.asarray(a)) / max(1, t1 - t0) / 3.0


def test_lane_margin_cone():
    ball = np.array([[0.0, 0.0]])
    mate = np.array([[400.0, 0.0]])
    # rival sobre la línea a 200 px: la corta
    assert PC.lane_margin(ball, mate, np.array([[[200.0, 5.0]]]), 23.3)[0] < 0
    # rival a 100 px de la línea, cerca de la pelota (s ≈ 50): no llega
    assert PC.lane_margin(ball, mate, np.array([[[50.0, 100.0]]]), 23.3)[0] > 0
    # el mismo rival cerca del receptor (s ≈ 380): KAPPA·s = 114 > 100 − 23 → llega a cortarla
    assert PC.lane_margin(ball, mate, np.array([[[380.0, 100.0]]]), 23.3)[0] < 0


def test_completed_progressive_pass_and_retention():
    ep = _episode(200)
    # el lugar 0 conduce en (-300, 0) y le pasa al lugar 1, adelantado a (150, 100): la pelota queda ≥ 25% más
    # cerca del arco rival (1430 → 1004 px)
    ep.pos[:, 1] = [150.0, 100.0]
    ep.pos[:, 4] = [300.0, 500.0]
    ep.ball[:20, :2] = ep.pos[:20, 0] + [20.0, 0]
    ep.kicked[19, 0] = True
    _move_ball(ep, 19, 40, ep.ball[19, :2], [150.0 - 20.0, 100.0])
    ep.ball[40:, :2] = [150.0 - 20.0, 100.0]
    E = PC.chain_events(ep)
    assert len(E["passes"]) == 1
    p = E["passes"][0]
    assert p["a"] == 0 and p["b"] == 1
    u = PC.unit_counts([ep])
    assert u["passes"] == 1 and u["failed"] == 0 and u["prog"] == 1 and u["rec_ret"] == 1
    r = PC.rates([u])
    assert r["precision_pase"] == 1.0 and r["retencion_tras_recibir"] == 1.0


def test_intercepted_aimed_pass_is_a_failed_attempt():
    ep = _episode(120)
    ep.pos[:, 1] = [100.0, 0.0]           # compañero adelante sobre el eje
    ep.pos[:, 4] = [-100.0, 5.0]          # rival en la línea
    ep.ball[:20, :2] = ep.pos[:20, 0] + [20.0, 0]
    ep.kicked[19, 0] = True
    _move_ball(ep, 19, 30, ep.ball[19, :2], [-100.0 - 22.0, 5.0])
    ep.ball[30:, :2] = [-100.0 - 22.0, 5.0]
    u = PC.unit_counts([ep])
    assert u["passes"] == 0 and u["failed"] == 1


def test_wall_pass_and_rates_are_finite():
    ep = _episode(300)
    ep.pos[:, 1] = [-300.0, 250.0]
    ep.ball[:20, :2] = ep.pos[0, 0] + [20, 0]
    ep.kicked[19, 0] = True
    _move_ball(ep, 19, 35, ep.ball[19, :2], [-300.0, 250.0 - 22])
    ep.kicked[37, 1] = True                # de primera, de vuelta al lugar 0, que avanzó
    ep.pos[40:, 0] = [-150.0, 0.0]
    _move_ball(ep, 37, 55, [-300.0, 250.0 - 22], [-150.0 - 22.0, 0.0])
    ep.ball[55:, :2] = [-150.0 - 22.0, 0.0]
    u = PC.unit_counts([ep])
    assert u["passes"] == 2 and u["wall"] == 1


def test_gate_directions():
    names = [m for m, _, _ in PC.METRICS]
    human = {m: dict(valor=1.0, ic90=[0.9, 1.1]) for m in names}
    band = {m: [0.5, 1.5] for m in names}
    agent = {m: dict(valor=1.0, ic90=[0.9, 1.1]) for m in names}
    g = PC.gate(agent, human, band)
    assert g["aprobado"] and abs(g["indice_cadena"] - 1.0) < 1e-9
    # significativamente peor en una "mas", peor en una "menos" y fuera de banda en una de estilo
    agent["pases_por_min"] = dict(valor=0.5, ic90=[0.4, 0.6])
    agent["circulacion_inutil_frac"] = dict(valor=2.0, ic90=[1.8, 2.2])
    agent["largo_pase_p50"] = dict(valor=3.0, ic90=[2.0, 4.0])
    g = PC.gate(agent, human, band)
    bad = {k for k, v in g["metricas"].items() if not v["ok"]}
    assert bad == {"pases_por_min", "circulacion_inutil_frac", "largo_pase_p50"} and not g["aprobado"]
    assert g["indice_cadena"] < 1.0
    # un poco por debajo (dentro de la tolerancia humana) no reprueba la métrica ni el gate
    agent = {m: dict(valor=1.0, ic90=[0.9, 1.1]) for m in names}
    agent["precision_pase"] = dict(valor=0.9, ic90=[0.85, 0.95])
    g = PC.gate(agent, human, band, strict=True)
    assert g["metricas"]["precision_pase"]["ok"] and not g["metricas"]["precision_pase"]["al_menos_humano"]
    assert g["metricas"]["precision_pase"]["significativamente_peor"] and g["aprobado"]
    # todo un 10% por debajo: el índice global queda en 0,9 < 0,95 y reprueba la certificación
    agent = {m: dict(valor=0.9, ic90=[0.85, 0.95]) for m in names}
    assert not PC.gate(agent, human, band, strict=True)["aprobado"]


def test_epv_is_antisymmetric():
    from learn import x4_epv as E
    path = ROOT / "runs" / "x4_epv" / "epv.pt"
    if not path.exists():
        pytest.skip("falta runs/x4_epv/epv.pt")
    epv = E.EPV(path)
    rng = np.random.default_rng(0)
    n = 64
    ball = np.concatenate([rng.uniform(-1100, 1100, (n, 1)), rng.uniform(-600, 600, (n, 1)), rng.normal(0, 3, (n, 2))], 1)
    pos = np.stack([rng.uniform(-1100, 1100, (n, 8)), rng.uniform(-600, 600, (n, 8))], -1)
    vel = rng.normal(0, 1, (n, 8, 2))
    rk = rng.integers(0, 4, n)
    rt = np.where(rk > 0, rng.integers(0, 2, n), -1)
    ko = np.zeros(n, bool)
    phi = epv.phi(ball, pos, vel, rk, rt, ko)
    mb, mp, mv, mt = E.mirror_state(ball, pos, vel, rt)
    assert np.allclose(epv.phi(mb, mp, mv, rk, mt, ko), -phi, atol=1e-6)
    # pelota frente al arco azul con un rojo encima vale más para el rojo que en su propio arco
    near = ball.copy()[:2]
    near[0, :4] = [1050, 0, 0, 0]
    near[1, :4] = [-1050, 0, 0, 0]
    p2 = pos[:2].copy()
    p2[0, 0] = [1020, 0]
    p2[1, 4] = [-1020, 0]
    v = epv.phi(near, p2, np.zeros((2, 8, 2)), np.zeros(2, int), np.full(2, -1), np.zeros(2, bool))
    assert v[0] > 0.1 and v[1] < -0.1


def test_epv_labels_respect_censoring():
    from learn.x4_epv import episode_labels
    ep = _episode(400)
    ep.goal_ev[399] = -1
    yr, yb, ok = episode_labels(ep, horizon_s=10.0)
    assert ok.all() and yb[399 - 200:].all() and not yb[:399 - 200].any() and not yr.any()
    ep2 = _episode(400)                           # sin gol: los últimos 10 s quedan censurados
    yr, yb, ok = episode_labels(ep2, horizon_s=10.0)
    assert ok[:200].all() and not ok[200:].any()


def test_team_filter_splits_events():
    ep = _episode(200)
    ep.pos[:, 1] = [150.0, 100.0]
    ep.pos[:, 4] = [300.0, 500.0]
    ep.ball[:20, :2] = ep.pos[:20, 0] + [20.0, 0]
    ep.kicked[19, 0] = True
    _move_ball(ep, 19, 40, ep.ball[19, :2], [130.0, 100.0])
    ep.ball[40:, :2] = [130.0, 100.0]
    both = PC.unit_counts([ep])
    red = PC.unit_counts([ep], team=0)
    blue = PC.unit_counts([ep], team=1)
    assert red["passes"] == both["passes"] == 1 and blue["passes"] == 0
    assert abs(red["minutes"] - both["minutes"] / 2) < 1e-9
