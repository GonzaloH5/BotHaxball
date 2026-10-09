"""D0–D3 de HaxEleven: lugares fijos, eventos en un partido sintético, Hungarian, formaciones y control del espacio."""
import itertools

import numpy as np

from tools import hx11_d0, hx11_d1, hx11_d2, hx11_d3


def test_slots_keep_place_until_leaving():
    # columnas 0 y 1 rojas, 2 azul; la 0 sale en el tick 2 y la 3 entra al rojo: toma el lugar libre más bajo
    team = np.array([[1, 1, 2, 0], [1, 1, 2, 0], [0, 1, 2, 1], [0, 1, 2, 1]], np.int8)
    s = hx11_d0.slots(team)
    assert s[0, 0] == 0 and s[0, 1] == 1 and s[0, 11] == 2
    assert s[2, 0] == 3 and s[2, 1] == 1
    assert (s[:, 2:11] == -1).all()


def test_q16_roundtrip():
    x = np.array([[1799.94, -1200.04], [np.nan, 3.3]])
    q = hx11_d0.q16(x, 10)
    assert q.dtype == np.int16 and q[1, 0] == 0
    assert np.abs(q[0] / 10 - x[0]).max() <= 0.05 + 1e-9


def _synthetic():
    """Rojo 1 le pasa a rojo 2 (patada en t=10), rojo 2 patea al arco (t=60) y es gol en t=156."""
    T = 200
    pos = np.zeros((T, 22, 2))
    pid = np.full((T, 22), -1, np.int32)
    for s, xy in {0: (-1700, 0), 1: (0, 0), 2: (400, 0), 11: (1700, 400), 12: (800, 300)}.items():
        pos[:, s] = xy
        pid[:, s] = 100 + s
    ball = np.zeros((T, 2))
    bv = np.zeros((T, 2))
    ball[:11, 0] = 23.0
    for t in range(11, 46):
        ball[t, 0] = min(377.0, 23.0 + 10.0 * (t - 10))
        bv[t, 0] = 10.0 if ball[t, 0] < 377 else 0.0
    ball[46:61, 0] = 377.0
    for t in range(61, T):
        ball[t, 0] = 377.0 + 15.0 * (t - 60)
        bv[t, 0] = 15.0
    kick = np.zeros((T, 22), np.int8)
    kick[10, 1] = kick[60, 2] = 1
    phase = np.zeros(T, np.int8)
    phase[156:] = hx11_d0.GOAL
    score = np.zeros((T, 2), np.int16)
    score[156:, 0] = 1
    o = dict(frame=np.arange(T, dtype=np.int32), ball_pos=hx11_d0.q16(ball, 10), ball_vel=hx11_d0.q16(bv, 1000),
             pos=hx11_d0.q16(pos, 10), vel=np.zeros((T, 22, 2), np.int16), slot_pid=pid,
             gk_pid=np.tile(np.array([100, 111], np.int32), (T, 1)), phase=phase, piece_kind=np.zeros(T, np.int8),
             piece_team=np.full(T, -1, np.int8), kick=kick, score=score, ball_inv=np.full(T, 1.65, np.float16),
             ball_g=np.zeros((T, 2), np.float16), spot=np.zeros((T, 2), np.int16))
    return o, dict(decisions=[], numbers=[])


def test_d1_pass_reception_shot_goal():
    o, meta = _synthetic()
    ev, owner, seg, ctl = hx11_d1.detect(o, meta)
    by = {}
    for e in ev:
        by.setdefault(e["tipo"], []).append(e)
    p = by["pase"][0]
    assert p["lugar"] == 1 and p["resultado"] == "completo" and p["a"] == 2 and p["patada"] and not p["largo"]
    r = by["recepcion"][0]
    assert r["lugar"] == 2 and r["de"] == 1
    s = by["remate"][0]
    assert s["lugar"] == 2 and s["resultado"] == "gol" and s["al_arco"]
    g = by["gol"][0]
    assert g["lugar"] == 2 and g["equipo"] == 0 and g["asistencia"] == 1 and not g["propio"]
    assert ctl["goles_marcador"] == ctl["goles_detectados"] == 1
    assert owner[30] == 1 and owner[50] == 2 and owner[170] == -1


def test_hungarian_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(20):
        a = rng.random((6, 6))
        ans = hx11_d2.hungarian(a)
        best = min(itertools.permutations(range(6)), key=lambda p: sum(a[i, p[i]] for i in range(6)))
        assert np.isclose(sum(a[i, ans[i]] for i in range(6)), sum(a[i, best[i]] for i in range(6)))
        assert sorted(ans) == list(range(6))


def test_classify_canonical_formations():
    rng = np.random.default_rng(1)
    for name, c in hx11_d2.CANON.items():
        means = c * np.array([350.0, 280.0]) + rng.normal(0, 8, c.shape)
        f, cost, asg = hx11_d2.classify(means[rng.permutation(10)])
        assert f == name, (name, f, cost)
    names = hx11_d2.role_names("4-3-3", np.arange(10))
    assert names[:4] == ["DEF-I", "DEF-CI", "DEF-CD", "DEF-D"] and names[-3:] == ["DEL-I", "DEL-C", "DEL-D"]


def test_control_prefers_the_closer_team():
    team = np.r_[np.zeros(11), np.ones(11)].astype(np.int64)
    pos = np.full((22, 2), 5000.0)
    on = np.zeros(22, np.bool_)
    pos[1], pos[12] = (-600.0, 0.0), (900.0, 0.0)
    on[1] = on[12] = True
    vel = np.zeros((22, 2))
    ball = np.array([0.0, 0.0])
    near_red = hx11_d3.control(pos, vel, team, on, ball, np.array([-580.0, 0.0]), 0, 15.0, 10.0)
    near_blue = hx11_d3.control(pos, vel, team, on, ball, np.array([880.0, 0.0]), 0, 15.0, 10.0)
    assert near_red > 0.9 and near_blue < 0.1


def test_vaep_flips_the_previous_team():
    rows = [dict(t=0, team=0), dict(t=10, team=1)]
    v = hx11_d3.vaep_values(rows, np.array([0.1, 0.3]), np.array([0.05, 0.02]), [])
    # la acción 2 es del azul: su "antes" es la visión del azul de la acción 1 (marcar = 0,05, recibir = 0,1)
    assert np.isclose(v[1], (0.3 - 0.05) - (0.02 - 0.1))


def test_infer_gk_takes_the_player_nearest_its_goal():
    T = 7200
    pos = np.zeros((T, 22, 2), np.int16)
    pid = np.full((T, 22), -1, np.int32)
    pid[:, 0], pid[:, 1], pid[:, 11] = 10, 11, 20
    pos[:, 0] = (-17000, 0)      # rojo 10 en su arco (×10, como D0)
    pos[:, 1] = (-5000, 0)
    pos[:100, 1] = (-17900, 0)   # rojo 11 pasa un momento más cerca: el bloque de 1 min no cambia
    pos[:, 11] = (17000, 0)
    g = hx11_d0.infer_gk(pos, pid)
    assert (g[:, 0] == 10).all() and (g[:, 1] == 20).all()


def test_viterbi_follows_the_emissions_and_stays():
    from tools import hx11_marcas
    logE = np.full((30, 3), -50.0)
    logE[:10, 0] = logE[10:, 2] = 0.0
    logE[15, 1] = 0.0            # un solo cuadro a favor del estado 1: la permanencia lo absorbe
    logE[15, 2] = -3.0
    path = hx11_marcas.viterbi(logE, 0.97)
    assert (path[:10] == 0).all() and (path[10:] == 2).all()


def test_follow_tells_man_marking_from_zonal():
    from tools import hx11_marcas
    F = 600
    fr = np.arange(F) * 6
    t = fr.astype(np.float64)
    rng = np.random.default_rng(2)
    q = np.zeros((F, 22, 2))
    q[:, :22] = rng.uniform(-800, 800, (22, 2))
    q[:, 11] = np.c_[300 * np.sin(2 * np.pi * t / 400), 200 * np.cos(2 * np.pi * t / 300)]
    q[:, 12] = np.c_[250 * np.cos(2 * np.pi * t / 350), 300 * np.sin(2 * np.pi * t / 500)]
    q[:, 0] = q[:, 11] + (-50.0, 0.0)          # el defensor 0 acompaña a su referencia: personal
    refs = np.full((F, 11), -1)
    refs[:, 0], refs[:, 1] = 0, 1             # el defensor 1 se queda en su lugar aunque la suya se vaya: zonal
    ok = np.ones((F, 11), bool)
    ok[:, 10] = False
    pids = np.tile(np.arange(22), (F, 1))
    res = hx11_marcas.follow(fr, q, np.zeros((F, 2)), np.arange(11), np.arange(11, 22), refs, ok, ok, pids)
    b0 = hx11_marcas.beta(res[0]["xtx"], res[0]["xty"])[0]
    b1 = hx11_marcas.beta(res[1]["xtx"], res[1]["xty"])[0]
    assert res[0]["n"] > 50 and b0 > 0.9 and abs(b1) < 0.2, (b0, b1)
    assert res[2]["n"] == 0


def test_l0_mirror_matches_a_y_flipped_match():
    import dataclasses
    from pathlib import Path
    import pytest
    from learn import hx11_data as H
    from learn.hx11_bc import mirror
    name = "AV_KJ_HaxEleven-1-10-2026-23h32m"
    if not (H.D0 / f"{name}.npz").exists() or not (H.D2 / f"{name}.json").exists():
        pytest.skip("sin datos locales de HaxEleven")
    m = H.load_match(name)
    fy = lambda a, k: a * np.array([1, -1], a.dtype) if k else a
    inp = m.inp.astype(np.int64)
    inp_f = np.where(inp >= 0, (inp & ~3) | ((inp & 1) << 1) | ((inp & 2) >> 1), inp).astype(np.int8)
    mf = dataclasses.replace(m, pos=fy(m.pos, 1), vel=fy(m.vel, 1), ball=fy(m.ball, 1), bvel=fy(m.bvel, 1),
                             bg=fy(m.bg, 1), spot=fy(m.spot, 1), inp=inp_f,
                             anchor={k: v * np.array([1, -1, 1, -1], np.float32) for k, v in m.anchor.items()})
    t, s = H.valid_samples(m, 7)
    r = np.random.default_rng(0).integers(0, len(t), 512)
    d = np.random.default_rng(1).integers(H.D_MIN, H.D_MAX + 1, 512)
    own, ents, y, ch = H.build(m, t[r], s[r], d)
    own_f, ents_f, y_f, ch_f = H.build(mf, t[r], s[r], d)
    own_m, ents_m, y_m = mirror(own, ents, y)
    assert np.allclose(own_m, own_f, atol=1e-5) and np.allclose(ents_m, ents_f, atol=1e-5)
    assert (y_m == y_f).all() and (ch == ch_f).all()
