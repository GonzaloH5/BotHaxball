"""Regla de patada en las pelotas paradas propias (learn.hx11_policy.kick_rule_ok)."""
import numpy as np

from learn.hx11_policy import BAR_MIN, BAR_WAIT, kick_rule_ok, seen_now


def test_lateral_paralelo_bloqueado():
    # pelota en el lateral de abajo (y = +1230), el que saca al costado sobre la línea: patada paralela, no entra
    assert not kick_rule_ok(np.array([-23.0, 1230.0]), np.array([0.0, 1230.0]), 1, 0, 1.8, 50)


def test_lateral_desde_adentro_bloqueado():
    assert not kick_rule_ok(np.array([0.0, 1207.0]), np.array([0.0, 1230.0]), 1, 0, 1.8, 50)
    assert not kick_rule_ok(np.array([0.0, -1207.0]), np.array([0.0, -1230.0]), 1, 1, 1.8, 50)


def test_lateral_desde_afuera_permitido():
    assert kick_rule_ok(np.array([-20.0, 1242.0]), np.array([0.0, 1230.0]), 1, 0, 1.8, 50)
    assert kick_rule_ok(np.array([10.0, -1253.0]), np.array([0.0, -1230.0]), 1, 1, 1.8, 50)


def test_lateral_apagado():
    assert kick_rule_ok(np.array([-23.0, 1230.0]), np.array([0.0, 1230.0]), 1, 0, 1.8, 50, lateral=False)


def test_espera_larga_siempre_permitido():
    assert kick_rule_ok(np.array([-23.0, 1230.0]), np.array([0.0, 1230.0]), 1, 0, 1.8, BAR_WAIT)


def test_corner_y_tiro_libre():
    # córner del lado derecho: patea desde afuera de la línea de fondo, con barra alta
    assert kick_rule_ok(np.array([1820.0, 1190.0]), np.array([1800.0, 1200.0]), 2, 0, BAR_MIN + 0.1, 50)
    assert not kick_rule_ok(np.array([1780.0, 1190.0]), np.array([1800.0, 1200.0]), 2, 0, BAR_MIN + 0.1, 50)
    assert not kick_rule_ok(np.array([1820.0, 1190.0]), np.array([1800.0, 1200.0]), 2, 0, BAR_MIN - 0.1, 50)
    # tiro libre del azul (ataca hacia −x): desde su lado (x mayor)
    assert kick_rule_ok(np.array([520.0, 0.0]), np.array([500.0, 0.0]), 4, 1, BAR_MIN + 0.1, 50)
    assert not kick_rule_ok(np.array([480.0, 0.0]), np.array([500.0, 0.0]), 4, 1, BAR_MIN + 0.1, 50)


def test_otros_tipos_libres():
    assert kick_rule_ok(np.array([0.0, 0.0]), np.array([10.0, 0.0]), 5, 0, 1.0, 0)
    assert kick_rule_ok(np.array([0.0, 0.0]), np.array([10.0, 0.0]), 0, 0, 1.0, 0)


def test_seen_now_lleva_la_posicion_al_momento_de_la_decision():
    class W:
        pos = np.zeros((1, 22, 2), np.int16)
        vel = np.zeros((1, 22, 2), np.int16)
        ball = np.array([[100, 12300]], np.int16)
        bvel = np.zeros((1, 2), np.int16)
    W.pos[0, 3] = [-500, 12000]       # (−50, 1200) px
    W.vel[0, 3] = [0, 2000]           # 2 px/tick hacia afuera
    me, ball = seen_now(W, 0, 3, 14)
    assert np.allclose(me, [-50.0, 1230.0]) and np.allclose(ball, [10.0, 1230.0])
