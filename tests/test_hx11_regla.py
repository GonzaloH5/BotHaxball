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


def test_juego_abierto_atras_y_banda():
    from learn.hx11_policy import open_kick_ok
    me, ball = np.array([-480.0, 0.0]), np.array([-500.0, 0.0])     # rojo en campo propio, patada hacia atrás
    assert not open_kick_ok(me, ball, 0, np.zeros((0, 2)))
    assert open_kick_ok(me, ball, 0, np.array([[-800.0, 10.0]]), np.array([[0.0, 500.0]]))       # compañero libre
    assert not open_kick_ok(me, ball, 0, np.array([[-800.0, 10.0]]), np.array([[-820.0, 40.0]]))  # marcado
    assert not open_kick_ok(me, ball, 0, np.array([[-800.0, 10.0]]), np.array([[-650.0, 20.0]]))  # rival en la línea
    assert open_kick_ok(np.array([-520.0, 0.0]), ball, 0, np.zeros((0, 2)))                       # hacia adelante
    # pegado a la banda, hacia afuera
    assert not open_kick_ok(np.array([300.0, 1080.0]), np.array([300.0, 1100.0]), 0, np.zeros((0, 2)))
    # en campo rival, hacia atrás: libre
    assert open_kick_ok(np.array([520.0, 0.0]), np.array([500.0, 0.0]), 0, np.zeros((0, 2)))


def test_distancia_en_saques_ajenos():
    from learn.hx11_policy import KICKOFF_KIND, keep_away
    own = np.zeros((1, 69), np.float32)
    own[0, 31 + 2] = 1          # pelota parada
    own[0, 37 + 2] = 1          # córner
    own[0, 45 + 0] = 1          # de mi equipo
    own[0, 4:6] = [0.03, 0.0]   # pelota 30 px adelante (coordenadas del equipo)
    own[0, 10] = 0.03
    a = np.array([3 + 9])       # hacia la pelota, pateando
    assert keep_away(own, a, (2, 3, 4))[0] == 7            # se aleja (dx −1), sin patear
    assert keep_away(own, a, ())[0] == 3 + 9               # si el córner lo saca el bot, no se toca
    own[0, 4:6] = [0.12, 0.0]
    own[0, 10] = 0.12
    assert keep_away(own, np.array([3]), (2, 3, 4))[0] == 0        # entre 90 y 140 px: no va hacia ella
    assert keep_away(own, np.array([7]), (2, 3, 4))[0] == 7        # alejarse sí
    own[0, 31:37] = 0
    own[0, 31 + 5] = 1          # saque inicial
    assert keep_away(own, np.array([3 + 9]), (KICKOFF_KIND,))[0] == 0
    assert keep_away(own, np.array([3 + 9]), ())[0] == 3 + 9
    own[0, 31:37] = 0
    own[0, 31 + 0] = 1          # juego abierto: nada
    assert keep_away(own, np.array([3 + 9]), (2, 3, 4, KICKOFF_KIND))[0] == 3 + 9


def test_destrabar():
    from learn.hx11_policy import FREEZE_DECISIONS, unfreeze
    own = np.zeros((1, 69), np.float32)
    own[0, 31] = 1                      # juego abierto
    own[0, 10] = 0.9                    # pelota a 900 px
    own[0, 64:66] = [-0.6, 0.0]         # centroide 600 px atrás: la zona está atrás
    a = np.array([0])
    act, idle = unfreeze(own, a, np.array([FREEZE_DECISIONS - 1]))
    assert act[0] == 7                  # va hacia su zona (−x)
    act, idle = unfreeze(own, a, np.array([0]))
    assert act[0] == 0 and idle[0] == 1 # recién se quedó quieto: no se toca
    own[0, 2:4] = [0.5, 0.0]            # se está moviendo: no se toca
    act, idle = unfreeze(own, a, np.array([FREEZE_DECISIONS]))
    assert act[0] == 0 and idle[0] == 0


def test_destrabar_no_saca_al_arquero():
    from learn.hx11_policy import FREEZE_DECISIONS, unfreeze
    own = np.zeros((1, 69), np.float32)
    own[0, 31] = 1
    own[0, 10] = 0.9
    own[0, 64:66] = [0.8, 0.0]
    own[0, 68] = 1                      # arquero
    act, _ = unfreeze(own, np.array([0]), np.array([FREEZE_DECISIONS]))
    assert act[0] == 0


def test_correa():
    from learn.hx11_policy import leash
    own = np.zeros((1, 69), np.float32)
    own[0, 31] = 1                      # juego abierto
    own[0, 59 + 3] = 1                  # línea MED (p95 ~425 px)
    own[0, 10] = 0.8                    # pelota a 800 px
    own[0, 64:66] = [0.0, -0.6]         # zona 600 px hacia −y
    ents = np.zeros((1, 21, 9), np.float32)
    ents[0, 0, 0] = 1
    ents[0, 0, 5:7] = [0.1, 0.0]        # un compañero a 100 px de la pelota: no soy el más cercano
    assert leash(own, ents, np.array([5]))[0] == 1          # vuelve hacia −y
    own[0, 64:66] = [0.0, -0.3]         # dentro del p95: no se toca
    assert leash(own, ents, np.array([5]))[0] == 5
    own[0, 64:66] = [0.0, -0.6]
    own[0, 10] = 0.2                    # en la jugada (pelota a 200 px): no se toca
    assert leash(own, ents, np.array([5]))[0] == 5
