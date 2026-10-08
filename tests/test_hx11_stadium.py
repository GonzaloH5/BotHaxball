from sim.stadium import BLUE, RED, load_stadium


def test_haxeleven_x11_keeps_script_barriers_and_drops_bar():
    st = load_stadium("haxeleven_x11")
    assert (st.field_half_w, st.field_half_h) == (1800.0, 1200.0)
    assert (st.goal_x, st.goal_half_height) == (1810.0, 170.0)
    # las barreras del script (discos 1 y 2 del mapa) arrancan fuera de la cancha pero se conservan
    assert list(st.d_src[:2]) == [0, 1]
    assert st.d_radius[0] == 50.0 and st.d_mask[0] == RED
    assert st.d_radius[1] == 17.5 and st.d_mask[1] == RED | BLUE
    # el disco móvil de la barra de fuerza (28 del mapa, 27 sin la pelota) queda afuera
    assert 27 not in set(st.d_src.tolist())
    assert st.ball["invMass"] == 1.65 and st.ball["radius"] == 8
    assert st.player["invMass"] == 0.69 and st.player["acceleration"] == 0.119
    assert len(st.red_spawn) == len(st.blue_spawn) == 12
