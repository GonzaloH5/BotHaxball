import json

import numpy as np

from tools.hx11_npz import convert, load
from tools.hx11_script import ball_mode, piece_kind


def test_piece_kind_by_spot():
    assert piece_kind(117.5, 1230) == "lateral"
    assert piece_kind(-1765, -1165) == "corner"
    assert piece_kind(1572, 0) == "saque_arco"
    assert piece_kind(-1275, 0) == "penal"
    assert piece_kind(-299.0, -552.8) == "tiro_libre"


def test_ball_mode_colors():
    assert ball_mode(0xFFFFFF) == "blanca"
    assert ball_mode(0x00050A) == "azul"
    assert ball_mode(float("nan")) == "?"


def test_npz_forward_fills_static_discs_and_props(tmp_path):
    ticks = [
        dict(type="header", stadium="x"),
        dict(type="tick", frame=10, state=1, score=[0, 0], time=1.0, nd=4,
             d={"0": [0, 0, 1, 0], "3": [5, 5, 0, 0]}, m={"1": [100, 9999, 0, 0]},
             p={"0": {"invMass": 1.65, "color": 16777215}, "1": {"radius": 50}, "3": {"invMass": 0.69}},
             players=[[7, 1, 16, 1, 3]]),
        # el disco 1 no se movió y la pelota no cambió de masa: no se repiten
        dict(type="tick", frame=11, state=1, score=[0, 0], time=1.02, nd=4,
             d={"0": [1, 0, 1, 0], "3": [6, 5, 1, 0]}, m={}, p={"0": {"invMass": 2.43}}, players=[[7, 1, 0, 0, 3]]),
        dict(type="event", name="kick", frame=11, playerId=7),
    ]
    src = tmp_path / "t.jsonl"
    src.write_text("\n".join(json.dumps(t) for t in ticks), encoding="utf-8")
    T, P = convert(open(src, encoding="utf-8"), tmp_path / "out")
    assert (T, P) == (2, 1)
    z, ev, header = load(tmp_path / "out")
    assert z["pid"].tolist() == [7]
    assert z["disc_pos"][1, list(z["disc_ids"]).index(1)].tolist() == [100, 9999]
    assert np.allclose(z["ball_invMass"], [1.65, 2.43])
    assert z["disc_radius"][1, list(z["disc_ids"]).index(1)] == 50
    assert z["pl_input"][:, 0].tolist() == [16, 0] and z["pl_invMass"][1, 0] == np.float32(0.69)
    assert ev[0]["name"] == "kick" and header["stadium"] == "x"
