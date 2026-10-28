import gzip
import json

from tools.x3futsal_physics import _read_replay, transition_is_valid


def make_tick(frame, players=None, state=1, score=None):
    if players is None:
        players = [{"id": pid, "team": team} for team in (1, 2) for pid in range(3)]
    return {"frame": frame, "state": state, "score": score or [0, 0], "players": players,
            "discs": [{"x": 0, "y": 0, "vx": 0, "vy": 0}]}


def test_physics_transition_requires_same_target_span_and_uninterrupted_game():
    span = {"start": 10, "end": 20}
    assert transition_is_valid(make_tick(10), make_tick(11), span, set())
    assert not transition_is_valid(make_tick(9), make_tick(10), span, set())
    assert not transition_is_valid(make_tick(10), make_tick(12), span, set())
    assert not transition_is_valid(make_tick(10), make_tick(11, score=[1, 0]), span, set())
    assert not transition_is_valid(make_tick(10, state=0), make_tick(11), span, set())
    assert transition_is_valid(make_tick(10, state=0), make_tick(11, state=0), span, set(),
                               allowed_states=(0, 1))
    assert not transition_is_valid(make_tick(10, state=0), make_tick(11, state=1), span, set())
    assert not transition_is_valid(make_tick(10), make_tick(11), span, {11})


def test_physics_transition_requires_stable_three_v_three_roster():
    span = {"start": 0, "end": 10}
    previous = make_tick(0)
    joined = make_tick(1, players=[*previous["players"], {"id": 6, "team": 1}])
    assert not transition_is_valid(previous, joined, span, set())


def test_physics_transition_rejects_coincident_player_discs():
    players = [{"id": pid, "team": team, "disc": pid} for pid, team in enumerate((1, 1, 1, 2, 2, 2))]
    previous = make_tick(0, players=players)
    previous["discs"] = [{"x": pid * 30, "y": 0, "vx": 0, "vy": 0} for pid in range(6)]
    current = make_tick(1, players=players)
    current["discs"] = [dict(disc) for disc in previous["discs"]]
    current["discs"][1]["x"] = current["discs"][0]["x"]
    assert not transition_is_valid(previous, current, {"start": 0, "end": 10}, set())


def test_physics_transition_rejects_missing_disc_coordinates():
    span = {"start": 0, "end": 10}
    previous = make_tick(0)
    current = make_tick(1)
    current["discs"][0]["x"] = None
    assert not transition_is_valid(previous, current, span, set())


def test_replay_reader_excludes_active_ticks_with_player_radius_overrides(tmp_path):
    player = [{"id": 8, "team": 1}]
    records = [
        {"type": "tick", **make_tick(0, players=player)},
        {"type": "tick", **make_tick(1, players=player)},
        {"type": "event", "frame": 1, "name": "disc_props", "id": 8, "kind": True,
         "data1": [None, None, None, None, None, None, 25]},
        {"type": "event", "frame": 1, "name": "disc_props", "id": 8, "kind": True,
         "data1": [None, None, None, None, None, None, None, None, 0.75]},
        {"type": "tick", **make_tick(2, players=player)},
        {"type": "tick", **make_tick(3, players=player)},
        {"type": "event", "frame": 4, "name": "disc_props", "id": 8, "kind": True,
         "data1": [None, None, None, None, None, None, 15, None, 0.5]},
        {"type": "tick", **make_tick(4, players=player)},
    ]
    path = tmp_path / "replay.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    ticks, excluded_frames, _, property_summary, _, _ = _read_replay(
        path, [{"start": 0, "end": 5}], default_player_properties={"radius": 15, "invMass": 0.5})

    assert len(ticks) == 5
    assert property_summary["playerPhysicsOverrideTicks"] == 2
    assert property_summary["playerPhysicsOverridePlayerTicks"] == 2
    assert property_summary["overriddenPlayerProperties"] == {"invMass": 2, "radius": 2}
    assert {2, 3}.issubset(excluded_frames)
    assert not transition_is_valid(ticks[1], ticks[2], {"start": 0, "end": 5}, excluded_frames)
