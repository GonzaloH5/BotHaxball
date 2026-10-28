import gzip
import json

import numpy as np
import pytest

from learn.x3futsal_data import (
    DATA_VERSION,
    TACTICAL_ENTITY_DIM,
    TACTICAL_SELF_DIM,
    X3Data,
    action_from_input,
    extract_recording,
    featurize,
    featurize_tactical,
    featurize_tactical_team,
    flip_action_y,
    mirror_states_y,
)
from learn.x3futsal_eval import STADIUM, _player_states
from learn.x3futsal_policy import decode_actions_np
from sim.physics import BatchSim
from sim.stadium import load_stadium


def _tick(frame, inputs, state=1):
    players = []
    discs = [{"x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0}]
    for index in range(6):
        team = 1 if index < 3 else 2
        x = (-1 if team == 1 else 1) * (120 + index * 20)
        discs.append({"x": x, "y": index * 25.0, "vx": 0.1 * index, "vy": 0.0})
        players.append({"id": index + 1, "team": team, "disc": index + 1,
                        "input": inputs[index], "kicking": False})
    return {"type": "tick", "frame": frame, "state": state, "score": [0, 0],
            "time": frame / 60, "discs": discs, "players": players}


def test_actions_mirror_horizontal_direction_and_keep_kick_bit():
    assert action_from_input(8, 1) == 3
    assert action_from_input(4, 2) == 3
    assert action_from_input(24, 1) == 12
    assert action_from_input(0, 2) == 0


def test_vertical_action_and_state_reflection_is_involution():
    actions = np.arange(18, dtype=np.int64)
    flipped = flip_action_y(actions)
    assert np.array_equal(flip_action_y(flipped), actions)
    assert flipped[1] == 5 and flipped[5] == 1
    assert flipped[2] == 4 and flipped[4] == 2
    assert flipped[10] == 14 and flipped[14] == 10

    states = np.random.default_rng(0).normal(size=(4, 7, 4)).astype(np.float32)
    assert np.allclose(mirror_states_y(mirror_states_y(states)), states)


def test_extraction_uses_input_at_transition_start_and_records_prior_action(tmp_path):
    path = tmp_path / "replay.jsonl.gz"
    first = _tick(0, [8, 0, 0, 4, 0, 0])
    second = _tick(1, [0, 8, 0, 0, 4, 0])
    third = _tick(2, [4, 0, 8, 16, 0, 0])
    fourth = _tick(3, [8, 8, 0, 4, 0, 0])
    fifth = _tick(4, [0, 0, 0, 0, 0, 0])
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for item in (first, second, third, fourth, fifth):
            handle.write(json.dumps(item) + "\n")

    extracted = extract_recording(
        "replay.hbr2",
        {"complete": True, "targetSpans": [{"start": 0, "end": 5}]},
        {"jsonl": path.name},
        tmp_path,
    )

    assert extracted["frame"].tolist() == [0, 1, 2, 3]
    assert extracted["actions"][0].tolist() == [3, 0, 0, 3, 0, 0]
    assert extracted["actions"][1].tolist() == [0, 3, 0, 0, 3, 0]
    assert extracted["has_prior"].tolist() == [False, True, True, True]
    assert extracted["prior_actions"][1].tolist() == extracted["actions"][0].tolist()
    assert extracted["changed"][1].tolist() == [True, True, False, True, True, False]
    assert extracted["stride3_has_prior"].tolist() == [False, False, False, True]
    assert extracted["stride3_prior_actions"][3].tolist() == extracted["actions"][0].tolist()


def test_extraction_includes_valid_kickoff_transitions(tmp_path):
    path = tmp_path / "kickoff.jsonl.gz"
    first = _tick(0, [8, 0, 0, 4, 0, 0], state=0)
    second = _tick(1, [0, 8, 0, 0, 4, 0], state=0)
    third = _tick(2, [4, 0, 8, 16, 0, 0], state=1)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for item in (first, second, third):
            handle.write(json.dumps(item) + "\n")

    extracted = extract_recording(
        "kickoff.hbr2",
        {"complete": True, "targetSpans": [{"start": 0, "end": 3}]},
        {"jsonl": path.name},
        tmp_path,
    )

    assert extracted["frame"].tolist() == [0, 1]
    assert extracted["actions"][0].tolist() == [3, 0, 0, 3, 0, 0]
    assert extracted["actions"][1].tolist() == [0, 3, 0, 0, 3, 0]
    assert extracted["has_prior"].tolist() == [False, True]


def _mirrored_observations():
    state = np.zeros((2, 7, 4), dtype=np.float32)
    positions = [(-100, 0), (-30, -40), (20, 35), (100, 0), (30, -40), (-20, 35)]
    for index, (x, y) in enumerate(positions):
        state[0, index + 1, :2] = (x, y)
        state[1, index + 1, :2] = (x, y)
    state[0, 0] = (0, 10, 0.5, 0.0)
    state[1, 0] = (0, 10, -0.5, 0.0)
    return featurize(state, np.asarray([0, 3]), np.zeros(2), np.ones(2),
                     field_half_w=400, field_half_h=200)


def test_observation_is_mirror_invariant():
    own, mates, rivals = _mirrored_observations()
    assert np.allclose(own[0], own[1])
    assert own.shape == (2, 12)
    assert mates.shape == (2, 2, 6)
    assert rivals.shape == (2, 3, 6)


def test_tactical_observation_is_mirror_invariant_and_detects_kick_reach():
    state = np.zeros((2, 7, 4), dtype=np.float32)
    positions = [(-18, 0), (-150, -40), (-300, 35), (18, 0), (150, -40), (300, 35)]
    for index, (x, y) in enumerate(positions):
        state[0, index + 1, :2] = (x, y)
        state[1, index + 1, :2] = (x, y)
    state[0, 0] = (0, 0, 0.5, 0.0)
    state[1, 0] = (0, 0, -0.5, 0.0)
    own, mates, rivals = featurize_tactical(state, np.asarray([0, 3]), field_half_w=550, field_half_h=240, goal_x=557.5)
    assert np.allclose(own[0], own[1], atol=1e-6)
    assert own.shape == (2, TACTICAL_SELF_DIM)
    assert mates.shape == (2, 2, TACTICAL_ENTITY_DIM)
    assert rivals.shape == (2, 3, TACTICAL_ENTITY_DIM)
    assert own[0, 12] == 1.0  # ball_in_reach (distance 18 <= 25.25)
    assert own[0, 24] == 1.0  # is_closest_to_ball


def test_featurize_tactical_team_matches_featurize_tactical():
    stadium = load_stadium(STADIUM)
    sim = BatchSim(16, 3, 3, stadium=stadium, seed=19)
    sim.reset_random(np.arange(16))
    sim.kick_cancel = np.random.default_rng(7).random((16, 6)) < 0.35
    for team in (0, 1):
        slots = np.arange(3 * team, 3 * team + 3, dtype=np.int64)
        states = _player_states(sim)
        rep = np.repeat(states[:, None, :, :], 3, axis=1).reshape(16 * 3, 7, 4)
        actor_slots = np.tile(slots, 16)
        kc = sim.kick_cancel[:, slots].reshape(-1)
        ref_o, ref_m, ref_r = featurize_tactical(
            rep, actor_slots, stadium.field_half_w, stadium.field_half_h, stadium.goal_x, kick_cancel=kc
        )
        fast_o, fast_m, fast_r = featurize_tactical_team(
            sim, team, stadium.field_half_w, stadium.field_half_h, stadium.goal_x
        )
        assert np.allclose(ref_o, fast_o, atol=1e-6)
        assert np.allclose(ref_m, fast_m, atol=1e-6)
        assert np.allclose(ref_r, fast_r, atol=1e-6)


def test_decode_actions_suppresses_out_of_reach_kicks_and_respects_kick_cancel():
    logits = np.zeros((3, 18), dtype=np.float32)
    # Row 0: wants to move right + kick, within reach, kick armed -> should output 12
    logits[0, 12] = 5.0
    # Row 1: wants to move right + kick, far from ball (120 px) -> should strip kick bit -> 3
    logits[1, 12] = 5.0
    # Row 2: wants to move right + kick, within reach, but kick_cancel is True -> should strip kick bit -> 3
    logits[2, 12] = 5.0
    decoded = decode_actions_np(
        logits,
        ball_dist_px=np.asarray([20.0, 120.0, 20.0]),
        kick_cancel=np.asarray([False, False, True]),
        shot_alignment=np.asarray([0.9, 0.9, 0.9]),
    )
    assert decoded.tolist() == [12, 3, 3]


def test_split_loader_returns_stable_batch_contract(tmp_path):
    states = np.zeros((2, 7, 4), dtype=np.float16)
    actions = np.tile(np.arange(6, dtype=np.uint8), (2, 1))
    np.savez_compressed(tmp_path / "shard.npz", states=states, actions=actions,
                        prior_actions=np.zeros((2, 6), dtype=np.uint8),
                        has_prior=np.asarray([False, True]),
                        changed=np.zeros((2, 6), dtype=bool),
                        score_diff=np.zeros(2, dtype=np.int16), time=np.zeros(2, dtype=np.float32),
                        dataVersion=np.asarray(DATA_VERSION), sourceSha256=np.asarray("a" * 64))
    (tmp_path / "manifest.json").write_text(json.dumps({
        "dataVersion": DATA_VERSION,
        "buildStatus": "complete",
        "shards": [{"split": "train", "file": "shard.npz", "ticks": 2,
                    "sourceSha256": "a" * 64}],
    }), encoding="utf-8")

    data = X3Data(tmp_path, "train")
    observation, labels, _, has_prior, _ = data.batch(np.asarray([1, 0]), np.asarray([4, 0]))

    assert labels.tolist() == [4, 0]
    assert has_prior.tolist() == [True, False]
    assert [part.shape for part in observation] == [(2, 12), (2, 2, 6), (2, 3, 6)]

    tactical_obs, _, _, _, _ = data.batch(np.asarray([1, 0]), np.asarray([4, 0]), tactical=True)
    assert [part.shape for part in tactical_obs] == [(2, TACTICAL_SELF_DIM), (2, 2, TACTICAL_ENTITY_DIM),
                                                      (2, 3, TACTICAL_ENTITY_DIM)]


def test_policy_is_mirror_and_permutation_invariant():
    torch = pytest.importorskip("torch")
    from learn.x3futsal_policy import X3SetPolicy

    own, mates, rivals = _mirrored_observations()
    prior = torch.tensor([5, 12], dtype=torch.long)
    has_prior = torch.tensor([True, False])
    torch.manual_seed(2)
    model = X3SetPolicy().eval()
    with torch.no_grad():
        expected = model(torch.from_numpy(own), torch.from_numpy(mates), torch.from_numpy(rivals),
                         prior, has_prior)
        permuted = model(torch.from_numpy(own), torch.from_numpy(mates[:, ::-1].copy()),
                         torch.from_numpy(rivals[:, ::-1].copy()), prior, has_prior)
        ignored_prior = model(torch.from_numpy(own), torch.from_numpy(mates), torch.from_numpy(rivals),
                              torch.tensor([1, 2]), torch.zeros(2, dtype=torch.bool))
        ignored_prior_alt = model(torch.from_numpy(own), torch.from_numpy(mates), torch.from_numpy(rivals),
                                  torch.tensor([3, 4]), torch.zeros(2, dtype=torch.bool))
    assert torch.allclose(expected[0], expected[1], atol=1e-6)
    assert torch.allclose(expected, permuted, atol=1e-6)
    assert torch.allclose(ignored_prior, ignored_prior_alt, atol=1e-6)