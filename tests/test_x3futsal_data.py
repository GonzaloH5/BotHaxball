import gzip
import json

import numpy as np
import pytest

from learn.x3futsal_data import DATA_VERSION, X3Data, action_from_input, extract_recording, featurize


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


def test_extraction_uses_input_at_transition_start_and_records_prior_action(tmp_path):
    path = tmp_path / "replay.jsonl.gz"
    first = _tick(0, [8, 0, 0, 4, 0, 0])
    second = _tick(1, [0, 8, 0, 0, 4, 0])
    third = _tick(2, [4, 0, 8, 16, 0, 0])
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for item in (first, second, third):
            handle.write(json.dumps(item) + "\n")

    extracted = extract_recording(
        "replay.hbr2",
        {"complete": True, "targetSpans": [{"start": 0, "end": 3}]},
        {"jsonl": path.name},
        tmp_path,
    )

    assert extracted["frame"].tolist() == [0, 1]
    assert extracted["actions"][0].tolist() == [3, 0, 0, 3, 0, 0]
    assert extracted["actions"][1].tolist() == [0, 3, 0, 0, 3, 0]
    assert extracted["has_prior"].tolist() == [False, True]
    assert extracted["prior_actions"][1].tolist() == extracted["actions"][0].tolist()
    assert extracted["changed"][1].tolist() == [True, True, False, True, True, False]


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
