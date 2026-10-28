"""Pressure-start selection and delayed raw-input restoration for full 4v4."""
from types import SimpleNamespace
import json

import numpy as np
import pytest

from env.rs4z import contract as C, obs_v3
from env.rs4z.core import RS4ZEnv
from learn.x4_data import move_of_input
from learn.x4_pressure_starts import PressureStartBank, restore_source_history, valid_source_window
from tests.rs4z.test_x4_coop_curriculum import _data


def _pressured_data():
    data = _data()
    data.names = ["train_recording_0"]
    # Exactly 75 px from the passer, but more than 75 px from the ball.
    # This defender never intercepts the initial 180 px pass.
    data.pos[:, 4] = [-23.0, 75.0]
    return data


def _bank(data=None, **kwargs):
    return PressureStartBank(_pressured_data() if data is None else data,
                             min_examples=kwargs.pop("min_examples", 1),
                             min_recordings=kwargs.pop("min_recordings", 1), **kwargs)


def _two_recordings():
    left, right = _pressured_data(), _pressured_data()
    result = {}
    for key, value in vars(left).items():
        if key == "names":
            result[key] = ["train_recording_0", "train_recording_1"]
        elif key == "valid":
            result[key] = np.arange(480)
        elif key == "rec_of_tick":
            result[key] = np.repeat(np.arange(2), 240)
        else:
            result[key] = np.concatenate([value, getattr(right, key)])
    return SimpleNamespace(**result)


def test_bank_selects_natural_passer_pressure_retention_and_prequeue_tick():
    data = _pressured_data()
    bank = _bank(data)
    assert bank.records.tolist() == [[60, 90, 0, 1, 0]]
    assert bank.indices.tolist() == [60 - 15 - 12]
    assert bank.sample(np.random.default_rng(9)) == 33
    assert np.linalg.norm(data.pos[60, 4] - data.pos[60, 0]) == 75
    assert np.linalg.norm(data.pos[60, 4] - data.ball[60, :2]) > 75
    manifest = json.loads(json.dumps(bank.manifest))
    assert manifest["examples"] == manifest["recordings"] == 1
    assert manifest["source_recording_names"] == ["train_recording_0"]
    assert manifest["retention_ticks"] == 120 and manifest["pressure_reference"] == "passer"
    assert manifest["endpoint_control"] == "any_teammate_physical"


def test_pressure_radius_is_a_strict_recorded_filter_without_relocation():
    data = _pressured_data()
    data.pos[:, 4, 1] = 75.001
    initial = data.pos.copy()
    with pytest.raises(ValueError, match="insufficient pressure starts: 0 examples"):
        _bank(data)
    assert np.array_equal(initial, data.pos)


@pytest.mark.parametrize("corruption", ["rival_body", "rival_kick", "out", "restart", "kickoff",
                                        "record_boundary", "player_change", "absent_player", "map_change",
                                        "invalid_tick", "nonfinite", "no_physical_endpoint_control"])
def test_bank_rejects_lost_or_unverifiable_two_second_continuations(corruption):
    data = _pressured_data()
    tick = 150  # after the pass miner's shorter initial-reception tail
    if corruption == "rival_body":
        data.pos[tick, 4] = data.ball[tick, :2]
    elif corruption == "rival_kick":
        data.kick_ev[tick, 4] = True
    elif corruption == "out":
        data.ball[tick, 1] = C.LINE_H + 1
    elif corruption == "restart":
        data.rkind[tick] = 1
    elif corruption == "kickoff":
        data.state[tick] = 0
    elif corruption == "record_boundary":
        data.rec_of_tick[tick:] = 1
    elif corruption == "player_change":
        data.pid[tick:, 2] = 88
    elif corruption == "absent_player":
        data.pid[tick, 2] = -1
    elif corruption == "map_change":
        data.map_idx[tick] = obs_v3.MAP_NAMES.index("haxarg_2k23")
    elif corruption == "invalid_tick":
        data.valid = data.valid[data.valid != tick]
    elif corruption == "nonfinite":
        data.pos[tick, 2, 0] = np.nan
    else:
        # The original receiver is only 37 px away, which the curriculum's
        # broad control margin accepts. Physical contact requires <=23.375 px.
        data.ball[210, :2] = [240.0, 0.0]
    with pytest.raises(ValueError, match="insufficient pressure starts: 0 examples"):
        _bank(data)


def test_bank_allows_repass_with_another_teammate_controlling_after_two_seconds():
    data = _pressured_data()
    # Receiver sends the ball back to the original passer before the endpoint.
    data.kick_ev[160, 1] = True
    data.inp[160, 1] = 16
    data.ball[160:211, 0] = np.linspace(180, -23, 51)
    data.ball[160:210, 2] = -203 / 50
    data.ball[211:, 0] = -23
    bank = _bank(data)
    assert bank.records.tolist() == [[60, 90, 0, 1, 0]]
    assert np.array_equal(data.ball[210, :2], data.pos[210, 0])
    assert np.linalg.norm(data.ball[210, :2] - data.pos[210, 1]) > 200


def test_bank_requires_both_example_count_and_independent_recording_count():
    data = _two_recordings()
    bank = _bank(data, min_examples=2, min_recordings=2)
    assert bank.indices.tolist() == [33, 273]
    assert bank.manifest["recording_counts"] == {"0": 1, "1": 1}
    with pytest.raises(ValueError, match="need >= 3 examples"):
        _bank(data, min_examples=3)
    with pytest.raises(ValueError, match="need >= 1 examples and >= 3 recordings"):
        _bank(data, min_recordings=3)
    with pytest.raises(ValueError, match="need >= 200 examples and >= 10 recordings"):
        PressureStartBank(data)
    seeded1, seeded2 = _bank(data, seed=7), _bank(data, seed=7)
    sequence1 = [seeded1.sample() for _ in range(20)]
    sequence2 = [seeded2.sample() for _ in range(20)]
    assert sequence1 == sequence2 and set(sequence1) == {33, 273}


def test_bank_rejects_explicit_nontrain_provenance():
    data = _pressured_data()
    data.split = "dev"
    with pytest.raises(ValueError, match="train split only"):
        _bank(data)


@pytest.mark.parametrize("phase", ["open", "restart", "kickoff"])
def test_source_history_validation_is_shared_across_human_start_phases(phase):
    data = _pressured_data()
    if phase == "restart":
        data.rkind[33] = 1
    elif phase == "kickoff":
        data.state[33] = 0
    assert valid_source_window(data, 33, np.array([0, 1, 4, 7, 8, 11, 14, 15]), 6)


@pytest.mark.parametrize("corruption", ["oldest_gap", "forward_gap", "player_change", "duplicate_pid",
                                        "absent_player", "map_change", "record_boundary", "negative_delay"])
def test_source_history_validation_covers_oldest_and_forward_inputs(corruption):
    data = _pressured_data()
    delays = np.array([0, 1, 4, 7, 8, 11, 14, 15])
    if corruption == "oldest_gap":
        data.valid = data.valid[data.valid != 16]
    elif corruption == "forward_gap":
        data.valid = data.valid[data.valid != 49]
    elif corruption == "player_change":
        data.pid[20, 2] = 88
    elif corruption == "duplicate_pid":
        data.pid[16:50, 2] = 1
    elif corruption == "absent_player":
        data.pid[16:50, 2] = -1
    elif corruption == "map_change":
        data.map_idx[20] = 2
    elif corruption == "record_boundary":
        data.rec_of_tick[20] = 1
    else:
        delays[0] = -1
    assert not valid_source_window(data, 33, delays, 6)


def test_restore_reconstructs_exact_world_queue_without_labels_or_physics_changes():
    data = _pressured_data()
    data.inp[:] = (np.arange(240)[:, None] + 5 * np.arange(8)[None]) % 32
    data.label = np.full((240, 8), 17)  # imitation labels may include future kicks
    env = RS4ZEnv(1, map="sanguchito_rs_x4", max_delay=15, frame_skip=3, seed=1)
    delay = np.array([0, 1, 4, 7, 8, 11, 14, 15])
    env.start_match([0], delay=delay[None])
    env.place(0, ball_pos=data.ball[33, :2], ball_vel=data.ball[33, 2:4],
              player_pos=data.pos[33], player_vel=data.vel[33])
    env.ev_kicked[:] = env.ev_touch[:] = True
    before = (env.ball_pos.copy(), env.player_pos.copy(), env.active.copy(), env.ri.copy())
    assert restore_source_history(env, 0, data, 33) == 33
    for h in range(env.H):
        idx = 33 + delay + 1 - 3 * (h + 1)
        raw = data.inp[idx, np.arange(8)]
        assert np.array_equal(env.act_hist[0, :, h], move_of_input(raw) + 9 * ((raw & 16) != 0))
    raw = data.inp[33]
    assert np.array_equal(env._s_act[0], move_of_input(raw) + 9 * ((raw & 16) != 0))
    assert not env.ev_kicked.any() and not env.ev_touch.any()
    for previous, after in zip(before, (env.ball_pos, env.player_pos, env.active, env.ri)):
        assert np.array_equal(previous, after)


def test_restore_fails_before_mutation_for_invalid_history_and_zeroes_inactive_slots():
    data = _pressured_data()
    env = RS4ZEnv(1, map="sanguchito_rs_x4", max_delay=15, frame_skip=3, seed=1)
    env.start_match([0], delay=np.full((1, 8), 15))
    env.act_hist[:] = 7
    data.valid = data.valid[data.valid != 49]
    with pytest.raises(ValueError, match="invalid source history"):
        restore_source_history(env, 0, data, 33)
    assert (env.act_hist == 7).all()
    data.valid = np.arange(240)
    data.inp[:] = 24
    env.active[0, 7] = False
    restore_source_history(env, 0, data, 33)
    assert not env.act_hist[0, 7].any() and env._s_act[0, 7] == 0


def test_still_policy_cannot_execute_the_selected_human_release_from_pending_queue():
    data = _pressured_data()
    bank = _bank(data)
    env = RS4ZEnv(1, map="sanguchito_rs_x4", max_delay=15, frame_skip=3, seed=1)
    env.start_match([0], delay=np.full((1, 8), 15))
    t = bank.sample()
    env.place(0, ball_pos=data.ball[t, :2], ball_vel=data.ball[t, 2:4],
              player_pos=data.pos[t], player_vel=data.vel[t])
    restore_source_history(env, 0, data, t)
    assert env.active.all() and not (env.act_hist >= 9).any()
    for _ in range(15):
        ev = env.step(np.zeros((1, 8), np.int64))
        assert not ev["kicked"].any()
