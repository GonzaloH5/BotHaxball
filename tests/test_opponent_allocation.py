"""Small task batches must retain rare opponents without inflating their share."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from train.multitask import MultiTrainer, TaskSlot, SELF, POOL, SCRIPTED


def setup(n=13, fractions=(.45, .5, .05), pool=True):
    trainer = MultiTrainer.__new__(MultiTrainer)
    trainer.cfg = {"curriculum": [dict(zip(("selfplay", "pool", "scripted"), fractions), scripted_eps=0)],
                   "league": {"opponents_per_iter": 4}}
    trainer.rng = np.random.default_rng(7)
    trainer.league = SimpleNamespace(members=[object()] if pool else [], sample=lambda *args: [0])
    slot = TaskSlot(SimpleNamespace(name="big_2v2"), SimpleNamespace(N=n, P=4, T=2))
    trainer.slots, trainer.task_state = [slot], {}
    return trainer, slot


@pytest.mark.parametrize("n", [1, 2, 9, 13, 14, 19, 20, 25, 32])
@pytest.mark.parametrize("fractions", [(.45, .5, .05), (.4, .2, .4), (.3, 0, .7)])
def test_long_run_proportions_and_valid_learner_masks(n, fractions):
    trainer, slot = setup(n, fractions)
    total = np.zeros(3, dtype=int)
    for iteration in range(400):
        trainer.assign_modes(slot)
        counts = np.bincount(slot.modes, minlength=3)
        assert counts.sum() == n
        total += counts
        # Cumulative rounding error stays small, including one-environment tasks.
        np.testing.assert_allclose(total, np.array(fractions) * n * (iteration + 1), atol=1.01, rtol=0)
        assert slot.learner[:, :2].all()
        assert slot.learner[slot.modes == SELF, 2:].all()
        assert not slot.learner[slot.modes != SELF, 2:].any()
        assert (slot.opp_id[slot.modes != POOL] == -1).all()
        assert (slot.opp_id[slot.modes == POOL] == 0).all()
    assert total[SCRIPTED] > 0


def test_missing_pool_is_excluded_and_appearance_resets_allocation_context():
    trainer, slot = setup(pool=False)
    for _ in range(30):
        trainer.assign_modes(slot)
        assert not (slot.modes == POOL).any()
    trainer.league.members = [object()]
    trainer.assign_modes(slot)
    fresh_trainer, fresh_slot = setup()
    fresh_trainer.assign_modes(fresh_slot)
    np.testing.assert_array_equal(np.bincount(slot.modes, minlength=3),
                                  np.bincount(fresh_slot.modes, minlength=3))
    np.testing.assert_allclose(slot.mode_carry, fresh_slot.mode_carry)


def test_allocation_carry_survives_serialized_task_state():
    trainer, slot = setup()
    for _ in range(7):
        trainer.assign_modes(slot)
    trainer.sync_task_state()
    state = json.loads(json.dumps(trainer.task_state))[slot.task.name]
    resumed_trainer, resumed_slot = setup()
    resumed_slot.mode_context = tuple(state["mode_context"])
    resumed_slot.mode_carry = np.array(state["mode_carry"])
    for _ in range(100):
        trainer.assign_modes(slot)
        resumed_trainer.assign_modes(resumed_slot)
        np.testing.assert_array_equal(np.bincount(slot.modes, minlength=3),
                                      np.bincount(resumed_slot.modes, minlength=3))
        np.testing.assert_allclose(slot.mode_carry, resumed_slot.mode_carry)


def test_rebalanced_batch_sizes_keep_cumulative_proportions():
    trainer, slot = setup()
    total = np.zeros(3, dtype=int)
    environments = 0
    for n in [13, 14, 9, 1, 25, 19] * 30:
        slot.N = n
        trainer.assign_modes(slot)
        environments += n
        total += np.bincount(slot.modes, minlength=3)
        np.testing.assert_allclose(total, np.array([.45, .5, .05]) * environments, atol=1.01, rtol=0)


def test_empty_goal_window_is_not_displayed_as_zero_percent(capsys):
    trainer, slot = setup()
    recorded = {}
    trainer.writer = SimpleNamespace(add_scalar=lambda key, value, step: recorded.update({key: value}))
    trainer.cfg["log"] = {"every": 1}
    trainer.iteration, trainer.steps, trainer.stage = 1, 100, 0
    trainer.league.learner_elo = 1200
    trainer.rcfg = SimpleNamespace(shaping_coef=1)
    slot.env.rcfg = trainer.rcfg
    trainer.cfg["reward"] = {"shaping_decay_steps": 400_000_000}
    stats = {"entropy": 2}
    trainer.log(stats, .001, 100, 1, .2)
    assert "big_2v2 r0 sin datos(0)" in capsys.readouterr().out
    assert np.isnan(recorded["task/big_2v2/goal_share_vs_scripted"])
    slot.wr_window = [(0, 5)]
    trainer.log(stats, .001, 100, 1, .2)
    assert "big_2v2 r0 0.00(5)" in capsys.readouterr().out
    assert recorded["task/big_2v2/goal_share_vs_scripted"] == 0
