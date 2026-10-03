"""RS4 goal logs must follow complete matches, learner color and saved history."""
from types import SimpleNamespace

import numpy as np
import pytest

from train.multitask import TaskSlot, SELF, POOL, SCRIPTED
from train.rs4_trainer import RS4V3Trainer


def test_goal_window_matches_scripted_results_and_excludes_drills():
    slot = TaskSlot(SimpleNamespace(name='rs4_4v4'), SimpleNamespace(N=6, P=8, T=4))
    slot.modes = np.array([SCRIPTED, SCRIPTED, SCRIPTED, SELF, POOL, SCRIPTED])
    slot.learner_color = np.array([0, 1, 0, 0, 0, 0])
    slot.opponent_key = np.array(['scripted'] * 6)
    # Saved match history predates the goal-window repair.
    slot.match_window = [(1, 0, 0, 3, 1, 0)]
    slot.wr_window = []
    trainer = object.__new__(RS4V3Trainer)
    trainer.league = SimpleNamespace(members=[], record=lambda *args: None)
    info = dict(match_done=np.array([1, 1, 1, 1, 1, 0], bool),
                is_drill=np.array([0, 0, 1, 0, 0, 0], bool),
                goal=np.array([1, -1, 1, 1, 1, 1]),
                done=np.ones(6, bool),
                final_score=np.array([[2, 1], [4, 1], [99, 0], [88, 0], [77, 0], [66, 0]]))
    trainer._record(slot, info)
    assert slot.wr_window == [(3, 1), (2, 1), (1, 4)]
    share, goals = slot.winrate()
    assert goals == 12 and share == pytest.approx(.5)
    assert slot.match_performance()[2] == (2, 0, 1)


def test_goal_window_is_bounded_and_scoreless_matches_are_not_fake_goals():
    slot = TaskSlot(SimpleNamespace(name='rs4_4v4'), SimpleNamespace(N=1, P=8, T=4))
    slot.modes = np.array([SCRIPTED])
    slot.learner_color = np.array([1])
    slot.opponent_key = np.array(['scripted'])
    slot.match_window = [(1, 0, 0, 1, 0, 0)] * 400
    trainer = object.__new__(RS4V3Trainer)
    trainer.league = SimpleNamespace(members=[], record=lambda *args: None)
    info = dict(match_done=np.array([True]), goal=np.array([0]), done=np.array([True]),
                final_score=np.array([[0, 0]]))
    trainer._record(slot, info)
    assert len(slot.wr_window) == len(slot.match_window) == 400
    assert slot.wr_window[-1] == (0, 0)
    assert slot.winrate() == (1., 399)
    slot.match_window = []
    trainer._record(slot, info)
    assert slot.winrate() == (0., 0)
