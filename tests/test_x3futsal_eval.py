import numpy as np
import pytest

pytest.importorskip("torch")

from learn.x3futsal_eval import STADIUM, Policy, ScriptedPolicy, _aggregate, action_to_sim, play_batch
from sim.stadium import load_stadium


class StillPolicy(Policy):
    name = "still"

    def actions(self, sim, team, scores, elapsed, previous, has_previous):
        return np.zeros((sim.N, 3), dtype=np.int64)


def test_model_actions_mirror_horizontally_for_blue():
    assert action_to_sim(3, 0) == 3
    assert action_to_sim(3, 1) == 7
    assert action_to_sim(2, 1) == 8
    assert action_to_sim(1, 1) == 1
    assert action_to_sim(11, 1) == 17


def test_scripted_striker_excludes_goalkeeper():
    class NearbyGoalkeeper:
        N = 1
        ball_pos = np.asarray([[0.0, 0.0]])
        player_pos = np.asarray([[
            [-5.0, 0.0], [-20.0, 0.0], [120.0, 80.0],
            [300.0, 0.0], [350.0, 40.0], [400.0, -40.0],
        ]])

    policy = ScriptedPolicy(load_stadium(STADIUM))
    actions = policy.actions(NearbyGoalkeeper(), 0, np.zeros(1), 0.0,
                             np.zeros((1, 6), dtype=np.int64), np.zeros((1, 6), dtype=bool))

    assert actions[0, 1] >= 9


def test_scripted_policy_can_complete_kickoff_without_safety_reset():
    games = play_batch(StillPolicy(), ScriptedPolicy(load_stadium(STADIUM)), matches=1,
                       minutes=0.1, seed=0)

    assert games[0]["safetyKickoffs"] == 0
    assert games[0]["teams"][1]["kicks"] > 0


def test_aggregation_tracks_each_policy_across_both_sides():
    stats = {
        "kicks": 1, "shots": 0, "completedPasses": 0, "turnovers": 0,
        "touches": 0, "movementDecisions": 0, "meanPlayerSpeed": 0.0,
        "meanTeamSpread": 0.0, "safetyKickoffs": 1,
    }
    games = [
        {"firstPolicyTeam": 0, "goals": [2, 0], "teams": [stats, stats]},
        {"firstPolicyTeam": 1, "goals": [0, 1], "teams": [stats, stats]},
    ]

    first = _aggregate(games, 0)
    second = _aggregate(games, 1)

    assert (first["wins"], first["draws"], first["losses"]) == (2, 0, 0)
    assert (second["wins"], second["draws"], second["losses"]) == (0, 0, 2)
    assert first["matches"] == second["matches"] == 2
    assert first["safetyKickoffsPerMatch"] == 1


def test_short_match_finishes_exact_tick_budget():
    games = play_batch(StillPolicy(), StillPolicy(), matches=2, minutes=0.01,
                       seed=5)
    repeated = play_batch(StillPolicy(), StillPolicy(), matches=2, minutes=0.01,
                          seed=5)

    assert games == repeated
    assert len(games) == 2
    assert all(game["ticks"] == 36 for game in games)
    assert all(len(game["teams"]) == 2 for game in games)
    assert all(game["safetyKickoffs"] == 0 for game in games)
