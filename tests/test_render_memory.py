"""Replay decisions advance memory once; actual actions and terminal resets match."""
import numpy as np

from eval import render


def test_replay_tracks_executed_actions_and_preserves_reset_between_decisions(monkeypatch):
    original = render.HaxballEnv
    step_count = 0

    def environment(*args, **kwargs):
        env = original(*args, **kwargs)
        step = env.step
        def controlled(actions):
            nonlocal step_count
            step_count += 1
            obs, reward, done, info = step(actions)
            info["executed_actions"] = np.full_like(actions, step_count % 18)
            if step_count == 2:  # Ends midway through a three-tick command.
                done[:] = True
            return obs, reward, done, info
        env.step = controlled
        return env

    class RecurrentSpy:
        name = "memory-spy"
        def __init__(self):
            self.previous = 18
            self.decisions = []
            self.executed = []
        def reset(self, env, done=None):
            self.previous = 18
        def __call__(self, env, obs, players):
            self.decisions.append(self.previous)
            return np.zeros((env.N, len(players)), dtype=np.int64)
        def record_executed(self, env, actions):
            self.previous = int(actions[0, 0])
            self.executed.append(self.previous)

    monkeypatch.setattr(render, "HaxballEnv", environment)
    red, blue = RecurrentSpy(), RecurrentSpy()
    _, frames = render.record(red, blue, minutes=6 / 3600, frame_skip=3)
    assert len(frames) == 6
    assert red.decisions == blue.decisions == [18, 18]
    assert red.executed == blue.executed == [1, 2, 4, 5, 6]
