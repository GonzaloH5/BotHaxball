"""Curriculum integration: task terminals, shared reward, and legacy critic compatibility."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from learn.x4_ppo import COOP_CRITIC_DIM, CRITIC_IN, Critic, Trainer, load_critic_weights, parse_args


def test_legacy_critic_preserves_predictions():
    torch.manual_seed(7)
    old_dim = CRITIC_IN - COOP_CRITIC_DIM
    old = Critic()
    state = old.state_dict()
    state["net.0.weight"] = state["net.0.weight"][:, :old_dim].clone()
    reference = torch.nn.Sequential(
        torch.nn.Linear(old_dim, 512), torch.nn.ReLU(), torch.nn.Linear(512, 512),
        torch.nn.ReLU(), torch.nn.Linear(512, 256), torch.nn.ReLU(), torch.nn.Linear(256, 1))
    reference.load_state_dict({k.removeprefix("net."): v for k, v in state.items()})
    restored = Critic()
    assert load_critic_weights(restored, state)
    x = torch.randn(9, old_dim)
    with torch.no_grad():
        assert torch.allclose(restored(torch.cat([x, torch.zeros(9, COOP_CRITIC_DIM)], -1)),
                              reference(x).squeeze(-1), atol=1e-6)
    assert not load_critic_weights(restored, restored.state_dict())


def test_curriculum_rejects_overwriting_full_match_training():
    with pytest.raises(SystemExit):
        parse_args(["--bc", "unused", "--out", "unused", "--coop-frac", "0.75"])
    with pytest.raises(SystemExit):
        parse_args(["--bc", "unused", "--out", "unused", "--coop-frac", "0.5"])


class TerminalTask:
    """Every second step completes a task; next state must not bootstrap its return."""
    def __init__(self, n):
        self.active = np.zeros(n, bool)
        self.active[0] = True
        self.defender_kind = np.zeros(n, np.int64)
        self.stage = 0
        self.calls = 0
        self.resets = []

    def reset(self, arena, rows):
        self.resets.extend(map(int, rows))
        self.active[rows] = False
        if 0 in rows:
            self.active[0] = True
            arena.learner[0] = np.array([True] * 4 + [False] * 4)
            arena.opp[0] = -2

    def before_step(self, env):
        pass

    def step(self, env, events):
        self.calls += 1
        reward = np.zeros((env.N, 2))
        done = np.zeros(env.N, bool)
        if self.calls % 2 == 0:
            reward[0] = [1.0, -1.0]
            done[0] = True
        return reward, done, {"task_episodes": int(done.sum())}

    def descriptor(self):
        out = np.zeros((len(self.active), COOP_CRITIC_DIM))
        out[:, 0] = self.active
        return out

    def advance(self):
        return False

    def state_dict(self):
        return {"stage": self.stage}


def test_task_done_is_a_gae_boundary_and_frozen_defenders_are_not_samples(tmp_path):
    from pathlib import Path
    from learn.x4_ppo import ROOT
    bc = ROOT / "runs/x4_bc/final_sangu_rsone/best.pt"
    if not bc.exists():
        pytest.skip("needs versioned BC checkpoint")
    a = parse_args(["--bc", str(bc), "--out", str(tmp_path), "--device", "cpu",
                    "--envs", "2", "--rollout", "4", "--threads", "1",
                    "--bank-recordings", "0", "--human-starts", "0", "--pool-frac", "0",
                    "--eval-every", "0", "--shaping-kind", "none", "--auto-pass-arm-update", "0"])
    trainer = Trainer(a)
    arena = trainer.arenas[0]
    task = TerminalTask(arena.N)
    arena.coop = task
    trainer.coop_defenders = (trainer.bc,)
    task.reset(arena, np.arange(arena.N))
    buf, stats = trainer.rollout()
    _, _, _, _, _, _, rewards, done, learner, _, _, _ = buf[0]
    assert done[:, 0].tolist() == [False, True, False, True]
    assert np.array_equal(rewards[1, 0], np.array([1.] * 4 + [-1.] * 4))
    assert not learner[:, 0, 4:].any()
    assert learner[:, 0, :4].all()
    assert stats["matches"] == 0
    assert stats["coop"]["task_episodes"] == 2
    assert task.resets.count(0) == 3
    # An enormous value in the following episode must not leak across done.
    vv = np.zeros((4, 2, 8), np.float32)
    vv[2, 0] = 1000
    adv, _ = trainer.gae(vv, rewards, done, np.zeros((2, 8)))
    assert np.allclose(adv[1, 0], rewards[1, 0])
    trainer.log.close()
