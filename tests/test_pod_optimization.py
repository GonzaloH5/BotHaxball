import copy
import numpy as np
import pytest
import torch

from tools.optimize_rs4_pod import choose_profile
from train.rs4_program import ProgramState, default_program


def test_runtime_selection_rejects_fast_but_different_actions_and_noisy_winners():
    rows = {'baseline': [{'speed':100.}]*3,
            'wrong': [{'speed':300., 'equivalent':False}]*3,
            'noisy': [{'speed':90.}, {'speed':150.}, {'speed':160.}],
            'stable': [{'speed':110.}, {'speed':111.}, {'speed':112.}]}
    assert choose_profile(rows, 'baseline', 'speed')['profile'] == 'stable'
    del rows['noisy']
    assert choose_profile(rows, 'baseline', 'speed')['profile'] == 'stable'


def test_no_gain_keeps_reference_runtime():
    rows = {'base':[{'speed':100.}]*3, 'tiny':[{'speed':103.}]*3}
    assert choose_profile(rows, 'base', 'speed')['profile'] == 'base'


def test_hardware_cost_is_separate_monotone_and_backward_compatible():
    state = ProgramState(default_program())
    state.charge_diagnostics(100)
    original = state.state_dict()
    original.pop('hardware_diagnostic_steps')
    restored = ProgramState(state.config, original)
    restored.charge_hardware_diagnostics(200)
    assert restored.diagnostic_steps == 100
    assert restored.effective_branch_limit == restored.config['branch_budget_steps'] - 300
    with pytest.raises(ValueError, match='disminuir'):
        restored.charge_hardware_diagnostics(199)
    saved = restored.state_dict()
    assert ProgramState(restored.config, saved).hardware_diagnostic_steps == 200
    finished = copy.deepcopy(restored)
    finished.relative_steps = finished.effective_branch_limit
    assert ProgramState(finished.config, finished.state_dict()).complete
    with pytest.raises(ValueError, match='reservado'):
        restored.charge_hardware_diagnostics(restored.config['branch_budget_steps'])


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_model_agent_runtime_keeps_recurrent_actions_and_reset_on_device(tmp_path, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from eval.agents import ModelAgent
    from env.haxball_env import HaxballEnv
    from train.recurrent_model import RecurrentSetActorCritic
    from env.haxball_env import U_SELF_DIM
    model = RecurrentSetActorCritic(U_SELF_DIM, ent_dim=8, hidden=16, ent_hidden=8, memory_size=8)
    path = tmp_path / 'model.pt'
    torch.save({'model':model.state_dict(), 'model_config':model.config()}, path)
    cpu = ModelAgent(str(path), True)
    candidate = ModelAgent(str(path), True, device)
    env = HaxballEnv(2, 1, 'classic', obs_layout='universal', seed=51)
    obs = env.reset()
    for _ in range(3):
        np.testing.assert_array_equal(cpu(env, obs, np.array([0, 1])), candidate(env, obs, np.array([0, 1])))
    candidate.record_executed(env, np.zeros((2, 2), dtype=np.int64))
    candidate.reset(env, np.array([True, False]))
    memory, previous = candidate._states[env]
    assert str(memory.device).startswith(device) and str(previous.device).startswith(device)
    assert memory[0].abs().sum() == 0 and (previous[0] == model.n_actions).all()
