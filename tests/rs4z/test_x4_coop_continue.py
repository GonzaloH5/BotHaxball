"""Continuation preserves learning while changing the curriculum's evidence."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from learn.x4_coop_continue import (
    continuation_command,
    evaluation_command,
    extension_targets,
    regression_checks,
    resumed_checkpoint,
)
from learn.x4_coop_curriculum import CoopCurriculum


def _args(tmp_path):
    return SimpleNamespace(
        root=str(tmp_path), init='frozen/original_init200.pt',
        bc='frozen/original_bc.pt', original='frozen/original_policy.pt',
        parent600='references/parent600.pt', device='cpu', envs=512,
        rollout=128, threads=5, recordings=80, eval_every=100, seed=7,
    )


def _option(command, name):
    return command[command.index(name) + 1]


def _same_state(actual, expected):
    """Compare nested optimizer/checkpoint states, including tensor moments."""
    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _same_state(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            _same_state(a, e)
    else:
        assert actual == expected


def _checkpoint():
    # An actual Adam step populates the step counter and both moment tensors.
    actor, critic = torch.nn.Linear(3, 2), torch.nn.Linear(3, 1)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=0.0002, eps=1e-5)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=0.0005, eps=1e-5)
    for model, optimizer in ((actor, actor_opt), (critic, critic_opt)):
        optimizer.zero_grad()
        model(torch.ones(2, 3)).square().mean().backward()
        optimizer.step()
    return dict(
        model=actor.state_dict(), critic=critic.state_dict(),
        opt_pi=actor_opt.state_dict(), opt_v=critic_opt.state_dict(),
        vnorm=[0.31, 1.74, 128000.0], update=600, decisions=39321600,
        args=dict(critic_warmup=10, init='frozen/original_init200.pt',
                  bc='frozen/original_bc.pt'),
        baseline={'update': 200, 'old_reward': True}, best_score=.91,
        best_pass=1.4, drift_count=2, pass_ok_streak=1,
        pool_names=['snapshot_00400', 'snapshot_00600'],
        coop_state={
            'sanguchito_rs_x4': dict(
                version=1, stage=2, window=[1, 0, 1], mastery_streak=1,
                stage_started_learning_update=103,
                total_episodes=[700, 2100, 1100, 0, 0],
                total_successes=[600, 1400, 380, 0, 0],
                exercise_order=list(range(16)),
                rng_state=np.random.default_rng(7).bit_generator.state,
            ),
        },
    )


def _evaluation(**overrides):
    metrics = dict(pases_por_min=8.0, precision_pase=.6,
                   retencion_tras_recibir=.5, retencion_bajo_presion=.5,
                   tiros_por_min=2.0, goles_por_min=1.0)
    metrics.update(overrides)
    return dict(cadena_pase=dict(metricas={k: [v, 32] for k, v in metrics.items()}))


def _match(point=.39, upper=.49, valid=True):
    return dict(valid_for_strength=valid,
                point_fraction=dict(valor=point, ic90=[.1, upper]))


def test_command_preserves_reference_paths_and_uses_absolute_resume_budget(tmp_path):
    a = _args(tmp_path)
    command = continuation_command(a, 700)
    assert _option(command, '--updates') == '700'
    assert '--resume' in command
    assert _option(command, '--bc') == a.bc
    assert _option(command, '--init') == a.init
    assert _option(command, '--pool') == a.init + ',' + a.original
    assert _option(command, '--out') == str(tmp_path / 'curriculum')
    assert _option(command, '--critic-warmup') == '10'
    assert _option(command, '--coop-start-stage') == '2'
    assert _option(command, '--coop-frac') == '0.5'
    assert _option(command, '--coop-replay-frac') == '0.25'
    assert _option(command, '--coop-finish-frac') == '0.25'
    assert _option(command, '--coop-min-stage-updates') == '100'
    assert _option(command, '--coop-mastery-windows') == '2'
    assert '--coop-reward-cooperative-goals' in command
    assert '--coop-possession-retention' in command
    assert '--pass-ready' in command and '--pass-macro' in command


def test_resume_preserves_actor_critic_adam_league_and_update_without_mutating_parent():
    parent = _checkpoint()
    before = copy.deepcopy(parent)
    resumed = resumed_checkpoint(parent)
    _same_state(parent, before)
    for key in ('model', 'critic', 'opt_pi', 'opt_v', 'vnorm', 'update',
                'decisions', 'args', 'pool_names'):
        _same_state(resumed[key], before[key])
    assert resumed['update'] == 600
    assert resumed['baseline'] is None
    assert resumed['best_score'] == resumed['best_pass'] == -1.0
    assert resumed['drift_count'] == resumed['pass_ok_streak'] == 0
    state = resumed['coop_state']['sanguchito_rs_x4']
    assert state['stage'] == 2
    assert state['window'] == [] and state['mastery_streak'] == 0
    assert state['stage_started_learning_update'] == 590
    for key in ('total_episodes', 'total_successes', 'exercise_order', 'rng_state'):
        _same_state(state[key], before['coop_state']['sanguchito_rs_x4'][key])
    # Independence also holds for nested tensors and list counters after return.
    resumed['model']['weight'].zero_()
    state['total_episodes'][2] += 1
    _same_state(parent, before)


def test_changed_criteria_require_100_new_actor_updates_and_two_new_windows():
    bank = SimpleNamespace(records=np.array([[0, 0, 0, 1, 0]]),
                           d=SimpleNamespace(ball=np.zeros((2, 4))))
    cur = CoopCurriculum(bank, 16, np.random.default_rng(7), start_stage=2,
                         min_episodes=32, window=32, threshold=.4,
                         mastery_windows=2, min_stage_updates=100)
    cur.load_state_dict(resumed_checkpoint(_checkpoint())['coop_state']['sanguchito_rs_x4'])
    assert not cur.window and cur.mastery_streak == 0
    cur.window.extend([1] * 32)
    # PPO checks before learning: update 700 has completed only 699-10 steps.
    assert cur.advance(learning_updates=700 - 1 - 10) is None
    assert len(cur.window) == 32 and cur.mastery_streak == 0 and cur.stage == 2
    assert cur.advance(learning_updates=701 - 1 - 10) is None
    assert not cur.window and cur.mastery_streak == 1 and cur.stage == 2
    assert cur.advance(learning_updates=701 - 1 - 10) is None
    cur.window.extend([1] * 32)
    promoted = cur.advance(learning_updates=702 - 1 - 10)
    assert promoted['from_stage'] == 2 and promoted['to_stage'] == 3


@pytest.mark.parametrize('missing', [None, float('nan'), float('inf')])
@pytest.mark.parametrize('side', ['baseline', 'current'])
def test_guard_handles_unknown_and_nonfinite_metrics(missing, side):
    baseline, current = _evaluation(), _evaluation()
    target = baseline if side == 'baseline' else current
    for values in target['cadena_pase']['metricas'].values():
        values[0] = missing
    result = regression_checks(baseline, current)
    assert len(result['missing_metrics']) == 6
    assert not any(result['regressed_metrics'].values())
    assert not result['material_regression']


def test_guard_handles_absent_metrics_without_claiming_a_pass():
    result = regression_checks(dict(cadena_pase=dict(metricas={})), _evaluation())
    assert len(result['missing_metrics']) == 6
    assert not result['material_regression']


def test_shot_and_goal_drop_are_material_even_when_cooperation_is_unchanged():
    result = regression_checks(_evaluation(), _evaluation(tiros_por_min=1.29, goles_por_min=.49))
    assert result['regressed_metrics']['tiros_por_min']
    assert result['regressed_metrics']['goles_por_min']
    assert sum(result['regressed_metrics'].values()) == 2
    assert result['material_regression']


def test_offensive_thresholds_are_strict_and_a_single_drop_is_insufficient():
    baseline = _evaluation()
    at_limit = regression_checks(baseline, _evaluation(tiros_por_min=1.3, goles_por_min=.5))
    assert not any(at_limit['regressed_metrics'].values())
    single = regression_checks(baseline, _evaluation(tiros_por_min=1.29))
    assert single['regressed_metrics']['tiros_por_min']
    assert not single['material_regression']


def test_goal_guard_reads_selfplay_context_outside_pass_chain_metrics():
    baseline = _evaluation()
    current = _evaluation(tiros_por_min=1.29)
    del baseline['cadena_pase']['metricas']['goles_por_min']
    del current['cadena_pase']['metricas']['goles_por_min']
    baseline['selfplay'] = dict(goals_per_min=1.0)
    current['selfplay'] = dict(goals_per_min=.49)
    result = regression_checks(baseline, current)
    assert result['regressed_metrics']['goles_por_min']
    assert 'goles_por_min' not in result['missing_metrics']
    assert result['material_regression']


@pytest.mark.parametrize('point,upper,valid,regression', [
    (.39, .49, True, True),
    (.40, .49, True, False),
    (.39, .50, True, False),
    (.39, .49, False, False),
    (None, None, False, False),
    (None, .49, True, False),
    (.39, None, True, False),
])
def test_fixed_parent_requires_complete_loss_and_confidence(point, upper, valid, regression):
    result = regression_checks(_evaluation(), _evaluation(), _match(point, upper, valid))
    assert result['fixed_parent_regression'] is regression
    assert result['material_regression'] is regression


def test_no_fixed_parent_match_does_not_create_strength_evidence():
    result = regression_checks(_evaluation(), _evaluation(), None)
    assert not result['fixed_parent_regression']
    assert not result['material_regression']


def test_adaptive_evaluation_uses_dev_and_all_three_frozen_references(tmp_path):
    a = _args(tmp_path)
    command = evaluation_command(a, 700)
    references = [command[i + 1] for i, value in enumerate(command) if value == '--reference']
    assert references == [a.parent600, a.init, a.original]
    assert _option(command, '--split') == 'dev'
    assert _option(command, '--bc') == a.bc
    assert _option(command, '--seed') == '6007'
    assert _option(command, '--policy') == str(tmp_path / 'curriculum/last.pt')
    assert _option(command, '--out') == str(tmp_path / 'evaluation_dev_00700.json')


@pytest.mark.parametrize('stage', [2, 3, 4])
def test_unchanged_extension_preserves_mastery_at_every_remaining_stage(stage):
    parent = _checkpoint()
    parent['update'] = 800
    state = parent['coop_state']['sanguchito_rs_x4']
    state.update(stage=stage, window=[1, 1, 0, 1], mastery_streak=1,
                 stage_started_learning_update=623)
    before = copy.deepcopy(parent)
    resumed = resumed_checkpoint(parent, reset_mastery=False)
    _same_state(parent, before)
    _same_state(resumed['coop_state'], before['coop_state'])
    for key in ('model', 'critic', 'opt_pi', 'opt_v', 'vnorm', 'update',
                'decisions', 'args', 'pool_names'):
        _same_state(resumed[key], before[key])
    assert resumed['update'] == 800
    # A new evaluation series starts independently of curriculum mastery.
    assert resumed['baseline'] is None
    assert resumed['best_score'] == resumed['best_pass'] == -1.0
    assert resumed['drift_count'] == resumed['pass_ok_streak'] == 0
    resumed['coop_state']['sanguchito_rs_x4']['window'].append(0)
    _same_state(parent, before)


@pytest.mark.parametrize('parent_update,additional_updates,expected', [
    (800, 400, [900, 1000, 1100, 1200]),
    (600, 200, [700, 800]),
])
def test_extension_targets_are_absolute_checkpoints(parent_update, additional_updates, expected):
    assert extension_targets(parent_update, additional_updates) == expected


def test_extension_dev_evaluation_protects_both_600_and_800(tmp_path):
    a = _args(tmp_path)
    a.parent_checkpoint = 'references/parent800.pt'
    command = evaluation_command(a, 900)
    references = [command[i + 1] for i, value in enumerate(command) if value == '--reference']
    assert references == [a.parent600, a.parent_checkpoint, a.init, a.original]
    assert len(references) == len(set(references))
    assert _option(command, '--split') == 'dev'
    assert _option(command, '--out') == str(tmp_path / 'evaluation_dev_00900.json')


def test_dev_references_are_unique_when_parent_is_600_and_initial_equals_original(tmp_path):
    a = _args(tmp_path)
    a.parent_checkpoint = a.parent600
    a.original = a.init
    command = evaluation_command(a, 700)
    references = [command[i + 1] for i, value in enumerate(command) if value == '--reference']
    assert references == [a.parent600, a.init]
    assert _option(command, '--split') == 'dev'
