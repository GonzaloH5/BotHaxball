"""Keep the registered experiment bounded, comparable and free of curriculum rewards."""
import json
from pathlib import Path

import pytest

from learn.x4_coop_ab import BRANCHES, branch_command, evaluation_command, parse_args, training_audit
from learn.x4_ppo import parse_args as ppo_args


def config(tmp_path):
    return parse_args(['--source800', 'parent.pt', '--ref200', 'ref.pt', '--bc', 'bc.pt',
                       '--original', 'original.pt', '--root', str(tmp_path)])


def test_one_treatment_only_and_exact_actor_budget(tmp_path):
    a = config(tmp_path)
    cmds = [branch_command(a, branch) for branch in BRANCHES]
    cfgs = [ppo_args(cmd[3:]) for cmd in cmds]
    assert cfgs[0].human_open_sampler == 'uniform'
    assert cfgs[1].human_open_sampler == 'pressure_pass'
    for cfg in cfgs:
        assert cfg.updates - cfg.critic_warmup == 200
        assert cfg.critic_warmup == 10
        assert cfg.human_restore_history and cfg.human_starts == .4 and cfg.human_restart_frac == .3
        assert cfg.coop_frac == 0 and cfg.pass_bonus == 0 and cfg.shaping_kind == 'none'
        assert cfg.style_coef == cfg.auto_pass_arm_update == cfg.eval_every == 0
        assert cfg.seed == 71 and not cfg.resume
    x, y = [dict(vars(cfg)) for cfg in cfgs]
    for obj in (x, y):
        for key in ('out', 'human_open_sampler'):
            obj.pop(key)
        for key in ('delay_values', 'delay_weights'):
            obj[key] = obj[key].tolist()
    assert x == y
    assert '--parent800' in evaluation_command(a) and '--ref200' in evaluation_command(a)


def test_pressure_mode_cannot_silently_use_empty_bank_or_history(tmp_path):
    with pytest.raises(SystemExit):
        ppo_args(['--bc', 'unused', '--out', str(tmp_path), '--human-open-sampler', 'pressure_pass'])
    with pytest.raises(SystemExit):
        ppo_args(['--bc', 'unused', '--out', str(tmp_path), '--human-open-sampler', 'pressure_pass',
                  '--human-restore-history', '--bank-recordings', '0'])


def test_registered_wall_budget_cannot_be_extended(tmp_path):
    args = config(tmp_path).original_argv
    with pytest.raises(SystemExit):
        parse_args(args + ['--wall-hours', '2.01'])


def test_training_audit_requires_all_actor_updates_and_no_bonus(tmp_path):
    rows = [dict(update=i+1, critic_only=i<10, nonfinite_skipped=0) for i in range(12)]
    (tmp_path/'last.pt').write_bytes(b'checkpoint')
    path = tmp_path/'log.jsonl'
    path.write_text('\n'.join(json.dumps(r) for r in rows))
    assert training_audit(tmp_path, 2)['actor_updates'] == 2
    rows[-1]['critic_only'] = True
    path.write_text('\n'.join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError, match='actor update'):
        training_audit(tmp_path, 2)
    rows[-1]['critic_only'] = False
    rows[-1]['pass_bonus_abs'] = .1
    path.write_text('\n'.join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError, match='bonus contaminated'):
        training_audit(tmp_path, 2)
