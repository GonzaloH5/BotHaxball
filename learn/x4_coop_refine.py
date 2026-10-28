"""Refine a preserved cooperative checkpoint with persistent skill practice.

Local update 0 is the preserved parent's update 200. Warm-start actor, critic,
and value normalization; reset optimizer momentum and curriculum mastery.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

from learn.x4_coop_run import ROOT, Run, atomic_json, common_args, read_rows


def training_command(a, updates):
    command = common_args(a)
    command[command.index('--critic-warmup') + 1] = '10'
    command[command.index('--pool') + 1] = a.init + ',' + a.original
    return [sys.executable, '-m', 'learn.x4_ppo', *command,
            '--out', str(Path(a.root) / 'curriculum'), '--updates', str(updates), '--resume',
            '--coop-frac', '0.5', '--coop-start-stage', '1', '--coop-min-episodes', '512',
            '--coop-threshold', '0.4', '--coop-mastery-windows', '2',
            '--coop-min-stage-updates', '100', '--coop-transfer-frac', '0.5',
            '--coop-replay-frac', '0.5', '--coop-possession-retention',
            '--coop-bank-limit', '6000']


def regression_checks(baseline, current):
    # Coarse guard against persistent degradation, not a significance test.
    before = baseline['cadena_pase']['metricas']
    after = current['cadena_pase']['metricas']
    limits = dict(pases_por_min=('ratio', .8), precision_pase=('drop', .05),
                  retencion_tras_recibir=('drop', .08), retencion_bajo_presion=('drop', .10))
    checks = {}
    for metric, (kind, margin) in limits.items():
        x, y = before[metric][0], after[metric][0]
        checks[metric] = bool(y < x * margin if kind == 'ratio' else y < x - margin)
    return dict(regressed_metrics=checks, material_regression=sum(checks.values()) >= 2,
                limitation='Requires two consecutive evaluations; does not certify improvement.')


def evaluation_command(a, update, split='dev'):
    return [sys.executable, '-m', 'learn.x4_coop_eval',
            '--policy', str(Path(a.root) / 'curriculum/last.pt'),
            '--reference', a.init, '--reference', a.original, '--reference', a.control,
            '--bc', a.bc, '--matches', '8' if split == 'dev' else '32',
            '--minutes', '2' if split == 'dev' else '3',
            '--probe-states', '64' if split == 'dev' else '128',
            '--recordings', '30' if split == 'dev' else '40',
            '--device', a.device, '--threads', str(a.threads), '--split', split,
            '--seed', str(a.seed + (5000 if split == 'dev' else 10000)),
            '--out', str(Path(a.root) / f'evaluation_{split}_{update:05d}.json')]


class RefineRun(Run):
    def main(self):
        a = self.a
        sources = dict(vars(a))
        refs = self.root / 'references'
        refs.mkdir()
        fingerprints = {}
        for name in ('init', 'bc', 'original', 'control'):
            source = Path(getattr(a, name)).resolve()
            dest = refs / (name + '.pt')
            shutil.copy2(source, dest)
            fingerprints[name] = dict(source=str(source), frozen_copy=str(dest),
                                     sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
            setattr(a, name, str(dest))
        import torch
        parent = torch.load(a.init, map_location='cpu', weights_only=True)
        assert parent['update'] == 200, 'must start from the preserved update-200 checkpoint'
        files = ['learn/x4_ppo.py', 'learn/x4_coop_curriculum.py', 'learn/x4_pass_macro.py',
                 'learn/x4_coop_refine.py', 'learn/x4_coop_eval.py', 'learn/x4_macro_trace.py',
                 'env/rs4z/core.py', 'env/rs4z/kernel.py', 'learn/x4_policy.py']
        atomic_json(self.root / 'manifest.json', dict(
            arguments=vars(a), source_arguments=sources, frozen_checkpoints=fingerprints,
            parent_update=200, parent_decisions=parent['decisions'],
            additional_updates=a.updates, additional_environment_decisions=a.envs*a.rollout*a.updates,
            code_sha256={f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files},
            optimizer_state='reset', curriculum_state='reset to execution stage 1',
            training_command=training_command(a, a.updates),
            mastery='100 actor updates per stage, two disjoint 512-episode windows with Wilson lower bound >= 40%',
            automatic_export=False, automatic_deployment=False))
        self.state['parent_update'] = 200
        self.state['cooperative_guards'] = []
        self.save()
        streak = 0
        for target in range(100, a.updates + 1, 100):
            self.execute(f'training_{target}', {'curriculum': training_command(a, target)})
            logs = read_rows(self.root / 'curriculum/log.jsonl')
            evs = read_rows(self.root / 'curriculum/eval.jsonl')
            if (logs[-1]['update'] != target or evs[-1]['update'] != target
                    or any(r.get('nonfinite_skipped', 0) or r.get('restored_nonfinite') for r in logs)):
                raise RuntimeError('incomplete/nonfinite training phase')
            guard = dict(update=target, parent_equivalent_update=200+target,
                         **regression_checks(evs[0], evs[-1]))
            streak = streak + 1 if guard['material_regression'] else 0
            guard['consecutive_regressions'] = streak
            self.state['cooperative_guards'].append(guard)
            self.save()
            self.execute(f'evaluation_dev_{target}', {'evaluation_dev': evaluation_command(a, target)})
            if streak >= 2:
                self.state.update(status='stopped_on_cooperative_regression', stopped_at=target)
                self.save()
                return 2
        self.execute('evaluation_test', {'evaluation_test': evaluation_command(a, a.updates, 'test')})
        self.state.update(status='finished', finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        self.save()
        return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'init', 'bc', 'original', 'control'):
        ap.add_argument('--'+name, required=True)
    ap.add_argument('--updates', type=int, default=600, help='additional updates after parent 200')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--envs', type=int, default=512)
    ap.add_argument('--rollout', type=int, default=128)
    ap.add_argument('--threads', type=int, default=5)
    ap.add_argument('--recordings', type=int, default=80)
    ap.add_argument('--eval-every', type=int, default=100)
    ap.add_argument('--wall-hours', type=float, default=3)
    ap.add_argument('--seed', type=int, default=7)
    a = ap.parse_args()
    if (a.updates < 100 or a.updates % 100 or a.eval_every != 100
            or min(a.envs, a.rollout, a.threads, a.recordings) < 1 or a.wall_hours <= 0):
        ap.error('invalid bounded refinement budget')
    if Path(a.root).exists() and any(Path(a.root).iterdir()):
        ap.error('refinement directory must be empty')
    run = RefineRun(a)
    try:
        return run.main()
    except Exception as e:
        run.state.update(status='failed', error=f'{type(e).__name__}: {e}')
        run.save()
        raise


if __name__ == '__main__':
    raise SystemExit(main())
