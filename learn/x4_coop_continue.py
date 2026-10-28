"""Bounded continuation with cooperative finishing practice and frozen rivals.

Preserves actor, critic, Adam and league. Changed task criteria start fresh
mastery evidence; unchanged criteria retain it. Development evaluation controls
continuation; no test reuse.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time

from learn.x4_coop_refine import training_command
from learn.x4_coop_run import ROOT, Run, atomic_json, read_rows


def continuation_command(a, update):
    command = training_command(a, update)
    command[command.index('--coop-start-stage') + 1] = '2'
    command[command.index('--coop-replay-frac') + 1] = '0.25'
    return command + ['--coop-finish-frac', '0.25', '--coop-reward-cooperative-goals']


def resumed_checkpoint(parent, reset_mastery=True):
    """Start a new evaluation baseline; retain valid curriculum evidence."""
    ck = copy.deepcopy(parent)
    ck.update(baseline=None, best_score=-1.0, best_pass=-1.0,
              drift_count=0, pass_ok_streak=0)
    if reset_mastery:
        for state in ck['coop_state'].values():
            state.update(window=[], mastery_streak=0,
                         stage_started_learning_update=max(0, int(ck['update'])
                                                           - int(ck['args']['critic_warmup'])))
    return ck


def extension_targets(parent_update, additional_updates):
    if (parent_update < 600 or parent_update % 100 or additional_updates < 100
            or additional_updates % 100):
        raise ValueError('continuation must use complete 100-update boundaries')
    return list(range(parent_update + 100, parent_update + additional_updates + 1, 100))


def regression_checks(baseline, current, parent_match=None):
    before = baseline['cadena_pase']['metricas']
    after = current['cadena_pase']['metricas']
    limits = dict(pases_por_min=('ratio', .8), precision_pase=('drop', .05),
                  retencion_tras_recibir=('drop', .08), retencion_bajo_presion=('drop', .10),
                  tiros_por_min=('ratio', .65), goles_por_min=('ratio', .5))
    checks, missing = {}, []
    for metric, (kind, margin) in limits.items():
        x, y = before.get(metric, [None])[0], after.get(metric, [None])[0]
        if metric == 'goles_por_min':
            # Goals are context, outside the pass-chain gate's METRICS.
            x = baseline.get('selfplay', {}).get('goals_per_min', x)
            y = current.get('selfplay', {}).get('goals_per_min', y)
        if x is None or y is None or not (math.isfinite(x) and math.isfinite(y)):
            missing.append(metric)
            checks[metric] = False
        else:
            checks[metric] = bool(y < x * margin if kind == 'ratio' else y < x - margin)
    rival_regression = False
    if parent_match is not None:
        value = parent_match['point_fraction']['valor']
        upper = parent_match['point_fraction']['ic90'][1]
        rival_regression = bool(parent_match.get('valid_for_strength') and value is not None
                                and upper is not None and value < .4 and upper < .5)
    return dict(regressed_metrics=checks, missing_metrics=missing,
                fixed_parent_regression=rival_regression,
                material_regression=sum(checks.values()) >= 2 or rival_regression,
                limitation='Two consecutive dev evaluations stop continuation; this does not certify improvement.')


def evaluation_command(a, update):
    references = list(dict.fromkeys([a.parent600, getattr(a, 'parent_checkpoint', a.parent600),
                                    a.init, a.original]))
    reference_args = [part for name in references for part in ('--reference', name)]
    return [sys.executable, '-m', 'learn.x4_coop_eval',
            '--policy', str(Path(a.root) / 'curriculum/last.pt'),
            *reference_args,
            '--bc', a.bc, '--matches', '8', '--minutes', '2', '--probe-states', '64',
            '--recordings', '30', '--device', a.device, '--threads', str(a.threads),
            '--split', 'dev', '--seed', str(a.seed + 6000),
            '--out', str(Path(a.root) / f'evaluation_dev_{update:05d}.json')]


class ContinueRun(Run):
    def main(self):
        import torch
        a = self.a
        source_run = Path(a.source_run).resolve()
        source = source_run / 'curriculum/last.pt'
        parent = torch.load(source, map_location='cpu', weights_only=True)
        targets = extension_targets(int(parent['update']), a.updates)
        if not parent.get('coop_state'):
            raise ValueError('requires a cooperative curriculum checkpoint')
        if any(not 2 <= s['stage'] <= 4 for s in parent['coop_state'].values()):
            raise ValueError('continuation requires pressure, finishing or transfer stage')
        if a.envs != parent['args']['envs'] or a.rollout != parent['args']['rollout']:
            raise ValueError('preserve the parent environment and rollout sizes')
        reset_mastery = not parent['args'].get('coop_reward_cooperative_goals', False)
        expected = dict(coop_frac=.5, coop_min_episodes=512, coop_threshold=.4,
                        coop_bank_limit=6000, coop_transfer_frac=.5, coop_replay_frac=.25,
                        coop_mastery_windows=2, coop_min_stage_updates=100,
                        coop_possession_retention=True, coop_finish_frac=.25,
                        critic_warmup=10, bank_recordings=a.recordings)
        if not reset_mastery:
            mismatches = {k: parent['args'].get(k) for k, v in expected.items()
                          if parent['args'].get(k) != v}
            if mismatches:
                raise ValueError(f'cannot retain mastery with changed criteria: {mismatches}')
        source_manifest = json.loads((source_run / 'manifest.json').read_text(encoding='utf-8'))
        for name in ('init', 'bc', 'original'):
            setattr(a, name, parent['args'][name] if name in ('init', 'bc')
                    else source_manifest['arguments'][name])
            if not Path(getattr(a, name)).is_file():
                raise FileNotFoundError(getattr(a, name))
        refs = self.root / 'references'
        refs.mkdir()
        parent_update = int(parent['update'])
        frozen_parent = refs / f'parent{parent_update}.pt'
        shutil.copy2(source, frozen_parent)
        frozen_parent.chmod(0o444)
        a.parent_checkpoint = str(frozen_parent)
        if parent_update == 600:
            a.parent600 = str(frozen_parent)
        else:
            anchor = Path(source_manifest['arguments']['parent600'])
            if not anchor.is_file():
                raise FileNotFoundError(anchor)
            frozen600 = refs / 'parent600.pt'
            shutil.copy2(anchor, frozen600)
            frozen600.chmod(0o444)
            a.parent600 = str(frozen600)
        # Keep the original baseline as well as the new parent baseline, so
        # repeatedly extending a run cannot hide cumulative cooperation loss.
        anchor_baseline_path = source_run / 'references/baseline600.json'
        if anchor_baseline_path.exists():
            baseline600 = json.loads(anchor_baseline_path.read_text(encoding='utf-8'))
        else:
            baseline600 = next(ev for ev in read_rows(source_run / 'curriculum/eval.jsonl')
                               if ev['update'] == 600)
        atomic_json(refs / 'baseline600.json', baseline600)
        (refs / 'baseline600.json').chmod(0o444)
        output = self.root / 'curriculum'
        output.mkdir()
        torch.save(resumed_checkpoint(parent, reset_mastery=reset_mastery), output / 'last.pt')
        # Preserve the opponent league rather than silently dropping snapshots.
        for name in parent.get('pool_names', []):
            filename = name + '.pt'
            if Path(filename).name != filename:
                raise ValueError('invalid checkpoint league filename')
            shutil.copy2(source_run / 'curriculum' / filename, output / filename)
        files = ['learn/x4_ppo.py', 'learn/x4_coop_curriculum.py', 'learn/x4_pass_macro.py',
                 'learn/x4_coop_continue.py', 'learn/x4_coop_eval.py', 'learn/x4_macro_trace.py',
                 'env/rs4z/core.py', 'env/rs4z/kernel.py', 'learn/x4_policy.py']
        atomic_json(self.root / 'manifest.json', dict(
            arguments=vars(a), source_run=str(source_run), parent_update=parent_update,
            parent_sha256=hashlib.sha256(frozen_parent.read_bytes()).hexdigest(),
            frozen600_sha256=hashlib.sha256(Path(a.parent600).read_bytes()).hexdigest(),
            preserved_200_sha256=hashlib.sha256(Path(a.init).read_bytes()).hexdigest(),
            additional_updates=a.updates, final_update=parent_update+a.updates,
            additional_environment_decisions=a.envs*a.rollout*a.updates,
            code_sha256={f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files},
            training_command=continuation_command(a, parent_update+a.updates),
            initial_mastery_stages={name: state['stage'] for name, state in parent['coop_state'].items()},
            task_mix_by_stage={
                '2': dict(full_match=.5, stage1=.125, stage2=.25, stage3=.125),
                '3': dict(full_match=.5, stage1=.0625, stage2=.0625, stage3=.375),
                '4': dict(full_match=.5, stage1=.0625, stage2=.0625, stage4=.375)},
            preserved_state=['actor', 'critic', 'Adam', 'value normalization', 'league', 'update']
                            + ([] if reset_mastery else ['curriculum mastery evidence and dwell clock']),
            reset_state=['evaluation baseline and records']
                        + (['mastery window and streak', 'stage dwell clock'] if reset_mastery else []),
            reset_mastery=reset_mastery,
            criterion='Own goals after a qualifying pass count; finishing requires a new own kick after the inherited queue.',
            mastery='100 actor updates per stage; two disjoint 512-episode windows with Wilson lower bound >= 40%; finishing rows cannot promote stage 2.',
            adaptive_evaluation_split='dev', final_test_reused=False,
            automatic_export=False, automatic_deployment=False))
        self.state.update(parent_update=parent_update, additional_updates=a.updates,
                          final_update=parent_update+a.updates, guards=[])
        self.save()
        streak = 0
        for target in targets:
            self.execute(f'training_{target}', {'curriculum': continuation_command(a, target)})
            logs = read_rows(output / 'log.jsonl')
            evs = read_rows(output / 'eval.jsonl')
            if (logs[-1]['update'] != target or evs[-1]['update'] != target
                    or any(r.get('nonfinite_skipped', 0) or r.get('restored_nonfinite') for r in logs)):
                raise RuntimeError('incomplete/nonfinite continuation')
            self.execute(f'evaluation_dev_{target}', {'evaluation_dev': evaluation_command(a, target)})
            report = json.loads((self.root / f'evaluation_dev_{target:05d}.json').read_text(encoding='utf-8'))
            candidate = str(output / 'last.pt')
            if not all(row.get('valid_for_strength') for row in report['matches'][candidate].values()):
                raise RuntimeError('incomplete dev matches; continuation guard has no valid strength evidence')
            match = report['matches'][candidate][a.parent_checkpoint]
            guard = dict(update=target, **regression_checks(evs[0], evs[-1], match))
            guard['cumulative_from600'] = regression_checks(
                baseline600, evs[-1], report['matches'][candidate][a.parent600])
            guard['material_regression'] |= guard['cumulative_from600']['material_regression']
            streak = streak + 1 if guard['material_regression'] else 0
            guard['consecutive_regressions'] = streak
            self.state['guards'].append(guard)
            self.save()
            if streak >= 2:
                self.state.update(status='stopped_on_regression', stopped_at=target)
                self.save()
                return 2
        self.state.update(status='finished', finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        self.save()
        return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-run', required=True)
    ap.add_argument('--root', required=True)
    ap.add_argument('--updates', type=int, default=200, help='additional updates after the source checkpoint')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--envs', type=int, default=512)
    ap.add_argument('--rollout', type=int, default=128)
    ap.add_argument('--threads', type=int, default=5)
    ap.add_argument('--recordings', type=int, default=80)
    ap.add_argument('--eval-every', type=int, default=100)
    ap.add_argument('--wall-hours', type=float, default=2)
    ap.add_argument('--seed', type=int, default=7)
    a = ap.parse_args()
    if (a.updates < 100 or a.updates % 100 or a.eval_every != 100
            or min(a.envs, a.rollout, a.threads, a.recordings) < 1 or a.wall_hours <= 0):
        ap.error('invalid bounded continuation budget')
    if Path(a.root).exists() and any(Path(a.root).iterdir()):
        ap.error('continuation directory must be empty')
    run = ContinueRun(a)
    try:
        return run.main()
    except Exception as e:
        run.state.update(status='failed', error=f'{type(e).__name__}: {e}')
        run.save()
        raise


if __name__ == '__main__':
    raise SystemExit(main())
