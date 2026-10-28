"""Bounded, sequential A/B from update 800: only human open-start sampling differs."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

from learn.x4_coop_run import ROOT, Run, atomic_json, read_rows

PARENT_HASH = '530c28cafaa1131a2435067f29ba8c8d53d0c1c501f2b15bb2bd10e4d83665c9'
MACRO_HASH = '1e15b9d9fc460467dc44ec5fb9354ee5a2d098f8c43e96d216d27e997e8ee4f5'
BRANCHES = ('A_uniform', 'B_pressure')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def branch_command(a, branch):
    if branch not in BRANCHES:
        raise ValueError('unknown A/B branch')
    refs = Path(a.root) / 'references'
    return [sys.executable, '-m', 'learn.x4_ppo',
            '--bc', str(refs / 'bc.pt'), '--init', str(refs / 'parent800.pt'),
            '--out', str(Path(a.root) / branch), '--device', a.device,
            '--envs', str(a.envs), '--rollout', str(a.rollout), '--threads', str(a.threads),
            '--updates', str(a.actor_updates + 10), '--critic-warmup', '10',
            '--epochs', '5', '--minibatches', '4', '--lr', '0.0003', '--critic-lr', '0.001',
            '--lambda-dist', '0.05', '--lambda-decay', '1', '--lambda-min', '0.05',
            '--gamma', '0.999', '--gae', '0.98', '--ent', '0.002',
            '--human-starts', '0.4', '--human-restart-frac', '0.3', '--human-restore-history',
            '--human-open-sampler', 'uniform' if branch == BRANCHES[0] else 'pressure_pass',
            '--bank-recordings', str(a.recordings), '--pool-frac', '0.5',
            '--pool', str(refs / 'ref200.pt') + ',' + str(refs / 'original.pt'),
            '--delays', '8:1,9:1,10:1,11:1', '--delay-feature', 'mixed',
            '--eval-every', '0', '--dev-recordings', '0', '--rsone-every', '0',
            '--snapshot-every', '100', '--pool-max', '6', '--no-keep-evals',
            '--shaping-kind', 'none', '--style-coef', '0', '--pass-bonus', '0',
            '--auto-pass-arm-update', '0', '--forfeit-reasons', '1,2,3,4', '--stop-on-drift', '0',
            '--pass-ref-key', 'sanguchito_train', '--pass-macro', '--pass-ready', '--macro-trace',
            '--coop-frac', '0', '--seed', str(a.seed)]


def evaluation_command(a):
    root, refs = Path(a.root), Path(a.root) / 'references'
    return [sys.executable, '-m', 'learn.x4_coop_ab_eval',
            '--a', str(root / BRANCHES[0] / 'last.pt'), '--b', str(root / BRANCHES[1] / 'last.pt'),
            '--parent800', str(refs / 'parent800.pt'), '--ref200', str(refs / 'ref200.pt'),
            '--device', a.device, '--threads', str(a.threads),
            '--out', str(root / 'evaluation.json'), '--replay-out', str(root / 'replays.html')]


def training_audit(path, actor_updates):
    rows = read_rows(Path(path) / 'log.jsonl')
    target = actor_updates + 10
    problems = []
    if len(rows) != target or not rows or rows[-1]['update'] != target:
        problems.append('incomplete update budget')
    if sum(not row['critic_only'] for row in rows) != actor_updates:
        problems.append('incorrect actor update count')
    if sum(row['critic_only'] for row in rows) != 10:
        problems.append('incorrect critic warmup count')
    if any(row.get('nonfinite_skipped') or row.get('restored_nonfinite') for row in rows):
        problems.append('nonfinite update')
    if any(row.get('coop_decisions', 0) or row.get('coop_actor_samples', 0)
           or row.get('pass_bonus_abs', 0) or row.get('shaping_abs', 0) for row in rows):
        problems.append('exercise or bonus contaminated full-match training')
    if problems:
        raise ValueError('; '.join(problems))
    return dict(total_updates=target, actor_updates=actor_updates, critic_only_updates=10,
                final_checkpoint_sha256=sha256(Path(path) / 'last.pt'),
                human_start_counts=rows[-1].get('human_start_counts', {}), nonfinite_updates=0)


def state_hash(state):
    h = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        h.update(key.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def preflight(a):
    """Validate the real train bank and both fresh optimizers before any branch runs."""
    import numpy as np
    import torch
    from unittest.mock import patch
    from env.rs4z import obs_v3
    from learn import x4_data as XD
    from learn.x4_pressure_starts import PressureStartBank
    from learn.x4_ppo import Trainer, parse_args as ppo_args

    torch.set_num_threads(a.threads)
    cfg = ppo_args(branch_command(a, BRANCHES[0])[3:])
    names = XD.split_names(cfg.splits, 'train', None, ('sanguchito_rs_x4',))[:a.recordings]
    data = XD.load(names, maps=('sanguchito_rs_x4',))
    if not set(data.names).issubset(set(names)):
        raise ValueError('bank includes recordings outside the train selection')
    pressure = PressureStartBank(data, seed=a.seed)
    parent = torch.load(Path(a.root) / 'references/parent800.pt', map_location='cpu', weights_only=False)
    expected_initial = dict(actor=state_hash(parent['model']), critic=state_hash(parent['critic']),
                            vnorm=[float(x) for x in parent['vnorm']],
                            fresh_actor_optimizer=True, fresh_critic_optimizer=True)
    del parent
    print(json.dumps(dict(event='bank_preflight', manifest=pressure.manifest)), flush=True)
    records = {}
    # Share read-only data during this diagnostic only; real branches construct
    # their banks independently from the same frozen split and configuration.
    with patch.object(XD, 'load', return_value=data), \
            patch('learn.x4_pressure_starts.PressureStartBank', return_value=pressure):
        for branch in BRANCHES:
            tr = Trainer(ppo_args(branch_command(a, branch)[3:] + ['--out', str(Path(a.root) / 'preflight' / branch)]))
            try:
                ar = tr.arenas[0]
                initial = dict(actor=state_hash(tr.policy.state_dict()),
                               critic=state_hash(tr.critic.state_dict()),
                               vnorm=[float(tr.vnorm.mean), float(tr.vnorm.sq), float(tr.vnorm.w)],
                               fresh_actor_optimizer=not tr.opt_pi.state,
                               fresh_critic_optimizer=not tr.opt_v.state)
                if not initial['fresh_actor_optimizer'] or not initial['fresh_critic_optimizer']:
                    raise ValueError('optimizers were inherited')
                if initial != expected_initial:
                    raise ValueError('actor, critic or normalization differs from the selected parent800')
                if ar.coop is not None or not ar.env.active.all():
                    raise ValueError('preflight does not contain complete 4v4 rosters')
                if not np.isfinite(obs_v3.observe(ar.env, delay=ar.delay_obs)).all():
                    raise ValueError('nonfinite initial observations')
                # A real rollout and actor/critic update at the planned batch size
                # catch queue indexing, masks, memory and finite-gradient failures.
                buf, stats = tr.rollout()
                info = tr.learn(buf, critic_only=False)
                if stats['coop_decisions'] or stats['pass_bonus'] or stats['shaping']:
                    raise ValueError('preflight reward is not the declared full-match reward')
                if info.get('nonfinite_skipped') or not all(
                        bool(torch.isfinite(p).all()) for p in list(tr.policy.parameters()) + list(tr.critic.parameters())):
                    raise ValueError('nonfinite preflight update')
                records[branch] = dict(initial=initial, actual_batch=dict(envs=a.envs, rollout=a.rollout),
                                       actor_samples=info['samples'], human_start_counts=dict(tr.bank.start_counts),
                                       finite_update=True, full_rosters=True)
                print(json.dumps(dict(event='branch_preflight', branch=branch, result=records[branch])), flush=True)
            finally:
                tr.log.close()
                del tr
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    if records[BRANCHES[0]]['initial'] != records[BRANCHES[1]]['initial']:
        raise ValueError('A/B do not start with identical actor, critic, normalization and fresh Adam')
    return dict(passed=True, bank=pressure.manifest, branches=records, split='train',
                limitation='Preflight validates feasibility; it does not establish improvement.')


class ABRun(Run):
    def main(self):
        a = self.a
        refs = self.root / 'references'
        refs.mkdir()
        fingerprints = {}
        for key, source in (('parent800', a.source800), ('ref200', a.ref200), ('bc', a.bc), ('original', a.original)):
            source = Path(source).resolve()
            target = refs / (key + '.pt')
            shutil.copy2(source, target)
            target.chmod(0o444)
            fingerprints[key] = dict(source=str(source), path=str(target), sha256=sha256(target))
        if fingerprints['parent800']['sha256'] != PARENT_HASH:
            raise ValueError('the selected source is not the declared update-800 checkpoint')
        if sha256(ROOT / 'learn/x4_pass_macro.py') != MACRO_HASH:
            raise ValueError('the corrected pass controller differs from the declared version')
        files = ('learn/x4_ppo.py', 'learn/x4_pressure_starts.py', 'learn/x4_coop_ab.py',
                 'learn/x4_coop_ab_eval.py', 'learn/x4_pass_macro.py', 'tools/x4_html_replay.py',
                 'env/rs4z/core.py', 'env/rs4z/kernel.py', 'tools/x4_pass_chain.py')
        manifest = dict(version='x4-pressure-start-ab-1', arguments=vars(a), references=fingerprints,
                        code_sha256={f: sha256(ROOT / f) for f in files},
                        branches={b: branch_command(a, b) for b in BRANCHES},
                        evaluation=evaluation_command(a), parent_update=800, local_actor_updates=a.actor_updates,
                        optimizer_state='fresh in both branches', human_start_fraction=.4,
                        human_restart_fraction=.3, pressure_fraction_within_open_human_starts=.5,
                        training_split='train', evaluation_seeds=[73001, 73002, 73003, 73004],
                        final_checkpoint_only=True, automatic_continuation=False, automatic_deployment=False,
                        wall_budget_hours=a.wall_hours)
        atomic_json(self.root / 'manifest.json', manifest)
        self.execute('preflight', {'preflight': [sys.executable, '-m', 'learn.x4_coop_ab',
                     *a.original_argv, '--preflight-only']})
        if not json.loads((self.root / 'preflight.json').read_text())['passed']:
            raise ValueError('preflight failed')
        if a.only_preflight:
            self.state.update(status='preflight_complete')
            self.save()
            return
        audit = {}
        for branch in BRANCHES:
            self.execute('training_' + branch, {branch: branch_command(a, branch)})
            audit[branch] = training_audit(self.root / branch, a.actor_updates)
            self.state['training_audit'] = audit
            self.save()
        self.execute('final_evaluation', {'evaluation': evaluation_command(a)})
        report = json.loads((self.root / 'evaluation.json').read_text())
        self.state.update(status='finished', decision=report['decision'], training_audit=audit)
        atomic_json(self.root / 'decision.json', report['decision'])
        self.save()


def parse_args(argv=None):
    raw = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('source800', 'ref200', 'bc', 'original', 'root'):
        ap.add_argument('--' + name, required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--envs', type=int, default=512)
    ap.add_argument('--rollout', type=int, default=128)
    ap.add_argument('--threads', type=int, default=5)
    ap.add_argument('--recordings', type=int, default=80)
    ap.add_argument('--seed', type=int, default=71)
    ap.add_argument('--actor-updates', type=int, default=200)
    ap.add_argument('--wall-hours', type=float, default=2.0)
    ap.add_argument('--preflight-only', action='store_true', help=argparse.SUPPRESS)
    ap.add_argument('--only-preflight', action='store_true', help='run the full batch preflight and stop')
    a = ap.parse_args(raw)
    if min(a.envs, a.rollout, a.threads, a.recordings, a.actor_updates) < 1 or not 0 < a.wall_hours <= 2:
        ap.error('invalid batch or wall-clock budget')
    a.original_argv = raw
    return a


def main(argv=None):
    a = parse_args(argv)
    if a.preflight_only:
        atomic_json(Path(a.root) / 'preflight.json', preflight(a))
        return
    if (Path(a.root) / 'manifest.json').exists() or (Path(a.root) / 'run_state.json').exists():
        raise SystemExit('A/B output already used: select a new directory; no automatic resume')
    run = ABRun(a)
    try:
        run.main()
    except Exception as error:
        run.state.update(status='budget_exhausted' if isinstance(error, TimeoutError) else 'failed',
                         error=str(error), decision=dict(status='not_demonstrated', reason=str(error)))
        run.save()
        atomic_json(run.root / 'decision.json', run.state['decision'])
        raise


if __name__ == '__main__':
    main()
