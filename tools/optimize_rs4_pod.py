"""Measure fixed-learning runtime profiles on disposable copies of a stopped RS4 run."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PROFILES = [(2, 4), (1, 2), (1, 4), (2, 2), (2, 6), (4, 4)]
EVAL_PROFILES = [('cpu', 2, 4), ('cpu', 1, 2), ('cpu', 1, 4), ('cpu', 2, 2), ('cuda', 1, 2), ('cuda', 1, 4)]


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def choose_profile(rows, baseline, key, minimum_gain=.05):
    """Require repeatable improvement and identical inputs, not a lucky best run."""
    reference = rows[baseline]
    eligible = {name: samples for name, samples in rows.items()
                if all(s.get('equivalent', True) for s in samples)
                and len(samples) == len(reference)}
    median = lambda samples: statistics.median(s[key] for s in samples)
    stable = {name:samples for name, samples in eligible.items()
              if median(samples) >= median(reference) * (1 + minimum_gain)
              and all(sample[key] >= original[key] for sample, original in zip(samples, reference))}
    best = max(stable, key=lambda name: median(stable[name])) if stable else baseline
    return dict(profile=best, speed=median(rows[best]), baseline_speed=median(reference),
                gain=median(rows[best]) / median(reference) - 1)


def evaluation_worker(args):
    import numpy as np
    import torch
    from numba import set_num_threads
    from eval.agents import ModelAgent, ScriptedAgent
    from eval.rs4_v3 import MixedTeamAgent, full_game
    torch.set_num_threads(args.torch_threads)
    set_num_threads(args.physics_threads)
    if args.device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
    run = args.run_dir
    learner = ModelAgent(str(run / 'latest.pt'), True, args.device)
    parent = ModelAgent(str(run / 'parent.pt'), True, args.device)
    teacher = ModelAgent(str(run / 'teacher.pt'), True, args.device)

    class Trace:
        def __init__(self, inner):
            self.inner, self.hash = inner, hashlib.sha256()
        def reset(self, env, done=None):
            if hasattr(self.inner, 'reset'):
                self.inner.reset(env, done)
        def record_executed(self, env, actions):
            if hasattr(self.inner, 'record_executed'):
                self.inner.record_executed(env, actions)
        def __call__(self, env, obs, players):
            actions = self.inner(env, obs, players)
            self.hash.update(np.asarray(actions, dtype='<i8').tobytes())
            return actions

    # Compile every physics/model path before timing it.
    full_game(learner, parent, games=args.games, minutes=.01, seed=51)
    full_game(MixedTeamAgent(learner, teacher, (0,)), ScriptedAgent(policy='r3', style=0),
              games=args.games, minutes=.01, seed=51)
    rows, signatures = [], []
    repetitions = []
    for repetition in range(args.repetitions):
        started = time.perf_counter()
        for seed in (51, 73, 91):
            for color in (0, 1):
                for kind in ('scripted', 'neural', 'mixed'):
                    own = MixedTeamAgent(learner, teacher, (0,)) if kind == 'mixed' else learner
                    rival = parent if kind == 'neural' else ScriptedAgent(policy='r3', style=0)
                    own, rival = Trace(own), Trace(rival)
                    torch.manual_seed(seed * 100 + color)
                    row = full_game(own, rival, args.games, args.minutes, seed, color)
                    rows.append(row)
                    signatures.append(own.hash.hexdigest() + rival.hash.hexdigest())
        elapsed = time.perf_counter() - started
        repetitions.append(dict(elapsed_seconds=elapsed, batches_per_second=18 / elapsed))
    atomic_json(args.out, dict(device=args.device, torch_threads=args.torch_threads,
                              physics_threads=args.physics_threads, games=args.games, minutes=args.minutes,
                              action_signatures=signatures, outcomes=rows, repetitions=repetitions))


def run_command(command, log):
    print('RUN', ' '.join(map(str, command)), flush=True)
    with Path(log).open('w', encoding='utf-8') as stream:
        subprocess.run(list(map(str, command)), cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)


def optimize(args):
    import fcntl
    import yaml
    run, out = args.run_dir.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (run / '.runner.lock').open('a+b')
    while True:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if not args.wait:
                raise RuntimeError('Runner active; use --wait to queue benchmarks')
            atomic_json(out / 'status.json', dict(state='waiting_for_runner', updated=time.time()))
            time.sleep(30)
    # Hold the same lock as the runner for the entire measurement/application.
    branch = json.loads((run / 'ledger.json').read_text())['selected']
    if branch not in ('control', 'memory'):
        raise ValueError('A selected RS4 branch is required')
    source = [run / branch / 'latest.pt', run / 'parent.pt', run / 'teacher.pt', run / 'ledger.json']
    hashes = {str(p): digest(p) for p in source}
    snapshot = out / 'snapshot'
    snapshot.mkdir(exist_ok=True)
    for path in source[:3]:
        shutil.copy2(path, snapshot / path.name)
    cfg = yaml.safe_load((run / branch / 'config.yaml').read_text())
    original_config = copy.deepcopy(cfg)
    cfg['bc_reference'] = str(snapshot / 'teacher.pt')
    config = snapshot / 'config.yaml'
    config.write_text(yaml.safe_dump(cfg, sort_keys=False))
    atomic_json(out / 'inputs.json', dict(branch=branch, hashes=hashes, config=original_config,
                                       benchmark_only=True, learning_parameters_unchanged=True))
    atomic_json(out / 'status.json', dict(state='training_profiles', updated=time.time()))
    training = {f'{t}/{n}': [] for t, n in TRAIN_PROFILES}
    summaries = []
    work_path = run / 'hardware_diagnostic_work.json'
    work = json.loads(work_path.read_text()) if work_path.exists() else {'attempts': []}
    maximum = (3 + args.iters) * cfg['ppo']['rollout_len'] * cfg['env']['agents']
    import torch
    from train.rs4_program import ProgramState
    budget_state = ProgramState.from_config(cfg, torch.load(snapshot / 'latest.pt', map_location='cpu', weights_only=False)['rs4_program_state'])
    for repetition in range(args.repetitions):
        profiles = list(TRAIN_PROFILES)
        random.Random(51 + repetition).shuffle(profiles)
        for t, n in profiles:
            name = f'train_{t}_{n}_{repetition}'
            dest = out / (name + '.json')
            attempt = dict(id=str(out / name), charged_steps=maximum, status='reserved')
            copy.deepcopy(budget_state).charge_hardware_diagnostics(sum(item['charged_steps'] for item in work['attempts']) + maximum)
            work['attempts'].append(attempt)
            atomic_json(work_path, work)
            run_command([sys.executable, '-u', '-m', 'tools.benchmark_multitask', '--config', config,
                         '--checkpoint', snapshot / 'latest.pt', '--warmup', '3', '--iters', str(args.iters),
                         '--device', 'cuda', '--torch-threads', t, '--numba-threads', n,
                         '--json-output', dest], out / (name + '.log'))
            value = json.loads(dest.read_text())
            actual = value['diagnostic_ppo_samples']
            if not isinstance(actual, int) or isinstance(actual, bool) or not 0 <= actual <= maximum:
                raise ValueError('Invalid benchmark sample count')
            attempt.update(charged_steps=actual, status='reconciled')
            atomic_json(work_path, work)
            summaries.append(value)
            value['equivalent'] = value['comparison_contract'] == summaries[0]['comparison_contract']
            training[f'{t}/{n}'].append(value)
            atomic_json(out / 'progress.json', dict(training_completed=len(summaries), training_total=6*args.repetitions))
    selected_train = choose_profile(training, '2/4', 'steps_per_second')
    atomic_json(out / 'status.json', dict(state='evaluation_profiles', updated=time.time()))
    evaluation = {}
    baseline = None
    for device, t, n in EVAL_PROFILES:
        name = f'eval_{device}_{t}_{n}'
        dest = out / (name + '.json')
        run_command([sys.executable, '-u', '-m', 'tools.optimize_rs4_pod', '--evaluation-worker',
                     '--run-dir', snapshot, '--out', dest, '--device', device,
                     '--torch-threads', t, '--physics-threads', n, '--games', '128',
                     '--minutes', '.25', '--repetitions', str(args.repetitions)], out / (name + '.log'))
        value = json.loads(dest.read_text())
        if baseline is None:
            baseline = value
        same = value['action_signatures'] == baseline['action_signatures'] and value['outcomes'] == baseline['outcomes']
        evaluation[f'{device}/{t}/{n}'] = [{**s, 'equivalent': same} for s in value['repetitions']]
    selected_eval = choose_profile(evaluation, 'cpu/2/4', 'batches_per_second')
    # Verify the selected route on complete two-minute batches before applying.
    if selected_eval['profile'] != 'cpu/2/4':
        verification = []
        for profile in ('cpu/2/4', selected_eval['profile']):
            device, t, n = profile.split('/')
            name = 'verify_' + profile.replace('/', '_')
            dest = out / (name + '.json')
            run_command([sys.executable, '-u', '-m', 'tools.optimize_rs4_pod', '--evaluation-worker',
                         '--run-dir', snapshot, '--out', dest, '--device', device,
                         '--torch-threads', t, '--physics-threads', n, '--games', '128',
                         '--minutes', '2', '--repetitions', '1'], out / (name + '.log'))
            verification.append(json.loads(dest.read_text()))
        if (verification[0]['action_signatures'] != verification[1]['action_signatures']
                or verification[0]['outcomes'] != verification[1]['outcomes']):
            selected_eval = choose_profile({'cpu/2/4': evaluation['cpu/2/4']}, 'cpu/2/4', 'batches_per_second')
    for path, expected in hashes.items():
        if digest(path) != expected:
            raise RuntimeError(f'Source changed during measurement: {path}')
    result = dict(training=selected_train, evaluation=selected_eval,
                  diagnostic_ppo_samples=sum(s['diagnostic_ppo_samples'] for s in summaries),
                  training_profiles=training, evaluation_profiles=evaluation, applied=False)
    if args.apply:
        if yaml.safe_load((run / branch / 'config.yaml').read_text()) != original_config:
            raise RuntimeError('Source config changed during measurement')
        target = run / branch / 'config.yaml'
        backup = target.with_name('config_before_pod_optimization.yaml')
        if not backup.exists():
            shutil.copy2(target, backup)
        t, n = map(int, selected_train['profile'].split('/'))
        original_config['ppo'].update(device='cuda', torch_threads=t, numba_threads=n)
        temp = target.with_suffix('.yaml.tmp')
        temp.write_text(yaml.safe_dump(original_config, sort_keys=False))
        temp.replace(target)
        device, t, n = selected_eval['profile'].split('/')
        atomic_json(run / 'evaluation_runtime.json', dict(device=device, torch_threads=int(t), physics_threads=int(n)))
        result['applied'] = True
    atomic_json(out / 'result.json', result)
    atomic_json(out / 'status.json', dict(state='complete', updated=time.time(), applied=result['applied']))
    print(json.dumps({k:result[k] for k in ('training','evaluation','diagnostic_ppo_samples','applied')}, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--wait', action='store_true')
    p.add_argument('--apply', action='store_true')
    p.add_argument('--repetitions', type=int, default=3)
    p.add_argument('--iters', type=int, default=10)
    p.add_argument('--evaluation-worker', action='store_true')
    p.add_argument('--device', default='cpu', choices=('cpu','cuda'))
    p.add_argument('--torch-threads', type=int, default=2)
    p.add_argument('--physics-threads', type=int, default=4)
    p.add_argument('--games', type=int, default=128)
    p.add_argument('--minutes', type=float, default=.25)
    args = p.parse_args()
    if min(args.repetitions, args.iters, args.torch_threads, args.physics_threads, args.games) < 1 or args.minutes <= 0:
        p.error('Counts, threads and duration must be positive')
    if not args.evaluation_worker and args.repetitions < 3:
        p.error('Optimization requires at least three repetitions')
    if args.evaluation_worker:
        evaluation_worker(args)
    else:
        try:
            optimize(args)
        except Exception as error:
            args.out.mkdir(parents=True, exist_ok=True)
            atomic_json(args.out / 'status.json', dict(state='failed', error=str(error), updated=time.time()))
            raise


if __name__ == '__main__':
    main()
