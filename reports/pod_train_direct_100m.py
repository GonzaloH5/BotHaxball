"""Train one bounded RS4 segment without running or fabricating evaluations."""
import fcntl
import argparse
import json
import signal
import subprocess
import sys
import time
import yaml
from pathlib import Path

ROOT = Path('/workspace/HaxballRL')
sys.path.insert(0, str(ROOT))
from tools.run_rs4_v3 import _checkpoint, atomic_json, refresh_ledger, file_hash
from train.checkpoints import atomic_torch_save

directory = ROOT / 'runs/rs4_v3_public'
status_path = ROOT / 'pod_logs/rs4_direct_status.json'

def status(state, **extra):
    atomic_json(dict(state=state, updated=time.time(), **extra), status_path)

def interrupted(signum, frame):
    raise KeyboardInterrupt

signal.signal(signal.SIGINT, interrupted)
signal.signal(signal.SIGTERM, interrupted)
parser = argparse.ArgumentParser()
parser.add_argument('--target-steps', type=int)
args = parser.parse_args()
with (directory / '.runner.lock').open('a+b') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ledger = refresh_ledger(directory, json.loads((directory / 'ledger.json').read_text()))
    try:
        manifest = json.loads((directory / 'specialization.json').read_text())
        assert file_hash(directory / 'parent.pt') == manifest['source_sha256']
        branch = ledger['selected']
        assert branch in ('control', 'memory') and not ledger['complete']
        candidate, checkpoint, cfg, program = _checkpoint(directory, branch)
        program.charge_diagnostics(ledger.get('diagnostic_ppo_steps', 0))
        program.charge_hardware_diagnostics(ledger.get('hardware_diagnostic_ppo_steps', 0))
        start = program.relative_steps
        target = min(program.effective_branch_limit,
                     args.target_steps if args.target_steps is not None else start + 100_000_000)
        assert target > start
        checkpoint['rs4_program_state'] = program.state_dict()
        atomic_torch_save(checkpoint, candidate)
        atomic_json(refresh_ledger(directory, ledger), directory / 'ledger.json')
        status('training', branch=branch, start=start, target=target, evaluation_deferred=True)
        print(f'ENTRENAMIENTO DIRECTO | {branch} | {start:,} -> {target:,} pasos utiles | evaluacion pendiente', flush=True)
        direct_cfg = yaml.safe_load((directory / branch / 'config.yaml').read_text())
        direct_cfg.setdefault('rs4_v3', {})['pause_for_evaluation'] = False
        direct_cfg_path = directory / branch / 'config_direct_100m.yaml'
        direct_cfg_path.write_text(yaml.safe_dump(direct_cfg, sort_keys=False))
        subprocess.run([sys.executable, '-u', '-m', 'train.multitask',
                        '--config', str(direct_cfg_path),
                        '--run', f'{directory.name}/{branch}', '--resume', '--override',
                        f"ppo.total_steps={cfg['rs4_program']['start_steps'] + target}"], cwd=ROOT, check=True)
        _, _, _, after = _checkpoint(directory, branch)
        assert after.relative_steps >= target, 'El entrenamiento termino antes del objetivo'
        ledger['last_direct_segment'] = dict(start=start, steps=after.relative_steps,
                                            target=target, evaluation_pending=True)
        status('training_complete_evaluation_pending', steps=after.relative_steps, target=target)
        print('ENTRENAMIENTO TERMINADO. Checkpoint guardado; evaluacion pendiente, no ejecutada.', flush=True)
    except BaseException as error:
        status('interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        raise
    finally:
        atomic_json(refresh_ledger(directory, ledger), directory / 'ledger.json')
