"""Read-only checkpoint audit and bounded behavioral probes; never runs PPO."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from numba import set_num_threads

from eval.agents import ModelAgent, ScriptedAgent
from eval.rs4_v3 import MixedTeamAgent, full_game, functional_trial
from train.rs4_program import ProgramState


class IdleAgent:
    def __call__(self, env, obs, players):
        return np.zeros((env.N, len(players)), dtype=np.int64)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', default='runs/rs4_v3_public/control/latest.pt')
    ap.add_argument('--out', default='reports/rs4_objective_audit_20261002.json')
    args = ap.parse_args()
    torch.set_num_threads(2)
    set_num_threads(4)
    path = Path(args.checkpoint)
    ck = torch.load(path, map_location='cpu', weights_only=False)
    saved = ck['rs4_program_state']
    state = ProgramState(saved['config'], saved)
    history = []
    for e in saved['evaluations']:
        r = e['report']
        groups = {}
        for row in r['full_games']['rows']:
            key = row['team_mode'] + ':' + Path(row['opponent']).name
            g = groups.setdefault(key, dict(games=0, wins=0, draws=0, losses=0, goals_for=0, goals_against=0))
            for k in g:
                g[k] += row[k]
        for g in groups.values():
            g['points'] = (g['wins'] + .5 * g['draws']) / g['games']
        history.append(dict(steps=e['steps'], phase=e['phase'], gates=e['gates'],
                            skills=r['skills'], baseline=r['baseline'], matches=r['matches'],
                            suite=r['suite'], source=r.get('source_fingerprint', {}).get('sha256'),
                            groups=groups, behavior=r.get('behavior'),
                            functional_rows=r['functional']['rows']))
    out = dict(checkpoint=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
               model=ck['model_config'], relative_steps=state.relative_steps,
               phase_start_steps=state.phase_start_steps, transitions=state.transitions,
               skill_debts=state.skill_debts, settings=state.settings(), history=history,
               lr_window=saved['lr_kl_window'], probes=[])
    output = Path(args.out)
    def save():
        output.write_text(json.dumps(out, indent=2), encoding='utf-8')
    save()
    # Can doing literally nothing receive a perfect defense-conceded score?
    for scenario in ('defense', 'transition'):
        for color in (0, 1):
            row = functional_trial(IdleAgent(), scenario, games=32, seed=51, color=color)
            row['agent'] = 'idle'
            out['probes'].append(row)
            print(json.dumps(row), flush=True)
            save()
    # Same saved weights, fixed opponent/seed/colors; isolate deployment argmax.
    for greedy in (False, True):
        agent = ModelAgent(str(path), greedy=greedy)
        for style in (0, 1, 2):
            for color in (0, 1):
                torch.manual_seed(5100 + color)
                row = full_game(agent, ScriptedAgent(policy='r3', style=style),
                                games=16, minutes=2, seed=51, color=color)
                row.update(agent='greedy' if greedy else 'sampled', style=style)
                out['probes'].append(row)
                print(json.dumps({k:v for k,v in row.items() if k not in ('behavior','restarts')}), flush=True)
                save()
    teacher = Path('runs/bc_rs4_v2_20261001/bc.pt')
    if teacher.exists():
        for greedy in (False, True):
            for color in (0, 1):
                team = MixedTeamAgent(ModelAgent(str(path), greedy=greedy),
                                      ModelAgent(str(teacher)), learner_slots=(0,))
                torch.manual_seed(5100 + color)
                row = full_game(team, ScriptedAgent(policy='r3', style=0),
                                games=16, minutes=2, seed=51, color=color)
                row.update(agent='mixed_greedy' if greedy else 'mixed_sampled', style=0)
                out['probes'].append(row)
                print(json.dumps({k:v for k,v in row.items() if k not in ('behavior','restarts')}), flush=True)
                save()


if __name__ == '__main__':
    main()
