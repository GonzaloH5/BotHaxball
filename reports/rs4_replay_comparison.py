"""Generate a small, reproducible replay comparison, not a promotion evaluation."""
import html
import json
import re
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
new = 'runs/multi/rs4_2300M.pt'
old = 'runs/multi/rs4_champion_1400M.pt'
cases = [
    ('Actual vs R3', 51, new, 'scripted:r3', 'rs4_2300M_vs_r3_seed51.html', 0),
    ('Actual vs R3', 73, 'scripted:r3', new, 'rs4_2300M_blue_vs_r3_seed73.html', 1),
    ('Actual vs R3', 91, new, 'scripted:r3', 'rs4_2300M_red_vs_r3_seed91.html', 0),
]
for seed in (51, 73, 91):
    blue = seed == 73
    cases.append(('Actual vs campeon 1400M', seed, old if blue else new,
                  new if blue else old, f'rs4_2300M_vs_champion_seed{seed}.html', int(blue)))
    cases.append(('Campeon 1400M vs R3', seed, 'scripted:r3' if blue else old,
                  old if blue else 'scripted:r3', f'rs4_champion_vs_r3_seed{seed}.html', int(blue)))
for style, label in enumerate(('equilibrado', 'agresivo', 'conservador')):
    for model, tag in ((new, 'actual'), (old, 'campeon')):
        cases.append((f'{tag} vs R3 {label}', 51, model, f'scripted:r3:{style}',
                      f'rs4_{tag}_vs_r3_style{style}.html', 0))

rows = []
for group, seed, red, blue, name, color in cases:
    path = root / 'replays' / name
    if not path.exists():
        subprocess.run([sys.executable, '-m', 'eval.render', red, blue, '--task', 'rs4_4v4',
                        '--minutes', '2', '--greedy', '--seed', str(seed), '--out', str(path)],
                       cwd=root, check=True)
    match = re.search(r'const M=(.*?), F=(.*?);\s*const c=', path.read_text(encoding='utf-8'), re.S)
    meta, frames = json.loads(match[1]), json.loads(match[2])
    score = frames[-1][-2:]
    rows.append(dict(group=group, seed=seed, color='azul' if color else 'rojo',
                     goals_for=score[color], goals_against=score[1-color],
                     kickoff_timeouts=meta['kickoff_stalls'], replay=name, ticks=len(frames)))
    print(json.dumps(rows[-1], ensure_ascii=False), flush=True)

(root / 'reports/rs4_replay_comparison_2300M.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
options = ''.join(f'<option value="{html.escape(r["replay"])}">{html.escape(r["group"])} | semilla {r["seed"]} | {r["color"]} | {r["goals_for"]}-{r["goals_against"]}</option>' for r in rows)
page = '''<!doctype html><html lang="es"><meta charset="utf-8"><title>Comparacion RS4 2300M</title>
<style>body{background:#1a1d1a;color:#e8ece6;font:16px system-ui;margin:20px}select{font:inherit;padding:8px;width:100%;max-width:800px}iframe{border:0;width:100%;height:760px}p{max-width:950px}</style>
<h2>RS4: checkpoint de 2300M y campeon de 1400M</h2>
<p>Quince grabaciones de dos minutos, decisiones greedy, semillas 51/73/91 y tres estilos de R3. Las semillas 51 y 91 repiten la trayectoria en los duelos deterministas con el mismo color; no son muestras independientes. El marcador del selector corresponde al modelo indicado, independientemente del color. Esta muestra sirve para inspeccion; no reemplaza la evaluacion completa.</p>
<p>Revisar: pase y receptor en saque central; separacion de apoyos; presion e intercepcion; cobertura del arquero; ejecucion de corners. Comparar el modelo actual y el campeon contra R3 con la misma semilla.</p>
<select id="pick">OPTIONS</select><iframe id="replay" src="FIRST"></iframe>
<script>document.getElementById('pick').onchange=e=>document.getElementById('replay').src=e.target.value;</script></html>'''.replace('OPTIONS', options).replace('FIRST', rows[0]['replay'])
(root / 'replays/rs4_comparacion_2300M.html').write_text(page, encoding='utf-8')
