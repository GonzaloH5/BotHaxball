"""Replay the paused checkpoint against matched rivals; do not train."""
import html
import json
import re
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
current = 'runs/multi/rs4_2568M.pt'
previous = 'runs/multi/rs4_2300M.pt'
champion = 'runs/multi/rs4_champion_1400M.pt'
cases = []
for style, label in enumerate(('equilibrado', 'agresivo', 'conservador')):
    cases.append((f'Actual 2568M vs R3 {label}', 51, current, f'scripted:r3:{style}',
                  f'rs4_2568M_vs_r3_style{style}.html', 0))
    cases.append((f'Antes 2300M vs R3 {label}', 51, previous, f'scripted:r3:{style}',
                  f'rs4_actual_vs_r3_style{style}.html', 0))
for seed, color in ((51, 0), (73, 1)):
    red, blue = (current, champion) if color == 0 else (champion, current)
    cases.append(('Actual 2568M vs campeon 1400M', seed, red, blue,
                  f'rs4_2568M_vs_champion_seed{seed}.html', color))
    cases.append(('Antes 2300M vs campeon 1400M', seed,
                  previous if color == 0 else champion, champion if color == 0 else previous,
                  f'rs4_2300M_vs_champion_seed{seed}.html', color))
    red, blue = (current, previous) if color == 0 else (previous, current)
    cases.append(('Actual 2568M vs anterior 2300M', seed, red, blue,
                  f'rs4_2568M_vs_2300M_seed{seed}.html', color))

rows = []
for label, seed, red, blue, name, color in cases:
    path = root / 'replays' / name
    if not path.exists():
        subprocess.run([sys.executable, '-m', 'eval.render', red, blue, '--task', 'rs4_4v4',
                        '--minutes', '2', '--greedy', '--seed', str(seed), '--out', str(path)],
                       cwd=root, check=True)
    match = re.search(r'const M=(.*?), F=(.*?);\s*const c=', path.read_text(encoding='utf-8'), re.S)
    meta, frames = json.loads(match[1]), json.loads(match[2])
    score = frames[-1][-2:]
    row = dict(label=label, seed=seed, color='rojo' if color == 0 else 'azul',
               goals_for=score[color], goals_against=score[1-color],
               central_restart_timeouts=meta['kickoff_stalls'], ticks=len(frames), replay=name)
    rows.append(row)
    print(json.dumps(row, ensure_ascii=False), flush=True)

(root / 'reports/rs4_paused_2568M_results.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
options = ''.join(f'<option value="{html.escape(r["replay"])}">{html.escape(r["label"])} | {r["color"]} | {r["goals_for"]}-{r["goals_against"]}</option>' for r in rows)
table = ''.join(f'<tr><td>{html.escape(r["label"])}</td><td>{r["color"]}</td><td>{r["goals_for"]}-{r["goals_against"]}</td></tr>' for r in rows)
page = '''<!doctype html><html lang="es"><meta charset="utf-8"><title>RS4: antes y despues de 268M</title>
<style>body{background:#1a1d1a;color:#e8ece6;font:16px system-ui;margin:20px}select{font:inherit;padding:8px;width:100%;max-width:1000px}iframe{border:0;width:100%;height:760px}p{max-width:1000px}td,th{padding:6px 16px;text-align:left}table{border-collapse:collapse}tr{border-bottom:1px solid #444}</style>
<h2>RS4: 2300M frente a 2568M</h2>
<p>Entrenamiento pausado y guardado en 2.567.851.912 pasos. Siete grabaciones nuevas de dos minutos, decisiones greedy, tres estilos R3 y rivales neuronales con ambos colores. Las grabaciones anteriores se reutilizan como referencia. Los marcadores se muestran desde el modelo indicado, no siempre desde rojo.</p>
<p>Comparar especialmente: cuantos van a sacar de banda, apoyos separados, entradas al area, pases interiores y cobertura del arquero. Los resultados no prueban por si solos mejor juego colectivo ni reemplazan una evaluacion completa.</p>
<select id="pick">OPTIONS</select><iframe id="replay" src="FIRST"></iframe>
<details><summary>Resultados</summary><table><tr><th>Partido</th><th>Modelo observado</th><th>Marcador</th></tr>TABLE</table></details>
<script>document.getElementById('pick').onchange=e=>document.getElementById('replay').src=e.target.value;</script></html>'''
page = page.replace('OPTIONS', options).replace('FIRST', rows[0]['replay']).replace('TABLE', table)
(root / 'replays/rs4_comparacion_2568M.html').write_text(page, encoding='utf-8')
