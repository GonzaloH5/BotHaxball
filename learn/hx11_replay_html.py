"""Visor HTML de los partidos simulados de `learn/hx11_selfplay.py` (sin nombres: solo posiciones).

Un archivo autocontenido por partido, con la cancha de HaxEleven [1800, 1200], los 22 jugadores, la pelota, el
marcador y controles de reproducción (play/pausa, velocidad, barra de tiempo).

  python -m learn.hx11_replay_html runs/hx11_l0/mini/self/trayectorias.json --game 0 --out runs/hx11_l0/mini/self/juego0.html
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PAGE = """<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>HaxEleven L0 simulado</title>
<style>
:root { --bg:#f4f4f2; --fg:#1d1d1b; --pitch:#5e8f4f; --line:#e9efe6; --red:#d24b4b; --blue:#3d6fd1; --ball:#fff; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#161716; --fg:#e8e8e4; --pitch:#3f6b35; --line:#cfd8cb; } }
:root[data-theme="dark"] { --bg:#161716; --fg:#e8e8e4; --pitch:#3f6b35; --line:#cfd8cb; }
body { margin:0; background:var(--bg); color:var(--fg); font:14px system-ui, sans-serif; }
main { max-width:1100px; margin:0 auto; padding:12px 16px; }
canvas { width:100%; height:auto; display:block; border-radius:6px; }
.bar { display:flex; gap:10px; align-items:center; margin:8px 0; flex-wrap:wrap; }
input[type=range] { flex:1; min-width:160px; }
button, select { font:inherit; padding:4px 10px; }
</style></head><body><main>
<h1 style="font-size:18px;margin:4px 0">HaxEleven: L0 simulado (los 22 con la imitación)</h1>
<div class="bar"><button id="play">Pausa</button>
<select id="speed"><option value="0.5">x0,5</option><option value="1" selected>x1</option><option value="2">x2</option><option value="4">x4</option></select>
<input id="seek" type="range" min="0" value="0"><span id="info"></span></div>
<canvas id="c" width="1960" height="1360"></canvas>
<p style="opacity:.7">Rojo ataca hacia la derecha. Fuente: __SRC__.</p>
</main><script>
const G = __DATA__;
const c = document.getElementById('c'), ctx = c.getContext('2d');
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const W = 1960, H = 1360, S = 0.5, ox = W / 2, oy = H / 2;
const X = x => ox + x * S, Y = y => oy + y * S;
let f = 0, playing = true, last = performance.now(), acc = 0;
const seek = document.getElementById('seek'); seek.max = G.ball.length - 1;
function draw() {
  ctx.fillStyle = css('--pitch'); ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = css('--line'); ctx.lineWidth = 3;
  ctx.strokeRect(X(-1800), Y(-1200), 3600 * S, 2400 * S);
  ctx.beginPath(); ctx.moveTo(X(0), Y(-1200)); ctx.lineTo(X(0), Y(1200)); ctx.stroke();
  ctx.beginPath(); ctx.arc(X(0), Y(0), 180 * S, 0, 7); ctx.stroke();
  for (const sx of [-1, 1]) { ctx.strokeRect(X(sx > 0 ? 1800 - 300 : -1800), Y(-450), 300 * S, 900 * S); }
  const p = G.pos[f];
  for (let i = 0; i < 22; i++) {
    if (!G.on[i]) continue;
    ctx.fillStyle = i < 11 ? css('--red') : css('--blue');
    ctx.beginPath(); ctx.arc(X(p[i][0] / 10), Y(p[i][1] / 10), 15 * S * 1.4, 0, 7); ctx.fill();
  }
  const b = G.ball[f];
  ctx.fillStyle = css('--ball'); ctx.strokeStyle = '#000'; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(X(b[0] / 10), Y(b[1] / 10), 8 * S * 1.6, 0, 7); ctx.fill(); ctx.stroke();
  const sc = G.score[f];
  document.getElementById('info').textContent = `${(f / 60).toFixed(1)} s · ${sc[0]}–${sc[1]}`;
  seek.value = f;
}
function loop(now) {
  if (playing) { acc += (now - last) / 1000 * 60 * Number(document.getElementById('speed').value);
    while (acc >= 1) { f = Math.min(f + 1, G.ball.length - 1); acc -= 1; } }
  last = now; draw(); requestAnimationFrame(loop);
}
document.getElementById('play').onclick = e => { playing = !playing; e.target.textContent = playing ? 'Pausa' : 'Seguir'; };
seek.oninput = () => { f = Number(seek.value); };
requestAnimationFrame(loop);
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traj")
    ap.add_argument("--game", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    g = json.loads(Path(a.traj).read_text(encoding="utf-8"))[a.game]
    data = dict(pos=g["pos"], ball=g["ball"], on=g["pid"], score=g["score"])
    html = PAGE.replace("__DATA__", json.dumps(data, separators=(",", ":"))).replace("__SRC__", Path(a.traj).name)
    Path(a.out).write_text(html, encoding="utf-8")
    print(a.out)


if __name__ == "__main__":
    main()
