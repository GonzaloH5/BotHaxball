"""Repetición HTML autónoma de partidos RS4-Z (un canvas, sin dependencias externas).

    rec = Recorder(env, rows=[0, 1])   # antes del partido
    rec.capture(env, ev)               # después de cada env.step
    rec.save("reports/rs4z/replays/x.html", title="L5 vs L5")
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from env.rs4z import kernel as K

KIND = {0: "", 1: "lateral", 2: "córner", 3: "saque de arco"}


class Recorder:
    def __init__(self, env, rows=(0,), labels=None, tick_ms=50):
        """tick_ms: duración de un cuadro al reproducir a 1× (50 ms = una decisión de 3 ticks;
        16,7 ms si se graba tick a tick con `record_ticks`)."""
        self.tick_ms = tick_ms
        self.rows = list(rows)
        self.frames = {r: [] for r in self.rows}
        self.labels = labels or {}

    def capture(self, env, ev=None):
        for r in self.rows:
            p = env.player_pos[r]
            act = env.active[r]
            kicked = ev["kicked"][r] if ev is not None else np.zeros(8, dtype=bool)
            self.frames[r].append(dict(
                b=[round(float(env.ball_pos[r, 0]), 1), round(float(env.ball_pos[r, 1]), 1)],
                p=[[round(float(p[i, 0]), 1), round(float(p[i, 1]), 1)] if act[i] else None for i in range(8)],
                k=[int(i) for i in np.flatnonzero(kicked)],
                s=[int(env.ri[r, K.RI_SCORE0]), int(env.ri[r, K.RI_SCORE1])],
                r=KIND.get(int(env.ri[r, K.RI_KIND]), ""), rt=int(env.ri[r, K.RI_TEAM]),
                ko=int(env.ri[r, K.RI_KO]), c=int(env.ri[r, K.RI_CLOCK]),
                m=int(env.ri[r, K.RI_MASS]), br=round(float(env.radius[r, 0]), 3),
            ))

    def save(self, path, title="RS4-Z"):
        data = {str(r): dict(label=self.labels.get(r, f"partido {r}"), frames=f, tick_ms=self.tick_ms)
                for r, f in self.frames.items()}
        html = TEMPLATE.replace("__TITLE__", title).replace("__DATA__", json.dumps(data, separators=(",", ":")))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")
        return path


def record_ticks(controllers, rows, n_envs, decisions, seed=0, match_ticks=None, path=None, title="RS4-Z", labels=None):
    """Graba partidos tick a tick (tamaños reales): el entorno avanza de a 1 tick y los controladores
    deciden cada 3 ticks con la acción sostenida, exactamente como en el entorno de entrenamiento."""
    from env.rs4z.core import RS4ZEnv
    env = RS4ZEnv(n_envs, seed=seed, frame_skip=1, deadline=0, max_delay=0)
    env.start_match(np.arange(n_envs), match_ticks=match_ticks)
    for c in controllers:
        c.sync(env, np.arange(n_envs))
    rec = Recorder(env, rows=rows, labels=labels, tick_ms=1000 / 60)
    act = np.zeros((n_envs, 8), dtype=np.int64)
    for d in range(decisions):
        act[:] = 0
        for c in controllers:
            c.act(env, env.active.copy(), act)
        for k in range(3):
            ev = env.step(act)
            rec.capture(env, ev)
            g = np.flatnonzero(ev["goal"] != 0)
            if len(g):
                for c in controllers:
                    c.sync(env, g)
        for c in controllers:
            c.push(env)
    if path is not None:
        rec.save(path, title=title)
    return rec


TEMPLATE = """<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title><style>
:root{--bg:#f4f4f1;--fg:#1d1d1b;--panel:#fff;--muted:#6b6b66}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecece8;--panel:#22221f;--muted:#9a9a93}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px system-ui,sans-serif}
main{max-width:1100px;margin:0 auto;padding:12px 16px}canvas{width:100%;height:auto;background:#5e8a4f;border-radius:6px}
.bar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:8px 0}select,button,input{font:inherit}
#info{color:var(--muted)}</style></head><body><main><h1 style="font-size:18px">__TITLE__</h1>
<div class="bar"><select id="game"></select><button id="play">⏸</button><input id="t" type="range" min="0" value="0" style="flex:1">
<select id="speed"><option value="1">1×</option><option value="2">2×</option><option value="4">4×</option></select></div>
<div id="info"></div><canvas id="c" width="1100" height="660"></canvas></main><script>
const DATA=__DATA__;const cv=document.getElementById('c'),g=cv.getContext('2d');
const sel=document.getElementById('game'),rng=document.getElementById('t'),info=document.getElementById('info');
for(const k in DATA){const o=document.createElement('option');o.value=k;o.textContent=DATA[k].label;sel.appendChild(o)}
let cur=Object.keys(DATA)[0],t=0,playing=true;const S=1100/2600,OX=1300,OY=780;
function X(x){return (x+OX)*S}function Y(y){return (y+OY)*S}
function draw(){const F=DATA[cur].frames;rng.max=F.length-1;const f=F[t];g.clearRect(0,0,cv.width,cv.height);
g.fillStyle='#5e8a4f';g.fillRect(0,0,cv.width,cv.height);g.strokeStyle='#e8f0e0';g.lineWidth=2;
g.strokeRect(X(-1150),Y(-670),1150*2*S,670*2*S);g.beginPath();g.moveTo(X(0),Y(-670));g.lineTo(X(0),Y(670));g.stroke();
g.beginPath();g.arc(X(0),Y(0),180*S,0,7);g.stroke();g.strokeRect(X(-1150),Y(-320),310*S,640*S);g.strokeRect(X(840),Y(-320),310*S,640*S);
g.lineWidth=4;g.strokeStyle='#fff';g.beginPath();g.moveTo(X(-1162),Y(-124));g.lineTo(X(-1162),Y(124));g.moveTo(X(1162),Y(-124));g.lineTo(X(1162),Y(124));g.stroke();
f.p.forEach((p,i)=>{if(!p)return;g.beginPath();g.arc(X(p[0]),Y(p[1]),15*S,0,7);g.fillStyle=i<4?'#e5534b':'#4a8fe7';g.fill();
g.lineWidth=f.k.includes(i)?4:1.5;g.strokeStyle=f.k.includes(i)?'#fff':'#111';g.stroke();g.fillStyle='#fff';g.font='9px system-ui';g.fillText(i%4,X(p[0])-2.5,Y(p[1])+3)});
g.beginPath();g.arc(X(f.b[0]),Y(f.b[1]),(f.br||8.325)*S,0,7);g.fillStyle=f.rt===0?'#ffb3ad':f.rt===1?'#b7d3ff':'#fff';g.fill();g.strokeStyle='#111';g.lineWidth=1;g.stroke();
const sec=(f.c/60).toFixed(1);info.textContent=`rojo ${f.s[0]} - ${f.s[1]} azul · reloj ${sec}s · ${f.ko?'saque inicial · ':''}${f.r?('saque: '+f.r+' ('+(f.rt===0?'rojo':'azul')+') · '):''}masa ${f.m?'0,3':'0,5'} · decisión ${t}`;rng.value=t}
sel.onchange=()=>{cur=sel.value;t=0;draw()};rng.oninput=()=>{t=+rng.value;draw()};
document.getElementById('play').onclick=e=>{playing=!playing;e.target.textContent=playing?'⏸':'▶'};
const STEP=DATA[cur].tick_ms||50;setInterval(()=>{if(!playing)return;const F=DATA[cur].frames;t=Math.min(F.length-1,t+(+document.getElementById('speed').value));draw()},STEP);draw();
</script></body></html>"""
