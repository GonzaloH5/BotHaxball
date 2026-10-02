"""Graba un partido y genera un HTML autocontenido para verlo en el navegador.

python -m eval.render runs/classic_1v1/latest.pt scripted --minutes 2 --out replay.html
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from env.haxball_env import HaxballEnv
from env.rewards import RewardConfig

from .agents import env_config, env_kwargs, make_agent, reset_agents


def stadium_lines(st):
    """Polilíneas visibles del estadio (arcos discretizados)."""
    lines = []
    for i in range(len(st.s_p0)):
        if not st.s_vis[i]:
            continue
        p0, p1 = st.s_p0[i], st.s_p1[i]
        if st.s_curved[i]:
            c, r = st.s_center[i], st.s_radius[i]
            a0 = math.atan2(p0[1] - c[1], p0[0] - c[0])
            a1 = math.atan2(p1[1] - c[1], p1[0] - c[0])
            # elegir el sentido que pasa por el lado "de adentro" (t0)
            mid_dir = st.s_t0[i]
            cand = []
            for sgn in (1, -1):
                da = (a1 - a0) % (2 * math.pi) if sgn == 1 else -((a0 - a1) % (2 * math.pi))
                am = a0 + da / 2
                pm = np.array([math.cos(am), math.sin(am)])
                cand.append((float(np.dot(pm, mid_dir) + np.dot(pm, st.s_t1[i])), da))
            da = max(cand)[1]
            pts = [[c[0] + r * math.cos(a0 + da * k / 16), c[1] + r * math.sin(a0 + da * k / 16)] for k in range(17)]
        else:
            pts = [list(p0), list(p1)]
        lines.append([[round(x, 1), round(y, 1)] for x, y in pts])
    return lines


def record(agent_red, agent_blue, minutes=2.0, n_per_team=1, stadium="classic", frame_skip=3, seed=0,
           env_kw=None):
    """Graba tick a tick. Usa el mismo entorno que el entrenamiento (con frame_skip=1 y la
    acción repetida `frame_skip` ticks), así goles, pelota afuera y saques se comportan igual."""
    kwargs = dict(env_kw or {})
    kwargs["corner_reset_prob"] = 0.0  # los replays no usan el currículo de entrenamiento
    kwargs.setdefault("kickoff_timeout", 180)
    env = HaxballEnv(1, n_per_team, stadium, 1, max_ticks=7200, random_reset_prob=0.0,
                     reward=RewardConfig(shaping_coef=0.0), seed=seed, **kwargs)
    env.reset()
    env.public_sample_period = frame_skip
    env._reset_envs(np.array([0]), kickoff_team=np.array([0]))
    env._phi = env._potentials()
    obs = env.observe()
    reset_agents((agent_red, agent_blue), env)
    red, blue = np.arange(env.T), np.arange(env.T, env.P)
    frames = []
    public_cues = []
    a = np.zeros((1, env.P), dtype=np.int64)
    s = env.sim
    stalls = 0
    decision_active = False
    for t in range(int(minutes * 3600)):
        if t % frame_skip == 0:
            a[:, red] = agent_red(env, obs, red)
            a[:, blue] = agent_blue(env, obs, blue)
            decision_active = True
        obs, _, done, info = env.step(a)
        if decision_active:
            seen = set()
            for agent in (agent_red, agent_blue):
                if id(agent) not in seen and hasattr(agent, "record_executed"):
                    agent.record_executed(env, info.get("executed_actions", a))
                seen.add(id(agent))
        stalls += int(info["stall"].sum())
        if done.any():
            reset_agents((agent_red, agent_blue), env, done)
            a[done] = 0
            # A terminal may land between repeated physics ticks. Preserve the
            # recurrent reset sentinel until a new policy decision, rather than
            # treating intervening neutral rendering ticks as policy actions.
            decision_active = False
        comba = int(s.ps_comba[0]) if s.ps_on else 0
        public_cues.append([int(colors[0]) for colors in env.public_colors()] if hasattr(env, "public_colors") else [0xFFFFFF, 0xFFFFFF])
        frames.append([round(float(v), 1) for v in s.pos[0, [0, *range(s.first_player, s.K)]].ravel()]
                      + [int(k) for k in (a[0] >= 9)] + [comba, int(env.score[0, 0]), int(env.score[0, 1])])
    st = s.st
    W = max(st.width, st.field_half_w + 30)
    meta = {"lines": stadium_lines(st), "posts": st.d_pos.tolist(), "post_r": st.d_radius.tolist(),
            "P": env.P, "T": env.T, "r_player": st.player["radius"], "r_ball": st.ball["radius"],
            "W": W, "H": st.height, "fw": st.field_half_w, "fh": st.field_half_h,
            "ko": st.kickoff_radius, "S": min(1.6, 1100 / (2 * W + 20)),
            "red": getattr(agent_red, "name", "rojo"), "blue": getattr(agent_blue, "name", "azul"),
            "kickoff_stalls": stalls, "public_cues": public_cues}
    return meta, frames


HTML = """<!doctype html><html lang="es"><head><meta charset="utf-8"><title>Replay HaxBall</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#1a1d1a;--fg:#e8ece6;--grass:#718c5a;--line:#c7e6bd;--red:#e56e56;--blue:#5689e5;--muted:#9aa396}
body{margin:0;background:var(--bg);color:var(--fg);font:14px system-ui,sans-serif;display:flex;flex-direction:column;align-items:center;padding:16px}
canvas{max-width:100%;height:auto;border-radius:8px}
.bar{display:flex;gap:12px;align-items:center;margin:10px 0;flex-wrap:wrap;justify-content:center}
.score{font-size:22px;font-weight:600;font-variant-numeric:tabular-nums}
.r{color:var(--red)}.b{color:var(--blue)} input[type=range]{width:min(600px,80vw)} button,select{background:#2a2f2a;color:var(--fg);border:1px solid #3a413a;border-radius:6px;padding:4px 10px}
small{color:var(--muted)}
</style></head><body>
<div class="bar"><span class="r" id="rn"></span><span class="score"><span class="r" id="rs">0</span> - <span class="b" id="bs">0</span></span><span class="b" id="bn"></span></div>
<canvas id="c"></canvas>
<div class="bar"><button id="play">⏸</button><select id="spd"><option value="0.5">0.5x</option><option value="1" selected>1x</option><option value="2">2x</option><option value="4">4x</option></select>
<input type="range" id="seek" min="0" value="0"><small id="time"></small></div>
<small id="restart-note"></small>
<script>
const M=__META__, F=__FRAMES__;
const c=document.getElementById('c'),x=c.getContext('2d'),S=M.S||1.6;
c.width=(M.W*2+20)*S;c.height=(M.H*2+20)*S;
const seek=document.getElementById('seek');seek.max=F.length-1;
document.getElementById('rn').textContent=M.red;document.getElementById('bn').textContent=M.blue;
document.getElementById('restart-note').textContent=M.kickoff_stalls?`Esta grabación contiene ${M.kickoff_stalls} saques centrales vencidos (reinicios sin gol).`:'';
let i=0,playing=true,acc=0,last=performance.now();
const tx=v=>(v+M.W+10)*S, ty=v=>(v+M.H+10)*S;
function disc(px,py,r,fill,stroke,w){x.beginPath();x.arc(tx(px),ty(py),r*S,0,7);x.fillStyle=fill;x.fill();x.lineWidth=w*S;x.strokeStyle=stroke;x.stroke();}
function draw(){const f=F[i];x.fillStyle='#718c5a';x.fillRect(0,0,c.width,c.height);
 x.strokeStyle='#c7e6bd';x.lineWidth=3*S;x.strokeRect(tx(-M.fw),ty(-M.fh),M.fw*2*S,M.fh*2*S);
 x.beginPath();x.moveTo(tx(0),ty(-M.fh));x.lineTo(tx(0),ty(M.fh));x.stroke();
 x.beginPath();x.arc(tx(0),ty(0),(M.ko||75)*S,0,7);x.stroke();
 x.strokeStyle='#000';x.lineWidth=2*S;for(const l of M.lines){x.beginPath();x.moveTo(tx(l[0][0]),ty(l[0][1]));for(const p of l.slice(1))x.lineTo(tx(p[0]),ty(p[1]));x.stroke();}
 M.posts.forEach((p,k)=>disc(p[0],p[1],M.post_r[k],p[0]<0?'#ffcccc':'#ccccff','#000',2));
 for(let p=0;p<M.P;p++){const k=f[2*M.P+2+p];disc(f[2+2*p],f[3+2*p],M.r_player,p<M.T?'#e56e56':'#5689e5',k?'#fff':'#000',2);}
 const bc=M.public_cues?.[i]?.[0];
 const ballColor=bc!=null&&bc!==0xFFFFFF?'#'+bc.toString(16).padStart(6,'0'):(f[3*M.P+2]?'#ff3030':'#fff');
 disc(f[0],f[1],M.r_ball,ballColor,'#000',2);
 document.getElementById('rs').textContent=f[f.length-2];document.getElementById('bs').textContent=f[f.length-1];
 const s=Math.floor(i/60);document.getElementById('time').textContent=Math.floor(s/60)+':'+String(s%60).padStart(2,'0');seek.value=i;}
function loop(t){const dt=t-last;last=t;if(playing){acc+=dt*60/1000*+document.getElementById('spd').value;while(acc>=1&&i<F.length-1){i++;acc--;}}draw();requestAnimationFrame(loop);}
document.getElementById('play').onclick=e=>{playing=!playing;e.target.textContent=playing?'⏸':'▶';};
seek.oninput=()=>{i=+seek.value;};
requestAnimationFrame(loop);
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("red")
    ap.add_argument("blue")
    ap.add_argument("--minutes", type=float, default=2.0)
    ap.add_argument("--n-per-team", type=int, default=None, help="por defecto, el del checkpoint")
    ap.add_argument("--stadium", default=None, help="por defecto, el del checkpoint")
    ap.add_argument("--task", default=None, help="tarea de train/tasks.yaml (estadio + formato + reglas)")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="replay.html")
    args = ap.parse_args()
    cfg = env_config([args.red, args.blue], args.stadium, args.n_per_team, args.task)
    meta, frames = record(make_agent(args.red, args.greedy), make_agent(args.blue, args.greedy),
                          args.minutes, cfg["n_per_team"], cfg["stadium"], cfg["frame_skip"],
                          seed=args.seed, env_kw=env_kwargs(cfg))
    html = HTML.replace("__META__", json.dumps(meta)).replace("__FRAMES__", json.dumps(frames, separators=(",", ":")))
    Path(args.out).write_text(html, encoding="utf-8")
    print(f"{args.out}: {len(frames)} ticks, resultado {frames[-1][-2]}-{frames[-1][-1]}")
    if meta["kickoff_stalls"]:
        print(f"Aviso: {meta['kickoff_stalls']} saques centrales vencidos; se reiniciaron episodios sin sumar goles.")


if __name__ == "__main__":
    main()
