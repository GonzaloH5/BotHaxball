"""Generate a self-contained HTML replay for one X3 Futsal checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from learn.x3futsal_eval import _load_policy_spec, load_stadium, play_batch


PAGE = r"""<!doctype html>
<html lang="en"><meta charset="utf-8"><title>__TITLE__</title>
<style>
body{margin:0;background:#15191d;color:#eee;font:14px system-ui,sans-serif}
main{max-width:1000px;margin:auto;padding:16px} canvas{width:100%;background:#167344;border:2px solid #ddd}
.bar{display:flex;gap:10px;align-items:center;margin:10px 0} input{flex:1}
</style><main><h2>__TITLE__</h2><div class="bar"><button id="play">Pausa</button>
<select id="speed"><option>.5</option><option selected>1</option><option>2</option><option>4</option></select>
<input id="seek" type="range" min="0" value="0"><span id="info"></span></div>
<canvas id="c" width="1200" height="700"></canvas>
<script>
const G=__DATA__, c=document.getElementById("c"),x=c.getContext("2d"), W=1200,H=700;
const sx=v=>W/2+v*0.52, sy=v=>H/2+v*0.52; let f=0,run=true,last=performance.now(),acc=0;
const seek=document.getElementById("seek");seek.max=G.length-1;
function draw(){x.fillStyle="#167344";x.fillRect(0,0,W,H);x.strokeStyle="#fff";x.lineWidth=3;
x.strokeRect(sx(-557.5),sy(-240),1115,480);x.beginPath();x.moveTo(sx(0),sy(-240));x.lineTo(sx(0),sy(240));x.stroke();
x.beginPath();x.arc(sx(0),sy(0),60,0,Math.PI*2);x.stroke();x.strokeRect(sx(-557.5),sy(-110),95,220);x.strokeRect(sx(462.5),sy(-110),95,220);
let q=G[f];q.players.forEach((p,i)=>{x.fillStyle=i<3?"#e64b4b":"#4d8cff";x.beginPath();x.arc(sx(p[0]),sy(p[1]),11,0,7);x.fill();});
x.fillStyle="#fff";x.beginPath();x.arc(sx(q.ball[0]),sy(q.ball[1]),7,0,7);x.fill();x.fillStyle="#fff";x.font="bold 22px system-ui";x.fillText(q.goals[0]+" - "+q.goals[1],W/2-25,35);
document.getElementById("info").textContent=(f/60).toFixed(1)+" s";seek.value=f}
function loop(t){if(run){acc+=(t-last)/1000*60*Number(document.getElementById("speed").value);while(acc>=1){f=Math.min(f+1,G.length-1);acc--}}last=t;draw();requestAnimationFrame(loop)}
play.onclick=()=>{run=!run;play.textContent=run?"Pausa":"Seguir"};seek.oninput=()=>f=+seek.value;requestAnimationFrame(loop);
</script></main>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--opponent", default="scripted")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20261030)
    ap.add_argument("--minutes", type=float, default=1)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    stadium = load_stadium(__import__("learn.x3futsal_eval", fromlist=["STADIUM"]).STADIUM)
    import torch
    device = torch.device(args.device)
    policy = _load_policy_spec(args.checkpoint, stadium, device, assist_mode="hybrid")
    opponent = _load_policy_spec(args.opponent, stadium, device, assist_mode="hybrid")
    game = play_batch(policy, opponent, matches=1, minutes=args.minutes, seed=args.seed,
                      capture_trajectory=True)[0]
    html = PAGE.replace("__TITLE__", Path(args.checkpoint).name + " vs " + Path(args.opponent).name)
    html = html.replace("__DATA__", json.dumps(game["trajectory"], separators=(",", ":")))
    Path(args.out).write_text(html, encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
