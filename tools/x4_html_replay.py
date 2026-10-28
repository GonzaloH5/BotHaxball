"""Write an offline HTML viewer for recorded RS4Z simulator frames.

``write_html(payload, output_path)`` embeds a JSON-compatible x4-replay-1 payload.
It only renders recorded states: it does not run physics or contact a server.
Frame schema (24 values): clock, red score, blue score, phase, restart team,
kick bitmask, ball x/y, then x/y for player slots 0..7 in world coordinates.
The match clock comes from the simulator; playback time uses sample_hz.
"""
from __future__ import annotations

import json
import math
from pathlib import Path


def _validate(payload):
    if not isinstance(payload, dict) or payload.get("version") != "x4-replay-1":
        raise ValueError("expected an x4-replay-1 payload")
    hz = payload.get("sample_hz", 20)
    if not isinstance(hz, (int, float)) or not math.isfinite(hz) or hz <= 0:
        raise ValueError("sample_hz must be positive and finite")
    pitch = payload.get("pitch")
    if not isinstance(pitch, dict):
        raise ValueError("pitch geometry is required")
    for key in ("line_w", "line_h", "goal_x", "goal_half_h", "player_r"):
        value = pitch.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"pitch.{key} must be positive and finite")
    matches = payload.get("matches")
    if not isinstance(matches, list) or not matches:
        raise ValueError("at least one recorded match is required")
    for i, match in enumerate(matches):
        if not isinstance(match, dict) or not match.get("frames"):
            raise ValueError(f"match {i} has no recorded frames")
        for j, frame in enumerate(match["frames"]):
            if len(frame) != 24:
                raise ValueError(f"match {i}, frame {j}: expected 24 values")
            if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in frame):
                raise ValueError(f"match {i}, frame {j}: values must be finite numbers")
            if frame[3] not in range(6) or frame[4] not in (-1, 0, 1):
                raise ValueError(f"match {i}, frame {j}: invalid phase/restart team")


def write_html(payload, output_path):
    """Validate and write a standalone UTF-8 replay; return its absolute Path."""
    _validate(payload)
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    # A script[type=application/json] still has an HTML raw-text end delimiter.
    # Escape it and the Unicode JS line separators without changing JSON values.
    for character, replacement in (("<", "\\u003c"), (">", "\\u003e"), ("&", "\\u0026"),
                                   ("\u2028", "\\u2028"), ("\u2029", "\\u2029")):
        data = data.replace(character, replacement)
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_HTML.replace("__REPLAY_DATA__", data), encoding="utf-8")
    return path


_HTML = r'''<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="dark">
<title>HaxBall · Grabación del simulador</title>
<style>
:root{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#ecf3f2;background:#111b20;font-synthesis:none;--muted:#afc0c6;--border:#34464d;--red:#ff746b;--blue:#70b8ff}
*{box-sizing:border-box}body{margin:0}button,select,input{font:inherit}button,select{border:1px solid var(--border);border-radius:9px;background:#213139;color:inherit;min-height:42px;padding:9px 12px}button{cursor:pointer}button:hover{background:#2d424b}button:focus-visible,select:focus-visible,input:focus-visible{outline:3px solid #f9df94;outline-offset:3px}button:disabled{opacity:.45;cursor:default}main{max-width:1240px;margin:auto;padding:25px 22px 30px}.eyebrow{margin:0 0 7px;color:#a3c7b4;font-size:12px;font-weight:650;letter-spacing:.12em;text-transform:uppercase}h1{font-size:clamp(23px,3vw,31px);line-height:1.2;letter-spacing:-.035em;margin:0 0 9px}.intro{color:var(--muted);font-size:14px;margin:0 0 22px;line-height:1.5}.picker{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:16px}.picker label{color:var(--muted);font-size:13px}.picker select{flex:1;min-width:200px;max-width:650px}.match-meta{color:var(--muted);font-size:12px;margin-left:auto}.scoreboard{display:grid;grid-template-columns:minmax(0,1fr) auto minmax(0,1fr);align-items:center;gap:16px;background:#1a282f;border:1px solid var(--border);border-radius:12px 12px 0 0;padding:15px 20px}.team{min-width:0;font-weight:600;font-size:15px;overflow-wrap:anywhere}.team small{display:block;font-size:11px;color:var(--muted);font-weight:400;margin-top:3px}.team.red{color:var(--red)}.team.blue{color:var(--blue);text-align:right}.score-center{text-align:center;white-space:nowrap}.score{font-size:30px;font-weight:750;font-variant-numeric:tabular-nums;line-height:1.1;letter-spacing:.025em}.score .dash{color:#8099a4;font-weight:400;padding:0 9px}.clock{display:block;color:#c8d8db;font-size:13px;font-variant-numeric:tabular-nums;margin-top:5px}.board{position:relative;width:100%;background:#263529;border-left:1px solid var(--border);border-right:1px solid var(--border);overflow:hidden}canvas{display:block;width:100%;aspect-ratio:1.75}.phase-tag{position:absolute;left:13px;top:11px;pointer-events:none;padding:5px 9px;border:1px solid #ffffff2b;border-radius:6px;background:#102019cf;font-size:11px;color:#e9f1de}.event-tag{position:absolute;left:50%;top:16px;transform:translateX(-50%);border-radius:8px;background:#14212fee;padding:8px 14px;font-weight:700;font-size:14px;pointer-events:none;white-space:nowrap}.event-tag[hidden]{display:none}.transport{border:1px solid var(--border);border-radius:0 0 12px 12px;background:#1a282f;padding:15px 18px}.timeline{display:flex;align-items:center;gap:12px;margin-bottom:12px}.timeline input{flex:1;min-width:0;accent-color:#b2d7bb;cursor:pointer;height:24px}.playback-time{font-size:12px;color:var(--muted);font-variant-numeric:tabular-nums;white-space:nowrap}.toolbar{display:flex;align-items:center;gap:9px;flex-wrap:wrap}.play-button{background:#b2d7bb;color:#132b22;border-color:#b2d7bb;font-weight:650;min-width:128px}.play-button:hover{background:#c4e2ca}.step-button{font-size:17px;line-height:1;min-width:44px}.speed-label{font-size:12px;color:var(--muted);margin-left:6px}.toolbar select{padding:6px 9px;min-width:70px}.trail-label{display:flex;align-items:center;gap:7px;font-size:12px;color:var(--muted);cursor:pointer;margin-left:5px;min-height:42px}.trail-label input{accent-color:#b2d7bb;width:16px;height:16px}.status{font-size:12px;color:var(--muted);margin-left:auto}.footer{display:flex;justify-content:space-between;gap:16px;font-size:12px;line-height:1.5;color:var(--muted);margin-top:14px}.footer p{margin:0}.keyboard{white-space:nowrap}.error{padding:15px;border:1px solid #bf6464;border-radius:8px;background:#3a2328;color:#ffcfcb;margin-top:15px}[hidden]{display:none!important}@media(max-width:650px){main{padding:17px 10px 23px}.intro{margin-bottom:16px;font-size:12px}.picker{gap:7px;margin-bottom:11px}.picker label{width:100%}.picker select{width:100%;max-width:none;min-width:0;font-size:12px}.match-meta{margin:0;font-size:11px}.scoreboard{padding:12px 11px;gap:8px}.team{font-size:12px}.team small{font-size:10px}.score{font-size:25px}.score .dash{padding:0 4px}.clock{font-size:11px}.transport{padding:11px 10px}.timeline{gap:7px}.playback-time{font-size:10px}.toolbar{gap:6px}.play-button{min-width:113px;font-size:12px}.speed-label{margin-left:0}.trail-label{margin-left:0}.status{width:100%;margin:2px 0 0}.footer{font-size:11px;display:block}.keyboard{display:none}.phase-tag{left:8px;top:8px;font-size:10px;padding:4px 7px}.event-tag{top:8px;font-size:11px;padding:5px 9px}}@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto}}
</style>
</head>
<body data-ready="loading">
<main>
<header><p class="eyebrow">HaxBall · Real Soccer 4v4</p><h1 id="replay-title">Grabación del simulador</h1><p class="intro" id="replay-description">Partidos simulados con el modelo y la física de Real Soccer.</p></header>
<div class="picker"><label for="match-select">Partido</label><select id="match-select" aria-label="Elegir partido"></select><span class="match-meta" id="match-meta"></span></div>
<section aria-label="Reproductor de la grabación">
<div class="scoreboard"><div class="team red"><span id="red-label">Rojo</span><small id="red-update"></small></div><div class="score-center"><div class="score" aria-label="Marcador"><span id="score-red">0</span><span class="dash">–</span><span id="score-blue">0</span></div><span class="clock" id="match-clock">Partido · 0:00</span></div><div class="team blue"><span id="blue-label">Azul</span><small id="blue-update"></small></div></div>
<div class="board"><canvas id="replay-canvas" aria-label="Cancha con posiciones de la pelota y los ocho jugadores">Tu navegador necesita admitir canvas para mostrar la grabación.</canvas><span class="phase-tag" id="phase-tag">Juego abierto</span><span class="event-tag" id="event-tag" hidden></span></div>
<div class="transport"><div class="timeline"><input id="seek" type="range" min="0" max="0" step="1" value="0" aria-label="Posición de la reproducción"><span class="playback-time" id="playback-time">0:00 / 0:00</span></div><div class="toolbar"><button type="button" class="play-button" id="play-toggle" aria-label="Reproducir grabación">▶ Reproducir</button><button type="button" class="step-button" id="step-back" aria-label="Retroceder un fotograma" title="Retroceder un fotograma">‹</button><button type="button" class="step-button" id="step-forward" aria-label="Avanzar un fotograma" title="Avanzar un fotograma">›</button><label class="speed-label" for="speed">Velocidad</label><select id="speed" aria-label="Velocidad de reproducción"><option value="0.5">0,5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option></select><label class="trail-label"><input type="checkbox" id="trail-toggle">Estela de pelota</label><span id="status" class="status" role="status" data-ready="loading">Cargando…</span></div></div>
</section>
<div class="footer"><p>El reloj de partido se pausa durante el saque inicial. La barra recorre toda la grabación.</p><p class="keyboard">Espacio: reproducir · ← / →: un fotograma</p></div>
<p id="error" class="error" role="alert" hidden></p>
</main>
<script id="replay-data" type="application/json">__REPLAY_DATA__</script>
<script>
"use strict";
(() => {
try {
  const data=JSON.parse(document.getElementById("replay-data").textContent);
  const byId=id=>document.getElementById(id);
  const canvas=byId("replay-canvas"), ctx=canvas.getContext("2d"), pitch=data.pitch;
  if(!ctx)throw new Error("El navegador no pudo abrir la cancha.");
  const hz=Number(data.sample_hz)||20, RED="#ff746b", BLUE="#70b8ff";
  const boundW=Math.max(Number(pitch.bound_w)||1350,pitch.line_w+90);
  const boundH=Math.max(Number(pitch.bound_h)||770,pitch.line_h+65);
  const ratio=boundW/boundH;
  canvas.style.aspectRatio=String(ratio);
  let matchIndex=0, match=data.matches[0], frames=match.frames;
  let position=0, playing=false, speed=1, previousTime=null, width=1, height=1, scale=1;
  const frameIndex=()=>Math.min(frames.length-1,Math.max(0,Math.floor(position)));
  const clock=seconds=>{seconds=Math.max(0,Math.floor(seconds));return `${Math.floor(seconds/60)}:${String(seconds%60).padStart(2,"0")}`;};
  const xy=(x,y)=>[width/2+x*scale,height/2+y*scale];
  const finitePair=p=>Array.isArray(p)&&p.length>=2&&Number.isFinite(p[0])&&Number.isFinite(p[1]);
  const color=(value,fallback="#d3e1d1")=>typeof value==="string"&&/^#[0-9a-f]{3,8}$/i.test(value)?value:fallback;
  function line(p0,p1,stroke,lineWidth=1){if(!finitePair(p0)||!finitePair(p1))return;const a=xy(...p0),b=xy(...p1);ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.strokeStyle=stroke;ctx.lineWidth=lineWidth;ctx.stroke();}
  function circle(x,y,r,fill,stroke=null,lineWidth=1){const p=xy(x,y);ctx.beginPath();ctx.arc(...p,Math.max(0,r),0,Math.PI*2);if(fill){ctx.fillStyle=fill;ctx.fill();}if(stroke){ctx.lineWidth=lineWidth;ctx.strokeStyle=stroke;ctx.stroke();}}
  function discontinuity(a,b){
    if(!a||!b||a[3]!==b[3]||a[1]!==b[1]||a[2]!==b[2])return true;
    for(let offset=6;offset<24;offset+=2)if(Math.hypot(a[offset]-b[offset],a[offset+1]-b[offset+1])>250)return true;
    return false;
  }
  function ground(){
    ctx.fillStyle="#263b2b";ctx.fillRect(0,0,width,height);
    const top=xy(-pitch.line_w,-pitch.line_h), fw=pitch.line_w*2*scale, fh=pitch.line_h*2*scale;
    ctx.fillStyle="#6d8c56";ctx.fillRect(...top,fw,fh);
    const stripe=fw/12;ctx.fillStyle="#ffffff06";for(let i=0;i<12;i+=2)ctx.fillRect(top[0]+stripe*i,top[1],stripe,fh);
    ctx.strokeStyle="#d5e3c0";ctx.lineWidth=Math.max(1,scale*2.5);ctx.strokeRect(...top,fw,fh);
    line([0,-pitch.line_h],[0,pitch.line_h],"#d5e3c0",Math.max(1,scale*2.5));
    circle(0,0,(Number(pitch.kickoff_r)||187.6)*scale,null,"#d5e3c0",Math.max(1,scale*2));
    circle(0,0,Math.max(1.5,3*scale),"#d5e3c0");
    if(!Array.isArray(pitch.segments)||!pitch.segments.length){
      for(const sign of [-1,1]){const x=sign*pitch.line_w;line([x,-pitch.goal_half_h],[x+sign*60,-pitch.goal_half_h],"#d3e1d1",1.5);line([x+sign*60,-pitch.goal_half_h],[x+sign*60,pitch.goal_half_h],"#d3e1d1",1.5);line([x+sign*60,pitch.goal_half_h],[x,pitch.goal_half_h],"#d3e1d1",1.5);}
    }
    for(const segment of pitch.segments||[]){
      if(!finitePair(segment.p0)||!finitePair(segment.p1))continue;
      const stroke=color(segment.color);
      if(segment.curved&&finitePair(segment.center)&&Number.isFinite(segment.radius)&&segment.radius>0){
        const c=xy(...segment.center),a=Math.atan2(segment.p0[1]-segment.center[1],segment.p0[0]-segment.center[0]);
        const b=Math.atan2(segment.p1[1]-segment.center[1],segment.p1[0]-segment.center[0]);
        // Stadium._arc already orders p0/p1 for positive sweep. curved=2
        // means a major (>180 degree) arc, not the opposite drawing direction.
        ctx.beginPath();ctx.arc(...c,segment.radius*scale,a,b,false);ctx.strokeStyle=stroke;ctx.lineWidth=Math.max(1,scale*2);ctx.stroke();
      }else line(segment.p0,segment.p1,stroke,Math.max(1,scale*2));
    }
    for(const disc of pitch.discs||[]){if(finitePair(disc.pos)&&Number.isFinite(disc.r)&&disc.r>0)circle(...disc.pos,Math.max(2,disc.r*scale),color(disc.color,"#f5cd54"),"#263b2b",.8);}
  }
  function trail(index){
    if(!byId("trail-toggle").checked||index<1)return;
    let start=Math.max(0,index-12);
    for(let i=start+1;i<=index;i++)if(discontinuity(frames[i-1],frames[i]))start=i;
    if(start===index)return;
    ctx.beginPath();ctx.moveTo(...xy(frames[start][6],frames[start][7]));
    for(let i=start+1;i<=index;i++)ctx.lineTo(...xy(frames[i][6],frames[i][7]));
    ctx.strokeStyle="#f3eddc88";ctx.lineWidth=1.8;ctx.lineCap="round";ctx.lineJoin="round";ctx.stroke();
  }
  function render(){
    const index=frameIndex(), f=frames[index], next=frames[Math.min(index+1,frames.length-1)];
    const alpha=discontinuity(f,next)?0:position-index;
    ground();trail(index);
    for(let slot=0;slot<8;slot++){
      const offset=8+slot*2,x=f[offset]+(next[offset]-f[offset])*alpha,y=f[offset+1]+(next[offset+1]-f[offset+1])*alpha;
      if(Math.abs(x)>boundW+100||Math.abs(y)>boundH+100)continue;
      const team=slot<4?RED:BLUE,r=pitch.player_r*scale,p=xy(x,y);
      let flash=false;for(let k=index;k>=Math.max(0,index-3);k--){if(k<index&&discontinuity(frames[k],frames[k+1]))break;if((frames[k][5]&(1<<slot))!==0){flash=true;break;}}
      if(flash)circle(x,y,r+3,null,"#fff4c8",1.8);
      circle(x,y,r,team,"#142a24",Math.max(1,scale*2));
      ctx.font=`700 ${Math.max(9,Math.min(12,r*1.35))}px system-ui,sans-serif`;ctx.textAlign="center";ctx.textBaseline="middle";
      const label=String(slot%4+1),labelY=r<6?p[1]-r-6:p[1]+.3;
      if(r<6){ctx.strokeStyle="#163025";ctx.lineWidth=2;ctx.strokeText(label,p[0],labelY);}
      ctx.fillStyle=r<6?"#f6f3e8":"#102026";ctx.fillText(label,p[0],labelY);
    }
    const bx=f[6]+(next[6]-f[6])*alpha,by=f[7]+(next[7]-f[7])*alpha;
    circle(bx,by,(Number(pitch.ball_r)||8.325)*scale,"#fff7e6","#203029",Math.max(.65,scale*2));
    byId("score-red").textContent=String(f[1]);byId("score-blue").textContent=String(f[2]);
    byId("match-clock").textContent=`Partido · ${clock(f[0]/60)}`;
    const phases=["Juego abierto","Saque inicial","Lateral","Córner","Saque de arco","Cobro pendiente"];
    let phase=phases[f[3]]||"Juego abierto";
    if(f[3]>=2&&f[3]<=4&&f[4]>=0)phase+=` · ${f[4]===0?"rojo":"azul"}`;
    byId("phase-tag").textContent=phase;
    const end=index===frames.length-1,event=byId("event-tag");
    let goalTeam=null;
    if(index>0&&(f[1]>frames[index-1][1]||f[2]>frames[index-1][2]))goalTeam=f[1]>frames[index-1][1]?0:1;
    if(goalTeam!==null){event.hidden=false;event.textContent=`Gol · ${goalTeam===0?"rojo":"azul"}`;event.style.color=goalTeam===0?RED:BLUE;}
    else if(end){event.hidden=false;event.textContent=match.finished?"Fin del partido":"Fin de la grabación";event.style.color="#ecf3f2";}
    else event.hidden=true;
    byId("seek").value=String(index);byId("seek").setAttribute("aria-valuetext",`${clock(index/hz)} de ${clock((frames.length-1)/hz)}`);
    byId("playback-time").textContent=`${clock(index/hz)} / ${clock((frames.length-1)/hz)}`;
    byId("play-toggle").textContent=playing?"Ⅱ Pausar":"▶ Reproducir";
    byId("play-toggle").setAttribute("aria-label",playing?"Pausar grabación":"Reproducir grabación");
    byId("step-back").disabled=index===0;byId("step-forward").disabled=end;
    byId("status").textContent=playing?"Reproduciendo":end?(match.finished?"Partido completo":"Grabación parcial"):"Pausado";
  }
  function resize(){
    width=Math.max(240,canvas.getBoundingClientRect().width);height=width/ratio;
    const dpr=Math.min(window.devicePixelRatio||1,2);canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);
    ctx.setTransform(dpr,0,0,dpr,0,0);scale=width/(2*boundW);render();
  }
  function pause(){playing=false;previousTime=null;render();}
  function play(){if(frameIndex()===frames.length-1)position=0;playing=true;previousTime=null;render();}
  function seek(value){position=Math.max(0,Math.min(frames.length-1,Number(value)||0));previousTime=null;render();}
  function setMatch(value){
    const index=Math.max(0,Math.min(data.matches.length-1,Number(value)||0));
    matchIndex=index;match=data.matches[index];frames=match.frames;position=0;playing=false;previousTime=null;
    byId("match-select").value=String(index);byId("seek").max=String(frames.length-1);
    byId("red-label").textContent=match.red_label||"Rojo";byId("blue-label").textContent=match.blue_label||"Azul";
    byId("red-update").textContent=match.red_update==null?"":"Actualización "+match.red_update;
    byId("blue-update").textContent=match.blue_update==null?"":"Actualización "+match.blue_update;
    byId("match-meta").textContent=match.finished?"Partido completo":"Grabación parcial";
    render();
  }
  byId("replay-title").textContent=data.title||"Grabación del simulador";
  if(data.checkpoint_update!=null)byId("replay-description").textContent=`Modelo de la actualización ${data.checkpoint_update} · Sanguchito Real Soccer · ${data.matches.length} ${data.matches.length===1?"partido":"partidos"}`;
  for(const [i,m]of data.matches.entries()){const option=document.createElement("option");option.value=String(i);option.textContent=m.label||`${m.red_label||"Rojo"} vs ${m.blue_label||"Azul"}`;byId("match-select").append(option);}
  byId("match-select").addEventListener("change",e=>setMatch(e.target.value));
  byId("play-toggle").addEventListener("click",()=>playing?pause():play());
  byId("step-back").addEventListener("click",()=>{pause();seek(frameIndex()-1);});
  byId("step-forward").addEventListener("click",()=>{pause();seek(frameIndex()+1);});
  byId("seek").addEventListener("input",e=>seek(e.target.value));
  byId("speed").addEventListener("change",e=>{speed=Number(e.target.value);previousTime=null;});
  byId("trail-toggle").addEventListener("change",render);
  document.addEventListener("keydown",e=>{
    if(/INPUT|SELECT|TEXTAREA|BUTTON/.test(document.activeElement?.tagName||"")||e.altKey||e.ctrlKey||e.metaKey)return;
    if(e.code==="Space"){e.preventDefault();playing?pause():play();}
    else if(e.code==="ArrowLeft"||e.code==="ArrowRight"){e.preventDefault();pause();seek(frameIndex()+(e.code==="ArrowLeft"?-1:1));}
  });
  document.addEventListener("visibilitychange",()=>{previousTime=null;});
  if(typeof ResizeObserver!=="undefined")new ResizeObserver(resize).observe(canvas);else window.addEventListener("resize",resize);
  window.replayTest={get frame_count(){return frames.length;},get current_frame(){return frameIndex();},get playing(){return playing;},get match_index(){return matchIndex;},seek,draw:render,pause,play,set_match:setMatch};
  setMatch(0);resize();document.body.dataset.ready="true";byId("status").dataset.ready="true";
  function tick(time){
    if(playing){if(previousTime!==null){position=Math.min(frames.length-1,position+Math.min((time-previousTime)/1000,.25)*hz*speed);if(position>=frames.length-1)playing=false;}previousTime=time;render();}
    else previousTime=null;
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}catch(error){document.body.dataset.ready="error";document.getElementById("status").dataset.ready="error";document.getElementById("status").textContent="No se pudo abrir la grabación";const node=document.getElementById("error");node.hidden=false;node.textContent=String(error.message||error);window.replayTest={error:String(error.message||error)};}
})();
</script>
</body>
</html>
'''
