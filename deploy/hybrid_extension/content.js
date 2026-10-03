(()=>{
  let port=null,retry=null,attempts=0,closing=false;
  const post=(type,data={})=>window.postMessage({...data,channel:'rs4-hybrid-v1',direction:'extension',type},location.origin);
  function lost(connection,reason){
    if(connection!==port)return;port=null;
    post('service',{connected:false,reason});
    try{connection?.disconnect();}catch{}
    let alive=false;try{alive=!!chrome.runtime.id;}catch{}
    if(!closing&&alive&&attempts<5&&!retry)retry=setTimeout(()=>{retry=null;connect();},Math.min(500*2**attempts,5000));
    else if(!closing)post('service',{connected:false,reason:'Extensión recargada o canal no disponible: recargá también HaxBall.'});
  }
  function send(message){
    // Never queue frames/actions across a disconnected generation.
    const connection=port;if(!connection)return false;
    try{connection.postMessage(message);return true;}
    catch{lost(connection,'Canal de la extensión desconectado; reconectando…');return false;}
  }
  function connect(){
    if(closing||port)return;attempts++;
    let connection;
    try{
      if(!chrome.runtime.id)throw Error('Extension context invalidated');
      connection=chrome.runtime.connect({name:'rs4-hybrid'});port=connection;
      connection.onMessage.addListener(m=>{
        if(port!==connection||closing)return;attempts=0;
        if(m.type==='shortcut'){shortcutLabel=m.label||'Sin atajo';if(elements)elements.key.textContent=shortcutLabel;}
        post(m.type,m);
      });
      connection.onDisconnect.addListener(()=>{
        // Reading lastError prevents an unchecked runtime.lastError diagnostic.
        let invalid=false;try{invalid=!chrome.runtime.id;void chrome.runtime.lastError;}catch{invalid=true;}
        lost(connection,invalid?'Extensión recargada: recargá también HaxBall.':'Canal de la extensión desconectado; reconectando…');
      });
    }catch{lost(port,'Extensión recargada o canal no disponible: recargá también HaxBall.');}
  }
  let status={},elements=null,lastQualification=null,shortcutLabel='Sin atajo',diagnostic={},events=[];
  const number=(value,digits=1)=>Number.isFinite(value)?value.toFixed(digits):'—';
  function record(message){
    if(events[0]?.message===message)return;
    events.unshift({time:new Date().toISOString(),message});events.length=Math.min(events.length,60);
    if(elements)elements.events.textContent=events.map(e=>`${new Date(e.time).toLocaleTimeString()}  ${e.message}`).join('\n');
  }
  function renderMetrics(){if(!elements)return;
    const fresh=diagnostic.at&&Date.now()-diagnostic.at<3000;
    elements.inference.textContent=fresh?number(diagnostic.inferenceMs)+' ms':'—';
    elements.age.textContent=fresh?number(diagnostic.stateAge)+' ms':'—';
    elements.telemetry.textContent=fresh?'Última muestra recibida':'Sin muestra reciente';
    elements.metrics.textContent=lastQualification?`Último resultado: Preflight ${lastQualification.passed?'aprobado':'fallido'} · p95 ${number(lastQualification.latencyP95)} ms · frame ×${number(lastQualification.frameRatio,3)} · ${lastQualification.samples??0} muestras`:'Todavía no se ejecutó una comprobación.';
  }
  function render(){if(!elements)return;
    elements.mode.textContent=status.mode||'HUMANO';elements.mode.dataset.mode=status.mode||'HUMANO';
    elements.toggle.textContent=status.mode==='BOT'?'Tomar control':'Ceder al bot';
    elements.toggle.disabled=status.mode!=='BOT'&&(!status.ready||!status.permission);
    elements.toggle.title=status.mode==='BOT'?'Volver al control humano':status.toggleReason||'';
    elements.reason.textContent=status.mode==='BOT'?status.reason||'Bot activo':status.checkReason||(status.qualifying?'Comprobando sin mover al jugador; mantené el juego visible.':!status.qualified?(status.reason?.includes('Preflight')?status.reason:'Comprobá la configuración antes de ceder al bot.'):status.reason==='Sin foco; teclas liberadas'?'Configuración lista: hacé clic para ceder al bot.':status.reason||'Listo');
    elements.model.textContent=`Modelo: ${status.model||'—'} · Adaptador: ${status.adapter||'—'}`;
    elements.permission.checked=!!status.permission;
    elements.qualify.disabled=!!status.qualifying||!status.canQualify;
    elements.qualify.title=status.checkReason||'';
    elements.qualify.textContent=status.qualifying?'Comprobando…':'Comprobar configuración';
    const room=status.room||{},teams=room.teams;
    elements.readiness.textContent=`Servicio: ${status.service?'conectado':'desconectado'} · jugador: ${room.playerId??'no detectado'} (${room.team===1?'rojo':room.team===2?'azul':'espectador'}) · equipos: ${teams?teams.join(' vs '):'—'} · mapa: ${room.mapName||'—'} (${status.mapReady?'preparado':'pendiente'}) · permiso: ${status.permission?'sí':'no'}`;
    elements.reload.disabled=!status.service||status.mode!=='HUMANO';
    elements.service.textContent=status.service?'Servicio conectado':'Servicio desconectado';
    elements.service.dataset.ok=String(!!status.service);
    const checks=[['Servicio local',status.service],['Adaptador compatible',!!status.adapter&&status.adapter!=='no compatible'],['Jugador en equipo',room.team===1||room.team===2],['Plantel 4 vs 4',teams?.[0]===4&&teams?.[1]===4],['Mapa preparado',status.mapReady],['Autorización',status.permission],['Preflight vigente',status.qualified]];
    elements.checklist.replaceChildren(...checks.map(([label,ok])=>{const item=document.createElement('li');item.textContent=`${ok?'✓':'○'} ${label}`;item.dataset.ok=String(!!ok);return item;}));
    renderMetrics();
  }
  function mount(){
    const host=document.createElement('div');host.id='rs4-hybrid-panel';
    const shadow=host.attachShadow({mode:'open'});
    shadow.innerHTML=`<style>
      :host{display:block;font:12px system-ui;color:#e6edf5;background:#101923;position:relative;z-index:10;border-bottom:1px solid #334354;color-scheme:dark}
      *{box-sizing:border-box}.bar,.actions,.consent{display:flex;align-items:center;gap:10px;flex-wrap:wrap}.bar{padding:9px 14px}.brand{font-weight:800;letter-spacing:1px}.brand span{color:#78ddbd}button{font:inherit;border:1px solid #496074;border-radius:7px;background:#223345;color:#f0f6fc;padding:7px 11px;cursor:pointer}button:hover:enabled{background:#30495f}button:disabled{opacity:.45;cursor:not-allowed}button:focus-visible,summary:focus-visible,input:focus-visible{outline:2px solid #78ddbd;outline-offset:3px}#toggle{background:#25654e;border-color:#408d6e;font-weight:700}#mode{border-radius:5px;background:#24384a;padding:5px 8px;color:#9dd6ff}#mode[data-mode=BOT]{color:#83edb6;background:#173f33}#mode[data-mode=SUSPENDIDO]{color:#ffd381;background:#483719}small,.muted{color:#a7b8c9}.reason{flex:1;min-width:180px;line-height:1.4}.consent{padding:0 14px 8px}.consent label{flex:1}input{accent-color:#78ddbd}details{padding:0 14px 8px}summary{cursor:pointer;color:#bdcddd;padding:5px 0}.workspace{max-height: min(380px,45vh);overflow:auto;padding:8px 2px}.grid{display:grid;grid-template-columns:1.2fr 1fr;gap:12px}.card{background:#172330;border:1px solid #304354;border-radius:8px;padding:12px;min-width:0}h2{font-size:11px;text-transform:uppercase;letter-spacing:1px;color:#a7b8c9;margin:0 0 10px}p{line-height:1.5;margin:8px 0;overflow-wrap:anywhere}ul{list-style:none;padding:0;margin:0;display:flex;gap:8px 16px;flex-wrap:wrap}li{color:#c4cddd}[data-ok=true]{color:#80dfb2}#service{font-size:11px}.stats{display:flex;gap:24px}.stats b{display:block;font-size:23px;font-variant-numeric:tabular-nums}.stats span{color:#a7b8c9}.actions{margin-top:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:11px/1.7 ui-monospace,monospace;max-height:130px;overflow:auto;margin:8px 0}.foot{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:10px}#feedback{color:#a7b8c9}@media(max-width:620px){.grid{grid-template-columns:1fr}.bar{gap:7px}.reason{flex-basis:100%}}
    </style><div class="bar"><span class="brand">RS4 <span>CONTROL</span></span><strong id="mode">HUMANO</strong><button id="toggle" disabled>Ceder al bot</button><small id="key"></small><small id="reason" class="reason" role="status"></small></div>
      <div class="consent"><label><input id="permission" type="checkbox"> Esta sala es propia o tengo permiso para usar el bot.</label><span id="service"></span></div>
      <details><summary>Opciones y diagnóstico</summary><div class="workspace"><div class="grid">
      <section class="card"><h2>Preparación del bot</h2><ul id="checklist"></ul><p id="readiness" class="muted"></p><div class="actions"><button id="qualify">Comprobar configuración</button><button id="pair">Emparejar / atajo</button><button id="manager">Salas y bots</button><button id="claim" title="Solicitar el servicio si no hay otra pestaña propietaria">Tomar esta pestaña</button><button id="reload" title="Releer el modelo local actual; requiere nueva comprobación">Recargar ONNX</button></div></section>
      <section class="card"><h2>Telemetría</h2><div class="stats"><div><b id="inference">—</b><span>Inferencia</span></div><div><b id="age">—</b><span>Edad de propuesta</span></div></div><p id="telemetry" class="muted"></p><p id="model"></p><p id="metrics"></p><p class="muted">Preflight: p95 ≤ 50 ms · frame ≤ ×1.05 · 30 muestras mínimas. Validación de 30 minutos en sala real: pendiente.</p></section></div>
      <section class="card" style="margin-top:12px"><h2>Eventos de esta pestaña</h2><pre id="events">Esperando cambios de estado…</pre><div class="foot"><button id="export">Descargar diagnóstico</button><button id="clear">Limpiar eventos</button><span id="feedback" role="status"></span></div><p class="muted">Hasta 60 eventos en memoria. El informe incluye estado y métricas; no incluye token ni chat. Esc recupera el control humano.</p></section></div></details>`;
    const find=id=>shadow.getElementById(id);
    find('key').textContent=shortcutLabel;
    elements=Object.fromEntries(['mode','toggle','reason','model','permission','qualify','reload','metrics','readiness','key','service','checklist','inference','age','telemetry','events'].map(id=>[id,find(id)]));
    find('toggle').onclick=()=>post('toggle');find('permission').onchange=()=>post('permission',{allowed:find('permission').checked});
    find('qualify').onclick=()=>post('qualify');find('pair').onclick=()=>send({type:'openOptions'});
    find('manager').onclick=()=>send({type:'openManager'});
    find('claim').onclick=()=>send({type:'claim'});find('reload').onclick=()=>post('reload');
    find('clear').onclick=()=>{events=[];elements.events.textContent='Historial limpio.';};
    find('export').onclick=()=>{
      // Explicit allowlist: never serialize bridge messages, credentials or gameplay frames.
      const report={version:chrome.runtime.getManifest().version,exportedAt:new Date().toISOString(),mode:status.mode,service:!!status.service,model:status.model,adapter:status.adapter,mapReady:!!status.mapReady,qualified:!!status.qualified,permission:!!status.permission,reason:elements.reason.textContent,readiness:elements.readiness.textContent,shortcut:shortcutLabel,diagnostic:{inferenceMs:diagnostic.inferenceMs,stateAge:diagnostic.stateAge,sampledAt:diagnostic.at?new Date(diagnostic.at).toISOString():null},qualification:lastQualification?{passed:lastQualification.passed,latencyP95:lastQualification.latencyP95,frameRatio:lastQualification.frameRatio,samples:lastQualification.samples}:null,events};
      const url=URL.createObjectURL(new Blob([JSON.stringify(report,null,2)],{type:'application/json'}));const link=document.createElement('a');link.href=url;link.download=`rs4-diagnostico-${Date.now()}.json`;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);find('feedback').textContent='Diagnóstico descargado.';
    };
    // Outside the gameplay canvas, no replacement of the official UI.
    document.body.prepend(host);
    const layout=document.createElement('style');layout.id='rs4-hybrid-layout';
    layout.textContent='.game-view{top:var(--rs4-hybrid-panel-height,0px)!important;height:calc(100vh - var(--rs4-hybrid-panel-height,0px))!important}';
    document.head.append(layout);
    const resize=new ResizeObserver(()=>document.documentElement.style.setProperty('--rs4-hybrid-panel-height',host.getBoundingClientRect().height+'px'));
    resize.observe(host);render();post('requestStatus');
    const metricsTimer=setInterval(renderMetrics,1000);
    window.addEventListener('pagehide',()=>clearInterval(metricsTimer),{once:true});
    // The official client clears body contents during boot/screen changes.
    const screens=new MutationObserver(()=>{if(!host.isConnected&&document.body)document.body.prepend(host);});
    screens.observe(document.documentElement,{childList:true,subtree:true});
  }
  window.addEventListener('message',event=>{
    const m=event.data;if(event.source!==window||event.origin!==location.origin||m?.channel!=='rs4-hybrid-v1'||m.direction!=='page')return;
    if(m.type==='status'){const before=[status.mode,status.service,status.checkReason,status.qualified].join('|');status={...status,...m};if(before!==[status.mode,status.service,status.checkReason,status.qualified].join('|'))record(`${status.mode||'HUMANO'} · ${status.checkReason||status.reason||'Estado actualizado'}`);render();}
    else if(m.type==='qualification'){lastQualification=m;record(`Preflight ${m.passed?'aprobado':'fallido'}`);renderMetrics();}
    else if(m.type==='diagnostic'){if(Number.isFinite(m.inferenceMs))diagnostic={inferenceMs:m.inferenceMs,stateAge:m.stateAge,at:Date.now()};}
    else if(['frame','map','control','ack','release','reload'].includes(m.type))send(m);
  });
  window.addEventListener('pagehide',()=>{closing=true;if(retry)clearTimeout(retry);const connection=port;port=null;try{connection?.disconnect();}catch{};},{once:true});
  connect();
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',mount,{once:true});else mount();
})();
