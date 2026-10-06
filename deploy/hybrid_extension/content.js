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
    elements['toggle-label'].textContent=status.mode==='BOT'?'Tomar control':'Ceder al bot';
    elements.pair.dataset.current=String(!status.service);
    elements.pair.dataset.done=String(!!status.service);
    elements.qualify.dataset.current=String(!!status.service&&!status.qualified&&!!status.canQualify);
    elements.qualify.dataset.done=String(!!status.qualified);
    elements.toggle.dataset.current=String(status.mode==='BOT'||(!!status.ready&&!!status.permission));
    elements.toggle.setAttribute('aria-pressed',String(status.mode==='BOT'));
    elements.qualify.setAttribute('aria-busy',String(!!status.qualifying));
    elements.toggle.disabled=status.mode!=='BOT'&&(!status.ready||!status.permission);
    elements.toggle.title=status.mode==='BOT'?'Volver al control humano':status.toggleReason||'';
    elements.reason.textContent=status.mode==='BOT'?(status.reason||'Bot activo'):
      status.qualifying?'Comprobando sin mover al jugador; mantené el juego visible.':
      status.checkReason||
      (status.ready&&status.permission?'Configuración lista. Podés ceder al bot.':
      !status.qualified?(status.reason?.includes('Preflight')?status.reason:'Comprobá la configuración antes de ceder al bot.'):
      status.reason||'Listo');
    elements.model.textContent=`Modelo: ${status.model||'—'} · Adaptador: ${status.adapter||'—'}`;
    elements.permission.checked=!!status.permission;
    elements.qualify.disabled=!!status.qualifying||!status.canQualify;
    elements.qualify.title=status.checkReason||'';
    elements['qualify-label'].textContent=status.qualifying?'Comprobando…':status.qualified?'Volver a comprobar':'Comprobar configuración';
    const room=status.room||{},teams=room.teams;
    elements.readiness.textContent=`Servicio: ${status.service?'conectado':'desconectado'} · jugador: ${room.playerId??'no detectado'} (${room.team===1?'rojo':room.team===2?'azul':'espectador'}) · equipos: ${teams?teams.join(' vs '):'—'} · mapa: ${room.mapName||'—'} (${status.mapReady?'preparado':'pendiente'}) · permiso: ${status.permission?'sí':'no'}`;
    elements.reload.disabled=!status.service||status.mode!=='HUMANO';
    elements.service.textContent=status.service?'Servicio conectado':'Servicio desconectado';
    elements.service.dataset.ok=String(!!status.service);
    const checks=[['Servicio local',status.service],['Adaptador compatible',!!status.adapter&&status.adapter!=='no compatible'],['Jugador en equipo',room.team===1||room.team===2],['Plantel 4 vs 4',teams?.[0]===4&&teams?.[1]===4],['Mapa preparado',status.mapReady],['Autorización',status.permission],['Comprobación vigente',status.qualified]];
    elements.checklist.replaceChildren(...checks.map(([label,ok])=>{const item=document.createElement('li');item.textContent=`${ok?'✓':'○'} ${label}`;item.dataset.ok=String(!!ok);return item;}));
    renderMetrics();
  }
  function mount(){
    const host=document.createElement('div');host.id='rs4-hybrid-panel';
    const shadow=host.attachShadow({mode:'open'});
    shadow.innerHTML=`<style>
      :host{display:block;font:12px/1.5 "Segoe UI",system-ui,sans-serif;color:#e8ede5;background:#202d27;position:relative;z-index:10;border-bottom:2px solid #8dac76;color-scheme:dark}
      *{box-sizing:border-box}button{font:600 12px/1.4 "Segoe UI",system-ui,sans-serif;border:1px solid #637464;border-radius:3px;background:transparent;color:#e8ede5;padding:8px 11px;cursor:pointer}button:hover:enabled{background:#354637}button:disabled{color:#94a192;border-color:#465647;cursor:not-allowed}button:focus-visible,summary:focus-visible,input:focus-visible{outline:2px solid #c3de9d;outline-offset:3px}.bar{display:flex;align-items:center;gap:16px;padding:10px 16px;flex-wrap:wrap}.identity{display:flex;align-items:center;gap:12px;margin-right:auto}.brand{font-size:22px;font-weight:850;letter-spacing:-1px;line-height:1}.competition{font:10px/1.5 ui-monospace,Consolas,monospace;color:#b3c3aa;border-left:1px solid #56644f;padding-left:12px}.competition span{display:block;color:#e8ede5;font-family:"Segoe UI",system-ui,sans-serif;font-size:12px}.steps{display:flex;align-items:center;gap:6px;flex-wrap:wrap}.steps button{display:flex;align-items:center;gap:8px}.step-number{font:10px ui-monospace,Consolas,monospace;color:#b4c3ab}.steps button[data-current=true]{background:#c3de9d;border-color:#c3de9d;color:#202d27}.steps button[data-current=true] .step-number{color:#425b34}.steps button[data-done=true]{border-color:#8dac76}.steps button[data-done=true] .step-number{color:#c3de9d}.steps button[data-current=true]:hover:enabled{background:#d3e9b5}#mode{font:600 10px ui-monospace,Consolas,monospace;letter-spacing:.4px;color:#d9e2d2;border:1px solid #677560;padding:4px 7px;border-radius:2px}#mode[data-mode=BOT]{background:#c3de9d;color:#202d27;border-color:#c3de9d}#mode[data-mode=SUSPENDIDO]{background:#594a29;color:#ffe1a0;border-color:#9a834d}.status-row{display:flex;align-items:center;gap:12px;flex-wrap:wrap;padding:0 16px 9px}.reason{flex:1;min-width:180px;color:#d2deca;overflow-wrap:anywhere}.key-help{font-size:11px;color:#afbea7}kbd{border:1px solid #64735c;padding:1px 4px;border-radius:2px;font:10px ui-monospace,Consolas,monospace}.consent{display:flex;align-items:center;gap:12px;flex-wrap:wrap;padding:9px 16px;border-top:1px solid #43503e;background:#1b271f;color:#becbb6}.consent label{display:flex;align-items:center;gap:6px;cursor:pointer;flex:1}.consent input{accent-color:#c3de9d;margin:0;flex:none}.consent #service{font:10px ui-monospace,Consolas,monospace}.consent [data-ok=true]{color:#c3de9d}details{border-top:1px solid #43503e}summary{cursor:pointer;padding:7px 16px;color:#bcc9b4;font-size:11px}summary:hover{color:#fff}.workspace{max-height:min(380px,45vh);overflow:auto;padding:8px 16px 16px;background:#18231c}.grid{display:grid;grid-template-columns:1.1fr 1fr;gap:24px}.card{min-width:0}h2{font-size:12px;color:#e8ede5;font-weight:600;margin:0 0 10px}p{line-height:1.5;margin:8px 0;overflow-wrap:anywhere}.muted{color:#adbea4;font-size:11px}ul{list-style:none;padding:0;margin:0;display:grid;grid-template-columns:1fr 1fr;gap:4px 12px}li{color:#aab8a2;font-size:11px}[data-ok=true]{color:#c3de9d}.stats{display:flex;gap:28px}.stats b{display:block;font:24px ui-monospace,Consolas,monospace}.stats span{color:#b8c7af;font-size:11px}.actions,.foot{display:flex;gap:7px;align-items:center;flex-wrap:wrap;margin-top:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:11px/1.7 ui-monospace,Consolas,monospace;max-height:100px;overflow:auto;margin:8px 0;color:#bdcdb3}.event-card{border-top:1px solid #43503e;margin-top:18px;padding-top:12px}#feedback{color:#bdcdb3}[hidden]{display:none!important}@media(max-width:700px){.bar{gap:10px}.identity{width:100%}.steps{width:100%;display:grid;grid-template-columns:auto 1fr 1fr}.steps button{justify-content:center;padding:8px;font-size:11px;gap:5px}.grid{grid-template-columns:1fr;gap:16px}.competition span{display:inline;margin-right:8px}.status-row{gap:7px}.reason{flex-basis:75%}.key-help{display:none}}@media(max-width:370px){.bar,.status-row{padding-left:10px;padding-right:10px}.steps button{font-size:10px}.consent{padding:8px 10px}}
    </style>
    <div class="bar"><div class="identity"><span class="brand">RS4</span><div class="competition"><span>Real Soccer · 4 vs 4</span>rs4_4v4</div></div><div class="steps" role="group" aria-label="Activación del bot"><button id="pair" title="Configurar conexión y atajo"><span class="step-number" aria-hidden="true">1</span>Conectar</button><button id="qualify"><span class="step-number" aria-hidden="true">2</span><span id="qualify-label">Comprobar configuración</span></button><button id="toggle" disabled><span class="step-number" aria-hidden="true">3</span><span id="toggle-label">Ceder al bot</span></button></div></div>
    <div class="status-row"><strong id="mode">HUMANO</strong><span id="reason" class="reason" role="status"></span><span class="key-help"><span id="key"></span> · <kbd>Esc</kbd> volver</span></div>
    <div class="consent"><label><input id="permission" type="checkbox"> Esta sala es propia o tengo permiso para usar el bot.</label><span id="service"></span></div>
    <details><summary>Opciones y diagnóstico</summary><div class="workspace"><div class="grid">
    <section class="card"><h2>Estado de la sala</h2><ul id="checklist"></ul><p id="readiness" class="muted"></p><div class="actions"><button id="manager">Salas y bots</button><button id="claim" title="Solicitar el servicio si no hay otra pestaña propietaria">Tomar esta pestaña</button><button id="reload" title="Releer el modelo local actual; requiere nueva comprobación">Recargar modelo</button></div></section>
    <section class="card"><h2>Rendimiento del modelo</h2><div class="stats"><div><b id="inference">—</b><span>Tiempo de inferencia</span></div><div><b id="age">—</b><span>Edad de propuesta</span></div></div><p id="telemetry" class="muted"></p><p id="model" class="muted"></p><p id="metrics" class="muted"></p><p class="muted">Comprobación: p95 ≤ 50 ms · frame ≤ ×1.05 · mínimo 30 muestras. Prueba de 30 minutos en sala real pendiente.</p></section></div>
    <section class="card event-card"><h2>Registro de esta sesión</h2><pre id="events">Esperando cambios de estado…</pre><div class="foot"><button id="export">Descargar diagnóstico</button><button id="clear">Limpiar eventos</button><span id="feedback" role="status"></span></div><p class="muted">Últimos 60 eventos. El archivo no incluye token, chat ni frames de juego.</p></section></div></details>`;
    const find=id=>shadow.getElementById(id);
    find('key').textContent=shortcutLabel;
    elements=Object.fromEntries(['pair','toggle-label','qualify-label','mode','toggle','reason','model','permission','qualify','reload','metrics','readiness','key','service','checklist','inference','age','telemetry','events'].map(id=>[id,find(id)]));
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
