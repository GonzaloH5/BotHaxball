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
        if(m.type==='shortcut'){shortcutLabel=m.label||'Sin atajo';if(elements)elements.mode.nextElementSibling.nextElementSibling.textContent=shortcutLabel;}
        post(m.type,m);
      });
      connection.onDisconnect.addListener(()=>{
        // Reading lastError prevents an unchecked runtime.lastError diagnostic.
        let invalid=false;try{invalid=!chrome.runtime.id;void chrome.runtime.lastError;}catch{invalid=true;}
        lost(connection,invalid?'Extensión recargada: recargá también HaxBall.':'Canal de la extensión desconectado; reconectando…');
      });
    }catch{lost(port,'Extensión recargada o canal no disponible: recargá también HaxBall.');}
  }
  let status={},elements=null,lastQualification=null,shortcutLabel='Sin atajo';
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
  }
  function mount(){
    const host=document.createElement('div');host.id='rs4-hybrid-panel';
    const shadow=host.attachShadow({mode:'open'});
    shadow.innerHTML=`<style>
      :host{display:block;font:13px system-ui;color:#e8edf7;background:#17212c;position:relative;z-index:10}
      .bar{display:flex;align-items:center;gap:10px;padding:6px 10px;flex-wrap:wrap}button{border:1px solid #738499;border-radius:5px;background:#26394b;color:white;padding:5px 10px;cursor:pointer}button:disabled{opacity:.5;cursor:default}
      strong{min-width:90px;color:#a6d7ff}strong[data-mode=BOT]{color:#79eda8}strong[data-mode=SUSPENDIDO]{color:#ffd381}
      small{color:#b5c2d1}details{padding:0 10px 6px}summary{cursor:pointer}label{display:block;margin:8px 0}input{accent-color:#63c99a}.reason{max-width:550px}
    </style><div class="bar"><strong id="mode">HUMANO</strong><button id="toggle" disabled>Ceder al bot</button><small id="key"></small><small id="reason" class="reason"></small></div>
      <label style="padding:0 10px"><input id="permission" type="checkbox"> Esta sala es propia o tengo permiso para usar el bot.</label>
      <details><summary>Opciones y diagnóstico</summary><p id="model"></p>
      <p id="readiness"></p>
      <button id="qualify">Comprobar configuración</button> <button id="pair">Emparejar / atajo</button> <button id="claim">Tomar esta pestaña</button> <button id="reload">Recargar ONNX</button>
      <p id="metrics">La validación de 30 minutos en sala real sigue pendiente.</p></details>`;
    const find=id=>shadow.getElementById(id);
    find('key').textContent=shortcutLabel;
    elements=Object.fromEntries(['mode','toggle','reason','model','permission','qualify','reload','metrics','readiness'].map(id=>[id,find(id)]));
    find('toggle').onclick=()=>post('toggle');find('permission').onchange=()=>post('permission',{allowed:find('permission').checked});
    find('qualify').onclick=()=>post('qualify');find('pair').onclick=()=>send({type:'openOptions'});
    find('claim').onclick=()=>send({type:'claim'});find('reload').onclick=()=>post('reload');
    // Outside the gameplay canvas, no replacement of the official UI.
    document.body.prepend(host);
    const layout=document.createElement('style');layout.id='rs4-hybrid-layout';
    layout.textContent='.game-view{top:var(--rs4-hybrid-panel-height,0px)!important;height:calc(100vh - var(--rs4-hybrid-panel-height,0px))!important}';
    document.head.append(layout);
    const resize=new ResizeObserver(()=>document.documentElement.style.setProperty('--rs4-hybrid-panel-height',host.getBoundingClientRect().height+'px'));
    resize.observe(host);render();post('requestStatus');
    // The official client clears body contents during boot/screen changes.
    const screens=new MutationObserver(()=>{if(!host.isConnected&&document.body)document.body.prepend(host);});
    screens.observe(document.documentElement,{childList:true,subtree:true});
  }
  window.addEventListener('message',event=>{
    const m=event.data;if(event.source!==window||event.origin!==location.origin||m?.channel!=='rs4-hybrid-v1'||m.direction!=='page')return;
    if(m.type==='status'){status={...status,...m};render();}
    else if(m.type==='qualification'){lastQualification=m;if(elements)elements.metrics.textContent=`Preflight ${m.passed?'aprobado':'fallido'} · p95 ${m.latencyP95.toFixed(1)} ms · frame ×${m.frameRatio.toFixed(3)} · sala real 30 min: pendiente`;}
    else if(m.type==='diagnostic'&&elements){const ms=value=>Number.isFinite(value)?value.toFixed(1):'—';
      elements.model.textContent=`${status.model||'—'} · Adaptador: ${status.adapter||'—'} · joints ${m.visibleJoints??'—'} · balón ${Number.isInteger(m.ballColor)?'#'+m.ballColor.toString(16).padStart(6,'0'):'—'} · inferencia ${ms(m.inferenceMs)} ms · estado ${ms(m.stateAge)} ms`;}
    else if(['frame','map','control','ack','release','reload'].includes(m.type))send(m);
  });
  window.addEventListener('pagehide',()=>{closing=true;if(retry)clearTimeout(retry);const connection=port;port=null;try{connection?.disconnect();}catch{};},{once:true});
  connect();
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',mount,{once:true});else mount();
})();
