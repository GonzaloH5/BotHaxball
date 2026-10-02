/* Native page adapter. No credentials, socket, room join or model execution here. */
(()=>{
  'use strict';
  if(globalThis.__rs4HybridInstalled)return;globalThis.__rs4HybridInstalled=true;
  const profile=globalThis.RS4Official,{Arbiter,captureControllers}=globalThis.RS4Control;
  const Shortcut=globalThis.RS4Shortcut,shortcutKeys=new Set();
  const emit=(type,data={})=>window.postMessage({...data,channel:'rs4-hybrid-v1',direction:'page',type},location.origin);
  let view=null,verified=false,service=false,mapReady=false,qualified=false,model=null,mapObject=null,compatibilityError='',serviceError='';
  let lastFrame=-1,shortcut=null,localWrites=false,restoreView=null,lastMask=0,lastIdentity='',qualification=null,lastTick=performance.now(),sportingSignature='',lastDiagnostic=0,visibleSignals=null;
  const arbiter=new Arbiter({backgroundBot:true,write:mask=>{if(view){localWrites=true;try{profile.writeMask(view,mask);lastMask=mask;}finally{localWrites=false;}}},
    notify:status=>{emit('status',status);emit('control',status);}});
  function prerequisites(){
    const room=profile.roomStatus(view);
    const blocker=compatibilityError||serviceError||(!verified?'Verificando versión del cliente…':!service?'Servicio desconectado: abrí Emparejar / atajo.':room.blocker||
      (!mapReady?(arbiter.mode==='SUSPENDIDO'&&arbiter.reason?arbiter.reason:'Preparando el mapa RS4…'):!arbiter.permission?'Marcá la casilla de permiso para esta sala.':arbiter.mode==='BOT'?'Tomá control humano antes de comprobar.':''));
    return {room,blocker};
  }
  function state(){const {room,blocker}=prerequisites();emit('status',{...arbiter.status(),reason:compatibilityError||arbiter.reason,adapter:verified?profile.VERSION:'no compatible',service,mapReady,qualified,
    model:model?.name||'',ready:arbiter.ready,permission:arbiter.permission,qualifying:!!qualification,canQualify:!blocker,
    checkReason:blocker,toggleReason:blocker||(!qualified?'Comprobá la configuración antes de ceder al bot.':''),room});}
  // A disabled button cannot itself focus the game iframe. Focus is checked at
  // the explicit handoff, not used to lock an otherwise qualified panel.
  function ready(){arbiter.ready=verified&&service&&mapReady&&qualified&&!!view&&!profile.roomStatus(view).blocker;state();}
  function stop(reason){
    qualification=null;qualified=false;arbiter.ready=false;
    // Bot safety gates must never take away an already-manual native keyboard.
    // Do not repeatedly neutralize HUMAN on every failed observation/frame.
    if(arbiter.mode==='BOT')arbiter.suspend(reason);
    else if(arbiter.mode==='SUSPENDIDO')arbiter.suspend(reason);
    else arbiter.reason=reason;
    state();
  }
  function resetGame(reason){qualified=false;mapReady=false;mapObject=null;lastFrame=-1;stop(reason);}
  function attach(candidate){
    if(!profile.validView(candidate)||view===candidate)return;
    if(restoreView)restoreView();view=candidate;lastIdentity='';resetGame('Jugador detectado; preparar mapa');
    const w=view.W, originalDown=w.Fa,originalUp=w.ld,originalSend=w.Bg,originalApply=w.A,originalFrame=view.sf,originalLeave=view.la;
    w.Fa=function(event){if(arbiter.mode==='HUMANO'&&arbiter.focused&&!arbiter.blocked.has(event.code))return originalDown.call(this,event);};
    w.ld=function(event){if(arbiter.mode==='HUMANO'&&arbiter.focused)return originalUp.call(this,event);};
    w.Bg=function(operation){if(localWrites||arbiter.mode==='HUMANO'&&arbiter.focused)return originalSend.call(this,operation);};
    // Gate before native ng is updated, not merely at transport: native releases
    // (e.g. opening chat) must not create an unsent mask that later looks applied.
    w.A=function(...args){if(localWrites||arbiter.mode==='HUMANO'&&arbiter.focused)return originalApply.apply(this,args);};
    view.sf=function(...args){const result=originalFrame.apply(this,args);try{tick();}catch(e){serviceError=`Estado no compatible: ${e.message}. Control humano disponible.`;stop(serviceError);}return result;};
    view.la=function(...args){stop('Sala desconectada');emit('release');const result=originalLeave.apply(this,args);restoreView?.();return result;};
    const owned=view;
    restoreView=()=>{owned.W.Fa=originalDown;owned.W.ld=originalUp;owned.W.Bg=originalSend;owned.W.A=originalApply;owned.sf=originalFrame;owned.la=originalLeave;
      if(view===owned)view=null;restoreView=null;};
  }
  const removeCapture=captureControllers(window,attach);
  async function verify(){
    try{
      const url=new URL('game-min.js',location.href),response=await fetch(url,{cache:'no-cache'});
      if(!response.ok)throw Error('No se pudo verificar el cliente');
      const bytes=await response.arrayBuffer(),hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),b=>b.toString(16).padStart(2,'0')).join('');
      if(hash!==profile.SHA256)throw Error('Cliente no compatible: bot deshabilitado');
      verified=true;ready();
    }catch(e){compatibilityError=e.message;verified=false;removeCapture();if(restoreView)restoreView();stop(e.message);}
  }
  const percentile=(values,q)=>values.length?values.slice().sort((a,b)=>a-b)[Math.min(values.length-1,Math.floor(values.length*q))]:Infinity;
  function qualifyTick(now){
    const q=qualification;if(!q)return;
    if(!arbiter.focused||document.hidden||arbiter.mode!=='HUMANO')return stop('Validación interrumpida');
    if(q.previous!==null){const dt=now-q.previous;if(dt>0)q.intervals[q.stage%2?'inference':'baseline'].push(dt);}
    q.previous=now;q.frames++;
    if(q.frames>=120){q.stage++;q.frames=0;q.previous=null;}
    if(q.stage>=6){
      const baseline=percentile(q.intervals.baseline,.5),inference=percentile(q.intervals.inference,.5),latency=percentile(q.latencies,.95);
      const passed=q.latencies.length>=30&&latency<=50&&inference<=baseline*1.05;
      qualification=null;qualified=passed;emit('qualification',{passed,latencyP95:latency,frameRatio:inference/baseline,
        samples:q.latencies.length,realRoomThirtyMinutes:false});ready();
      if(!passed)arbiter.reason='Preflight falló: latencia o frame time';state();
    }
  }
  function tick(){
    const now=performance.now();qualifyTick(now);lastTick=now;
    if(!verified||!service||!view)return;
    const room=profile.roomStatus(view);
    if(room.blocker){
      if(arbiter.mode==='BOT'||qualification)stop(room.blocker);
      arbiter.ready=false;if(now-lastDiagnostic>1000){lastDiagnostic=now;state();}return;
    }
    if(!arbiter.ready&&qualified)ready();
    const raw=profile.snapshot(view);
    if(!raw){if(arbiter.mode==='BOT')stop('Jugador espectador o partido detenido');return;}
    visibleSignals={visibleJoints:raw.joints.length,ballColor:raw.ball.color};
    if(now-lastDiagnostic>1000){lastDiagnostic=now;emit('diagnostic',{...visibleSignals,inferenceMs:null,stateAge:0});}
    const identity=[raw.playerId,raw.players.find(p=>p.id===raw.playerId).team,raw.players.map(p=>p.id+':'+p.team).join(',')].join('|');
    if(identity!==lastIdentity){lastIdentity=identity;arbiter.setSession(crypto.randomUUID());resetGame('Cambió jugador/equipo');}
    if(raw.paused||!raw.synchronized){stop('Partido pausado o desincronizado');return;}
    if(view.za.T.U!==mapObject){mapObject=view.za.T.U;mapReady=false;qualified=false;qualification=null;serviceError='';arbiter.prepareHuman('Preparando mapa');
      let stadium;
      try{stadium=profile.stadium(view);}catch(e){
        serviceError=`No se pudo preparar este mapa: ${e.message}. Servicio conectado; control humano disponible.`;stop(serviceError);return;
      }
      if(stadium.name.trim()!=='Real Soccer ONE'){
        serviceError=`Mapa «${stadium.name}» no compatible con este modelo (Real Soccer ONE). Servicio conectado; podés jugar manualmente.`;
        stop(serviceError);return;
      }
      emit('map',{stadium});return;}
    if(!mapReady)return;
    const sporting=[raw.phase,...raw.score].join(':');
    if(sporting!==sportingSignature){sportingSignature=sporting;qualification=null;
      arbiter.transition(arbiter.mode==='BOT'?'BOT':'HUMANO','Reinicio deportivo');state();}
    if(raw.phase===2||raw.phase===3){arbiter.lastResponse=now;return;}
    if(raw.frame<lastFrame){lastFrame=-1;stop('Reloj del partido reiniciado');emit('control',arbiter.status());}
    if(raw.frame-lastFrame<(model?.frameSkip||3))return;lastFrame=raw.frame;
    const f={...raw,session:arbiter.session,epoch:arbiter.epoch,capturedAt:now,qualify:!!qualification&&qualification.stage%2===1};
    emit('frame',{frameData:f}); // Baseline measures HUMAN relay versus relay + dry inference.
  }
  window.addEventListener('message',event=>{
    if(event.source!==window||event.origin!==location.origin||event.data?.channel!=='rs4-hybrid-v1'||event.data.direction!=='extension')return;
    const m=event.data;
    if(m.type==='service'){service=!!m.connected;serviceError=!service?String(m.reason||''):'';model=m.model||model;if(!service)stop(serviceError||'Servicio desconectado');else{
      qualified=false;mapReady=false;mapObject=null;qualification=null;arbiter.prepareHuman('Preparar configuración');ready();}}
    else if(m.type==='poll'){
      // Server-paced messages remain usable when the native rAF is paused.
      // Advance the same native client; never create another room connection.
      if(verified&&service&&view&&(!arbiter.focused||document.hidden||!document.hasFocus()||performance.now()-lastTick>100)){
        arbiter.watchdog();try{if(typeof view.za.A!=='function')throw Error('Cliente sin actualización nativa en segundo plano');view.za.A();tick();}catch(e){stop(e.message);}
      }
    }
    else if(m.type==='mapReady'){mapReady=true;ready();}
    else if(m.type==='permission'){arbiter.permission=m.allowed===true;if(!arbiter.permission)arbiter.transition('HUMANO','Permiso retirado');ready();}
    else if(m.type==='toggle'){arbiter.focus(!document.hidden&&document.hasFocus());ready();arbiter.toggle();state();}
    else if(m.type==='qualify'){
      arbiter.focus(!document.hidden&&document.hasFocus());
      if(prerequisites().blocker||!arbiter.focused||arbiter.mode!=='HUMANO'){state();return;}
      qualified=false;qualification={stage:0,frames:0,previous:null,intervals:{baseline:[],inference:[]},latencies:[]};state();
    }
    else if(m.type==='action'){
      const a=m.action;
      if(a.qualify){if(qualification&&a.epoch===arbiter.epoch&&a.session===arbiter.session){const age=performance.now()-a.capturedAt;
        if(age>=0)qualification.latencies.push(age);}return;}
      const accepted=arbiter.accept(a);emit('ack',{...a,accepted,mask:accepted?lastMask:-1});
      emit('diagnostic',{...visibleSignals,inferenceMs:a.inferenceMs,stateAge:performance.now()-a.capturedAt});
    }
    else if(m.type==='error'){serviceError=String(m.message||'Error del servicio');stop(serviceError);}
    else if(m.type==='shortcut')shortcut=Shortcut.normalize(m.shortcut);
    else if(m.type==='reload'&&arbiter.mode==='HUMANO'){arbiter.transition('HUMANO','Recargando modelo');qualified=false;mapReady=false;mapObject=null;emit('reload');}
    else if(m.type==='requestStatus')state();
  });
  document.addEventListener('keydown',event=>{
    if(!event.isTrusted)return;
    const editing=(event.composedPath?.()||[event.target]).some(element=>element?.matches?.('input,textarea,select,[contenteditable]:not([contenteditable="false"])'))||event.target?.isContentEditable;
    // A trusted gameplay press is an explicit human takeover after BOT failure.
    // Recover before tracking this key, so it is not treated as previously held.
    const gameKey=/^(?:Key[WASDX]|Arrow(?:Up|Down|Left|Right)|Space|Control(?:Left|Right)|Shift(?:Left|Right)|Numpad0)$/.test(event.code);
    if(arbiter.mode==='SUSPENDIDO'&&!editing&&(gameKey||event.code==='Escape')){
      qualification=null;qualified=false;arbiter.ready=false;arbiter.transition('HUMANO','Control humano recuperado; bot requiere validación');state();
    }
    // Focus notifications can lag an iframe/popup switch. The trusted native
    // key event is authoritative for foreground manual control, not for BOT.
    if(arbiter.mode==='HUMANO'&&document.hasFocus()&&!document.hidden)arbiter.focused=true;
    arbiter.key(event.code,true);
    if(event.code==='Escape')arbiter.transition('HUMANO');
    if(Shortcut.matches(event,shortcut)&&!editing){event.preventDefault();event.stopImmediatePropagation();shortcutKeys.add(event.code);if(!event.repeat)arbiter.toggle();}
  },true);
  document.addEventListener('keyup',event=>{if(event.isTrusted){arbiter.key(event.code,false);
    if(shortcutKeys.delete(event.code)){event.preventDefault();event.stopImmediatePropagation();}}},true);
  window.addEventListener('blur',()=>{arbiter.focus(false);qualification=null;state();});
  window.addEventListener('focus',()=>{arbiter.focus(!document.hidden);ready();});
  document.addEventListener('visibilitychange',()=>{arbiter.focus(!document.hidden&&document.hasFocus());qualification=null;ready();});
  document.addEventListener('freeze',()=>stop('El navegador congeló la pestaña'));
  let lastStatus=0;
  setInterval(()=>{arbiter.watchdog();if(arbiter.mode==='BOT'&&performance.now()-lastTick>250)stop('El navegador dejó de actualizar la partida');
    if(performance.now()-lastStatus>1000){lastStatus=performance.now();state();}},50);
  window.addEventListener('pagehide',()=>{arbiter.suspend('Página cerrada');emit('release');removeCapture();restoreView?.();},{once:true});
  verify();
})();
