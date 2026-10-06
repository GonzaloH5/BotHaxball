(()=>{
  const $=id=>document.getElementById(id),pending=new Map();
  let socket=null,authenticated=false,sequence=0,data={models:[],maps:[],bots:[],limit:8},profiles=[],lastInstances='',connectingGeneration=0;
  const Presets=globalThis.RS4Presets;
  let savedPresets=[],loadedId='',baseline='',presetBusy=false,undo=null,launchBusy=false,catalogReady=false;
  const statusLabels={queued:'En espera',starting:'Conectando',connected:'Conectado',stopping:'Deteniendo',stopped:'Detenido',failed:'Error'};
  function message(value){$('action-status').textContent=value;}
  function disconnected(reason){authenticated=false;catalogReady=false;$('connection-title').textContent='Gestor desconectado';$('connection-status').textContent=reason;$('connection-dot').dataset.connected='false';$('start').disabled=true;$('stop-all').disabled=true;lastInstances='';renderInstances();for(const p of pending.values()){clearTimeout(p.timer);p.reject(Error(reason));}pending.clear();}
  async function connect(){
    const generation=++connectingGeneration;
    if(socket){socket.close();socket=null;}disconnected('Conectando al servicio local…');
    try{
      const settings=await chrome.storage.local.get(['port','token']);if(generation!==connectingGeneration)return;
      const port=Number(settings.port||17841);
      if(!Number.isInteger(port)||port<1||port>65535||typeof settings.token!=='string'||settings.token.length<32)throw Error('Guardá puerto y token local en Configuración.');
      const ws=socket=new WebSocket(`ws://127.0.0.1:${port}/fleet`);
      const timeout=setTimeout(()=>{if(socket===ws){ws.close();disconnected('El gestor no respondió. Reiniciá el servicio con la versión actualizada.');}},6000);
      ws.onopen=()=>{if(socket===ws)ws.send(JSON.stringify({type:'auth',token:settings.token}));};
      ws.onmessage=event=>{
        if(socket!==ws)return;let m;try{m=JSON.parse(event.data);}catch{return;}
        if(m.type==='authenticated'){clearTimeout(timeout);authenticated=true;$('connection-title').textContent='Gestor disponible';$('connection-status').textContent=`Servicio local · 127.0.0.1:${port}`;$('connection-dot').dataset.connected='true';catalogReady=false;refresh(true);return;}
        const p=pending.get(m.id);if(!p)return;pending.delete(m.id);clearTimeout(p.timer);if(m.ok)p.resolve(m.result);else p.reject(Error(m.error||'No se pudo completar la acción.'));
      };
      ws.onerror=()=>{};
      ws.onclose=e=>{clearTimeout(timeout);if(socket===ws){socket=null;disconnected(e.code===1008?'Token local rechazado. Actualizalo en Configuración.':'Servicio no disponible. Iniciá o actualizá hybrid_server.js y pulsá Reconectar.');}};
    }catch(e){if(generation===connectingGeneration)disconnected(e.message);}
  }
  function rpc(type,extra={}){return new Promise((resolve,reject)=>{
    if(!authenticated||socket?.readyState!==WebSocket.OPEN)return reject(Error('Conectá el gestor primero.'));
    const id=++sequence,timer=setTimeout(()=>{pending.delete(id);reject(Error('Sin respuesta del gestor. Actualizá la lista antes de repetir la acción.'));},8000);
    pending.set(id,{resolve,reject,timer});try{socket.send(JSON.stringify({id,type,...extra}));}catch(e){clearTimeout(timer);pending.delete(id);reject(e);}
  });}
  function options(select,items,value){select.replaceChildren(...items.map(([id,label])=>{const o=document.createElement('option');o.value=id;o.textContent=label;return o;}));if(value&&!items.some(([id])=>id===value)){const o=document.createElement('option');o.value=value;o.textContent=`${catalogReady?'No disponible':'Pendiente de conexión'} — ${value}`;select.append(o);}select.value=value;if(select.selectedIndex<0)select.selectedIndex=0;}
  function renderProfiles(){
    $('profiles').replaceChildren(...profiles.map((p,i)=>{
      const card=document.createElement('div');card.className='profile';
      card.innerHTML='<div class="profile-head"><strong></strong><button type="button">Quitar</button></div><div class="profile-grid"><label>Nombre<input maxlength="25" required></label><label>Avatar<input maxlength="2"></label><label>País<input maxlength="2" pattern="[a-z]{2}" required></label><label class="wide">Modelo<select></select></label><label>Equipo<select></select></label></div>';
      card.querySelector('strong').textContent=`Bot ${i+1}`;
      const remove=card.querySelector('button');remove.disabled=profiles.length===1;remove.onclick=()=>{profiles.splice(i,1);renderProfiles();edited();};
      const inputs=card.querySelectorAll('input');['name','avatar','flag'].forEach((key,j)=>{inputs[j].value=p[key];inputs[j].oninput=()=>{p[key]=inputs[j].value;edited();};inputs[j].setAttribute('aria-label',`${key==='name'?'Nombre':key==='avatar'?'Avatar':'País'} del bot ${i+1}`);});
      const selects=card.querySelectorAll('select');options(selects[0],data.models.map(m=>[m.id,m.label]),p.model);p.model=selects[0].value;selects[0].onchange=()=>{p.model=selects[0].value;edited();};selects[0].setAttribute('aria-label',`Modelo del bot ${i+1}`);
      options(selects[1],[['0','Espectador'],['1','Rojo'],['2','Azul']],String(p.team));selects[1].onchange=()=>{p.team=Number(selects[1].value);edited();};selects[1].setAttribute('aria-label',`Equipo del bot ${i+1}`);return card;
    }));
    $('add-profile').disabled=profiles.length>=data.limit;$('capacity').textContent=`${profiles.length} / ${data.limit} bots por grupo. Límite compartido con las instancias activas.`;updateLaunch();
  }
  function addProfile(){let n=profiles.length+1;while(profiles.some(p=>p.name===`RS4 Bot ${n}`))n++;profiles.push({name:`RS4 Bot ${n}`,avatar:String(n).slice(-2),flag:'uy',team:0,model:data.models[0]?.id||'current'});renderProfiles();edited();}
  async function action(type,extra){try{data=await rpc(type,extra);renderInstances();message('Solicitud enviada. El estado se actualizará al confirmarse en la sala.');}catch(e){message(e.message);}}
  function renderInstances(){
    updateLaunch();const signature=JSON.stringify([authenticated,data.bots,data.models,data.limit]);if(signature===lastInstances)return;lastInstances=signature;
    const active=data.bots.filter(b=>['queued','starting','connected','stopping'].includes(b.status));$('active-count').textContent=authenticated?String(active.length):'—';$('active-limit').textContent='/ '+data.limit;$('stop-all').disabled=!authenticated||!active.length;
    if(!data.bots.length){$('instances').innerHTML='<div class="empty-state"><strong>El plantel todavía está vacío.</strong><p>Elegí una sala y agregá tu primer grupo arriba.</p></div>';return;}
    $('instances').replaceChildren(...data.bots.map(b=>{
      const row=document.createElement('article');row.className='instance';
      const head=document.createElement('div');head.className='instance-head';const name=document.createElement('strong');name.textContent=`${b.avatar||'·'}  ${b.name}${b.host?' · Host':''}`;const badge=document.createElement('span');badge.className='badge';badge.dataset.state=authenticated?b.status:'offline';badge.textContent=authenticated?(statusLabels[b.status]||b.status):'Sin actualizar';head.append(name,badge);row.append(head);
      const info=document.createElement('p');info.textContent=`Modelo: ${data.models.find(m=>m.id===b.model)?.label||b.model} · Equipo: ${['Espectador','Rojo','Azul'][b.team]||'—'} · Control: ${b.enabled?'bot activo':'bot pausado'}${b.playing?' · Partido en curso':''}`;row.append(info);
      if(b.link){const link=document.createElement('a');link.textContent='Abrir sala ↗';link.href=b.link;link.target='_blank';link.rel='noreferrer';row.append(link);}
      if(b.error){const error=document.createElement('p');error.className='error';error.textContent=b.error;row.append(error);}
      const controls=document.createElement('div');controls.className='actions';controls.style.marginTop='10px';
      const button=(label,command,disabled)=>{const el=document.createElement('button');el.type='button';el.textContent=label;if(command.action==='stop')el.className='danger';el.disabled=!authenticated||disabled;el.onclick=()=>action('command',{command:{id:b.id,...command}});controls.append(el);};
      const connected=b.status==='connected';button(b.enabled?'Pausar bot':'Reanudar bot',{action:'enabled',enabled:!b.enabled},!connected);
      const team=document.createElement('select');team.setAttribute('aria-label',`Equipo de ${b.name}`);options(team,[['0','Espectador'],['1','Rojo'],['2','Azul']],String(b.team));team.disabled=!authenticated||!connected;team.onchange=()=>action('command',{command:{id:b.id,action:'team',team:Number(team.value)}});controls.append(team);
      if(b.host){button('Iniciar partido',{action:'start'},!connected||b.playing);button('Pausar / continuar partido',{action:'pause'},!connected||!b.playing);button('Detener partido',{action:'stopGame'},!connected||!b.playing);}
      button(b.host?'Cerrar sala':'Desconectar bot',{action:'stop'},!['queued','starting','connected'].includes(b.status));row.append(controls);return row;
    }));
  }
  async function refresh(initial=false){try{data=await rpc('list');catalogReady=true;if(initial){options($('stadium'),data.maps.map(m=>[m,m==='rs_one'?'Real Soccer ONE':m]),$('stadium').value||'rs_one');renderProfiles();}renderInstances();}catch(e){message(e.message);}}
  function mode(){const host=$('mode').value==='host';$('host-fields').hidden=!host;$('join-fields').hidden=host;$('room').required=!host;$('room-name').required=host;$('headless-token').required=host;for(const input of $('host-fields').querySelectorAll('input,select'))input.disabled=!host;$('room').disabled=host;$('start').textContent=host?'Crear sala e iniciar bots':'Agregar bots a la sala';updateLaunch();}
  $('mode').onchange=()=>{mode();edited();};$('add-profile').onclick=addProfile;$('reconnect').onclick=connect;$('stop-all').onclick=()=>action('stopAll');
  $('launch').onsubmit=async event=>{
    event.preventDefault();if(launchBusy)return;launchBusy=true;updateLaunch();
    try{
      const config={mode:$('mode').value,room:$('room').value,name:$('room-name').value,password:$('room-password').value,token:$('headless-token').value,stadium:$('stadium').value,maxPlayers:Number($('max-players').value),timeLimit:Number($('time-limit').value),scoreLimit:Number($('score-limit').value),public:$('public-room').checked,profiles};
      data=await rpc('start',{config});$('headless-token').value='';$('room-password').value='';renderInstances();message('Grupo en preparación. Podés seguir el estado de cada bot abajo.');
      try{await chrome.storage.local.set({botProfiles:Presets.clean({profiles}).profiles});}catch{message('Grupo iniciado. No se pudo recordar el plantel en este navegador.');}
    }catch(e){message(e.message);}finally{launchBusy=false;updateLaunch();}
  };
  function capture(){return Presets.clean({mode:$('mode').value,room:$('room').value,name:$('room-name').value,stadium:$('stadium').value,maxPlayers:Number($('max-players').value),timeLimit:Number($('time-limit').value),scoreLimit:Number($('score-limit').value),public:$('public-room').checked,profiles});}
  function apply(config){
    const c=Presets.clean(config);profiles=c.profiles;
    for(const [id,key] of [['mode','mode'],['room','room'],['room-name','name'],['max-players','maxPlayers'],['time-limit','timeLimit'],['score-limit','scoreLimit']])$(id).value=c[key];
    $('public-room').checked=c.public;
    options($('stadium'),data.maps.map(m=>[m,m==='rs_one'?'Real Soccer ONE':m]),c.stadium);
    $('headless-token').value='';$('room-password').value='';mode();renderProfiles();
  }
  function updateLaunch(){
    const active=data.bots.filter(b=>['queued','starting','connected','stopping'].includes(b.status)).length;
    const missing=profiles.find(p=>!data.models.some(m=>m.id===p.model));
    let reason=!authenticated?'Conectá el servicio para iniciar. Podés guardar y cargar presets sin conexión.':!catalogReady?'Consultando modelos y mapas…':
      missing?`El modelo de ${missing.name||'un bot'} no está disponible. Elegí otro modelo o iniciá el servicio correspondiente.`:
      $('mode').value==='host'&&!data.maps.includes($('stadium').value)?'El mapa guardado no está disponible. Elegí uno de la lista.':
      active+profiles.length>data.limit?`Quedan ${Math.max(0,data.limit-active)} lugares libres; este grupo necesita ${profiles.length}. Detené bots o reducí el plantel.`:'';
    $('start').disabled=launchBusy||!!reason;$('start').setAttribute('aria-busy',String(launchBusy));
    $('launch-help').textContent=launchBusy?'Iniciando el grupo…':reason;
  }
  function presetMessage(value){$('preset-status').textContent=value;}
  function presetControls(){
    const selected=savedPresets.find(p=>p.id===$('preset-select').value);
    for(const id of ['preset-load','preset-delete'])$(id).disabled=presetBusy||!selected;
    $('preset-copy').disabled=presetBusy||!loadedId;$('preset-save').disabled=presetBusy;
    $('preset-select').disabled=presetBusy;$('preset-undo').disabled=presetBusy;
    $('preset-save').textContent=loadedId?'Actualizar preset':'Guardar preset';
    $('preset-current').textContent=loadedId?'En edición: '+(savedPresets.find(p=>p.id===loadedId)?.name||'preset cargado'):'Configuración nueva. Guardala con un nombre para volver a usarla.';
  }
  function renderPresets(selected=loadedId){
    options($('preset-select'),[['','Elegí un preset'],...savedPresets.map(p=>[p.id,p.name])],savedPresets.some(p=>p.id===selected)?selected:'');presetControls();
  }
  function edited(){
    $('preset-dirty').textContent=baseline&&(JSON.stringify(capture())!==baseline||$('preset-name').value!==savedPresets.find(p=>p.id===loadedId)?.name)?'Cambios sin guardar':'';
    updateLaunch();
  }
  async function presetOperation(fn){
    if(presetBusy)return;presetBusy=true;presetControls();
    try{await fn();}catch{presetMessage('No se pudo guardar el cambio en el navegador. Tu configuración sigue en el formulario; volvé a intentar.');}
    finally{presetBusy=false;presetControls();}
  }
  function rememberUndo(value){undo=value;$('preset-undo').hidden=false;$('preset-undo').textContent=value.kind==='delete'?'Deshacer eliminación':'Deshacer carga';}
  async function loadPreset(){
    const preset=savedPresets.find(p=>p.id===$('preset-select').value);if(!preset)return;
    const previous={kind:'load',config:capture(),id:loadedId,name:$('preset-name').value,baseline};
    await chrome.storage.local.set({[Presets.lastKey]:preset.id});
    rememberUndo(previous);loadedId=preset.id;apply(preset.config);$('preset-name').value=preset.name;baseline=JSON.stringify(capture());edited();presetControls();
    presetMessage(`«${preset.name}» cargado. Revisá la sala${preset.config.mode==='host'?' e ingresá un token headless reciente':''} antes de iniciar. Los bots no se inician al cargar.`);
  }
  async function savePreset(copy=false){
    const name=$('preset-name').value.trim();
    if(!name){presetMessage('Poné un nombre para reconocer esta configuración.');$('preset-name').focus();return;}
    const fields=[...$('profiles').querySelectorAll('input')];
    if($('mode').value==='host')fields.push(...['room-name','max-players','time-limit','score-limit'].map($));
    const invalid=fields.find(input=>!input.checkValidity());
    if(invalid){presetMessage('Revisá el campo marcado antes de guardar el preset.');invalid.reportValidity();return;}
    const id=copy||!loadedId?crypto.randomUUID():loadedId;
    if(savedPresets.some(p=>p.id!==id&&p.name.toLocaleLowerCase()===name.toLocaleLowerCase())){presetMessage('Ya existe un preset con ese nombre. Elegí otro nombre para la copia.');$('preset-name').focus();return;}
    const preset={id,name,version:Presets.version,config:capture()};
    await chrome.storage.local.set({[Presets.prefix+id]:preset,[Presets.lastKey]:id});
    savedPresets=savedPresets.filter(p=>p.id!==id).concat(preset);loadedId=id;baseline=JSON.stringify(preset.config);$('preset-name').value=name;renderPresets();edited();
    presetMessage(`«${name}» guardado en este navegador. Podés cerrar el gestor y retomarlo después.`);
  }
  async function deletePreset(){
    const preset=savedPresets.find(p=>p.id===$('preset-select').value);if(!preset)return;
    await chrome.storage.local.remove(Presets.prefix+preset.id);
    rememberUndo({kind:'delete',preset,wasLoaded:loadedId===preset.id,baseline});savedPresets=savedPresets.filter(p=>p.id!==preset.id);
    if(loadedId===preset.id){loadedId='';baseline='';}
    renderPresets();edited();presetMessage(`«${preset.name}» eliminado. Tu sala y plantel siguen en el formulario. Podés deshacerlo.`);
  }
  async function undoPreset(){
    if(!undo)return;
    if(undo.kind==='delete'){
      const p=undo.preset;await chrome.storage.local.set({[Presets.prefix+p.id]:p});savedPresets.push(p);if(undo.wasLoaded){loadedId=p.id;baseline=undo.baseline;}renderPresets(p.id);edited();presetMessage(`«${p.name}» restaurado.`);
    }else{
      await chrome.storage.local.set({[Presets.lastKey]:undo.id});loadedId=undo.id;apply(undo.config);$('preset-name').value=undo.name;baseline=undo.baseline;renderPresets();edited();presetMessage('Se recuperó la configuración anterior. Volvé a ingresar las credenciales si hacen falta.');
    }
    undo=null;$('preset-undo').hidden=true;
  }
  $('preset-select').onchange=presetControls;
  $('preset-name').oninput=edited;
  $('preset-load').onclick=()=>presetOperation(loadPreset);
  $('preset-save').onclick=()=>presetOperation(()=>savePreset());
  $('preset-copy').onclick=()=>presetOperation(()=>savePreset(true));
  $('preset-delete').onclick=()=>presetOperation(deletePreset);
  $('preset-undo').onclick=()=>presetOperation(undoPreset);
  $('launch').addEventListener('input',edited);$('launch').addEventListener('change',edited);
  async function initialize(){
    try{
      const stored=await chrome.storage.local.get(null);
      savedPresets=Object.entries(stored).filter(([key,p])=>key.startsWith(Presets.prefix)&&p?.version===Presets.version&&key===Presets.prefix+p.id&&typeof p.name==='string').map(([,p])=>({id:p.id,name:p.name.slice(0,60),version:Presets.version,config:Presets.clean(p.config)}));
      const last=savedPresets.find(p=>p.id===stored[Presets.lastKey]);
      if(last){loadedId=last.id;apply(last.config);$('preset-name').value=last.name;baseline=JSON.stringify(capture());presetMessage(`Retomaste «${last.name}». Revisá la sala y las credenciales antes de iniciar.`);}
      else apply({profiles:stored.botProfiles});
      renderPresets();edited();
    }catch{apply({});presetMessage('No se pudieron leer los presets. Podés preparar el grupo e intentar guardarlo.');}
    connect();
  }
  initialize();
  mode();setInterval(()=>{if(authenticated&&!document.hidden)refresh();},2000);
  window.addEventListener('pagehide',()=>{++connectingGeneration;socket?.close();});
})();
