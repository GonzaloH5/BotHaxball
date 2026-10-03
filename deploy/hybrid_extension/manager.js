(()=>{
  const $=id=>document.getElementById(id),pending=new Map();
  let socket=null,authenticated=false,sequence=0,data={models:[],maps:[],bots:[],limit:8},profiles=[],lastInstances='',connectingGeneration=0;
  const statusLabels={queued:'En espera',starting:'Conectando',connected:'Conectado',stopping:'Deteniendo',stopped:'Detenido',failed:'Error'};
  function message(value){$('action-status').textContent=value;}
  function disconnected(reason){authenticated=false;$('connection-title').textContent='Gestor desconectado';$('connection-status').textContent=reason;$('connection-dot').dataset.connected='false';$('start').disabled=true;$('stop-all').disabled=true;lastInstances='';renderInstances();for(const p of pending.values()){clearTimeout(p.timer);p.reject(Error(reason));}pending.clear();}
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
        if(m.type==='authenticated'){clearTimeout(timeout);authenticated=true;$('connection-title').textContent='Gestor disponible';$('connection-status').textContent=`Servicio local · 127.0.0.1:${port}`;$('connection-dot').dataset.connected='true';$('start').disabled=false;refresh(true);return;}
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
  function options(select,items,value){select.replaceChildren(...items.map(([id,label])=>{const o=document.createElement('option');o.value=id;o.textContent=label;return o;}));select.value=value;if(select.selectedIndex<0)select.selectedIndex=0;}
  function renderProfiles(){
    $('profiles').replaceChildren(...profiles.map((p,i)=>{
      const card=document.createElement('div');card.className='profile';
      card.innerHTML='<div class="profile-head"><strong></strong><button type="button">Quitar</button></div><div class="profile-grid"><label>Nombre<input maxlength="25" required></label><label>Avatar<input maxlength="2"></label><label>País<input maxlength="2" pattern="[a-z]{2}" required></label><label class="wide">Modelo<select></select></label><label>Equipo<select></select></label></div>';
      card.querySelector('strong').textContent=`Bot ${i+1}`;
      const remove=card.querySelector('button');remove.disabled=profiles.length===1;remove.onclick=()=>{profiles.splice(i,1);renderProfiles();};
      const inputs=card.querySelectorAll('input');['name','avatar','flag'].forEach((key,j)=>{inputs[j].value=p[key];inputs[j].oninput=()=>p[key]=inputs[j].value;inputs[j].setAttribute('aria-label',`${key==='name'?'Nombre':key==='avatar'?'Avatar':'País'} del bot ${i+1}`);});
      const selects=card.querySelectorAll('select');options(selects[0],data.models.map(m=>[m.id,m.label]),p.model);p.model=selects[0].value;selects[0].onchange=()=>p.model=selects[0].value;selects[0].setAttribute('aria-label',`Modelo del bot ${i+1}`);
      options(selects[1],[['0','Espectador'],['1','Rojo'],['2','Azul']],String(p.team));selects[1].onchange=()=>p.team=Number(selects[1].value);selects[1].setAttribute('aria-label',`Equipo del bot ${i+1}`);return card;
    }));
    $('add-profile').disabled=profiles.length>=data.limit;$('capacity').textContent=`${profiles.length} / ${data.limit} bots por grupo. Límite compartido con las instancias activas.`;
  }
  function addProfile(){let n=profiles.length+1;while(profiles.some(p=>p.name===`RS4 Bot ${n}`))n++;profiles.push({name:`RS4 Bot ${n}`,avatar:String(n).slice(-2),flag:'uy',team:0,model:data.models[0]?.id||'current'});renderProfiles();}
  async function action(type,extra){try{data=await rpc(type,extra);renderInstances();message('Solicitud enviada. El estado se actualizará al confirmarse en la sala.');}catch(e){message(e.message);}}
  function renderInstances(){
    const signature=JSON.stringify([authenticated,data.bots]);if(signature===lastInstances)return;lastInstances=signature;
    const active=data.bots.filter(b=>['queued','starting','connected','stopping'].includes(b.status));$('stop-all').disabled=!authenticated||!active.length;
    if(!data.bots.length){$('instances').textContent='No hay bots iniciados desde este servicio.';return;}
    $('instances').replaceChildren(...data.bots.map(b=>{
      const row=document.createElement('article');row.className='instance';
      const head=document.createElement('div');head.className='instance-head';const name=document.createElement('strong');name.textContent=`${b.avatar||'·'}  ${b.name}${b.host?' · Host':''}`;const badge=document.createElement('span');badge.className='badge';badge.textContent=authenticated?(statusLabels[b.status]||b.status):'Sin actualizar';head.append(name,badge);row.append(head);
      const info=document.createElement('p');info.textContent=`Modelo: ${b.model} · Equipo: ${['Espectador','Rojo','Azul'][b.team]||'—'} · Control: ${b.enabled?'bot activo':'bot pausado'}${b.playing?' · Partido en curso':''}`;row.append(info);
      if(b.link){const link=document.createElement('a');link.textContent='Abrir sala ↗';link.href=b.link;link.target='_blank';link.rel='noreferrer';row.append(link);}
      if(b.error){const error=document.createElement('p');error.className='error';error.textContent=b.error;row.append(error);}
      const controls=document.createElement('div');controls.className='actions';controls.style.marginTop='10px';
      const button=(label,command,disabled)=>{const el=document.createElement('button');el.type='button';el.textContent=label;el.disabled=!authenticated||disabled;el.onclick=()=>action('command',{command:{id:b.id,...command}});controls.append(el);};
      const connected=b.status==='connected';button(b.enabled?'Pausar bot':'Reanudar bot',{action:'enabled',enabled:!b.enabled},!connected);
      const team=document.createElement('select');team.setAttribute('aria-label',`Equipo de ${b.name}`);options(team,[['0','Espectador'],['1','Rojo'],['2','Azul']],String(b.team));team.disabled=!authenticated||!connected;team.onchange=()=>action('command',{command:{id:b.id,action:'team',team:Number(team.value)}});controls.append(team);
      if(b.host){button('Iniciar partido',{action:'start'},!connected||b.playing);button('Pausar / continuar partido',{action:'pause'},!connected||!b.playing);button('Detener partido',{action:'stopGame'},!connected||!b.playing);}
      button(b.host?'Cerrar sala':'Desconectar bot',{action:'stop'},!['queued','starting','connected'].includes(b.status));row.append(controls);return row;
    }));
  }
  async function refresh(initial=false){try{data=await rpc('list');if(initial){options($('stadium'),data.maps.map(m=>[m,m]),'rs_one');renderProfiles();}renderInstances();}catch(e){message(e.message);}}
  function mode(){const host=$('mode').value==='host';$('host-fields').hidden=!host;$('join-fields').hidden=host;$('room').required=!host;$('room-name').required=host;$('headless-token').required=host;$('start').textContent=host?'Crear sala e iniciar bots':'Agregar bots a la sala';}
  $('mode').onchange=mode;$('add-profile').onclick=addProfile;$('reconnect').onclick=connect;$('stop-all').onclick=()=>action('stopAll');
  $('launch').onsubmit=async event=>{
    event.preventDefault();$('start').disabled=true;
    try{
      const config={mode:$('mode').value,room:$('room').value,name:$('room-name').value,password:$('room-password').value,token:$('headless-token').value,stadium:$('stadium').value,maxPlayers:Number($('max-players').value),timeLimit:Number($('time-limit').value),scoreLimit:Number($('score-limit').value),public:$('public-room').checked,profiles};
      data=await rpc('start',{config});$('headless-token').value='';$('room-password').value='';renderInstances();message('Grupo en preparación. Podés seguir el estado de cada bot abajo.');
      await chrome.storage.local.set({botProfiles:profiles});
    }catch(e){message(e.message);}finally{$('start').disabled=!authenticated;}
  };
  chrome.storage.local.get('botProfiles').then(s=>{profiles=Array.isArray(s.botProfiles)?s.botProfiles.slice(0,8).filter(p=>p&&typeof p.name==='string').map(p=>({name:p.name,avatar:p.avatar||'',flag:p.flag||'uy',team:[0,1,2].includes(p.team)?p.team:0,model:p.model||'current'})):[];if(!profiles.length)addProfile();else renderProfiles();connect();}).catch(()=>{addProfile();connect();});
  mode();setInterval(()=>{if(authenticated&&!document.hidden)refresh();},2000);
  window.addEventListener('pagehide',()=>{++connectingGeneration;socket?.close();});
})();
