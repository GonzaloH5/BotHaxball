/* Only this privileged context holds the local pairing token/socket. */
importScripts('shortcut.js');
const Shortcut=globalThis.RS4Shortcut;
let socket=null,authenticated=false,owner=null,reconnect=null,settings={},model=null,connecting=null,connectionError='';
chrome.storage.local.setAccessLevel({accessLevel:'TRUSTED_CONTEXTS'}).catch(()=>{connectionError='No se pudo proteger el almacenamiento de la extensión';});
const ports=new Map();
const shortcutMessage=()=>({type:'shortcut',shortcut:Shortcut.normalize(settings.shortcut),label:Shortcut.label(settings.shortcut)});
const reply=(tab,value)=>{try{ports.get(tab)?.postMessage(value);}catch{}};
function disconnect(){const previous=socket;socket=null;authenticated=false;model=null;owner=null;
  if(previous){previous.onopen=previous.onmessage=previous.onclose=null;try{previous.close();}catch{}}
  for(const tab of ports.keys())reply(tab,{type:'service',connected:false,reason:connectionError});}
function sendSocket(ws,data){if(ws!==socket||ws?.readyState!==WebSocket.OPEN)return false;
  try{ws.send(JSON.stringify(data));return true;}catch{connectionError='Conexión local cerrada; reconectá el servicio';disconnect();return false;}}
function send(data){if(authenticated)return sendSocket(socket,data);return false;}
function connect(){if(!connecting)connecting=connectImpl().catch(()=>{
  connectionError='No se pudo conectar al servicio local; revisá puerto y token';disconnect();
}).finally(()=>{connecting=null;});return connecting;}
async function connectImpl(){
  if(reconnect){clearTimeout(reconnect);reconnect=null;}disconnect();
  settings=await chrome.storage.local.get(['token','port','shortcut']);
  for(const tab of ports.keys())reply(tab,shortcutMessage());
  if(typeof settings.token!=='string'||settings.token.length<32){connectionError='Falta emparejar: abrí Emparejar / atajo e ingresá el token del servicio';disconnect();return;}
  const port=Number(settings.port||17841);if(!Number.isInteger(port)||port<1||port>65535){connectionError='Puerto local inválido';disconnect();return;}
  connectionError='';
  const ws=socket=new WebSocket(`ws://127.0.0.1:${port}/rs4`);
  ws.onopen=()=>sendSocket(ws,{type:'auth',token:settings.token});
  ws.onmessage=event=>{
    if(socket!==ws)return;
    let m;try{m=JSON.parse(event.data);}catch{return;}
    if(m.type==='authenticated'){authenticated=true;connectionError='';model=m.model;
      const first=ports.keys().next().value;if(first!==undefined)send({type:'claim',tabId:first});}
    else if(m.type==='claimed'){owner=m.tabId;reply(owner,{type:'service',connected:true,model:m.model});}
    else if(m.type==='denied')reply(m.tabId,{type:'error',message:m.message});
    else if(m.type==='mapReady')reply(owner,{type:'mapReady'});
    else if(m.type==='poll'&&m.tabId===owner)reply(owner,{type:'poll'});
    else if(m.type==='action')reply(owner,{type:'action',action:m});
    else if(m.type==='error')reply(owner,{type:'error',message:m.message});
    else if(m.type==='reloaded'){model=m.model;reply(owner,{type:'service',connected:true,model:m.model});}
  };
  ws.onerror=()=>{if(socket===ws)connectionError='Servicio local no disponible: revisá que esté iniciado y el puerto';};
  ws.onclose=event=>{if(socket!==ws)return;if(event.code===1008)connectionError='Emparejamiento rechazado: copiá el token del servidor actual';disconnect();if(ports.size)reconnect=setTimeout(connect,2000);};
}
chrome.runtime.onConnect.addListener(port=>{
  const tab=port.sender?.tab?.id,url=port.sender?.url;
  // MessageSender can report the official /play document rather than the game's
  // cached iframe URL. Both are same-origin; content injection remains narrowly
  // scoped by the manifest, with no additional origins or external messaging.
  const official=/^https:\/\/www\.haxball\.com\/(?:[^?#]*\/__cache_static__\/g\/game\.html|play)(?:[?#]|$)/.test(url||'');
  if(port.sender?.id!==chrome.runtime.id||port.name!=='rs4-hybrid'||!Number.isInteger(tab)||!official){port.disconnect();return;}
  if(ports.has(tab)){port.disconnect();return;}ports.set(tab,port);
  reply(tab,shortcutMessage());
  if(authenticated){if(owner===null)send({type:'claim',tabId:tab});else reply(tab,{type:'error',message:'Otra pestaña activa'});}
  else if(!socket||socket.readyState>1)connect();
  port.onMessage.addListener(m=>{
    if(m?.type==='openManager'){chrome.tabs.create({url:chrome.runtime.getURL('manager.html')}).catch(()=>reply(tab,{type:'error',message:'Abrí el gestor desde la configuración de la extensión.'}));return;}
    if(m?.type==='openOptions'){chrome.runtime.openOptionsPage().catch(()=>reply(tab,{type:'error',message:'No se pudo abrir opciones; abrí el popup de la extensión'}));return;}
    if(m?.type==='claim'){if(owner!==tab)send({type:'claim',tabId:tab});return;}
    if(ports.get(tab)!==port||owner!==tab)return;
    if(m.type==='frame')send({...m.frameData,type:'frame',tabId:tab});
    else if(['map','control','ack','reload'].includes(m.type)){
      const clean={...m,tabId:tab};delete clean.channel;delete clean.direction;send(clean);
    }else if(m.type==='release'){send({type:'release',tabId:tab});owner=null;}
  });
  port.onDisconnect.addListener(()=>{void chrome.runtime.lastError;if(ports.get(tab)!==port)return;ports.delete(tab);if(owner===tab){send({type:'release',tabId:tab});owner=null;}
    if(!ports.size)disconnect();});
});
chrome.runtime.onMessage.addListener((m,sender,respond)=>{
  if(sender.id!==chrome.runtime.id||sender.url!==chrome.runtime.getURL('options.html'))return;
  if(m?.type==='pair'){connect().then(()=>respond({ok:true}));return true;}
  if(m?.type==='status')respond({connected:authenticated,model:model?.name||'',owner,error:connectionError});
});
chrome.storage.onChanged.addListener(async()=>{settings=await chrome.storage.local.get(['token','port','shortcut']);
  for(const tab of ports.keys())reply(tab,shortcutMessage());});
// Chrome 116+ websocket messages keep the service worker alive.
setInterval(()=>send({type:'ping'}),20000);
