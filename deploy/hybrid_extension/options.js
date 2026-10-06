const byId=id=>document.getElementById(id);
const Shortcut=globalThis.RS4Shortcut;let shortcut=null,shortcutEdited=false;
byId('show-guide').onclick=()=>{byId('startup-guide').open=true;byId('startup-guide').querySelector('summary').focus();};
const showShortcut=()=>{byId('shortcut').value=Shortcut.label(shortcut);};
chrome.storage.local.get(['port','token','shortcut']).then(s=>{byId('port').value=s.port||17841;byId('token').value=s.token||'';
  if(!shortcutEdited){shortcut=Shortcut.normalize(s.shortcut);showShortcut();}}).catch(()=>{byId('save-status').textContent='No se pudo leer la configuración. Volvé a abrir la extensión.';});
byId('version').textContent='v'+chrome.runtime.getManifest().version;
byId('show-token').onclick=()=>{const visible=byId('token').type==='password';byId('token').type=visible?'text':'password';byId('show-token').textContent=visible?'Ocultar':'Mostrar';byId('show-token').setAttribute('aria-label',visible?'Ocultar token':'Mostrar token');byId('show-token').setAttribute('aria-pressed',String(visible));};
byId('copy-command').onclick=async()=>{try{await navigator.clipboard.writeText(byId('command').textContent);byId('copy-status').textContent='Comando copiado.';}catch{byId('copy-status').textContent='No se pudo copiar. Seleccioná el comando y copialo manualmente.';}};
byId('shortcut').addEventListener('keydown',event=>{
  if(event.code==='Tab')return; // retain keyboard navigation
  event.preventDefault();event.stopPropagation();
  const selected=Shortcut.fromEvent(event);if(!selected)return;
  shortcutEdited=true;shortcut=selected;showShortcut();byId('shortcut-status').textContent='Atajo elegido; pulsá «Guardar atajo».';
});
byId('shortcut').addEventListener('keyup',event=>{if(event.code!=='Tab'){event.preventDefault();event.stopPropagation();}});
async function saveShortcut(){try{await chrome.storage.local.set({shortcut});byId('shortcut-status').textContent=shortcut?'Atajo guardado: '+Shortcut.label(shortcut):'Atajo desactivado; usá el botón del panel.';}catch{byId('shortcut-status').textContent='No se pudo guardar el atajo. Volvé a abrir la extensión.';}}
byId('save-shortcut').onclick=saveShortcut;
byId('clear-shortcut').onclick=()=>{shortcutEdited=true;shortcut=null;showShortcut();saveShortcut();};
document.getElementById('settings').onsubmit=async event=>{
  event.preventDefault();const port=Number(byId('port').value),token=byId('token').value.trim();
  if(!Number.isInteger(port)||port<1||port>65535||token.length<32)return;
  byId('connect').disabled=true;
  try{await chrome.storage.local.set({port,token,shortcut});await chrome.runtime.sendMessage({type:'pair'});
    byId('save-status').textContent='Configuración guardada. Conectando al servicio local…';await refreshStatus();
  }catch{byId('save-status').textContent='No se pudo conectar. Volvé a abrir la extensión e intentá nuevamente.';}
  finally{byId('connect').disabled=false;}
};
async function refreshStatus(){try{const s=await chrome.runtime.sendMessage({type:'status'});byId('status').textContent=s?.connected?`Conectado · ${s.model}`:s?.error||'Servicio desconectado: revisar arranque y token';byId('connection-title').textContent=s?.connected?'Servicio disponible':'Sin conexión';byId('connection-dot').dataset.connected=String(!!s?.connected);byId('next-step').hidden=!s?.connected;byId('owner').textContent=s?.connected?(s.owner==null?'Sin pestaña propietaria':`Pestaña propietaria: ${s.owner}`):'';}catch{byId('status').textContent='Extensión reiniciada; volver a abrir opciones';byId('connection-title').textContent='Canal no disponible';byId('connection-dot').dataset.connected='false';byId('next-step').hidden=true;byId('owner').textContent='';}}
refreshStatus();setInterval(refreshStatus,1000);
