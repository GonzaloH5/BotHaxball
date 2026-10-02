const byId=id=>document.getElementById(id);
const Shortcut=globalThis.RS4Shortcut;let shortcut=null,shortcutEdited=false;
const showShortcut=()=>{byId('shortcut').value=Shortcut.label(shortcut);};
chrome.storage.local.get(['port','token','shortcut']).then(s=>{byId('port').value=s.port||17841;byId('token').value=s.token||'';
  if(!shortcutEdited){shortcut=Shortcut.normalize(s.shortcut);showShortcut();}});
byId('shortcut').addEventListener('keydown',event=>{
  if(event.code==='Tab')return; // retain keyboard navigation
  event.preventDefault();event.stopPropagation();
  const selected=Shortcut.fromEvent(event);if(!selected)return;
  shortcutEdited=true;shortcut=selected;showShortcut();byId('shortcut-status').textContent='Atajo elegido; pulsá «Guardar atajo».';
});
byId('shortcut').addEventListener('keyup',event=>{if(event.code!=='Tab'){event.preventDefault();event.stopPropagation();}});
async function saveShortcut(){await chrome.storage.local.set({shortcut});byId('shortcut-status').textContent=shortcut?'Atajo guardado: '+Shortcut.label(shortcut):'Atajo desactivado; usá el botón del panel.';}
byId('save-shortcut').onclick=saveShortcut;
byId('clear-shortcut').onclick=()=>{shortcutEdited=true;shortcut=null;showShortcut();saveShortcut();};
document.getElementById('settings').onsubmit=async event=>{
  event.preventDefault();const port=Number(byId('port').value),token=byId('token').value.trim();
  if(!Number.isInteger(port)||port<1||port>65535||token.length<32)return;
  await chrome.storage.local.set({port,token,shortcut});await chrome.runtime.sendMessage({type:'pair'});
  byId('status').textContent='Conectando al servicio local…';
};
setInterval(async()=>{try{const s=await chrome.runtime.sendMessage({type:'status'});byId('status').textContent=s?.connected?`Conectado · ${s.model}`:s?.error||'Servicio desconectado: revisar arranque y token';}catch{byId('status').textContent='Extensión reiniciada; volver a abrir opciones';}},1000);
