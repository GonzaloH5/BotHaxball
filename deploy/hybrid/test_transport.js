// Extension transport regression tests: no browser, room or real credentials.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
class Signal{constructor(){this.listeners=[];}addListener(fn){this.listeners.push(fn);}emit(...args){for(const fn of this.listeners)fn(...args);}}
function port(sender={}){return {sender,name:'rs4-hybrid',onMessage:new Signal,onDisconnect:new Signal,sent:[],closed:false,
  postMessage(m){if(this.closed)throw Error('Attempting to use a disconnected port object');this.sent.push(m);},
  disconnect(){if(this.closed)return;this.closed=true;this.onDisconnect.emit();}};}
const code=name=>fs.readFileSync(path.join(__dirname,'../hybrid_extension',name),'utf8');
const flush=()=>new Promise(resolve=>setImmediate(resolve));
async function run(){
  const events={},messages=[],timers=new Map,connections=[];
  const window={addEventListener(type,fn){events[type]=fn;},postMessage(m){messages.push(m);}};
  const runtime={id:'transport-test',lastError:null,connect(){const p=port();connections.push(p);return p;}};
  const context=vm.createContext({window,location:{origin:'https://www.haxball.com'},chrome:{runtime},
    document:{readyState:'loading',addEventListener(){}},setTimeout(fn){const id=timers.size+1;timers.set(id,fn);return id;},clearTimeout(id){timers.delete(id);}});
  vm.runInContext(code('content.js'),context);
  const relay=n=>events.message({source:window,origin:'https://www.haxball.com',data:{channel:'rs4-hybrid-v1',direction:'page',type:'frame',frameData:{frame:n}}});
  relay(1);assert.equal(connections[0].sent.length,1);
  connections[0].disconnect();assert.equal(timers.size,1);
  relay(2);assert.equal(connections[0].sent.length,1,'no sends/queued frames after disconnect');
  const retry=timers.values().next().value;timers.clear();retry();assert.equal(connections.length,2);
  relay(3);assert.equal(connections[1].sent[0].frameData.frame,3,'only fresh frames reach replacement port');
  const previous=messages.length;connections[0].onMessage.emit({type:'action',mask:16});assert.equal(messages.length,previous,'old connection responses ignored');
  connections[1].closed=true;relay(4);assert.equal(timers.size,1,'synchronous postMessage failure is contained');
  assert(messages.some(m=>m.type==='service'&&!m.connected),'disconnect tells native arbiter to neutralize');
  events.pagehide();assert.equal(timers.size,0,'closing page cancels reconnect');
  runtime.id=null;runtime.connect=()=>{throw Error('Extension context invalidated');};
  const invalid=vm.createContext({window,location:{origin:'https://www.haxball.com'},chrome:{runtime},document:{readyState:'loading',addEventListener(){}},
    setTimeout(){throw Error('invalid context must not retry');},clearTimeout(){}});
  vm.runInContext(code('content.js'),invalid);assert.match(messages.at(-1).reason,/recargá también HaxBall/);
  runtime.id='transport-test';const refused=[];
  runtime.connect=()=>{const p=port();refused.push(p);return p;};
  const bounded=vm.createContext({window,location:{origin:'https://www.haxball.com'},chrome:{runtime},document:{readyState:'loading',addEventListener(){}},
    setTimeout(fn){timers.set(1,fn);return 1;},clearTimeout(id){timers.delete(id);}});
  vm.runInContext(code('content.js'),bounded);
  for(let i=0;i<5;i++){refused.at(-1).disconnect();const next=timers.get(1);timers.clear();if(next)next();}
  assert.equal(refused.length,5,'repeated rejection has a bounded retry budget');assert.equal(timers.size,0);
  assert.match(messages.at(-1).reason,/recargá también HaxBall/);

  class Socket{static OPEN=1;constructor(){this.readyState=0;this.sent=[];Socket.all.push(this);}send(m){if(this.readyState!==1||this.fail)throw Error('socket closed');this.sent.push(JSON.parse(m));}close(){this.readyState=3;}}Socket.all=[];
  const local={setAccessLevel:async()=>{},get:async()=>({token:'test-token-32-characters-minimum!!',port:17841}),onChanged:new Signal};
  const chrome={runtime:{id:'transport-test',lastError:null,onConnect:new Signal,onMessage:new Signal,getURL:p=>'chrome-extension://transport-test/'+p,openOptionsPage:async()=>{}},
    storage:{local,onChanged:local.onChanged}};
  const worker=vm.createContext({chrome,WebSocket:Socket,importScripts(){},RS4Shortcut:require('./shortcut'),setInterval(){},setTimeout(){},clearTimeout(){}});
  vm.runInContext(code('background.js'),worker);
  const sender={id:chrome.runtime.id,url:'https://www.haxball.com/play?c=test',tab:{id:4},frameId:1};
  const official=port(sender);chrome.runtime.onConnect.emit(official);await flush();assert(!official.closed,'official /play sender allowed');
  const first=Socket.all[0];first.readyState=1;const staleOpen=first.onopen;
  await vm.runInContext('connect()',worker);staleOpen();assert.equal(first.sent.length,0,'stale websocket cannot send auth');
  const current=Socket.all.at(-1);current.readyState=1;current.onopen();assert.equal(current.sent[0].type,'auth');
  current.onmessage({data:JSON.stringify({type:'authenticated',model:{name:'fixture'}})});
  current.onmessage({data:JSON.stringify({type:'claimed',tabId:4,model:{name:'fixture'}})});
  assert(official.sent.some(m=>m.type==='service'&&m.connected));
  const duplicate=port(sender);chrome.runtime.onConnect.emit(duplicate);assert(duplicate.closed,'duplicate frame is not another owner');
  const foreign=port({...sender,tab:{id:9},url:'https://evil.example/play'});chrome.runtime.onConnect.emit(foreign);assert(foreign.closed);
  const wrongExtension=port({...sender,id:'different-extension',tab:{id:10}});chrome.runtime.onConnect.emit(wrongExtension);assert(wrongExtension.closed);
  const cached=port({...sender,url:'https://www.haxball.com/H4mQWPXF/__cache_static__/g/game.html',tab:{id:11}});
  chrome.runtime.onConnect.emit(cached);assert(!cached.closed,'cached official iframe remains supported');
  current.fail=true;official.onMessage.emit({type:'frame',frameData:{frame:3}});
  assert.equal(vm.runInContext('authenticated',worker),false,'failed socket send invalidates connection safely');
  assert(official.sent.some(m=>m.type==='service'&&!m.connected));
}
module.exports={run};
if(require.main===module)run().then(()=>console.log('PASS extension port lifecycle, official origin and websocket races')).catch(e=>{console.error(e);process.exitCode=1;});
