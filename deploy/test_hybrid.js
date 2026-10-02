const assert=require('node:assert/strict'),fs=require('fs'),path=require('path'),vm=require('vm');
const {Arbiter,captureControllers}=require('./hybrid/arbiter');
const profile=require('./hybrid/official_profile');
const Shortcut=require('./hybrid/shortcut');
const {Engine,validateMeta,maskForAction}=require('./hybrid/engine');
const {buildObsUniversal}=require('./obs_universal');
const {startServer,loadModel}=require('./hybrid_server');
const {WebSocket}=require('ws');
const tests=[];const test=(name,fn)=>tests.push({name,fn});
test('extension transport disconnect/reconnect, context reload and websocket races',require('./hybrid/test_transport').run);
const meta={training_program:'rs4_v3',layout:'universal',rule_observation:'masked',self_dim:71,ent_dim:8,n_actions:18,
  frame_skip:3,max_ticks:7200,ps_cfg:{charge:96,grav:.1,power_inv:2.3},public_signals:{version:1,offset:56,auto_joints:true}};
const geometry={name:'Real Soccer ONE',field_half_w:1150,field_half_h:600,goal_x:1150,goal_half_height:120,
  kickoff_radius:100,player_radius:15,ball_radius:10,ball_invmass:1,
  obstacles:Object.fromEntries(['red','blue','ball'].map(k=>[k,{seg_a:[],seg_b:[],circ_c:[],circ_r:[],pl_n:[],pl_d:[]}]))};
const stadium=JSON.parse(fs.readFileSync(path.join(__dirname,'../stadiums/rs_one.hbs'),'utf8'));
function disc(pos=[0,0],radius=15){return {pos,vel:[0,0],gravity:[0,0],radius,color:0xFFFFFF,invMass:1,playerId:null};}
function frame(n=3){const players=Array.from({length:8},(_,i)=>({id:i+1,team:i<4?0:1,...disc([i*40-160,i*5]),input:0,isKicking:false}));
  return {frame:n,playerId:1,players,discs:[disc([0,0],10),...players],joints:[],score:[0,0],phase:1,paused:false,elapsed:n/60,
    synchronized:true,capturedAt:performance.now(),session:'test',epoch:1};}
const fakeOrt={Tensor:class{constructor(type,data,dims){Object.assign(this,{type,data,dims});}}};
const logits=()=>({logits:{data:Float32Array.from({length:18},(_,i)=>i===12?1:0)},memory_out:{data:new Float32Array(32).fill(.5)}});
const flush=()=>new Promise(resolve=>setImmediate(resolve));
test('user shortcuts, exact modifiers, legacy settings and disabled default',()=>{
  assert.equal(Shortcut.normalize(null),null);assert.equal(Shortcut.label(null),'Sin atajo');
  const event={code:'KeyB',ctrlKey:true,altKey:true,shiftKey:false,metaKey:false};
  const selected=Shortcut.fromEvent(event);assert.equal(Shortcut.label(selected),'Ctrl + Alt + B');
  assert(Shortcut.matches(event,selected));assert(!Shortcut.matches({...event,ctrlKey:false},selected));
  assert(!Shortcut.matches({...event,shiftKey:true},selected));assert(!Shortcut.matches({...event,isComposing:true},selected));
  assert.equal(Shortcut.fromEvent({...event,repeat:true}),null);
  for(const code of ['Escape','ControlLeft','ShiftRight','AltLeft','MetaRight','Dead','Unidentified'])assert.equal(Shortcut.fromEvent({...event,code}),null);
  assert.equal(Shortcut.label(Shortcut.fromEvent({code:'KeyV'})),'V');
  assert.equal(Shortcut.label('F8'),'F8');assert(!Shortcut.matches(event,null));
});
test('arbiter handoffs, held keys, focus, late responses and watchdog',()=>{
  let now=100;const writes=[];const a=new Arbiter({write:mask=>writes.push(mask),now:()=>now});a.session='test';a.ready=a.permission=true;
  a.key('KeyX',true);a.toggle();const old=a.epoch;
  assert(a.accept({session:'test',epoch:a.epoch,frame:1,capturedAt:now,mask:24}));a.toggle();assert.equal(writes.at(-1),0);
  assert.equal(a.key('KeyX',true),false);a.key('KeyX',false);assert(a.key('KeyX',true));
  assert(!a.accept({session:'test',epoch:old,frame:2,capturedAt:now,mask:16}));
  a.toggle();assert(!a.accept({session:'test',epoch:a.epoch,frame:2,capturedAt:now-101,mask:16}));
  assert(!a.accept({session:'foreign',epoch:a.epoch,frame:2,capturedAt:now,mask:16}));
  now+=251;a.watchdog();assert.equal(a.mode,'SUSPENDIDO');assert.equal(writes.at(-1),0);
  a.focus(true);assert.equal(a.mode,'HUMANO');a.toggle();a.focus(false);assert.equal(a.mode,'SUSPENDIDO');
  a.focus(true);assert.equal(a.mode,'HUMANO');assert.equal(a.epoch>old,true);
  for(let i=0;i<100;i++){a.toggle();assert.equal(a.mode,'BOT');a.toggle();assert.equal(writes.at(-1),0);}
});
test('background bot retains generation and memory eligibility but still rejects stale actions',()=>{
  let now=100;const writes=[];const a=new Arbiter({write:mask=>writes.push(mask),now:()=>now,backgroundBot:true});
  a.session='background';a.ready=a.permission=true;a.toggle();const epoch=a.epoch;
  a.focus(false);assert.equal(a.mode,'BOT');assert.equal(a.epoch,epoch);assert.equal(a.reason,'Bot en segundo plano');
  assert(a.accept({session:a.session,epoch,frame:1,capturedAt:now,mask:24}));assert.equal(writes.at(-1),24);
  assert(!a.accept({session:a.session,epoch,frame:2,capturedAt:now-101,mask:16}));
  a.focus(true);assert.equal(a.mode,'BOT');assert.equal(a.epoch,epoch);
  a.toggle();a.key('KeyX',true);a.focus(false);assert.equal(a.mode,'HUMANO');assert.equal(writes.at(-1),0);
  a.toggle();assert.equal(a.mode,'HUMANO','cannot enable a bot without foreground focus');
  a.focus(true);a.toggle();a.focus(false);now+=251;a.watchdog();assert.equal(a.mode,'SUSPENDIDO');assert.equal(writes.at(-1),0);
});
test('capture is transparent, scoped and removable',async()=>{
  const context=vm.createContext({captureControllers,captured:null});
  vm.runInContext(`class Target{addEventListener(t,f){this.listener=f;return 17;}}
    globalThis.EventTarget=Target;globalThis.document=new Target;globalThis.queueMicrotask=fn=>fn();
    globalThis.originalBind=Function.prototype.bind;globalThis.originalAdd=Target.prototype.addEventListener;
    globalThis.remove=captureControllers(globalThis,owner=>{captured=owner});
    globalThis.owner={value:9,handler(){return this.value}};
    globalThis.bound=owner.handler.bind(owner);
    globalThis.result=document.addEventListener('keydown',bound);
    globalThis.value=bound();remove();
    globalThis.restored=Function.prototype.bind===originalBind&&Target.prototype.addEventListener===originalAdd;`,context);
  assert.equal(context.result,17);assert.equal(context.value,9);assert.equal(context.captured,context.owner);assert(context.restored);
});
test('public-v1 model validation',()=>{
  validateMeta(meta);assert.throws(()=>validateMeta({...meta,rule_observation:'full'}));
  assert.throws(()=>validateMeta({...meta,training_program:'multi'}));assert.throws(()=>validateMeta({...meta,n_actions:20}));
  assert.equal(maskForAction(12,0),24);assert.equal(maskForAction(12,1),20);
});
test('new sessions keep manual control without model, connection or permission',()=>{
  const writes=[];const a=new Arbiter({write:mask=>writes.push(mask)});
  a.setSession('room-one');assert.equal(a.mode,'HUMANO');assert(!a.ready);assert(!a.permission);
  assert(a.key('KeyW',true));a.key('KeyW',false);assert(a.key('KeyX',true));a.key('KeyX',false);
  a.key('KeyW',true);const count=writes.length,epoch=a.epoch;
  a.prepareHuman('Servicio conectado; preparando mapa');assert.equal(writes.length,count,'connecting does not neutralize manual input');
  assert(a.epoch>epoch);assert(!a.blocked.has('KeyW'));a.key('KeyW',false);
  a.toggle();assert.equal(a.mode,'HUMANO','unavailable bot cannot steal keyboard');
  a.ready=a.permission=true;a.toggle();const old=a.epoch;
  a.setSession('room-two');assert.equal(a.mode,'HUMANO');assert(!a.ready);assert(!a.permission);assert.equal(writes.at(-1),0);
  assert(!a.accept({session:'room-one',epoch:old,frame:1,capturedAt:performance.now(),mask:16}));
});
test('panel prerequisites explain spectator, stopped games, pause and non-4v4',()=>{
  assert.match(profile.roomStatus(null).blocker,/controlador oficial/);
  const nativeDisc=()=>({a:{x:0,y:0},G:{x:0,y:0},ra:{x:0,y:0},V:15,S:0xFFFFFF,ca:1,i:39,C:2});
  const controls={Qc:new Set,A(){},Al(){},Fa(){},ld(){}};
  const players=Array.from({length:8},(_,i)=>({Z:i+1,I:nativeDisc(),fa:{ba:i<4?1:2},Ud:false}));
  const room={K:players,U:{D:'Real Soccer ONE'},M:{va:{H:[]},Cb:1,Ta:0}};
  const view={za:{T:room,yc:1,Y:0},W:controls,Fa(){},ld(){},sf(){},la(){}};
  assert.equal(profile.roomStatus(view).blocker,'');assert.equal(profile.roomStatus(view).mapName,'Real Soccer ONE');
  players[0].fa.ba=0;assert.match(profile.roomStatus(view).blocker,/espectador/);players[0].fa.ba=1;
  const game=room.M;room.M=null;assert.match(profile.roomStatus(view).blocker,/detenido/);room.M=game;
  game.Cb=3;assert.match(profile.roomStatus(view).blocker,/terminado/);game.Cb=1;
  game.Ta=120;assert.match(profile.roomStatus(view).blocker,/pausado/);game.Ta=0;
  players[0].Ud=true;assert.match(profile.roomStatus(view).blocker,/desincronizado/);players[0].Ud=false;
  players.pop();assert.match(profile.roomStatus(view).blocker,/rojo 4\/4 · azul 3\/4/);
});
test('human history does not infer; bot memory advances only on native acknowledgement',async()=>{
  let runs=0;const m={...meta,recurrent:true,memory_size:32};const e=new Engine(m,{run:async()=>{runs++;return logits();}},fakeOrt);
  e.map(stadium,geometry);e.control({mode:'HUMANO',session:'test',epoch:1});const sent=[];
  await e.frame(frame(),v=>sent.push(v));assert.equal(runs,0);assert.equal(e.lastFrame,3);
  e.control({mode:'BOT',session:'test',epoch:2});await e.frame({...frame(6),epoch:2},v=>sent.push(v));
  assert.equal(runs,1);assert(e.pending);assert.equal(e.memory.previousAction,18);assert.equal(e.memory.memory[0],0);
  await e.ack({...sent[0],accepted:true});assert.equal(e.memory.previousAction,12);assert.equal(e.memory.memory[0],.5);
  e.control({mode:'HUMANO',session:'test',epoch:3});assert.equal(e.memory.previousAction,18);assert.equal(e.memory.memory[0],0);
  assert.equal(e.tracker.previousBall.length,2,'controller reset preserves public history');
});
test('in-flight results discarded on handoff; queue retains only newest state',async()=>{
  let complete;const e=new Engine(meta,{run:()=>new Promise(resolve=>{complete=resolve;})},fakeOrt);
  e.map(stadium,geometry);e.control({mode:'BOT',session:'test',epoch:1});const sent=[];
  const first=e.frame(frame(3),v=>sent.push(v));await flush();await e.frame(frame(6),v=>sent.push(v));await e.frame(frame(9),v=>sent.push(v));
  assert.equal(e.queue.f.frame,9);e.control({mode:'HUMANO',session:'test',epoch:2});complete(logits());await first;
  assert.equal(sent.length,0);assert(!e.pending);assert(!e.queue);
});
test('invalid frame, duplicate frame, terminal and wrong ack stop safely',async()=>{
  const e=new Engine(meta,{run:async()=>logits()},fakeOrt);e.map(stadium,geometry);e.control({mode:'BOT',session:'test',epoch:1});const sent=[];
  await assert.rejects(e.frame({...frame(),players:frame().players.slice(1)},()=>{}));
  await e.frame(frame(),v=>sent.push(v));await e.ack({...sent[0],accepted:true,mask:0});assert.equal(e.mode,'SUSPENDIDO');
  e.control({mode:'HUMANO',session:'test',epoch:2});await assert.rejects(e.frame({...frame(),epoch:2},()=>{}),/monotonic/);
});
test('prepared observations use existing public decoder and ray casting',()=>{
  const e=new Engine(meta,{},fakeOrt);e.map(stadium,geometry);const f=frame();
  f.discs[0].color=0x0FBCF9;f.discs[0].pos=[0,-688];const result=e.observe(f);
  assert.equal(result.obs.length,127);assert(result.obs.every(Number.isFinite));
  assert.equal(result.obs[58],1);assert.equal(result.obs[60],1);
  assert.deepEqual(result.obs,buildObsUniversal(result.state,0,geometry,{maxEntities:7,psOn:true,outOfBounds:true,publicSignalsVersion:1,publicSignalConfig:meta.public_signals}));
});
test('malformed logits and recurrent memory never reach native inputs',async()=>{
  for(const bad of [
    {logits:{data:new Float32Array(19)}},
    {logits:{data:new Float32Array(18).fill(NaN)}},
    {logits:logits().logits,memory_out:{data:new Float32Array(32).fill(Infinity)}}
  ]){
    const e=new Engine({...meta,recurrent:true,memory_size:32},{run:async()=>bad},fakeOrt);
    e.map(stadium,geometry);e.control({mode:'BOT',session:'test',epoch:1});const sent=[];
    await e.frame(frame(),v=>sent.push(v));assert.equal(sent[0].type,'error');assert.equal(e.mode,'SUSPENDIDO');assert(!e.pending);
  }
});
function inbox(socket){const messages=[];socket.on('message',raw=>messages.push(JSON.parse(raw)));
  return async(type)=>{const end=Date.now()+5000;while(Date.now()<end){const i=messages.findIndex(m=>m.type===type);if(i>=0)return messages.splice(i,1)[0];await new Promise(r=>setTimeout(r,5));}throw Error('Timeout waiting for '+type);};}
async function open(port,origin='chrome-extension://'+'a'.repeat(32)){const ws=new WebSocket(`ws://127.0.0.1:${port}/rs4`,{origin});
  await new Promise((resolve,reject)=>{ws.once('open',resolve);ws.once('error',reject);});return ws;}
test('loopback service authentication, exclusive tab, reload and disconnect',async()=>{
  let loads=0;const server=await startServer({model:'test.onnx',port:0,token:'test-token-32-characters-minimum!!',
    loader:async()=>{loads++;return {meta,session:{run:async()=>logits()},name:'test.onnx',hash:'test'};},geometry:()=>geometry});
  const sockets=[];try{
    assert.equal(server.server.address().address,'127.0.0.1');
    await assert.rejects(open(server.port,'https://www.haxball.com'));
    const bad=await open(server.port);sockets.push(bad);const closed=new Promise(r=>bad.once('close',r));bad.send(JSON.stringify({type:'auth',token:'wrong'}));await closed;
    const a=await open(server.port);sockets.push(a);const get=inbox(a),send=m=>a.send(JSON.stringify(m));
    send({type:'auth',token:server.token});await get('authenticated');send({type:'claim',tabId:1});await get('claimed');
    assert.equal((await get('poll')).tabId,1,'local service supplies the background cadence');
    const b=await open(server.port);sockets.push(b);const other=inbox(b);b.send(JSON.stringify({type:'auth',token:server.token}));await other('authenticated');
    b.send(JSON.stringify({type:'claim',tabId:2}));assert.equal((await other('denied')).message,'Otra pestaña activa');
    send({type:'control',session:'test',epoch:1,mode:'BOT',tabId:1});send({type:'reload',tabId:1});assert.match((await get('error')).message,/HUMAN/);
    send({type:'control',session:'test',epoch:2,mode:'HUMANO',tabId:1});send({type:'reload',tabId:1});await get('reloaded');assert.equal(loads,2);
    send({type:'release',tabId:1});await flush();await new Promise(r=>setTimeout(r,10));b.send(JSON.stringify({type:'claim',tabId:2}));await other('claimed');
  }finally{for(const s of sockets)s.terminate();await server.close();}
});
test('browser modules generated from exactly the tested source',()=>{
  for(const name of ['official_profile.js','arbiter.js','shortcut.js'])assert.equal(fs.readFileSync(path.join(__dirname,'hybrid',name),'utf8'),fs.readFileSync(path.join(__dirname,'hybrid_extension',name),'utf8'));
  const manifest=JSON.parse(fs.readFileSync(path.join(__dirname,'hybrid_extension/manifest.json')));
  assert.equal(manifest.manifest_version,3);assert(!manifest.permissions.includes('debugger'));assert(!JSON.stringify(manifest).includes('<all_urls>'));
});
async function main(){let failures=0;for(const t of tests){try{await t.fn();console.log('PASS '+t.name);}catch(e){failures++;console.error('FAIL '+t.name+'\n'+e.stack);}}
  if(process.argv.includes('--model')){
    const filename=process.argv[process.argv.indexOf('--model')+1];try{
      const loaded=await loadModel(filename);assert.equal(loaded.meta.public_signals.version,1);
      const e=new Engine(loaded.meta,loaded.session,require('onnxruntime-node'));e.map(stadium,geometry);
      e.control({mode:'BOT',session:'test',epoch:1});const sent=[];
      await e.frame(frame(),v=>sent.push(v));assert.equal(sent[0].type,'action');assert(e.pending);
      const executed=sent[0].action;await e.ack({...sent[0],accepted:true});assert(!e.pending);
      if(loaded.meta.recurrent)assert.equal(e.memory.previousAction,executed);
      console.log('PASS actual ONNX inference and native acknowledgement '+loaded.name);
    }catch(e){failures++;console.error(e.stack);}
  }
  if(failures)process.exitCode=1;else console.log(`${tests.length} hybrid checks passed`);
}
module.exports={meta,geometry,stadium,frame};if(require.main===module)main();
