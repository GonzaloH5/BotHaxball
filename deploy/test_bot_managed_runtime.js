// Exercise the actual managed bot entry point with an offline Room API.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const {EventEmitter}=require('node:events');
const source=fs.readFileSync(path.join(__dirname,'bot.js'),'utf8');
async function scenario(join){
  const events=[],messages=[],processMock=new EventEmitter();
  Object.assign(processMock,{argv:['node','bot.js','--managed','--model',path.join(__dirname,'model.onnx'),'--player','Ana','--avatar','GK','--flag','uy','--team','2','--name','Prueba','--max-players','12',...(join?['--join','room1234']:['--stadium','classic'])],env:{RS4_BOT_SECRETS:JSON.stringify({token:'secret-host',password:'secret-room'})},connected:true,send:m=>messages.push(m),exit:code=>events.push(['exit',code])});
  let call;
  const room={isHost:!join,name:'Prueba',currentPlayerId:join?3:0,currentPlayer:{team:{id:0}},setKeyState:v=>events.push(['keys',v]),leave:()=>events.push(['leave']),setAvatar:v=>events.push(['avatar',v]),setTimeLimit:v=>events.push(['time',v]),setScoreLimit:v=>events.push(['score',v]),setPlayerTeam:(id,team)=>events.push(['team',id,team]),setCurrentStadium:()=>events.push(['map']),startGame:()=>events.push(['start']),stopGame:()=>events.push(['stop']),pauseGame:()=>events.push(['pause'])};
  const open=(kind,config,common)=>{call={kind,config,common};common.plugins[0].room=room;common.onOpen(room);};
  const api={Plugin:function(){},AllowFlags:{CreateRoom:1,JoinRoom:2},Utils:{generateAuth:async()=>['auth',{}],parseStadium:()=>({})},Room:{create:(c,o)=>open('host',c,o),join:(c,o)=>open('join',c,o)}};
  const context={require:id=>id==='../bridge/haxball'?()=>api:id==='onnxruntime-node'?{InferenceSession:{create:async()=>({})}}:require(id),process:processMock,__dirname,console:{log(){},warn(){},error(){}},setTimeout,clearTimeout,setInterval:()=>({unref(){}})};
  vm.runInNewContext(source,context);await new Promise(r=>setImmediate(r));
  assert.equal(call.kind,join?'join':'host');assert.equal(call.config.password,'secret-room');assert.equal(call.common.storage.player_name,'Ana');assert.equal(call.common.storage.avatar,'GK');assert.equal(call.common.storage.geo.flag,'uy');assert.equal(processMock.env.RS4_BOT_SECRETS,undefined);
  if(join)assert.equal(call.config.id,'room1234');else{assert.equal(call.config.token,'secret-host');assert.equal(call.config.maxPlayerCount,12);assert(events.some(e=>e[0]==='map'));}
  assert(events.some(e=>e[0]==='team'&&e[2]===2));
  processMock.emit('message',{type:'enabled',enabled:false});assert(messages.some(m=>m.type==='enabled'&&!m.enabled));assert(events.some(e=>e[0]==='keys'&&e[1]===0));
  processMock.emit('message',{type:'game',action:'start'});assert.equal(events.some(e=>e[0]==='start'),!join);
  processMock.emit('message',{type:'stop'});assert(events.some(e=>e[0]==='leave'));assert(events.some(e=>e[0]==='exit'&&e[1]===0));
}
Promise.resolve().then(()=>scenario(false)).then(()=>scenario(true)).then(()=>console.log('PASS managed bot entry: host/join, credentials, identity, team, game controls and shutdown')).catch(e=>{console.error(e);process.exitCode=1;});
