const assert=require('node:assert/strict'),{EventEmitter}=require('node:events'),path=require('node:path');
const {BotManager,roomId}=require('./bot_manager');
const wait=ms=>new Promise(r=>setTimeout(r,ms));
class FakeChild extends EventEmitter {
  constructor(){super();this.connected=true;this.sent=[];this.stdout={resume(){}};this.stderr={resume(){}};}
  send(m,callback){this.sent.push(m);callback?.();if(m.type==='stop')setImmediate(()=>this.kill());if(m.type==='enabled')setImmediate(()=>this.emit('message',{type:'enabled',enabled:m.enabled}));}
  kill(){if(!this.connected)return;this.connected=false;this.emit('exit',0);}
}
const profile=(name='Uno')=>({name,avatar:'1',flag:'uy',model:'current',team:1});
const model=path.join(__dirname,'rs4_public/model.onnx');
async function run(){
  const children=[],calls=[];
  const manager=new BotManager({model,spawn:(...args)=>{calls.push(args);const c=new FakeChild();children.push(c);return c;},startTimeout:1000});
  assert.equal(roomId('https://www.haxball.com/play?c=room1234'),'room1234');
  assert.throws(()=>roomId('https://evil.example/play?c=room1234'));
  const join={mode:'join',room:'room1234',profiles:[profile()]};
  assert.throws(()=>manager.start({...join,profiles:[profile(),profile()]}),/distinto/);
  assert.throws(()=>manager.start({...join,profiles:[{...profile(),model:'../../evil'}]}),/Modelo/);
  assert.throws(()=>manager.start({...join,profiles:[profile('--join')]}),/inválido/);
  assert.equal(children.length,0);
  const host={mode:'host',name:'Sala test',token:'private-headless-token',password:'private-password',stadium:'rs_one',maxPlayers:8,timeLimit:3,scoreLimit:3,profiles:[profile(),profile('Dos')]};
  const state=manager.start(host);assert.equal(children.length,1,'only host starts before room link');
  assert(!JSON.stringify(state).includes(host.token));assert(!JSON.stringify(state).includes(host.password));
  assert(!calls[0][1].includes(host.password));assert.equal(calls[0][2].windowsHide,true);
  children[0].emit('message',{type:'connected',playerId:0,isHost:true});children[0].emit('message',{type:'link',link:'https://www.haxball.com/play?c=room1234'});
  await wait(20);assert.equal(children.length,2);children[1].emit('message',{type:'connected',playerId:7,isHost:false});
  assert(children[0].sent.some(m=>m.type==='team'&&m.playerId===7&&m.team===1),'host assigns joined bot team');
  const bot=state.bots[1];manager.command({id:bot.id,action:'enabled',enabled:false});await wait(5);assert.equal(manager.snapshot().bots[1].enabled,false);
  manager.command({id:state.bots[0].id,action:'start'});assert(children[0].sent.some(m=>m.type==='game'&&m.action==='start'));
  assert.throws(()=>manager.command({id:bot.id,action:'start'}),/host/);
  manager.stopAll();await wait(5);assert(manager.snapshot().bots.every(b=>b.status==='stopped'));
  const queued=manager.start({...join,profiles:[profile('A'),profile('B'),profile('C')]});manager.stopAll();await wait(10);assert.equal(children.length,2,'cancelled queue never spawns');
  await manager.close();assert.throws(()=>manager.start(join),/cerrando/);
  const timeoutChild=new FakeChild(),timeoutManager=new BotManager({model,spawn:()=>timeoutChild,startTimeout:10});timeoutManager.start(join);await wait(30);assert.equal(timeoutManager.snapshot().bots[0].status,'failed');await timeoutManager.close();
  const capped=new BotManager({model,limit:1,spawn:()=>new FakeChild()});capped.start(join);assert.throws(()=>capped.start(join),/Máximo/);await capped.close();
  console.log('PASS bot manager: validation, credentials, host sequencing, teams, pause, limits, timeout and cleanup');
}
if(require.main===module)run().catch(e=>{console.error(e);process.exitCode=1;});
module.exports={FakeChild,profile,run};
