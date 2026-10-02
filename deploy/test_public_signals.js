const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {PublicSignalTracker, publicFeatures, colorTeam} = require('./public_signals');
const {PolicyMemory} = require('./policy_memory');
const {assertObservationContract} = require('./observation_contract');

const tracker = new PublicSignalTracker({barrier_discs:[1],barrier_segments:[0]});
const discs = [{color:0xff0000},{color:0xffffff},{color:0x0000ff}];
assert.equal(tracker.barrierColor(discs, []), -1, 'ignore colored decoration not explicitly configured');
discs[1].color = 0xff0000;
assert.equal(tracker.barrierColor(discs, [{color:0x0000ff,vis:false}]), 0xff0000);
assert.equal(tracker.barrierColor(discs, [{color:0x0000ff,vis:true}]), -2);
assert.throws(()=>new PublicSignalTracker({barrier_discs:[0]}).barrierColor(discs), /ball/);
assert.equal(colorTeam('#ff0000'), 0);
assert.equal(colorTeam([0,0,255]), 1);
assert.equal(colorTeam(0x123456), -1);
assert.throws(()=>publicFeatures({ball:{pos:[0,0],vel:[0,0]},players:[{team:0}]},0,{}), /packet/);
assert.throws(()=>assertObservationContract({layout:'universal',rule_observation:'full',public_signals:{version:1,offset:56}},{psOn:false}), /contract/);
assert.doesNotThrow(()=>assertObservationContract({layout:'universal',rule_observation:'masked',public_signals:{version:1,offset:56}},{psOn:false}));

// Exercise the real BotPlugin: cadence, unknown cues, contact history, reset,
// and sampling while inference is busy. Never connects to a room.
(async()=>{
  const code=fs.readFileSync(path.join(__dirname,'bot.js'),'utf8');
  const plugin=code.slice(code.indexOf('function BotPlugin(session)'),code.indexOf('\n(async () =>'));
  const disc=(x=0,y=0,color=0xffffff)=>({pos:{x,y},speed:{x:0,y:0},color});
  const me={id:1,team:{id:1},disc:disc(20),isKicking:true};
  const ball=disc(0,0,0xff0000), samples=[], pending=[];
  const room={isHost:true,currentPlayer:me,currentPlayerId:1,currentFrameNo:0,
    state:{players:[me]},stadium:{name:'RS4',segments:[]},
    gameState:{state:1,physicsState:{discs:[ball],segments:[]}},setKeyState:()=>{}};
  const geometry={field_half_w:1000,field_half_h:450,goal_x:1000,player_radius:15,ball_radius:10};
  const context={Plugin:function(){this.room=room;},AllowFlags:{CreateRoom:1,JoinRoom:2},
    PublicSignalTracker,PolicyMemory,PUBLIC_CUES:{},
    META:{public_signals:{version:1,offset:56},rule_observation:'masked',layout:'universal',frame_skip:3,n_actions:18,max_ticks:7200},
    UNIVERSAL:true,extrapMs:null,KICK_REACH:4,console,
    stadiumGeometry:()=>geometry,rulesFor:()=>({psOn:false}),assertObservationContract,
    buildObsUniversal:(s,p,g,o)=>{samples.push({s,o});return publicFeatures(s,p,g);},
    ort:{Tensor:class{constructor(t,data,dims){this.data=data;this.dims=dims;}}},
    sample:()=>0,decodeAction:()=>({dirX:0,dirY:0,kick:false}),Utils:{keyState:()=>0}};
  vm.createContext(context);vm.runInContext(plugin,context);
  const bot=new context.BotPlugin({run:()=>new Promise(resolve=>pending.push(resolve))});
  const flush=()=>new Promise(resolve=>setImmediate(resolve));
  for(let i=0;i<3;i++){room.currentFrameNo=i;bot.onGameTick();}
  assert.equal(samples.length,1);
  assert.equal(samples[0].s.publicSignals.contactTeam,0);
  assert.equal(samples[0].o.publicSignalsVersion,1);
  me.disc.pos.x=200;
  for(let i=3;i<6;i++){room.currentFrameNo=i;bot.onGameTick();}
  assert.equal(samples.length,1,'pending inference must not issue another action');
  pending.shift()({logits:{data:[1]}});await flush();
  for(let i=6;i<9;i++){room.currentFrameNo=i;bot.onGameTick();}
  assert.equal(samples[1].s.publicSignals.contactAge,6,'tracker continues while inference is busy');
  pending.shift()({logits:{data:[1]}});await flush();
  bot.onPlayerTeamChange();
  ball.color=0xffffff;
  for(let i=9;i<12;i++){room.currentFrameNo=i;bot.onGameTick();}
  assert.equal(samples[2].s.publicSignals.contactTeam,-1);
  assert.equal(publicFeatures(samples[2].s,0,geometry)[1],0,'unknown never infers ownership from kickoff/private state');
  pending.shift()({logits:{data:[1]}});await flush();
  // The actual BotPlugin must pass raw runtime joints and configure topology on map change.
  const map=JSON.parse(fs.readFileSync(path.join(__dirname,'../stadiums/rs_one.hbs'),'utf8'));
  Object.assign(geometry,map.haxballrl,{goal_x:1150});
  context.PUBLIC_CUES.auto_joints=true;
  room.stadium=map;
  room.gameState.physicsState.joints=map.joints;
  room.gameState.physicsState.discs=map.discs.map(d=>({...d,pos:{x:d.pos?.[0]||0,y:d.pos?.[1]||0},speed:{x:0,y:0}}));
  room.gameState.physicsState.discs[0]=ball;
  for(const j of map.joints.slice(0,4))room.gameState.physicsState.discs[j.d1].pos={...room.gameState.physicsState.discs[j.d0].pos};
  room.gameState.physicsState.discs[map.joints[1].d1].pos.x=1149;
  Object.assign(ball,{color:0x0FBCF9,pos:{x:200,y:-688}});
  for(let i=12;i<15;i++){room.currentFrameNo=i;bot.onGameTick();}
  assert.equal(samples[3].s.publicSignals.barrierColor,0x0000FF);
  assert.equal(publicFeatures(samples[3].s,0,geometry)[2],1);
  pending.shift()({logits:{data:[1]}});await flush();
  console.log('OK: public RS4 cues, configured barriers, busy inference cadence and resets');
})().catch(e=>{console.error(e);process.exitCode=1;});
