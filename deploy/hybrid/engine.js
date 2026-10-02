const {buildObsUniversal}=require('../obs_universal');
const {decodeAction}=require('../obs');
const {PolicyMemory}=require('../policy_memory');
const {PublicSignalTracker}=require('../public_signals');
const {assertObservationContract}=require('../observation_contract');
const {sampleLogits}=require('../runtime');
function validateMeta(meta){
  assertObservationContract(meta,{psOn:true});
  if(meta.training_program!=='rs4_v3'||meta.public_signals?.version!==1||meta.self_dim!==71||meta.ent_dim!==8
    ||meta.n_actions!==18||!Number.isInteger(meta.frame_skip)||meta.frame_skip<1||meta.frame_skip>12
    ||!Number.isFinite(meta.max_ticks)||meta.max_ticks<=0||!meta.ps_cfg||![meta.ps_cfg.charge,meta.ps_cfg.grav,meta.ps_cfg.power_inv].every(v=>Number.isFinite(v)&&v>0)
    ||(meta.recurrent&&(!Number.isInteger(meta.memory_size)||meta.memory_size<=0)))throw Error('Requires an exported public-v1 RS4 v3 model');
  return meta;
}
function validateFrame(f){
  const point=p=>Array.isArray(p)&&p.length===2&&p.every(v=>Number.isFinite(v)&&Math.abs(v)<1e7);
  if(!f||!Number.isInteger(f.frame)||f.frame<0||!Number.isInteger(f.playerId)||!Number.isFinite(f.capturedAt)
    ||!Array.isArray(f.players)||f.players.length!==8||[0,1].some(t=>f.players.filter(p=>p.team===t).length!==4)
    ||new Set(f.players.map(p=>p.id)).size!==8||!f.players.some(p=>p.id===f.playerId)
    ||f.players.some(p=>!Number.isInteger(p.id)||![0,1].includes(p.team)||!point(p.pos)||!point(p.vel)||!Number.isInteger(p.input)||p.input<0||p.input>31)
    ||!Array.isArray(f.discs)||f.discs.length<9||f.discs.length>512
    ||f.discs.some(d=>!point(d.pos)||!point(d.vel)||!point(d.gravity)||!Number.isFinite(d.radius)||d.radius<0||!Number.isInteger(d.color)||!Number.isFinite(d.invMass))
    ||!Array.isArray(f.joints)||f.joints.length>256||f.joints.some(j=>!Number.isInteger(j.d0)||j.d0<0||j.d0>=f.discs.length||!Number.isInteger(j.d1)||j.d1<0||j.d1>=f.discs.length||!Number.isInteger(j.color))
    ||!Array.isArray(f.score)||f.score.length!==2||!f.score.every(Number.isInteger)
    ||![0,1,2,3].includes(f.phase)||!Number.isFinite(f.elapsed)||!f.synchronized||f.paused)throw Error('RS4 state unavailable, paused, desynchronized or not 4v4');
  return f;
}
function maskForAction(action,team){const k=decodeAction(action,team);return (k.dirY<0?1:k.dirY>0?2:0)|(k.dirX<0?4:k.dirX>0?8:0)|(k.kick?16:0);}
class Engine{
  constructor(meta,session,ort){this.meta=validateMeta(meta);this.session=session;this.ort=ort;this.memory=new PolicyMemory(meta);
    this.tracker=new PublicSignalTracker(meta.public_signals);this.mode='HUMANO';this.epoch=0;this.id='';this.geom=null;
    this.pending=null;this.queue=null;this.running=false;this.lastFrame=-1;this.charge=0;this.ticks=0;this.kickoffTeam=-1;this.score=null;this.phase=null;this.modelGeneration=0;
  }
  map(stadium,geom){
    if(stadium.name.trim()!=='Real Soccer ONE'||geom.name.trim()!=='Real Soccer ONE'||geom.field_half_w!==1150||geom.field_half_h!==600)throw Error('v1 supports the RS ONE stadium only');
    this.geom=geom;this.tracker.configureStadium(stadium,geom);this.tracker.reset();this.lastFrame=-1;this.score=null;this.phase=null;this.resetPolicy();
  }
  resetPolicy(){this.memory.reset();this.pending=null;this.queue=null;this.modelGeneration++;this.charge=0;}
  control({mode,epoch,session}){
    if(!['HUMANO','BOT','SUSPENDIDO'].includes(mode)||!Number.isInteger(epoch)||epoch<0||typeof session!=='string'||session.length>100)throw Error('Invalid control generation');
    if(session===this.id&&epoch<this.epoch)throw Error('Old control generation');
    if(this.id!==session||this.epoch!==epoch||this.mode!==mode)this.resetPolicy();
    this.id=session;this.epoch=epoch;this.mode=mode;
  }
  observe(f){
    validateFrame(f);if(!this.geom)throw Error('Map is not ready');
    if(f.frame<=this.lastFrame)throw Error('Non-monotonic frame');
    const ball=f.discs[0], me=f.players.find(p=>p.id===f.playerId);
    const players=[me,...f.players.filter(p=>p!==me)].map(p=>({...p,canKick:true,touching:false}));
    const dt=this.lastFrame<0?0:f.frame-this.lastFrame;
    if(dt>120){this.tracker.reset();this.resetPolicy();throw Error('State gap too large');}
    if(this.phase!==null&&(this.phase!==f.phase||this.score?.some((v,i)=>v!==f.score[i]))){this.resetPolicy();this.tracker.reset();this.ticks=0;}
    if(this.score){if(f.score[0]>this.score[0])this.kickoffTeam=1;else if(f.score[1]>this.score[1])this.kickoffTeam=0;}
    this.score=f.score.slice();this.phase=f.phase;this.lastFrame=f.frame;this.ticks+=dt;
    players[0].touching=Math.hypot(...ball.pos.map((v,i)=>v-me.pos[i]))-this.geom.player_radius-this.geom.ball_radius<4;
    players[0].canKick=!((me.input&16)&&!me.isKicking);
    this.charge=players[0].touching&&(me.input&16)?this.charge+dt:0;
    const cfg=this.meta.ps_cfg, gravity=ball.gravity;
    const ps={comba:gravity.some(v=>v!==0)?1:0,prog:Math.min(this.charge/cfg.charge,1),
      invb:Math.min(1,Math.max(0,(ball.invMass-this.geom.ball_invmass)/(cfg.power_inv-this.geom.ball_invmass))),
      grav:gravity.map(v=>v/cfg.grav),holder:this.charge?0:-1};
    const barrier=this.tracker.barrierColor(f.discs,[],f.joints,ball);
    const packet=this.tracker.sample(ball,players,dt,this.geom.player_radius,this.geom.ball_radius,barrier);
    const state={ball,players,ps,publicSignals:packet,kickoff:f.phase===0,kickoffTeam:this.kickoffTeam,tfrac:Math.min(1,this.ticks/this.meta.max_ticks)};
    return {state,obs:buildObsUniversal(state,0,this.geom,{maxEntities:7,psOn:true,outOfBounds:true,publicSignalsVersion:1,publicSignalConfig:this.meta.public_signals})};
  }
  async frame(f,send){
    if(f.session!==this.id||f.epoch!==this.epoch)return;
    const sample=this.observe(f);
    if(this.mode!=='BOT'&&!f.qualify)return; // History continues; PPO/learning never runs.
    this.queue={f,sample,send};await this.drain();
  }
  async drain(){
    if(this.running||this.pending||!this.queue)return;
    const job=this.queue;this.queue=null;this.running=true;
    const generation=this.modelGeneration, memoryGeneration=this.memory.generation, start=performance.now();
    try{
      const out=await this.session.run({obs:new this.ort.Tensor('float32',Float32Array.from(job.sample.obs),[1,127]),...this.memory.feeds(this.ort)});
      if(generation!==this.modelGeneration||job.f.epoch!==this.epoch||job.f.session!==this.id)return;
      if(!out.logits||out.logits.data.length!==18)throw Error('Wrong policy output width');
      const action=sampleLogits(out.logits.data);
      if(this.meta.recurrent&&(!out.memory_out||out.memory_out.data.length!==this.meta.memory_size
        ||!Array.from(out.memory_out.data).every(Number.isFinite)))throw Error('Invalid recurrent policy output');
      const response={type:'action',session:this.id,epoch:this.epoch,frame:job.f.frame,capturedAt:job.f.capturedAt,
        action,mask:maskForAction(action,job.sample.state.players[0].team),inferenceMs:performance.now()-start,qualify:!!job.f.qualify};
      if(!job.f.qualify)this.pending={out,action,mask:response.mask,frame:job.f.frame,generation:memoryGeneration,created:performance.now()};
      job.send(response);
    }catch(e){if(generation===this.modelGeneration){this.mode='SUSPENDIDO';this.resetPolicy();job.send({type:'error',message:e.message});}}
    finally{this.running=false;if(this.queue&&!this.pending)await this.drain();}
  }
  async ack(message){
    const p=this.pending;
    if(!p||message.session!==this.id||message.epoch!==this.epoch||message.frame!==p.frame)return;
    if(message.accepted!==true||message.mask!==p.mask||performance.now()-p.created>100){this.mode='SUSPENDIDO';this.resetPolicy();return;}
    this.memory.accept(p.out,p.action,p.generation);this.pending=null;await this.drain();
  }
}
module.exports={Engine,validateMeta,validateFrame,maskForAction};
