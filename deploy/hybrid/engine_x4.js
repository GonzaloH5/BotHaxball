// Motor del servicio local para modelos X4 (export/to_onnx_x4.py: observación v3, Sanguchito).
// Misma interfaz que engine.js (control, map, frame, ack), pero la observación sale del rastreador de sala de
// deploy/bot.js (deploy/rs4z/room_state.js, paridad con el dataset en deploy/test_x4_room_state.js). Los cuadros llegan
// cada frame_skip ticks desde la extensión; el rastreador cuenta los ticks con el número de frame de la sala.
const {RS4ZTracker,inputToAction,decodeOwnAction}=require('../rs4z/room_state');
const {mapFromStadiumName,OBS_DIM}=require('../rs4z/obs_v3');
const {sampleLogits}=require('../runtime');
const {validateFrame}=require('./engine');

function validateX4Meta(meta){
  if(meta?.obs_version!=='x4-obs-v3'||meta.obs_dim!==OBS_DIM||meta.n_actions!==18||!Number.isInteger(meta.frame_skip)
    ||meta.frame_skip<1||meta.frame_skip>12)throw Error('Requiere un modelo X4 exportado (obs x4-obs-v3)');
  return meta;
}
function maskForKey(key){return (key.dirY<0?1:key.dirY>0?2:0)|(key.dirX<0?4:key.dirX>0?8:0)|(key.kick?16:0);}

class X4Engine{
  constructor(meta,session,ort){this.meta=validateX4Meta(meta);this.session=session;this.ort=ort;
    this.tracker=new RS4ZTracker(meta);this.mode='HUMANO';this.epoch=0;this.id='';this.mapName=null;this.kickStrength=5.85;
    this.stadiumName='';this.pending=null;this.queue=null;this.running=false;this.lastFrame=-1;this.kickoffTeam=0;
    this.score=null;this.modelGeneration=0;this.temperature=meta.temperature??1;this.delay=meta.default_delay??10;
  }
  map(stadium){
    const name=String(stadium?.name||'').trim(),map=mapFromStadiumName(name);
    if(this.meta.maps&&!this.meta.maps.includes(map))
      throw Error(`El modelo se entrenó en ${this.meta.maps.join(', ')}; esta sala es «${name}» (${map})`);
    this.mapName=map;this.stadiumName=name;this.kickStrength=stadium.playerPhysics?.kickStrength??5.85;
    this.tracker=new RS4ZTracker({...this.meta,map});this.lastFrame=-1;this.score=null;this.resetPolicy();
  }
  resetPolicy(){this.pending=null;this.queue=null;this.modelGeneration++;this.tracker.reset();this.lastFrame=-1;}
  control({mode,epoch,session}){
    if(!['HUMANO','BOT','SUSPENDIDO'].includes(mode)||!Number.isInteger(epoch)||epoch<0||typeof session!=='string'||session.length>100)throw Error('Invalid control generation');
    if(session===this.id&&epoch<this.epoch)throw Error('Old control generation');
    // al pasar de humano a bot se conserva la historia (las teclas del humano son las decisiones propias recientes)
    if(this.id!==session||this.epoch!==epoch||this.mode!==mode){this.pending=null;this.queue=null;this.modelGeneration++;}
    if(this.id!==session)this.tracker.reset();
    this.id=session;this.epoch=epoch;this.mode=mode;
  }
  observe(f){
    validateFrame(f);if(!this.mapName)throw Error('Map is not ready');
    if(f.frame<=this.lastFrame)throw Error('Non-monotonic frame');
    if(this.lastFrame>=0&&f.frame-this.lastFrame>120){this.tracker.reset();this.pending=null;throw Error('State gap too large');}
    if(this.score){if(f.score[0]>this.score[0])this.kickoffTeam=1;else if(f.score[1]>this.score[1])this.kickoffTeam=0;}
    this.score=f.score.slice();this.lastFrame=f.frame;
    // en el saque inicial el motor les pone a todos los jugadores la barrera del equipo que saca (redKO 8, blueKO 16;
    // env/rs4z/kernel.reset_kickoff): sirve también si el bot entró con el partido empezado
    if(f.phase===0){const m=f.players.find(p=>p.id===f.playerId)?.cMask;
      if(Number.isInteger(m)&&(m&8))this.kickoffTeam=0;else if(Number.isInteger(m)&&(m&16))this.kickoffTeam=1;}
    const ball=f.discs[0];
    const frame={tick:f.frame,state:f.phase===0?0:1,kickoffTeam:this.kickoffTeam,kickStrength:this.kickStrength,stadiumName:this.stadiumName,
      ball:{x:ball.pos[0],y:ball.pos[1],vx:ball.vel[0],vy:ball.vel[1],r:ball.radius,color:ball.color},
      players:f.players.map(p=>({id:p.id,team:p.team,x:p.pos[0],y:p.pos[1],vx:p.vel[0],vy:p.vel[1],input:p.input,
        isKicking:!!p.isKicking,invMass:p.invMass}))};
    this.tracker.update(frame);
    const built=this.tracker.build(frame,f.playerId,this.delay);
    if(!built)throw Error('Jugador fuera del 4v4');
    if(this.mode!=='BOT'){
      // jugando el humano, sus teclas son las decisiones propias que la red espera ver en la historia
      const me=f.players.find(p=>p.id===f.playerId);this.tracker.pushDecision(inputToAction(me.input));
    }
    return built;
  }
  async frame(f,send){
    if(f.session!==this.id||f.epoch!==this.epoch)return;
    const built=this.observe(f);
    if(this.mode!=='BOT'&&!f.qualify)return;
    this.queue={f,built,send};await this.drain();
  }
  async drain(){
    if(this.running||this.pending||!this.queue)return;
    const job=this.queue;this.queue=null;this.running=true;
    const generation=this.modelGeneration,start=performance.now();
    try{
      const out=await this.session.run({obs:new this.ort.Tensor('float32',Float32Array.from(job.built.obs),[1,OBS_DIM])});
      if(generation!==this.modelGeneration||job.f.epoch!==this.epoch||job.f.session!==this.id)return;
      if(!out.logits||out.logits.data.length!==18)throw Error('Wrong policy output width');
      const action=sampleLogits(out.logits.data,this.temperature);
      const key=decodeOwnAction(action,job.built.team),mask=maskForKey(key);
      const response={type:'action',session:this.id,epoch:this.epoch,frame:job.f.frame,capturedAt:job.f.capturedAt,
        action,mask,inferenceMs:performance.now()-start,qualify:!!job.f.qualify};
      if(!job.f.qualify)this.pending={world:key.world,mask,frame:job.f.frame,created:performance.now()};
      job.send(response);
    }catch(e){if(generation===this.modelGeneration){this.mode='SUSPENDIDO';this.resetPolicy();job.send({type:'error',message:e.message});}}
    finally{this.running=false;if(this.queue&&!this.pending)await this.drain();}
  }
  async ack(message){
    const p=this.pending;
    if(!p||message.session!==this.id||message.epoch!==this.epoch||message.frame!==p.frame)return;
    if(message.accepted!==true||message.mask!==p.mask||performance.now()-p.created>100){this.mode='SUSPENDIDO';this.resetPolicy();return;}
    this.tracker.pushDecision(p.world);this.pending=null;await this.drain();
  }
}
module.exports={X4Engine,validateX4Meta,maskForKey};
