// Local bot lifecycle. Only fixed bot.js and server-discovered models/maps can run.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const {fork}=require('node:child_process');
const ACTIVE=new Set(['queued','starting','connected','stopping']);
function roomId(value){
  if(typeof value!=='string')throw Error('Ingresá el enlace o ID de la sala.');
  let id=value.trim();
  if(id.includes('://')){let url;try{url=new URL(id);}catch{throw Error('Enlace de sala inválido.');}
    if(url.protocol!=='https:'||url.hostname!=='www.haxball.com'||url.pathname!=='/play')throw Error('Usá un enlace https://www.haxball.com/play?c=...');id=url.searchParams.get('c')||'';}
  if(!/^[A-Za-z0-9_-]{6,100}$/.test(id))throw Error('ID de sala inválido.');return id;
}
function text(value,label,max,empty=false){if(typeof value!=='string'||(!empty&&!value.trim())||value.length>max||/[\x00-\x1f]/.test(value)||value.trim().startsWith('--'))throw Error(`${label} inválido (máximo ${max} caracteres; no puede empezar con --).`);return value.trim();}
class BotManager {
  constructor({model,spawn=fork,limit=8,startTimeout=60000}={}){
    this.spawn=spawn;this.limit=limit;this.startTimeout=startTimeout;this.bots=new Map();this.closed=false;
    this.models=new Map();
    for(const [id,label,file] of [['current','Modelo del servicio',model],['classic','Clásico',path.join(__dirname,'model.onnx')],['multi','Multitarea',path.join(__dirname,'multi/model.onnx')]]){
      if(file&&fs.existsSync(file)&&fs.existsSync(file.replace(/\.onnx$/,'.json')))this.models.set(id,{id,label,path:path.resolve(file)});
    }
    const dir=path.resolve(__dirname,'../stadiums');
    this.maps=fs.readdirSync(dir).filter(f=>f.endsWith('.hbs')).map(f=>f.slice(0,-4));
  }
  snapshot(){return {limit:this.limit,models:[...this.models.values()].map(({id,label})=>({id,label})),maps:this.maps,bots:[...this.bots.values()].map(b=>({...b.public}))};}
  validate(input){
    if(!input||!['host','join'].includes(input.mode))throw Error('Elegí crear sala o unirse.');
    if(!Array.isArray(input.profiles)||!input.profiles.length||input.profiles.length>this.limit)throw Error(`Elegí entre 1 y ${this.limit} bots.`);
    const profiles=input.profiles.map(p=>{
      if(!p||!this.models.has(p.model))throw Error('Modelo no disponible.');
      if(![0,1,2].includes(p.team))throw Error('Equipo inválido.');
      const name=text(p.name,'Nombre',25),avatar=text(p.avatar??'','Avatar',2,true),flag=text(p.flag||'uy','País',2);
      if(!/^[a-z]{2}$/.test(flag))throw Error('País: usá dos letras minúsculas, por ejemplo uy.');
      return {name,avatar,flag,team:p.team,model:p.model};
    });
    if(new Set(profiles.map(p=>p.name.toLowerCase())).size!==profiles.length)throw Error('Cada bot del grupo necesita un nombre distinto.');
    const password=input.password??'';
    if(typeof password!=='string'||password.length>64||/[\x00-\x1f]/.test(password))throw Error('Contraseña inválida (máximo 64 caracteres).');
    const result={mode:input.mode,profiles,password};
    if(input.mode==='join')result.roomId=roomId(input.room);
    else {
      result.name=text(input.name,'Nombre de sala',40);result.token=text(input.token,'Token headless',2048);
      if(!this.maps.includes(input.stadium))throw Error('Elegí un mapa disponible.');result.stadium=input.stadium;
      for(const [key,min,max] of [['maxPlayers',profiles.length,30],['timeLimit',0,99],['scoreLimit',0,99]]){
        if(!Number.isInteger(input[key])||input[key]<min||input[key]>max)throw Error(`${key}: valor entre ${min} y ${max}.`);result[key]=input[key];
      }
      result.public=input.public===true;
    }
    return result;
  }
  start(input){
    if(this.closed)throw Error('El gestor se está cerrando.');
    const config=this.validate(input);
    if([...this.bots.values()].filter(b=>ACTIVE.has(b.public.status)).length+config.profiles.length>this.limit)throw Error(`Máximo ${this.limit} bots activos. Detené alguno antes de agregar más.`);
    // Retain at most 40 finished records, without removing active instances.
    const finished=[...this.bots.values()].filter(b=>!ACTIVE.has(b.public.status));
    for(const b of finished.slice(0,Math.max(0,finished.length-32)))this.bots.delete(b.public.id);
    const group=crypto.randomUUID();
    const batch=config.profiles.map((p,i)=>{
      const id=crypto.randomUUID(),b={public:{id,group,name:p.name,avatar:p.avatar,model:p.model,team:0,requestedTeam:p.team,host:config.mode==='host'&&i===0,status:'queued',enabled:true,link:config.roomId?`https://www.haxball.com/play?c=${config.roomId}`:'',error:'',createdAt:new Date().toISOString()},profile:p,child:null,timer:null,deadline:null};this.bots.set(id,b);return b;
    });
    if(config.mode==='host'){
      batch[0].onLink=id=>batch.slice(1).forEach((b,i)=>this.schedule(b,{...config,mode:'join',roomId:id,token:''},i*800));
      batch[0].dependents=batch.slice(1);
      this.launch(batch[0],config);
    }else batch.forEach((b,i)=>this.schedule(b,config,i*800));
    return this.snapshot();
  }
  schedule(b,config,delay){if(b.public.status!=='queued')return;b.timer=setTimeout(()=>{b.timer=null;this.launch(b,config);},delay);}
  launch(b,config){
    if(this.closed||b.public.status!=='queued')return;
    b.public.status='starting';
    if(config.mode==='join')b.public.link=`https://www.haxball.com/play?c=${config.roomId}`;
    const p=b.profile,args=['--managed','--model',this.models.get(p.model).path,'--player',p.name,'--avatar',p.avatar,'--flag',p.flag,'--team',String(p.team)];
    if(config.mode==='join')args.push('--join',config.roomId);
    else {args.push('--name',config.name,'--stadium',config.stadium,'--max-players',String(config.maxPlayers),'--time-limit',String(config.timeLimit),'--score-limit',String(config.scoreLimit));if(config.public)args.push('--public');}
    const redact=value=>{let s=String(value||'');for(const secret of [config.token,config.password])if(secret)s=s.split(secret).join('[oculto]');return s.slice(0,500);};
    const fail=message=>{b.public.error=redact(message);this.stop(b.public.id,'failed');};
    try{
      b.child=this.spawn(path.join(__dirname,'bot.js'),args,{windowsHide:true,silent:true,env:{...process.env,RS4_BOT_SECRETS:JSON.stringify({token:config.token,password:config.password})}});
      // Drain logs but never relay them: third-party errors can include credentials.
      b.child.stdout?.resume();b.child.stderr?.resume();
      b.deadline=setTimeout(()=>fail('Tiempo de conexión agotado. Revisá token, enlace, contraseña y red.'),this.startTimeout);
      b.child.on('message',m=>{
        if(!m||!['starting','connected'].includes(b.public.status))return;
        if(m.type==='connected') {b.public.status='connected';b.public.playerId=m.playerId;if(!b.public.host||b.public.link)clearTimeout(b.deadline);this.assignTeam(b);}
        else if(m.type==='link'&&b.public.host){let id;try{id=roomId(m.link);}catch{return;}b.public.link=`https://www.haxball.com/play?c=${id}`;clearTimeout(b.deadline);b.onLink?.(id);b.onLink=null;}
        else if(m.type==='state'){b.public.team=m.team;b.public.playing=!!m.playing;}
        else if(m.type==='enabled')b.public.enabled=!!m.enabled;
        else if(m.type==='closed') {b.public.error=redact(m.error);b.public.status=m.error?'failed':'stopped';}
        else if(m.type==='commandError')b.public.error=redact(m.error);
      });
      b.child.once('error',()=>fail('No se pudo iniciar el proceso del bot.'));
      b.child.once('exit',code=>{clearTimeout(b.deadline);clearTimeout(b.killTimer);if(!['failed','stopped'].includes(b.public.status)){b.public.status=code?'failed':'stopped';if(code&&!b.public.error)b.public.error='El proceso terminó con un error. Revisá el modelo y las dependencias de Node.';}b.child=null;for(const other of b.dependents||[])if(other.public.status==='queued')this.stop(other.public.id);});
    }catch{fail('No se pudo iniciar el bot.');}
  }
  send(b,message){if(!b.child?.connected)throw Error('El bot no está conectado.');b.child.send(message,()=>{});}
  assignTeam(b){
    const host=[...this.bots.values()].find(h=>h.public.host&&h.public.status==='connected'&&h.public.link&&h.public.link===b.public.link);
    if(host&&Number.isInteger(b.public.playerId))this.send(host,{type:'team',playerId:b.public.playerId,team:b.public.requestedTeam});
  }
  command(input){
    const b=this.bots.get(input.id);if(!b)throw Error('Bot no encontrado.');
    if(input.action==='stop'){this.stop(input.id);return this.snapshot();}
    if(b.public.status!=='connected')throw Error('Esperá a que el bot se conecte.');
    if(input.action==='enabled'&&typeof input.enabled==='boolean')this.send(b,{type:'enabled',enabled:input.enabled});
    else if(input.action==='team'&&[0,1,2].includes(input.team)){b.public.requestedTeam=input.team;this.send(b,{type:'team',team:input.team});this.assignTeam(b);}
    else if(['start','stopGame','pause'].includes(input.action)&&b.public.host)this.send(b,{type:'game',action:input.action==='stopGame'?'stop':input.action});
    else throw Error('Comando inválido o requiere ser host.');return this.snapshot();
  }
  stop(id,finalStatus='stopped'){
    const b=this.bots.get(id);if(!b)return;clearTimeout(b.timer);clearTimeout(b.deadline);b.onLink=null;
    for(const other of b.dependents||[])if(other.public.status==='queued')this.stop(other.public.id);
    if(!b.child){b.public.status=finalStatus;return;}
    b.public.status=finalStatus==='failed'?'failed':'stopping';
    try{if(b.child.connected)b.child.send({type:'stop'},()=>{});}catch{}
    if(!b.killTimer)b.killTimer=setTimeout(()=>{b.child?.kill();},2000);
  }
  stopAll(){for(const b of this.bots.values())if(ACTIVE.has(b.public.status))this.stop(b.public.id);return this.snapshot();}
  async close(){this.closed=true;this.stopAll();await Promise.all([...this.bots.values()].filter(b=>b.child).map(b=>new Promise(resolve=>{b.child.once('exit',resolve);setTimeout(()=>{b.child?.kill();resolve();},2500).unref();})));}
}
module.exports={BotManager,roomId};
