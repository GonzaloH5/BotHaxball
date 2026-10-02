// Local inference only. This file never imports Room, joins rooms, or trains.
const http=require('http'),fs=require('fs'),path=require('path'),crypto=require('crypto');
const {WebSocketServer}=require('ws'),ort=require('onnxruntime-node');
const {Engine,validateMeta}=require('./hybrid/engine');
const {geometryFromText}=require('./runtime');
async function loadModel(filename){
  const meta=validateMeta(JSON.parse(fs.readFileSync(filename.replace(/\.onnx$/,'.json'),'utf8')));
  const session=await ort.InferenceSession.create(filename,{intraOpNumThreads:2,interOpNumThreads:1});
  if(!session.inputNames.includes('obs')||!session.outputNames.includes('logits')||!!meta.recurrent!==session.inputNames.includes('memory'))throw Error('ONNX/JSON contract mismatch');
  return {meta,session,hash:crypto.createHash('sha256').update(fs.readFileSync(filename)).digest('hex'),name:path.basename(filename)};
}
async function startServer({model,port=17841,token=crypto.randomBytes(32).toString('base64url'),loader=loadModel,geometry=geometryFromText}={}){
  let current=await loader(model),owner=null;const clients=new Set(),maps=new Map();
  const server=http.createServer((req,res)=>{res.writeHead(404,{'Content-Type':'text/plain','Cache-Control':'no-store'});res.end('Local RS4 inference service');});
  const wss=new WebSocketServer({noServer:true,maxPayload:2<<20,perMessageDeflate:false});
  server.on('upgrade',(request,socket,head)=>{
    if(request.url!=='/rs4'||!/^chrome-extension:\/\/[a-p]{32}$/.test(request.headers.origin||''))return socket.destroy();
    wss.handleUpgrade(request,socket,head,ws=>wss.emit('connection',ws,request));
  });
  wss.on('connection',socket=>{
    let authenticated=false,engine=null,tab=null,pollTimer=null;clients.add(socket);
    const send=value=>{if(socket.readyState===1)socket.send(JSON.stringify(value));};
    const authTimer=setTimeout(()=>{if(!authenticated)socket.close(1008,'Authentication required');},5000);
    const startPolling=()=>{clearInterval(pollTimer);pollTimer=setInterval(()=>{
      if(engine&&authenticated&&owner?.socket===socket)send({type:'poll',tabId:tab});
    },1000*current.meta.frame_skip/60);};
    const reset=()=>{clearInterval(pollTimer);pollTimer=null;if(engine){engine.control({mode:'HUMANO',epoch:engine.epoch+1,session:engine.id});}if(owner?.socket===socket)owner=null;engine=null;tab=null;};
    socket.on('close',()=>{clearTimeout(authTimer);reset();clients.delete(socket);});
    socket.on('error',()=>{});
    // Serialize protocol operations, but do not await ONNX in this queue.
    let chain=Promise.resolve();
    socket.on('message',raw=>{chain=chain.then(async()=>{
      let msg;try{msg=JSON.parse(raw.toString());}catch{throw Error('Invalid JSON');}
      if(!msg||typeof msg.type!=='string')throw Error('Invalid protocol message');
      if(!authenticated){
        const supplied=Buffer.from(typeof msg.token==='string'?msg.token:'');const expected=Buffer.from(token);
        if(msg.type!=='auth'||supplied.length!==expected.length||!crypto.timingSafeEqual(supplied,expected)){socket.close(1008,'Invalid token');return;}
        authenticated=true;clearTimeout(authTimer);send({type:'authenticated',model:{name:current.name,hash:current.hash,frameSkip:current.meta.frame_skip}});return;
      }
      if(msg.type==='ping'){send({type:'pong'});return;}
      if(msg.type==='claim'){
        if(!Number.isInteger(msg.tabId)||msg.tabId<0)throw Error('Invalid tab owner');
        if(owner&&(owner.socket!==socket||owner.tab!==msg.tabId)){send({type:'denied',tabId:msg.tabId,message:'Otra pestaña activa'});return;}
        if(!owner){owner={socket,tab:msg.tabId};tab=msg.tabId;engine=new Engine(current.meta,current.session,ort);startPolling();}
        send({type:'claimed',tabId:tab,model:{name:current.name,hash:current.hash,frameSkip:current.meta.frame_skip}});return;
      }
      if(owner?.socket!==socket||msg.tabId!==tab)throw Error('Not the active tab');
      if(msg.type==='release'){reset();return;}
      if(msg.type==='control'){engine.control(msg);return;}
      if(msg.type==='map'){
        if(engine.mode==='BOT')throw Error('Map preparation requires human control');
        if(!msg.stadium||msg.stadium.name!=='Real Soccer ONE')throw Error('v1 supports RS ONE only');
        const text=JSON.stringify(msg.stadium);if(text.length>1<<20)throw Error('Map too large');
        const key=crypto.createHash('sha256').update(text).digest('hex');
        if(!maps.has(key)){if(maps.size>=4)maps.delete(maps.keys().next().value);maps.set(key,geometry(text));}
        engine.map(msg.stadium,maps.get(key));send({type:'mapReady',tabId:tab,mapHash:key});return;
      }
      if(msg.type==='frame'){const frameEngine=engine;frameEngine.frame(msg,send).catch(e=>{if(engine!==frameEngine)return;
        frameEngine.mode='SUSPENDIDO';frameEngine.resetPolicy();send({type:'error',message:e.message});});return;}
      if(msg.type==='ack'){const ackEngine=engine;ackEngine.ack(msg).catch(e=>{if(engine!==ackEngine)return;
        ackEngine.mode='SUSPENDIDO';ackEngine.resetPolicy();send({type:'error',message:e.message});});return;}
      if(msg.type==='reload'){
        if(engine.mode!=='HUMANO')throw Error('Reload only in HUMAN mode');
        engine.resetPolicy();current=await loader(model);engine=new Engine(current.meta,current.session,ort);maps.clear();startPolling();
        send({type:'reloaded',tabId:tab,model:{name:current.name,hash:current.hash,frameSkip:current.meta.frame_skip}});return;
      }
      throw Error('Unsupported protocol message');
    }).catch(e=>{if(engine){engine.mode='SUSPENDIDO';engine.resetPolicy();}send({type:'error',message:e.message});});});
  });
  await new Promise((resolve,reject)=>{server.once('error',reject);server.listen(port,'127.0.0.1',resolve);});
  return {server,wss,token,port:server.address().port,close:()=>new Promise(resolve=>{for(const socket of clients)socket.terminate();wss.close();server.close(resolve);})};
}
module.exports={startServer,loadModel};
if(require.main===module){
  const args=process.argv.slice(2),arg=(name,fallback)=>args.includes(name)?args[args.indexOf(name)+1]:fallback;
  if(args.includes('--help')){console.log('node deploy/hybrid_server.js --model path/model.onnx [--port 17841]');}
  else{
    const model=arg('--model',null),port=Number(arg('--port','17841'));
    if(!model||!model.endsWith('.onnx')||!Number.isInteger(port)||port<1||port>65535){console.error('Requires --model *.onnx and a valid port');process.exitCode=1;}
    else startServer({model:path.resolve(model),port}).then(service=>{
      console.log(`RS4 local: ws://127.0.0.1:${service.port}/rs4\nToken para emparejar: ${service.token}\nNo conecta a salas ni modifica checkpoints.`);
      const stop=()=>service.close().then(()=>process.exit(0));process.once('SIGINT',stop);process.once('SIGTERM',stop);
    }).catch(e=>{console.error(e.message);process.exitCode=1;});
  }
}
