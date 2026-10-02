// Offline gate: downloads public code into memory only; never opens a room.
const assert=require('node:assert/strict'),vm=require('node:vm'),crypto=require('node:crypto');
const profile=require('./official_profile');
const {Arbiter}=require('./arbiter');
async function officialSource(){
  const html=await (await fetch('https://www.haxball.com/play')).text();
  const frame=/src="([^"]+game\.html)"/.exec(html)?.[1];if(!frame)throw Error('Official frame not found');
  const url=new URL('game-min.js',new URL(frame,'https://www.haxball.com/play')).href;
  const bytes=Buffer.from(await (await fetch(url)).arrayBuffer());
  if(crypto.createHash('sha256').update(bytes).digest('hex')!==profile.SHA256)throw Error('Official client changed; adapter must stay disabled');
  return {source:bytes.toString('utf8'),url};
}
function extractClass(source,name){
  const start=source.indexOf('class '+name+'{');const next=/class [\w$]+(?: extends [\w$]+)?\{/.exec(source.slice(start+8));
  if(start<0||!next)throw Error('Missing native class '+name);
  return source.slice(start,start+8+next.index);
}
async function probe(){
  const {source,url}=await officialSource();
  const context={Set,window:{document:{addEventListener(){},removeEventListener(){}}},
    M:(self,fn)=>fn.bind(self),Ja:class{},D:{h:()=>{}},m:{j:{Jd:{v:()=>({v:key=>({ArrowUp:'Up',ArrowRight:'Right',KeyX:'Kick'})[key]})}}}};
  vm.createContext(context);
  vm.runInContext(extractClass(source,'bc')+';globalThis.NativeControls=bc',context);
  const controls=new context.NativeControls(),sent=[];
  controls.Bg=operation=>sent.push(operation.input);
  const nativeDisc=()=>({a:{x:200,y:-688},G:{x:0,y:0},ra:{x:0,y:0},V:15,S:0xFFFFFF,ca:1,i:39,C:2});
  const d=nativeDisc(),ball={...nativeDisc(),V:10,S:0xFF3F34};
  const room={K:[{Z:7,I:d,fa:{ba:1},W:0,Yb:false,Ud:false}],U:{rb:[{he:0,ie:1,S:0xEC7458}],us:()=>({name:'Real Soccer ONE',discs:[{pos:[0,0]}],joints:[]})},
    M:{va:{H:[ball,d]},Cb:1,Tb:0,Ob:0,Nc:12,Ta:0}};
  const view={za:{T:room,yc:7,Y:60},W:controls,Fa(){},ld(){},sf(){},la(){}};
  const visible=profile.snapshot(view);
  assert.deepEqual(visible.ball.pos,[200,-688]);assert.equal(visible.ball.color,0xFF3F34);
  assert.equal(visible.joints[0].color,0xEC7458);assert.equal(visible.playerId,7);
  assert.equal(profile.stadium(view).name,'Real Soccer ONE');
  let time=0;const arbiter=new Arbiter({write:mask=>profile.writeMask(view,mask),now:()=>time});
  arbiter.session='gate';arbiter.permission=arbiter.ready=true;
  for(let i=0;i<100;i++){
    arbiter.toggle();assert.equal(arbiter.mode,'BOT');
    assert(arbiter.accept({session:'gate',epoch:arbiter.epoch,frame:i,capturedAt:time,mask:24}));
    arbiter.toggle();assert.equal(controls.ng,0);assert.equal(view.za.yc,7);assert.equal(view.za.T,room);
  }
  assert(sent.includes(24));assert.equal(sent.at(-1),0);
  controls.Fa({code:'ArrowUp',preventDefault(){}});assert.equal(controls.ng,1);
  controls.ld({code:'ArrowUp'});assert.equal(controls.ng,0);
  return {gate:'offline_native_controls',passed:true,url,sha256:profile.SHA256,
    native_input_class:true,visible_state_fixture:true,handoffs:100,second_connection:false,
    real_room_tested:false,browser_performance_qualified:false};
}
module.exports={officialSource,extractClass,probe};
if(require.main===module)probe().then(report=>console.log(JSON.stringify(report,null,2))).catch(e=>{console.error(e.message);process.exitCode=1;});
