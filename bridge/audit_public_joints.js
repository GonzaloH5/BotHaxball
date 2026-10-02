// Offline audit: public map geometry/colors only. Never joins a room.
const fs=require('fs'),crypto=require('crypto');
process.env.HAXBALL_BIND='off';
const api=require('./haxball')({}, {language:false});
const {PublicSignalTracker,colorTeam}=require('../deploy/public_signals');
const os=require('os'),nodePath=require('path'),{spawnSync}=require('child_process');
function stadiumGeometry(stadium) {
  const temporary=fs.mkdtempSync(nodePath.join(os.tmpdir(),'rs4-joint-audit-'));
  const file=nodePath.join(temporary,'map.hbs');
  try {
    fs.writeFileSync(file,api.Utils.exportStadium(stadium));
    const root=nodePath.resolve(__dirname,'..');
    const python=process.env.HAXBALL_PYTHON||nodePath.join(root,'.venv',process.platform==='win32'?'Scripts/python.exe':'bin/python');
    const result=spawnSync(python,['-m','export.stadium_geom',file],{cwd:root,encoding:'utf8'});
    if(result.status!==0)throw new Error(result.stderr||String(result.error));
    return JSON.parse(result.stdout);
  } finally {fs.unlinkSync(file);fs.rmdirSync(temporary);}
}
const args=process.argv.slice(2), path=args[0];
if (!path) throw new Error('Usage: node bridge/audit_public_joints.js replay.hbr2 [--max-frames N] [--out report.json]');
const option=name=>args.includes(name)?args[args.indexOf(name)+1]:null;
const limit=Number(option('--max-frames')||Infinity), input=fs.readFileSync(path);
const tracker=new PublicSignalTracker({auto_joints:true});
const report={source:path,sha256:crypto.createHash('sha256').update(input).digest('hex'),
  private_state_used:false,ticks:0,maps:[],stationary_colored_ticks:0,barrier_ticks:0,
  agreeing_ticks:0,conflicting_ticks:0,missing_barrier_ticks:0,examples:[]};
let reader,stadium=null,previous='',done=false;
function finish() {
  if(done)return; done=true;
  if(reader)reader.setSpeed(0);
  const output=JSON.stringify(report,null,2);
  if(option('--out'))fs.writeFileSync(option('--out'),output+'\n');
  console.log(output);process.exit(0);
}
reader=api.Replay.read(new Uint8Array(input),{onGameTick(){
  const gs=reader.gameState, st=reader.state.stadium;
  if(!gs||!st)return;
  if(st!==stadium){
    stadium=st;
    const geom=stadiumGeometry(st);
    const ids=tracker.configureStadium(st,geom);
    report.maps.push({name:st.name,candidates:ids,field_half_w:geom.field_half_w,field_half_h:geom.field_half_h});
  }
  const world=gs.physicsState, raw=world.discs[0];
  const ball={pos:[raw.pos.x,raw.pos.y],vel:[raw.speed.x,raw.speed.y]};
  const barrier=tracker.barrierColor(world.discs,[],world.joints,ball);
  const team=Math.hypot(...ball.vel)<=.05?colorTeam(raw.color):-1, other=colorTeam(barrier);
  report.ticks++;
  if(team>=0){
    report.stationary_colored_ticks++;
    if(other>=0)report.barrier_ticks++;
    if(other===team)report.agreeing_ticks++;
    else if(other>=0||barrier===-2)report.conflicting_ticks++;
    else report.missing_barrier_ticks++;
  }
  const current=team+':'+barrier;
  if(current!==previous&&team>=0&&report.examples.length<16)
    report.examples.push({frame:reader.getCurrentFrameNo(),ball_color:raw.color,ball_position:ball.pos,barrier_color:barrier});
  previous=current;
  if(reader.getCurrentFrameNo()>=limit)finish();
},onEnd:finish},{requestAnimationFrame:cb=>setImmediate(()=>cb(performance.now())),cancelAnimationFrame:clearImmediate});
reader.setSpeed(1000);
setInterval(()=>{if(reader.getCurrentFrameNo()>=reader.maxFrameNo-1)finish();},500);
setTimeout(()=>{report.stopped_by_timeout=true;finish();},55000);
