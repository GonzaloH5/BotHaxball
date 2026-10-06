// Shared CLI/hybrid inference utilities. No room connection or training side effects.
const fs=require('fs'),os=require('os'),path=require('path'),{spawnSync}=require('child_process');
function sampleLogits(logits,temperature=0){
  if(!logits.length||!Array.from(logits).every(Number.isFinite))throw Error('Invalid policy logits');
  if(temperature<=0)return Array.from(logits).indexOf(Math.max(...logits));
  const mx=Math.max(...logits),weights=Array.from(logits,l=>Math.exp((l-mx)/temperature));
  let value=Math.random()*weights.reduce((a,b)=>a+b,0);
  for(let i=0;i<weights.length;i++){value-=weights[i];if(value<=0)return i;}return weights.length-1;
}
function geometryFromText(text){
  const root=path.resolve(__dirname,'..');
  const directory=fs.mkdtempSync(path.join(os.tmpdir(),'haxballrl-geometry-')),file=path.join(directory,'map.hbs');
  try{
    fs.writeFileSync(file,text);
    const python=process.env.HAXBALL_PYTHON||path.join(root,'.venv',process.platform==='win32'?'Scripts/python.exe':'bin/python');
    const result=spawnSync(python,['-m','export.stadium_geom',file],{cwd:root,encoding:'utf8',maxBuffer:8<<20,timeout:30000});
    if(result.status!==0)throw Error('Geometry preparation failed: '+(result.stderr||result.error));
    return JSON.parse(result.stdout);
  }finally{if(fs.existsSync(file))fs.unlinkSync(file);fs.rmdirSync(directory);}
}
// Opciones de sesión ONNX para un bot. Con varios bots en la misma PC, el valor por defecto de onnxruntime
// (un hilo por núcleo en cada sesión) hace que los procesos compitan por la CPU: en la prueba del 2026-10-06
// con 7 bots cada uno decidía cada 9–10 ticks en vez de cada 3. La red es chica: 1 hilo alcanza.
function ortSessionOptions(threads=1){
  const n=Math.max(1,Math.floor(Number(threads))||1);
  return {intraOpNumThreads:n,interOpNumThreads:1,executionMode:'sequential'};
}
module.exports={sampleLogits,geometryFromText,ortSessionOptions};
