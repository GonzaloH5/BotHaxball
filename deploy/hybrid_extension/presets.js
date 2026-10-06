/* Saved setups contain only reusable room settings, never credentials. */
(function(root){
  const defaults={mode:'join',room:'',name:'Entrenamiento RS4',stadium:'rs_one',maxPlayers:16,timeLimit:3,scoreLimit:3,public:false};
  const str=(v,n,fallback='')=>typeof v==='string'?v.slice(0,n):fallback;
  const integer=(v,min,max,fallback)=>Number.isInteger(v)&&v>=min&&v<=max?v:fallback;
  function clean(value={}){
    value=value&&typeof value==='object'?value:{};
    const profiles=(Array.isArray(value.profiles)?value.profiles:[]).slice(0,8).filter(p=>p&&typeof p==='object').map((p,i)=>({
      name:str(p.name,25,`RS4 Bot ${i+1}`),avatar:str(p.avatar,2),flag:/^[a-z]{2}$/.test(p.flag)?p.flag:'uy',team:integer(p.team,0,2,0),model:str(p.model,100,'current')||'current'
    }));
    return {mode:value.mode==='host'?'host':'join',room:str(value.room,2048),name:str(value.name,40,defaults.name),stadium:str(value.stadium,150,'rs_one')||'rs_one',maxPlayers:integer(value.maxPlayers,1,30,16),timeLimit:integer(value.timeLimit,0,99,3),scoreLimit:integer(value.scoreLimit,0,99,3),public:value.public===true,
      profiles:profiles.length?profiles:[{name:'RS4 Bot 1',avatar:'1',flag:'uy',team:0,model:'current'}]};
  }
  const api={clean,defaults,prefix:'rs4Preset:',lastKey:'rs4LastPreset',version:1};
  if(typeof module==='object'&&module.exports)module.exports=api;else root.RS4Presets=api;
})(globalThis);
