// Field mappings only, not a redistributed copy of the official client.
(function(root,factory){const api=factory();if(typeof module==='object')module.exports=api;else root.RS4Official=api;})(globalThis,()=>{
  'use strict';
  const SHA256='322d1904fc09896e5a16482f53e35a6e046c50a40c0270646305700b7420e885';
  const VERSION='0349dd60';
  const finite=n=>typeof n==='number'&&Number.isFinite(n);
  const point=p=>{if(!p||!finite(p.x)||!finite(p.y))throw Error('Invalid visible point');return [p.x,p.y];};
  function validView(view){
    const client=view?.za, controls=view?.W;
    return !!(client?.T&&Array.isArray(client.T.K)&&Number.isInteger(client.yc)&&Number.isInteger(client.Y)
      &&controls?.Qc instanceof Set&&typeof controls.A==='function'&&typeof controls.Al==='function'
      &&typeof controls.Fa==='function'&&typeof controls.ld==='function'&&typeof view.Fa==='function'&&typeof view.ld==='function'
      &&typeof view.sf==='function'&&typeof view.la==='function');
  }
  function disc(d,playerId=null){
    if(!d||![d.V,d.S,d.ca].every(finite))throw Error('Invalid visible disc');
    return {pos:point(d.a),vel:point(d.G),gravity:point(d.ra),radius:d.V,color:d.S,
      invMass:d.ca,cMask:d.i,cGroup:d.C,playerId};
  }
  function stadium(view){
    if(!validView(view)||typeof view.za.T.U?.us!=='function')throw Error('Unsupported native stadium');
    // Respect maps whose native export is disabled. Never forward room/player objects.
    const data=JSON.parse(JSON.stringify(view.za.T.U.us()));
    if(!data||typeof data.name!=='string'||!Array.isArray(data.discs)||!Array.isArray(data.joints||[]))throw Error('Invalid native stadium export');
    return data;
  }
  function snapshot(view){
    if(!validView(view))throw Error('Unsupported official controller');
    const c=view.za,room=c.T,game=room.M;
    if(!game||!game.va||!Array.isArray(game.va.H))return null;
    const players=room.K.filter(p=>p.I&&[1,2].includes(p.fa?.ba)).map(p=>({
      id:p.Z,team:p.fa.ba-1,...disc(p.I,p.Z),input:p.W,isKicking:!!p.Yb,desynchronized:!!p.Ud}));
    const me=players.find(p=>p.id===c.yc);
    const discs=game.va.H.map(d=>disc(d,players.find(p=>room.K.find(q=>q.Z===p.id)?.I===d)?.id??null));
    const joints=(room.U.rb||[]).map(j=>({d0:j.he,d1:j.ie,color:j.S}));
    if(!me)return null;
    if(!Number.isInteger(c.Y)||!Number.isInteger(game.Cb)||![game.Tb,game.Ob,game.Nc,game.Ta].every(finite))throw Error('Invalid visible game clock');
    return {frame:c.Y,playerId:c.yc,players,discs,joints,ball:discs[0],
      score:[game.Tb,game.Ob],phase:game.Cb,paused:game.Ta>0,elapsed:game.Nc,
      synchronized:!room.K.find(p=>p.Z===c.yc)?.Ud};
  }
  function roomStatus(view){
    if(!validView(view))return {blocker:'Esperando al controlador oficial: entrá a una sala. Si ya estás dentro, recargá HaxBall después de recargar la extensión.'};
    const room=view.za.T, game=room.M, me=room.K.find(p=>p.Z===view.za.yc);
    const teams=[1,2].map(t=>room.K.filter(p=>p.I&&p.fa?.ba===t).length);
    const mapName=typeof room.U?.D==='string'?room.U.D:'';
    let blocker='';
    if(!me||![1,2].includes(me.fa?.ba))blocker='Estás como espectador: necesitás un jugador en rojo o azul.';
    else if(!game||!game.va||!me.I)blocker='Partido detenido: esperá a que empiece el juego.';
    else if(game.Cb===3)blocker='Partido terminado: esperá el próximo inicio.';
    else if(game.Ta>0)blocker='Partido pausado: reanudalo para comprobar.';
    else if(me.Ud)blocker='Jugador desincronizado: esperando estado válido.';
    else if(teams.some(n=>n!==4))blocker=`Este modelo necesita 4v4: rojo ${teams[0]}/4 · azul ${teams[1]}/4.`;
    return {blocker,playerId:view.za.yc,team:me?.fa?.ba??0,teams,mapName};
  }
  const actions=['Up','Down','Left','Right','Kick'];
  function writeMask(view,mask){
    if(!validView(view)||!Number.isInteger(mask)||mask<0||mask>31)throw Error('Invalid native control');
    const w=view.W;w.Qc.clear();actions.forEach((name,i)=>{if(mask&(1<<i))w.Qc.add(name);});w.A();
    if(w.ng!==mask)throw Error('Native controller did not accept mask');
    return mask;
  }
  return {SHA256,VERSION,validView,disc,stadium,snapshot,roomStatus,writeMask};
});
