// RS4 public restart lines. Map discovery is structural, not a scan of every
// red/blue object. Runtime activation is measured from visible endpoints.
const RED_JOINT = 0xEC7458, BLUE_JOINT = 0x48BEF9;
function rgb(v) {
  if (typeof v === 'string' && /^[0-9a-f]{6}$/i.test(v.replace(/^#/, ''))) return parseInt(v.replace(/^#/, ''),16);
  return Number.isInteger(v) ? v : -1;
}
function xy(disc) {
  const p = disc?.pos;
  return Array.isArray(p) && p.length===2 && p.every(Number.isFinite) ? p : p && Number.isFinite(p.x) && Number.isFinite(p.y) ? [p.x,p.y] : null;
}
function candidateRows(stadium, geom) {
  const w=geom?.field_half_w, h=geom?.field_half_h;
  if (!(w>0 && h>0)) return [];
  const rows=[];
  (stadium?.joints || []).forEach((j,id)=>{
    const color=rgb(j.color), team=color===RED_JOINT?0:color===BLUE_JOINT?1:-1;
    if (team<0 || !Number.isInteger(j.d0) || !Number.isInteger(j.d1) || j.d0<=0 || j.d1<=0 || j.d0===j.d1) return;
    const a=stadium.discs?.[j.d0], b=stadium.discs?.[j.d1], p=xy(a), q=xy(b);
    const auxiliary=d=>d?.radius===0 && (d.cMask===0 || Array.isArray(d.cMask) && d.cMask.length===0);
    if (!p || !q || !auxiliary(a) || !auxiliary(b)) return;
    const y=(p[1]+q[1])/2, span=Math.abs(p[0]-q[0]);
    if (Math.abs(p[1]-q[1])>3 || Math.abs(y)<.7*h || Math.abs(y)>1.2*h
        || span<1.5*w || span>2.5*w || Math.abs((p[0]+q[0])/2)>.1*w) return;
    rows.push({id,team,y,d0:j.d0,d1:j.d1});
  });
  return rows;
}
function discoverJointBarriers(stadium, geom) {
  const rows=candidateRows(stadium,geom), upper=rows.filter(r=>r.y<0), lower=rows.filter(r=>r.y>0);
  // Exactly one red and one blue line per band, independent auxiliary discs.
  const pair=rs=>rs.length===2 && rs[0].team!==rs[1].team && Math.abs(rs[0].y-rs[1].y)<=3;
  if (!pair(upper) || !pair(lower) || new Set(rows.flatMap(r=>[r.d0,r.d1])).size!==8) return [];
  return rows.map(r=>r.id);
}
function visibleJointColors(discs, joints, ids, ball, geom) {
  const w=geom?.field_half_w, h=geom?.field_half_h;
  const p=xy(ball), v=ball?.vel || (ball?.speed && [ball.speed.x,ball.speed.y]);
  if (!(w>0 && h>0) || !p || !Array.isArray(v) || v.length!==2 || !v.every(Number.isFinite) || Math.hypot(...v)>.05 || Math.abs(p[1])<.8*h) return [];
  const colors=[];
  for (const id of ids) {
    if (!Number.isInteger(id) || id<0) throw new Error('Invalid barrier joint ID');
    const j=joints?.[id];
    if (!j || rgb(j.color)<0) continue;
    const a=typeof j.d0==='number'?discs[j.d0]:j.d0, b=typeof j.d1==='number'?discs[j.d1]:j.d1;
    const x=xy(a), y=xy(b);
    if (!x || !y || a?.playerId!=null || b?.playerId!=null || a?.radius!==0 || b?.radius!==0) continue;
    const band=(x[1]+y[1])/2, span=Math.abs(x[0]-y[0]);
    // Collapsed/short, off-screen, opposite-band and open-play lines aren't cues.
    if (span<1.5*w || span>2.5*w || Math.abs(x[1]-y[1])>3 || Math.abs(band)<.7*h
        || Math.abs(band)>1.2*h || band*p[1]<=0 || Math.abs(band-p[1])>.35*h
        || Math.abs((x[0]+y[0])/2)>.1*w || Math.abs(p[0])>1.15*w) continue;
    colors.push(j.color);
  }
  return colors;
}
module.exports={discoverJointBarriers,visibleJointColors};
