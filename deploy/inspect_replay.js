// Resume uno o más replays .hbr2: estadio, física, jugadores, duración, goles y cuánto
// "interviene" la sala (eventos de script como mover la pelota en laterales/córners).
// Uso: node inspect_replay.js <archivo.hbr2> [...] [--export-hbs carpeta]
const fs = require("fs");
const path = require("path");
const API = require("node-haxball")();
const { Replay, Utils } = API;

const args = process.argv.slice(2);
const exportIdx = args.indexOf("--export-hbs");
const exportDir = exportIdx >= 0 ? args[exportIdx + 1] : null;
const files = args.filter((a, i) => i !== exportIdx && i !== exportIdx + 1);

for (const f of files) {
  const data = Replay.readAll(new Uint8Array(fs.readFileSync(f)));
  const room = data.roomData;
  const st = room.stadium;
  let hbs = null;
  try { hbs = JSON.parse(Utils.exportStadium(st)); } catch (e) { /* algunos estadios no exportan limpio */ }
  const pp = (hbs && hbs.playerPhysics) || {};
  const counts = {};
  for (const ev of data.events) {
    const name = ev.constructor?.name || String(ev.type);
    counts[name] = (counts[name] || 0) + 1;
  }
  const players = (room.players || []).map((p) => `${p.name}(${p.team?.id ?? "?"})`);
  console.log("=".repeat(70));
  console.log(path.basename(f));
  console.log(`  estadio: ${st.name}  |  ${hbs ? `${hbs.width}x${hbs.height}, spawnDistance ${hbs.spawnDistance}` : "(no exportable)"}`);
  console.log(`  física jugador: ${JSON.stringify(pp)}`);
  if (hbs) {
    const ball = hbs.ballPhysics === "disc0" ? hbs.discs?.[0] : hbs.ballPhysics;
    console.log(`  pelota: ${JSON.stringify(ball)}`);
    console.log(`  objetos: ${hbs.vertexes?.length} vértices, ${hbs.segments?.length} segmentos, ${hbs.planes?.length} planos, ${hbs.discs?.length} discos, ${hbs.goals?.length} arcos`);
  }
  console.log(`  duración: ${(data.totalFrames / 60 / 60).toFixed(1)} min (${data.totalFrames} frames), goles: ${data.goalMarkers.length}`);
  console.log(`  jugadores al inicio (${players.length}): ${players.join(", ")}`);
  console.log(`  eventos: ${Object.entries(counts).sort((a, b) => b[1] - a[1]).map(([k, v]) => `${k}=${v}`).join(", ")}`);
  if (exportDir && hbs) {
    fs.mkdirSync(exportDir, { recursive: true });
    const out = path.join(exportDir, path.basename(f).replace(/\.hbr2$/, ".hbs"));
    fs.writeFileSync(out, JSON.stringify(hbs));
    console.log(`  estadio exportado: ${out}`);
  }
}
process.exit(0);
