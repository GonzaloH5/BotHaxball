// Lista los replays de una carpeta con su mapa, duración, jugadores y cuánto interviene el script de la sala
// (eventos que mueven/modifican discos por minuto). Salida JSON por stdout para tools/import_replays.py.
// Uso: node bridge/replays_by_map.js <carpeta>
const fs = require("fs");
const path = require("path");
const API = require("./haxball")({}, { language: false });
const { Replay } = API;

const dir = process.argv[2];
const out = [];
for (const f of fs.readdirSync(dir).filter((x) => /\.hbr2?$/i.test(x))) {
  const file = path.join(dir, f);
  try {
    const data = Replay.readAll(new Uint8Array(fs.readFileSync(file)));
    const room = data.roomData;
    let script = 0;
    for (const ev of data.events) {
      const n = ev.constructor?.name || "";
      if (/Disc/i.test(n)) script++;  // cambios de discos que manda el host (script de la sala)
    }
    const minutes = data.totalFrames / 3600;
    out.push({ file, name: f, stadium: room.stadium?.name || "?", minutes,
               players: (room.players || []).filter((p) => p.team && p.team.id !== 0).length,
               names: (room.players || []).map((p) => p.name),
               script_per_min: minutes > 0 ? script / minutes : 0, goals: data.goalMarkers.length });
  } catch (e) {
    out.push({ file, name: f, error: e.message });
  }
}
process.stdout.write(JSON.stringify(out));
process.exit(0);
