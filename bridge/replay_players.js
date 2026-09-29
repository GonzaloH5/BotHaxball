// Jugadores de un replay (id, nombre, equipo al inicio). Uso: node bridge/replay_players.js <replay.hbr2>
const fs = require("fs");
const API = require("./haxball")({}, { language: false });
const data = API.Replay.readAll(new Uint8Array(fs.readFileSync(process.argv[2])));
const room = data.roomData;
console.log(JSON.stringify({
  stadium: room.stadium && room.stadium.name, frames: data.totalFrames, goals: data.goalMarkers.length,
  players: (room.players || []).map((p) => ({ id: p.id, name: p.name, team: p.team ? p.team.id : null })),
}));
process.exit(0);
