// Mete N bots RS4-Z (deploy/bot.js) en una sala existente, cada uno en su proceso.
//   node deploy/rs4z/join_bots.js --join <link o id> [--count 7] [--password x] [--model deploy/rs4z/model.onnx]
//        [--hyst 1] [--temp 0] [--trace]
// Registros en runs/rs4z_room/logs/<prefijo>_<n>.log (y trace_<prefijo>_<n>.jsonl con --trace). Ctrl+C detiene a todos.
// Dos grupos con modelos distintos: correrlo dos veces con --prefix y --model distintos.
"use strict";
const { spawn } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");

const args = process.argv.slice(2);
const arg = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const raw = arg("--join", null);
if (!raw) { console.error("falta --join <link o id de la sala>"); process.exit(1); }
const id = raw.includes("?c=") ? new URL(raw).searchParams.get("c") : raw;
const count = Number(arg("--count", "7"));
const model = arg("--model", path.join(__dirname, "model.onnx"));
const password = arg("--password", null);
const prefix = arg("--prefix", "RL-Bot");
const logs = path.resolve(__dirname, "..", "..", "runs", "rs4z_room", "logs");
fs.mkdirSync(logs, { recursive: true });
const children = [];
for (let k = 1; k <= count; k++) {
  setTimeout(() => {
    const out = fs.openSync(path.join(logs, `${prefix}_${k}.log`), "w");
    const botArgs = [path.join(__dirname, "..", "bot.js"), "--model", model, "--join", id, "--player", `${prefix}-${k}`];
    if (password) botArgs.push("--password", password);
    if (arg("--hyst", null)) botArgs.push("--hyst", arg("--hyst"));
    if (args.includes("--trace")) botArgs.push("--trace", path.join(logs, `trace_${prefix}_${k}.jsonl`));
    if (arg("--temp", null)) botArgs.push("--temp", arg("--temp"));
    if (arg("--ort-threads", null)) botArgs.push("--ort-threads", arg("--ort-threads"));
    if (arg("--delay", null)) botArgs.push("--delay", arg("--delay"));
    if (arg("--map", null)) botArgs.push("--map", arg("--map"));
    if (arg("--extrapolate", null)) botArgs.push("--extrapolate", arg("--extrapolate"));
    if (k === 1) botArgs.push("--dump-stadium", path.join(logs, `room_stadium_${prefix}.hbs`));
    const child = spawn(process.execPath, botArgs, { stdio: ["ignore", out, out] });
    child.on("exit", (code) => console.log(`${prefix}-${k} salió (${code})`));
    children.push(child);
    console.log(`${prefix}-${k} uniéndose a ${id}`);
  }, (k - 1) * 2500);
}
const stop = () => { for (const c of children) c.kill(); process.exit(0); };
process.on("SIGINT", stop);
process.on("SIGTERM", stop);
