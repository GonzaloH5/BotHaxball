// Proceso del cerebro (learn/hx11_cerebro.py): un JSON por línea en cada sentido.
"use strict";

const { spawn } = require("child_process");
const path = require("path");
const readline = require("readline");

const REPO = path.join(__dirname, "..", "..");

function startCerebro({ model, pos, temp = 0.7, hyst = 0, every = 3, ens = 0, ensM = 0, renovarPatada = 1,
                        python, onDecision, onReady }) {
  const py = python || path.join(REPO, ".venv", process.platform === "win32" ? "Scripts\\python.exe" : "bin/python");
  const argv = ["-u", "-m", "learn.hx11_cerebro", "--model", model, "--temp", String(temp)];
  if (pos) argv.push("--pos", pos);
  if (hyst > 0) argv.push("--hyst", String(hyst));
  if (every !== 3) argv.push("--every", String(every));
  if (ens > 1) argv.push("--ens", String(ens), "--ens-m", String(ensM));   // acciones en bloque (ACT)
  if (renovarPatada < 1) argv.push("--renovar-patada", String(renovarPatada));   // tecla de patear persistente   // sin puesto: modo de varios bots (cada uno se registra)
  const proc = spawn(py, argv, {
    cwd: REPO, env: { ...process.env, PYTHONIOENCODING: "utf-8" }, stdio: ["pipe", "pipe", "pipe"], windowsHide: true,
  });
  readline.createInterface({ input: proc.stdout }).on("line", (line) => {
    let msg;
    try { msg = JSON.parse(line); } catch { return; }
    if (msg.listo) onReady && onReady(msg);
    else if (msg.input != null) onDecision && onDecision(msg);
  });
  proc.stderr.on("data", (d) => process.stderr.write(`[cerebro] ${d}`));
  return {
    proc,
    send: (fr) => { if (proc.stdin.writable) proc.stdin.write(JSON.stringify(fr) + "\n"); },
    setPos: (p) => { if (proc.stdin.writable) proc.stdin.write(JSON.stringify({ set_pos: p }) + "\n"); },
    stop: () => { try { proc.stdin.end(); proc.kill(); } catch {} },
  };
}

module.exports = { startCerebro };
