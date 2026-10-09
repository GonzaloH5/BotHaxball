// Proceso del cerebro (learn/hx11_cerebro.py): un JSON por línea en cada sentido.
"use strict";

const { spawn } = require("child_process");
const path = require("path");
const readline = require("readline");

const REPO = path.join(__dirname, "..", "..");

function startCerebro({ model, pos, temp = 0.7, python, onDecision, onReady }) {
  const py = python || path.join(REPO, ".venv", process.platform === "win32" ? "Scripts\\python.exe" : "bin/python");
  const proc = spawn(py, ["-u", "-m", "learn.hx11_cerebro", "--model", model, "--pos", pos, "--temp", String(temp)], {
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
