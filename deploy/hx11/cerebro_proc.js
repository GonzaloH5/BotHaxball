// Proceso del cerebro (learn/hx11_cerebro.py): un JSON por línea en cada sentido.
"use strict";

const { spawn } = require("child_process");
const path = require("path");
const readline = require("readline");

const REPO = path.join(__dirname, "..", "..");

function startCerebro({ model, pos, temp = 0.7, hyst = 0, every = 3, ens = 0, ensM = 0, renovarPatada = 1,
                        renovar = 1, sinSaques = "", reglaSaques = false, reglaJuego = false, distanciaSaques = false,
                        anclaVivo = false, destrabar = false, correa = false, python, onDecision, onReady,
                        onInfo }) {
  const py = python || path.join(REPO, ".venv", process.platform === "win32" ? "Scripts\\python.exe" : "bin/python");
  const argv = ["-u", "-m", "learn.hx11_cerebro", "--model", model, "--temp", String(temp)];
  if (pos) argv.push("--pos", pos);
  if (hyst > 0) argv.push("--hyst", String(hyst));
  if (every !== 3) argv.push("--every", String(every));
  if (ens > 1) argv.push("--ens", String(ens), "--ens-m", String(ensM));   // acciones en bloque (ACT)
  if (renovarPatada < 1) argv.push("--renovar-patada", String(renovarPatada));   // tecla de patear persistente
  if (renovar < 1) argv.push("--renovar", String(renovar));   // dirección persistente (menos cambios de tecla)
  if (sinSaques) argv.push("--sin-saques", sinSaques);   // pelotas paradas propias que el bot deja a los humanos
  if (reglaJuego) argv.push("--regla-juego");
  if (distanciaSaques) argv.push("--distancia-saques");
  if (anclaVivo) argv.push("--ancla-vivo");
  if (destrabar) argv.push("--destrabar");
  if (correa) argv.push("--correa");                        // fuera de la jugada, cerca de su zona (p95 humano)                  // quieto lejos de su zona → a su zona                // ancla del rol libre del equipo real   // no se acerca en saques que saca otro            // juego abierto: no afuera por la banda ni atrás sin compañero
  if (reglaSaques) argv.push("--regla-saques");          // en las que saca: desde atrás y con la barra alta   // sin puesto: modo de varios bots (cada uno se registra)
  const proc = spawn(py, argv, {
    cwd: REPO, env: { ...process.env, PYTHONIOENCODING: "utf-8" }, stdio: ["pipe", "pipe", "pipe"], windowsHide: true,
  });
  readline.createInterface({ input: proc.stdout }).on("line", (line) => {
    let msg;
    try { msg = JSON.parse(line); } catch { return; }
    if (msg.listo) onReady && onReady(msg);
    else if (msg.input != null) onDecision && onDecision(msg);
    else if (msg.ancla) onInfo && onInfo(msg);
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
