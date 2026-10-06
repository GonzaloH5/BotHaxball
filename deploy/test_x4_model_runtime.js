// Ruta del bot de sala con un modelo X4 exportado (export/to_onnx_x4.py), sin conectarse a una sala: metadatos del
// modelo → observación armada con lo que ve la sala (rs4z/room_state.js) en cuadros reales de Sanguchito → ONNX con las
// mismas opciones de sesión que deploy/bot.js → logits → acción con la temperatura que el bot usa por defecto.
// Correrlo con el modelo que dejó la cola antes de llevarlo a la sala:
//   node deploy/test_x4_model_runtime.js deploy/rs4z/x4_rl.onnx [logits_python.json]
// logits_python.json (opcional): {"cuadro:lugar": logits} de la misma política en torch, para compararlos.
"use strict";
const fs = require("fs");
const path = require("path");
const ort = require("onnxruntime-node");
const { RS4ZTracker, inputToAction, decodeOwnAction } = require("./rs4z/room_state");
const { OBS_DIM } = require("./rs4z/obs_v3");
const { sampleLogits, ortSessionOptions } = require("./runtime");

const fail = (msg) => { console.error(`FALLA: ${msg}`); process.exit(1); };

(async () => {
  const model = process.argv[2] || path.join(__dirname, "rs4z", "x4_rl.onnx");
  const meta = JSON.parse(fs.readFileSync(model.replace(/\.onnx$/, ".json"), "utf8"));
  if (meta.obs_version !== "x4-obs-v3") fail(`el modelo no es X4 (obs_version ${meta.obs_version})`);
  if (meta.obs_dim !== OBS_DIM) fail(`dimensión de la observación ${meta.obs_dim}, el bot arma ${OBS_DIM}`);
  if (meta.frame_skip !== 3) fail(`frame_skip ${meta.frame_skip} (el entrenamiento decide cada 3 ticks)`);
  if (meta.default_delay !== 10) fail(`retardo informado ${meta.default_delay} (el entrenamiento ve 10 la mitad de las veces)`);
  // la misma resolución que deploy/bot.js cuando no se pasa --temp
  const temperature = parseFloat(String(meta.temperature ?? 1));
  if (!(temperature > 0)) fail(`temperatura ${temperature}: el bot jugaría en greedy, no la política certificada`);
  if (!meta.maps || !meta.maps.includes("sanguchito_rs_x4")) fail(`el modelo no lista Sanguchito entre sus mapas (${meta.maps})`);
  const session = await ort.InferenceSession.create(model, ortSessionOptions(1));

  const fixture = JSON.parse(fs.readFileSync(path.join(__dirname, "rs4z", "fixture_x4_room_state.json"), "utf8"));
  const ref = process.argv[3] ? JSON.parse(fs.readFileSync(process.argv[3], "utf8")) : null;
  const trackers = new Map();
  let runs = 0, compared = 0, worst = 0, firstLogits = null;
  for (const frame of fixture.frames) {
    const order = [0, 1].flatMap((t) => frame.players.filter((p) => p.team === t).sort((a, b) => a.id - b.id));
    for (const p of order) {
      if (!trackers.has(p.id)) trackers.set(p.id, new RS4ZTracker({ obs_version: "x4-obs-v3", map: "sanguchito_rs_x4",
        ball_radii: meta.ball_radii, kick_strengths: meta.kick_strengths }));
      trackers.get(p.id).update(frame);
    }
    if (frame.frame % 3 !== 0) continue;
    for (const p of order) {
      const tr = trackers.get(p.id);
      const built = tr.build(frame, p.id, 0);       // retardo 0, como la referencia de Python del fixture
      if (built.obs.length !== OBS_DIM) fail(`observación de ${built.obs.length}`);
      if (runs < 400) {
        const out = await session.run({ obs: new ort.Tensor("float32", built.obs, [1, built.obs.length]) });
        const logits = Array.from(out.logits.data);
        if (logits.length !== 18 || !logits.every(Number.isFinite)) fail("logits inválidos");
        firstLogits = firstLogits || logits;
        const want = ref && ref[`${frame.frame}:${built.slot}`];
        if (want) {
          compared++;
          for (let j = 0; j < 18; j++) worst = Math.max(worst, Math.abs(want[j] - logits[j]));
        }
        const key = decodeOwnAction(sampleLogits(logits, temperature), built.team);
        if (![-1, 0, 1].includes(key.dirX) || ![-1, 0, 1].includes(key.dirY)) fail(`tecla inválida ${JSON.stringify(key)}`);
        runs++;
      }
      tr.pushDecision(inputToAction(p.nextInput));
    }
  }
  if (runs < 100) fail(`sólo ${runs} decisiones`);
  // con la temperatura del modelo se muestrea la distribución de la política, no la acción más probable
  const z = firstLogits.map((l) => Math.exp((l - Math.max(...firstLogits)) / temperature));
  const sum = z.reduce((a, b) => a + b, 0);
  const counts = new Array(18).fill(0);
  const N = 20000;
  for (let i = 0; i < N; i++) counts[sampleLogits(firstLogits, temperature)]++;
  const tv = 0.5 * counts.reduce((acc, c, j) => acc + Math.abs(c / N - z[j] / sum), 0);
  if (tv > 0.03) fail(`el muestreo no sigue la política (distancia de variación total ${tv.toFixed(3)})`);
  console.log(`modelo ${path.basename(model)}: mapas ${meta.maps.join(", ")}, temperatura ${temperature}, ` +
    `${runs} decisiones en cuadros reales de Sanguchito, muestreo fiel a la política (VT ${tv.toFixed(3)})`);
  if (ref) {
    if (compared < 100) fail(`sólo ${compared} decisiones con referencia de Python`);
    console.log(`logits contra Python: ${compared} decisiones, diferencia máxima ${worst.toExponential(2)}`);
    if (worst > 1e-3) fail("los logits de la sala no coinciden con los de Python");
  }
  console.log("OK");
})().catch((e) => fail(e.stack || String(e)));
