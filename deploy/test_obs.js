// node test_obs.js  -> verifica obs.js y el modelo ONNX contra fixture.json (generado en Python)
const fs = require("fs");
const path = require("path");
const ort = require("onnxruntime-node");
const { buildObs } = require("./obs");

(async () => {
  // carpeta con model.json / model.onnx / fixture.json (por defecto, la de este script)
  const dir = process.argv[2] ? path.resolve(process.argv[2]) : __dirname;
  const meta = JSON.parse(fs.readFileSync(path.join(dir, "model.json"), "utf8"));
  const fx = JSON.parse(fs.readFileSync(path.join(dir, "fixture.json"), "utf8"));
  const sess = await ort.InferenceSession.create(path.join(dir, "model.onnx"));
  let maxObs = 0, maxLog = 0, maxMemory = 0;
  if (meta.layout === "universal") {
    // obs universal: varias tareas (mapas y formatos), cada una con la geometría de su estadio
    const { buildObsUniversal } = require("./obs_universal");
    const perTask = {};
    for (const f of fx.states) {
      const geom = fx.geoms[f.stadium];
      for (let p = 0; p < f.players.length; p++) {
        const o = buildObsUniversal(f, p, geom, f.opts);
        if (o.length !== f.obs[p].length) throw new Error(`${f.task}: obs ${o.length} != ${f.obs[p].length}`);
        let e = 0;
        o.forEach((v, i) => { e = Math.max(e, Math.abs(v - f.obs[p][i])); });
        maxObs = Math.max(maxObs, e);
        perTask[f.task] = Math.max(perTask[f.task] || 0, e);
        const feeds = { obs: new ort.Tensor("float32", Float32Array.from(o), [1, o.length]) };
        if (meta.recurrent) {
          feeds.memory = new ort.Tensor("float32", Float32Array.from(f.memory[p]), [1, meta.memory_size]);
          feeds.previous_action = new ort.Tensor("int64", BigInt64Array.from([BigInt(f.previous_action[p])]), [1]);
        }
        const out = await sess.run(feeds);
        out.logits.data.forEach((v, i) => { maxLog = Math.max(maxLog, Math.abs(v - f.logits[p][i])); });
        if (meta.recurrent) out.memory_out.data.forEach((v, i) => {
          maxMemory = Math.max(maxMemory, Math.abs(v - f.memory_out[p][i]));
        });
      }
    }
    for (const [t, e] of Object.entries(perTask)) console.log(`  ${t}: error máx obs ${e.toExponential(2)}`);
    console.log(`${fx.states.length} estados | error máx obs ${maxObs.toExponential(2)} | error máx logits ${maxLog.toExponential(2)}`);
    if (meta.recurrent) console.log(`error máx memoria ${maxMemory.toExponential(2)}`);
    if (maxObs > 1e-4 || maxLog > 1e-3 || maxMemory > 1e-4) { console.error("FALLO: JS no coincide con Python"); process.exit(1); }
    console.log("OK: la observación universal y el modelo en JS coinciden con Python");
    return;
  }
  for (const f of fx) {
    for (let p = 0; p < f.players.length; p++) {
      const o = buildObs(f, p, meta);
      if (o.length !== meta.obs_dim) throw new Error(`obs_dim ${o.length} != ${meta.obs_dim}`);
      o.forEach((v, i) => { maxObs = Math.max(maxObs, Math.abs(v - f.obs[p][i])); });
      const out = await sess.run({ obs: new ort.Tensor("float32", Float32Array.from(o), [1, o.length]) });
      out.logits.data.forEach((v, i) => { maxLog = Math.max(maxLog, Math.abs(v - f.logits[p][i])); });
    }
  }
  console.log(`${fx.length} estados | error máx obs ${maxObs.toExponential(2)} | error máx logits ${maxLog.toExponential(2)}`);
  if (maxObs > 1e-4 || maxLog > 1e-3) { console.error("FALLO: JS no coincide con Python"); process.exit(1); }
  console.log("OK: la observación y el modelo en JS coinciden con Python");
})();
