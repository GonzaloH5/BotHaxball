// Paridad del actor RS4-Z en Node (onnxruntime-node) contra PyTorch.
//   python -m export.to_onnx_rs4z <ckpt> --out <dir>/model && node deploy/test_onnx_rs4z.js <dir>
"use strict";
const fs = require("fs");
const path = require("path");
const ort = require("onnxruntime-node");
const { OBS_DIM } = require("./rs4z/obs_v2");

(async () => {
  const dir = process.argv[2] || path.join(__dirname, "rs4z");
  const meta = JSON.parse(fs.readFileSync(path.join(dir, "model.json"), "utf8"));
  if (meta.obs_dim !== OBS_DIM || meta.obs_version !== "rs4z-obs-v2") throw new Error("contrato de observación distinto");
  const ref = JSON.parse(fs.readFileSync(path.join(dir, "model_parity.json"), "utf8"));
  const sess = await ort.InferenceSession.create(path.join(dir, "model.onnx"));
  const n = ref.obs.length;
  const x = new Float32Array(n * OBS_DIM);
  ref.obs.forEach((row, i) => x.set(row, i * OBS_DIM));
  const out = await sess.run({ obs: new ort.Tensor("float32", x, [n, OBS_DIM]) });
  const logits = out.logits.data;
  let worst = 0;
  ref.logits.forEach((row, i) => row.forEach((v, j) => { worst = Math.max(worst, Math.abs(v - logits[i * 18 + j])); }));
  console.log(`ONNX RS4-Z en Node: ${n} casos, diferencia máxima ${worst.toExponential(2)}`);
  if (worst > 1e-4) { console.error("FALLA"); process.exit(1); }
  console.log("OK");
})();
