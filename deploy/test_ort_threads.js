// Las sesiones ONNX de los bots usan 1 hilo por defecto (varios bots en la misma PC compiten por la CPU).
const assert = require("assert");
const fs = require("fs");
const path = require("path");
const { ortSessionOptions } = require("./runtime");
assert.deepStrictEqual(ortSessionOptions(), { intraOpNumThreads: 1, interOpNumThreads: 1, executionMode: "sequential" });
assert.strictEqual(ortSessionOptions("2").intraOpNumThreads, 2);
assert.strictEqual(ortSessionOptions("x").intraOpNumThreads, 1);
assert.strictEqual(ortSessionOptions(0).intraOpNumThreads, 1);
const bot = fs.readFileSync(path.join(__dirname, "bot.js"), "utf8");
assert.ok(/InferenceSession\.create\(MODEL, ortSessionOptions\(/.test(bot), "bot.js debe crear la sesión con ortSessionOptions");
console.log("OK: sesiones ONNX con 1 hilo por bot");
