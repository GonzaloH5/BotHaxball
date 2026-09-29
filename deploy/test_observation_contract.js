const assert = require("node:assert/strict");
const { assertObservationContract } = require("./observation_contract");
assert.throws(() => assertObservationContract({layout: "universal"}, {psOn: true}), /privado/);
assert.doesNotThrow(() => assertObservationContract({layout: "universal", rule_observation: "masked"}, {psOn: true}));
assert.doesNotThrow(() => assertObservationContract({layout: "universal"}, {psOn: false}));
assert.doesNotThrow(() => assertObservationContract({layout: "flat"}, {psOn: false}));
console.log("OK: contrato de observación y compatibilidad legacy");
