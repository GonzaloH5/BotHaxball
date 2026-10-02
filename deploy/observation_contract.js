// El cliente no recibe el estado privado del script. No sustituirlo silenciosamente
// por ceros para políticas entrenadas con esa información.
function assertObservationContract(meta, rules) {
  if (meta.public_signals && (meta.layout !== "universal" || meta.rule_observation !== "masked"
      || meta.public_signals.version !== 1 || meta.public_signals.offset !== 56)) {
    const error = new Error("Unsupported public RS4 observation contract");
    error.code = "OBSERVATION_CONTRACT";
    throw error;
  }
  if (meta.layout !== "universal" || !rules.psOn) return;
  if (meta.rule_observation !== "masked") {
    const error = new Error("Este modelo usa estado privado de reglas que la sala no expone. " +
      "Adaptarlo en un run nuevo con model.rule_observation=masked y volver a exportar.");
    error.code = "OBSERVATION_CONTRACT";
    throw error;
  }
}
module.exports = { assertObservationContract };
