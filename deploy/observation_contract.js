// El cliente no recibe el estado privado del script. No sustituirlo silenciosamente
// por ceros para políticas entrenadas con esa información.
function assertObservationContract(meta, rules) {
  if (meta.layout !== "universal" || !rules.psOn) return;
  if (meta.rule_observation !== "masked") {
    const error = new Error("Este modelo usa estado privado de reglas que la sala no expone. " +
      "Adaptarlo en un run nuevo con model.rule_observation=masked y volver a exportar.");
    error.code = "OBSERVATION_CONTRACT";
    throw error;
  }
}
module.exports = { assertObservationContract };
