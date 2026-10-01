// Estado explícito por sesión. La generación invalida inferencias pendientes de otra partida.
class PolicyMemory {
  constructor(meta) {
    this.meta = meta;
    this.generation = 0;
    this.reset();
  }
  reset() {
    this.generation++;
    this.memory = new Float32Array(this.meta.memory_size || 0);
    this.previousAction = this.meta.n_actions;
  }
  feeds(ort) {
    if (!this.meta.recurrent) return {};
    return {
      memory: new ort.Tensor("float32", this.memory, [1, this.meta.memory_size]),
      previous_action: new ort.Tensor("int64", BigInt64Array.from([BigInt(this.previousAction)]), [1]),
    };
  }
  recordExecutedAction(action, generation = this.generation) {
    if (generation !== this.generation) return false;
    if (!Number.isInteger(action) || action < 0 || action >= this.meta.n_actions) {
      throw new Error("La acción ejecutada está fuera del espacio de la política");
    }
    this.previousAction = action;
    return true;
  }
  accept(output, action, generation) {
    if (generation !== this.generation) return false;
    if (this.meta.recurrent) {
      if (!output.memory_out || output.memory_out.data.length !== this.meta.memory_size) {
        throw new Error("El ONNX devolvió una memoria de tamaño incorrecto");
      }
      if (!Array.from(output.memory_out.data).every(Number.isFinite)) {
        throw new Error("El ONNX devolvió una memoria no finita");
      }
      // Validate the actual action before mutating state. Integrations which
      // override a sampled command must pass the executed command here.
      this.recordExecutedAction(action, generation);
      this.memory = Float32Array.from(output.memory_out.data);
    }
    return true;
  }
}
module.exports = { PolicyMemory };
