"""Inferencia determinista capturable; muestreo y PPO quedan fuera del CUDA Graph.

Un graph vive sólo durante un rollout. No retenerlo después de actualizar pesos,
normalizadores o asignaciones de rivales (RunningNorm reemplaza algunos buffers).
"""
from __future__ import annotations

import time
import warnings

import torch


DECISION_BACKENDS = ("legacy", "eager", "auto", "graph")


def sample_decisions(logits, values):
    """Una categórica por fila, con logits de la política que realmente controla esa fila."""
    dist = torch.distributions.Categorical(logits=logits, validate_args=False)
    actions = dist.sample()
    return actions, dist.log_prob(actions), values


class CudaDecisionGraph:
    """Captura sólo forward/gather/scatter; nunca RNG, física, normalización ni Adam."""

    def __init__(self, device, mode="auto", reuse_capture_pool=True):
        if mode not in ("auto", "graph"):
            raise ValueError(f"modo de CUDA Graph desconocido: {mode}")
        self.device, self.mode = torch.device(device), mode
        self.graph = self.input = self.outputs = self.stream = None
        self.reuse_capture_pool = reuse_capture_pool
        # El graph anterior mantiene vivo su pool hasta la siguiente captura.
        # Nunca se vuelve a ejecutar: pesos/buffers/rivales se recapturan siempre.
        self._pool_owner = None
        self.disabled_reason = None
        self.capture_seconds = 0.0
        self.captures = self.replays = 0

    def reset(self):
        """Llamar con el último act ya sincronizado, antes de tocar normalizadores/pesos."""
        if self.reuse_capture_pool and self.graph is not None:
            self._pool_owner = self.graph
        self.graph = self.input = self.outputs = None
        if not self.reuse_capture_pool:
            self.stream = None
        self.capture_seconds = 0.0
        self.captures = self.replays = 0
        # Una incompatibilidad en auto se mantiene desactivada; no repetir captura/aviso 128 veces.

    @staticmethod
    def _unsupported(error):
        message = str(error).lower()
        if any(word in message for word in ("out of memory", "illegal memory", "device-side assert")):
            return False
        return any(phrase in message for phrase in (
            "not permitted when stream is capturing", "not permitted when stream is recording",
            "not supported during cuda graph capture", "cuda graph capture is not supported",
            "operation failed due to a previous error during capture", "cuda error: stream capture",
        ))

    @torch.no_grad()
    def _capture(self, forward, flat):
        if flat.device.type != "cuda":
            raise ValueError("CUDA Graph requiere un tensor CUDA")
        with torch.cuda.device(self.device):
            current = torch.cuda.current_stream(self.device)
            stream = self.stream if self.stream is not None else torch.cuda.Stream(device=self.device)
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                for _ in range(3):
                    forward(flat)
            current.wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            # torch.cuda.graph sincroniza antes de capturar. El H2D inicial ya debe estar completo.
            pool = self._pool_owner.pool() if self._pool_owner is not None else None
            with torch.cuda.graph(graph, pool=pool, stream=stream):
                outputs = forward(flat)
            current.wait_stream(stream)
        # Mantener vivos input/output/stream y los tensores de los modelos durante todo el rollout.
        self.graph, self.input, self.outputs, self.stream = graph, flat, outputs, stream
        self._pool_owner = None

    @torch.no_grad()
    def run(self, forward, flat):
        if self.disabled_reason is not None:
            return forward(flat)
        if self.graph is None:
            start = time.perf_counter()
            try:
                self._capture(forward, flat)
            except RuntimeError as error:
                if self.mode != "auto" or not self._unsupported(error):
                    raise
                self.disabled_reason = str(error)
                self._pool_owner = self.stream = None
                warnings.warn(f"CUDA Graph no compatible; usando decisiones eager: {error}", RuntimeWarning)
                return forward(flat)
            finally:
                self.capture_seconds += time.perf_counter() - start
            self.captures += 1
        if (flat.shape != self.input.shape or flat.stride() != self.input.stride()
                or flat.dtype != self.input.dtype or flat.device != self.input.device):
            raise ValueError("Cambió el layout del graph: resetear antes del siguiente rollout")
        if flat.data_ptr() != self.input.data_ptr():
            self.input.copy_(flat)
        self.graph.replay()
        self.replays += 1
        return self.outputs


class CudaDecisionProfile:
    """Eventos por fase; se leen tras la sincronización que act ya necesita.

    No agrega synchronize. Los intervalos GPU pueden incluir huecos de lanzamiento CPU,
    por lo que no prueban saturación de la GPU y no se suman a los tiempos host.
    """

    def __init__(self, device):
        self.device = device
        self.events = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
        self.totals, self.calls = {}, 0
        self.capture_seconds, self.captures, self.replays = 0.0, 0, 0

    def reset(self):
        self.totals.clear()
        self.calls = 0
        self.capture_seconds, self.captures, self.replays = 0.0, 0, 0

    def mark(self, index):
        self.events[index].record(torch.cuda.current_stream(self.device))

    def finish(self, host_submit, host_wait):
        names = ("H2D", "inferencia", "muestreo", "D2H")
        for i, name in enumerate(names):
            seconds = self.events[i].elapsed_time(self.events[i + 1]) / 1000
            self.totals[name] = self.totals.get(name, 0.0) + seconds
        for name, seconds in (("CPU envío", host_submit), ("CPU espera final", host_wait)):
            self.totals[name] = self.totals.get(name, 0.0) + seconds
        self.calls += 1

    def report(self, iterations):
        print("\nDecisiones CUDA: intervalos del stream y tiempos CPU (NO sumar entre sí):")
        for name, total in self.totals.items():
            print(f"  {name}: {total / iterations:.3f} s/iter")
        print(f"  preparación/captura graph (incluida en decisiones): {self.capture_seconds / iterations:.3f} s/iter")
        print(f"  capturas: {self.captures / iterations:.1f}/iter | replays: {self.replays / iterations:.0f}/iter")
