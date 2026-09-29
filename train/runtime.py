"""Configuración y transferencias para entrenar con CPU y CUDA."""
from __future__ import annotations

import copy
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path, seen=None):
    """Un YAML puede extender otro sin duplicar recompensas/currículo."""
    path = Path(path).resolve()
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError(f"Herencia circular de configuración: {path}")
    seen.add(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    base = cfg.pop("extends", None)
    if base is None:
        return cfg
    parent = load_config(path.parent / base, seen)

    def merge(a, b):
        out = copy.deepcopy(a)
        for key, value in b.items():
            out[key] = merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
        return out

    return merge(parent, cfg)


def cpu_budget():
    """Respeta afinidad y cuotas de CPU de contenedores Linux."""
    count = os.cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        count = min(count, len(os.sched_getaffinity(0)))
    for filename, v2 in (("/sys/fs/cgroup/cpu.max", True),
                         ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", False)):
        try:
            raw = Path(filename).read_text().split()
            if v2:
                if raw[0] == "max":
                    continue
                quota, period = map(int, raw)
            else:
                quota = int(raw[0])
                period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
            if quota > 0:
                count = min(count, max(1, quota // period))
        except (OSError, ValueError, IndexError, ZeroDivisionError):
            pass
    return max(1, count)


class CudaRolloutTransfer:
    """Buffers pinned reutilizables: una subida de obs y una bajada de decisiones por tick.

    La CPU puede calcular bots tras enqueue_output(), antes de wait_output().
    No se permite reutilizar los buffers mientras sus copias estén pendientes.
    """

    def __init__(self, device, reuse_device=True):
        self.device = device
        self.reuse_device = reuse_device
        self.obs = self.output = None
        self.ready = torch.cuda.Event()
        self.device_obs = None

    def upload(self, flat):
        return self.upload_many([flat])

    def upload_many(self, arrays):
        """Copia cada tarea directamente al buffer pinned, sin concatenate intermedio."""
        shape = (sum(a.shape[0] for a in arrays), arrays[0].shape[1])
        if self.obs is None or tuple(self.obs.shape) != shape:
            self.obs = torch.empty(shape, dtype=torch.float32, pin_memory=True)
            if self.reuse_device:
                self.device_obs = torch.empty(shape, dtype=torch.float32, device=self.device)
        host, offset = self.obs.numpy(), 0
        for array in arrays:
            np.copyto(host[offset:offset + len(array)], array)
            offset += len(array)
        if not self.reuse_device:
            return self.obs.to(self.device, non_blocking=True)
        self.device_obs.copy_(self.obs, non_blocking=True)
        return self.device_obs

    def enqueue_output(self, actions, logp, values):
        packed = torch.stack((actions.float(), logp, values))
        if self.output is None or self.output.shape != packed.shape:
            self.output = torch.empty(packed.shape, dtype=packed.dtype, pin_memory=True)
        self.output.copy_(packed, non_blocking=True)
        self.record_ready()

    def record_ready(self):
        self.ready.record(torch.cuda.current_stream(self.device))

    def wait_output(self):
        self.ready.synchronize()
        actions, logp, values = self.output.numpy()
        # Los buffers viven hasta el siguiente act; el rollout los copia a sus buffers antes.
        return actions.astype(np.int64), logp, values


class _BatchStorage:
    def __init__(self, rows, trailing_shape, dtype, device):
        self.capacity = 1 << max(0, (max(1, rows) - 1).bit_length())
        shape = (self.capacity, *trailing_shape)
        torch_dtype = torch.from_numpy(np.empty(0, dtype=dtype)).dtype
        self.host = torch.empty(shape, dtype=torch_dtype, pin_memory=device.type == "cuda")
        self.device = torch.empty(shape, dtype=torch_dtype, device=device)
        self.array = self.host.numpy()


class PpoBatchTransfer:
    """Un par host/device persistente por campo del lote PPO, con capacidad creciente.

    CUDA usa host pinned y H2D asíncrono; CPU existe para probar empaquetado/equivalencia.
    Las salidas son vistas, válidas hasta el próximo upload. Tras encolar sus consumidores,
    llamar a mark_consumed en ese stream antes de reutilizar, también si se cambia de stream.
    No cambiar el host mientras H2D está pendiente. Nunca serializar estos buffers en el checkpoint.
    """

    def __init__(self, device):
        self.device = torch.device(device)
        self.buffers = {}
        self.ready = torch.cuda.Event() if self.device.type == "cuda" else None
        self.pending = False
        self.allocations = 0
        self.last_timings = {}
        self.last_allocations = 0

    def mark_consumed(self):
        if self.ready is not None:
            self.ready.record(torch.cuda.current_stream(self.device))
            self.pending = True

    @staticmethod
    def _layout(arrays):
        if not arrays or any(a.ndim < 1 for a in arrays):
            raise ValueError("Cada campo PPO requiere una lista de arrays con eje de muestras")
        trailing_shape = arrays[0].shape[1:]
        if any(a.shape[1:] != trailing_shape for a in arrays):
            raise ValueError("Formas incompatibles al empaquetar un campo PPO")
        return (sum(len(a) for a in arrays), trailing_shape,
                np.result_type(*(a.dtype for a in arrays)))

    def upload(self, parts):
        started = time.perf_counter()
        if self.pending and not self.ready.query():
            self.ready.synchronize()  # normalmente ya terminó: update descarga métricas antes de retornar
        waited = time.perf_counter() - started
        layouts = {key: self._layout(arrays) for key, arrays in parts.items()}
        if len({layout[0] for layout in layouts.values()}) > 1:
            raise ValueError("Todos los campos PPO deben tener el mismo número de muestras")
        batch, allocated, packed, enqueued, reallocations = {}, 0.0, 0.0, 0.0, 0
        for key, arrays in parts.items():
            rows, trailing_shape, dtype = layouts[key]
            start = time.perf_counter()
            storage = self.buffers.get(key)
            if (storage is None or storage.capacity < rows or storage.array.shape[1:] != trailing_shape
                    or storage.array.dtype != dtype):
                storage = _BatchStorage(rows, trailing_shape, dtype, self.device)
                self.buffers[key] = storage
                self.allocations += 1
                reallocations += 1
            allocated += time.perf_counter() - start
            start, offset = time.perf_counter(), 0
            for array in arrays:
                np.copyto(storage.array[offset:offset + len(array)], array)
                offset += len(array)
            packed += time.perf_counter() - start
            start = time.perf_counter()
            output = storage.device[:rows]
            output.copy_(storage.host[:rows], non_blocking=self.device.type == "cuda")
            batch[key] = output
            enqueued += time.perf_counter() - start
        self.mark_consumed()  # protege el H2D incluso si aún no se llamó al update
        self.last_allocations = reallocations
        self.last_timings = dict(wait=waited, allocate=allocated, pack=packed, enqueue=enqueued)
        return batch


def batch_to_device(parts, device, transfer=None):
    if transfer is not None:
        if transfer.device != device:
            raise ValueError("El cache PPO y el lote deben estar en el mismo dispositivo")
        return transfer.upload(parts)
    batch = {}
    for key, arrays in parts.items():
        host = torch.from_numpy(np.concatenate(arrays))
        if device.type == "cuda":
            host = host.pin_memory()
        batch[key] = host.to(device, non_blocking=device.type == "cuda")
    return batch
