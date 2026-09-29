"""Configuración y transferencias para entrenar con CPU y CUDA."""
from __future__ import annotations

import copy
import os
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


def batch_to_device(parts, device):
    batch = {}
    for key, arrays in parts.items():
        host = torch.from_numpy(np.concatenate(arrays))
        if device.type == "cuda":
            host = host.pin_memory()
        batch[key] = host.to(device, non_blocking=device.type == "cuda")
    return batch
