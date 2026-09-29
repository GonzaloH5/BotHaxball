"""Reparto de muestras entre tareas en train/multitask.py (repaso y piso mínimo)."""
import shutil
from pathlib import Path

import yaml

from train.multitask import ROOT, MultiTrainer


def _trainer(stage):
    cfg = yaml.safe_load((ROOT / "train" / "config_multi.yaml").read_text(encoding="utf-8"))
    cfg["env"]["agents"] = 384
    run = "_pytest_multitask"
    t = MultiTrainer(cfg, run, resume=False)
    t.stage = stage
    t.build_envs()
    return t, ROOT / "runs" / run


def test_shaping_decays_per_task():
    """Una tarea que entra en la etapa B arranca con shaping completo aunque las de la etapa A ya lo hayan perdido."""
    t, d = _trainer(0)
    try:
        decay = t.cfg["reward"]["shaping_decay_steps"]
        t.steps = int(0.6 * decay)
        t.stage = 1
        t.sync_task_state()
        t.build_envs()
        old = {n for n in t.stages[0]["tasks"]}
        for s in t.slots:
            expected = 0.4 if s.task.name in old else 1.0
            assert abs(t.shaping_for(s) - expected) < 1e-6, (s.task.name, t.shaping_for(s))
    finally:
        t.writer.close()
        shutil.rmtree(d, ignore_errors=True)


def test_low_scoring_task_can_advance():
    """Una tarea con pocos goles por iteración (p. ej. 6 por iteración) igual junta la ventana y avanza."""
    t, d = _trainer(0)
    try:
        s = t.slots[0]
        for _ in range(60):          # 60 iteraciones x 6 goles = 360 goles, 90% a favor
            s.goals[0][:] = [0, 0]
            s.goals[2][:] = [5, 1] if _ % 2 else [6, 0]
            t.opponent_curriculum()
            if s.opp_stage:
                break
        assert s.opp_stage == 1
        wr, n = s.winrate()
    finally:
        t.writer.close()
        shutil.rmtree(d, ignore_errors=True)


def test_shares_rehearsal_and_samples():
    t, d = _trainer(1)
    try:
        sh = t.task_shares()
        assert abs(sum(sh.values()) - 1.0) < 1e-9
        new = t.stages[1]["tasks"]
        old = t.stages[0]["tasks"]
        # las tareas de la etapa anterior siguen activas (repaso) con menos peso que las nuevas
        assert all(n in sh for n in old + new)
        assert max(sh[n] for n in old) < min(sh[n] for n in new)
        assert min(sh.values()) >= t.cfg["schedule"]["min_share"] - 1e-9
        # las muestras por paso (partidos x jugadores) respetan las fracciones
        samples = {s.task.name: s.N * s.P for s in t.slots}
        tot = sum(samples.values())
        for n, s in samples.items():
            assert abs(s / tot - sh[n]) < 0.05, (n, s / tot, sh[n])
        # todas las tareas comparten el ancho de obs de la etapa
        assert len({s.env.obs_dim for s in t.slots}) == 1
    finally:
        t.writer.close()
        shutil.rmtree(d, ignore_errors=True)
