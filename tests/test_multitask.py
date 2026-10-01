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


def test_shaping_follows_opponent_stage_when_configured():
    """El perfil actual liga el shaping a la dificultad del rival, no a la edad de la tarea."""
    t, d = _trainer(0)
    try:
        t.steps = int(0.6 * t.cfg["reward"]["shaping_decay_steps"])
        t.stage = 1
        t.sync_task_state()
        t.build_envs()
        for s in t.slots:
            expected = t.cfg["reward"]["shaping_by_opp_stage"][s.opp_stage]
            assert abs(t.shaping_for(s) - expected) < 1e-6, (s.task.name, t.shaping_for(s))
    finally:
        t.writer.close()
        shutil.rmtree(d, ignore_errors=True)


def test_curriculum_needs_128_matches_and_two_consecutive_point_windows():
    """Ni muchos goles ni una sola ventana promueven al rival."""
    t, d = _trainer(0)
    try:
        s = t.slots[0]
        s.goals[2][:] = [1000, 0]
        t.opponent_curriculum()
        assert s.opp_stage == 0
        s.match_results[2][:] = [110, 18, 0, 200, 10, 3]
        t.opponent_curriculum()
        assert s.opp_stage == 0 and s.promotion_streak == 1
        # Sin partidos nuevos no cuenta como una segunda evaluación.
        t.opponent_curriculum()
        assert s.opp_stage == 0 and s.promotion_streak == 1
        s.match_results[2][:] = [110, 18, 0, 190, 11, 4]
        t.opponent_curriculum()
        assert s.opp_stage == 1
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
