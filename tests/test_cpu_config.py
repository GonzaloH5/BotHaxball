"""El perfil CPU sólo cambia ejecución, no parámetros del aprendizaje."""
import copy
from pathlib import Path

from train.runtime import load_config


def test_cpu_profile_preserves_training_contract():
    root = Path(__file__).resolve().parents[1]
    original = load_config(root / "train/config_runpod.yaml")
    expected = copy.deepcopy(original)
    expected["ppo"].update(device="cpu", torch_threads=8, numba_threads=4)
    expected["runtime"].update(cuda_decisions="legacy", reuse_ppo_batch=False, cache_bc_logits_cpu=True,
                               optimize_cpu=True)
    assert load_config(root / "train/config_cpu.yaml") == expected
