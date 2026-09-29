"""python -m sim.benchmark  -> ticks por segundo del simulador."""
import time

import numpy as np

from sim.physics import BatchSim

if __name__ == "__main__":
    for n_envs, nr in [(1, 1), (256, 1), (1024, 1), (1024, 3)]:
        sim = BatchSim(n_envs, nr, nr, seed=0)
        sim.reset_random(np.arange(n_envs))
        acts = np.random.default_rng(0).integers(0, 18, (n_envs, 2 * nr))
        sim.step(acts)
        steps = max(200, 200_000 // n_envs)
        t = time.perf_counter()
        for _ in range(steps):
            sim.step(acts)
        dt = time.perf_counter() - t
        print(f"{n_envs:5d} partidos {nr}v{nr}: {n_envs * steps / dt:,.0f} ticks/s")
