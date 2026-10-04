"""Controladores para `eval/rs4z/runner.py`: RS-Pro con nivel/estilo por equipo."""
from __future__ import annotations

import numpy as np

from .policy import STYLE_BALANCED, RSPro


class RSProController:
    """RS-Pro para los jugadores que se le asignen; nivel y estilo por equipo (o por partido)."""

    def __init__(self, level=5, style=None, seed=0, levels=None, styles=None):
        self.level = level
        self.style = STYLE_BALANCED if style is None else np.asarray(style, dtype=np.float64)
        self.levels = levels      # opcional: (N, 2) niveles por partido y equipo
        self.styles = styles      # opcional: (N, 2, 6)
        self.seed = seed
        self.bot = None

    def _setup(self, env):
        self.bot = RSPro(env, seed=self.seed)
        rows = np.arange(env.N)
        for t in (0, 1):
            if self.levels is None:
                self.bot.configure(rows, t, self.level, self.style)
            else:
                for n in rows:
                    sty = self.style if self.styles is None else self.styles[n, t]
                    self.bot.configure([n], t, int(self.levels[n, t]), sty)

    def sync(self, env, rows):
        if self.bot is None or self.bot.N != env.N:
            self._setup(env)
        self.bot.sync(env, rows)

    def push(self, env):
        self.bot.push(env)

    def act(self, env, ctrl, out):
        return self.bot.act(env, ctrl, out)
