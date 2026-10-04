"""Motor de ejercicios y partidos del curriculum RS4-Z.

Cada tarea fija: plantel, qué lugares controla el aprendiz (el resto, RS-Pro), cómo se coloca el
estado (siempre a través del árbitro, F9), cómo termina y qué cuenta como éxito. Los objetivos se
expresan con elementos que el agente ve (pelota, arcos, compañeros, rivales): no hay marcadores
invisibles. El equipo aprendiz juega de rojo o azul al azar (colocación espejada).

Resultado de un episodio de ejercicio (`outcome`): +1 éxito, -1 fracaso grave (gol recibido en
ejercicios de defensa), 0 fracaso. El tiempo agotado es éxito, fracaso o truncación según la tarea.
Los partidos no terminan por eventos: el fin de partido es el terminal.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import contract as C
from . import kernel as K

GOAL_X = 1162.0
REGAIN_TICKS = 60          # posesión recuperada: toque propio sin toque rival durante 1 s
LOSS_TICKS = 30            # posesión perdida: toque rival sin respuesta propia durante 0,5 s
CLEAN_LOSS_PX = 60.0      # en ataque: el rival toca la pelota sin ningún aprendiz a menos de esto → la ganó limpio
MATCH_STALL_FACTOR = 2     # un partido que dura más de 2× su reloj + 1 min real se corta (truncación)
MATCH_STALL_EXTRA = 3600


@dataclass(frozen=True)
class Task:
    name: str
    stage: str
    n_own: int                     # jugadores del equipo aprendiz
    n_opp: int                     # rivales (RS-Pro)
    start: str                     # colocador (ver PLACERS)
    timeout: int = 0               # ticks; 0 = partido (termina con el reloj)
    timeout_outcome: str = "fail"  # fail | success | truncate
    success: str = "goal"          # goal | touch | regain
    end_on_rival_touch: bool = False   # termina cuando el rival CONTROLA la pelota (no ante un roce)
    end_on_own_out: bool = True    # salida por el equipo aprendiz = fracaso
    concede_outcome: float = 0.0   # resultado si el rival anota (defensa: -1)
    opp_levels: tuple = (0, 5)     # rango de nivel de RS-Pro (la dificultad lo recorre)
    match_ticks: int = 0           # duración de reloj (sólo partidos)
    scripted_mates: int = 0        # compañeros del aprendiz controlados por RS-Pro
    require_pass: bool = False     # el gol cuenta sólo después de un pase entre aprendices (la tarea ES un pase)
    notes: str = ""


TASKS = {t.name: t for t in [
    # ------------------------------------------------------------- S1 control (1v0, arco vacío)
    Task("touch", "S1", 1, 0, "touch", timeout=360, success="touch", notes="llegar a la pelota (quieta o rodando)"),
    Task("empty_goal", "S1", 1, 0, "empty_goal", timeout=540, notes="definir a arco vacío desde el campo rival"),
    Task("carry_goal", "S1", 1, 0, "carry_goal", timeout=1080, notes="conducir desde campo propio y definir"),
    Task("receive_shot", "S1", 1, 0, "receive_shot", timeout=540, notes="recibir una pelota en movimiento y definir"),
    # ------------------------------------------------------------- S2 pelota y 1v1
    Task("shot_vs_last", "S2", 1, 1, "shot_vs_last", timeout=540, end_on_rival_touch=True, opp_levels=(0, 4),
         notes="definir contra el último hombre"),
    Task("attack_1v1", "S2", 1, 1, "attack_1v1", timeout=900, end_on_rival_touch=True, opp_levels=(0, 4),
         notes="1v1 desde mitad de cancha"),
    Task("defend_1v1", "S2", 1, 1, "defend_1v1", timeout=720, timeout_outcome="success", success="regain",
         concede_outcome=-1.0, opp_levels=(0, 4), notes="defender un 1v1: recuperar o aguantar sin gol"),
    Task("loose_ball", "S2", 1, 1, "loose_ball", timeout=360, success="regain", end_on_rival_touch=True,
         opp_levels=(0, 4), notes="ganar una pelota dividida"),
    Task("match_1v1", "S2", 1, 1, "kickoff", match_ticks=3600 * 2, opp_levels=(0, 3), notes="1v1 a cancha completa"),
    # ------------------------------------------------------------- S3 tiro, pase y rebotes (2 aprendices)
    Task("attack_2v1", "S3", 2, 1, "attack_2v1", timeout=780, end_on_rival_touch=True, opp_levels=(1, 5),
         notes="2v1: el pase es la solución eficiente"),
    Task("one_two", "S3", 2, 1, "one_two", timeout=660, end_on_rival_touch=True, opp_levels=(1, 5),
         require_pass=True, notes="pared contra un presionante con espacio a la espalda"),
    Task("through_ball", "S3", 2, 2, "through_ball", timeout=720, end_on_rival_touch=True, opp_levels=(1, 5),
         require_pass=True, notes="pase en profundidad a un compañero que corre"),
    Task("redirect", "S3", 2, 1, "redirect", timeout=540, end_on_rival_touch=True, opp_levels=(1, 5),
         notes="centro rodando al área: desvío o control y definición"),
    Task("attack_2v2", "S3", 2, 2, "attack_2v2", timeout=900, end_on_rival_touch=True, opp_levels=(1, 5),
         notes="2v2 en campo rival"),
    # ------------------------------------------------------------- S4 defensa y posicionamiento
    Task("defend_2v2", "S4", 2, 2, "defend_2v2", timeout=780, timeout_outcome="success", success="regain",
         concede_outcome=-1.0, opp_levels=(2, 5), notes="defender un ataque 2v2"),
    Task("defend_4v4", "S4", 4, 4, "defend_4v4", timeout=900, timeout_outcome="success", success="regain",
         concede_outcome=-1.0, opp_levels=(3, 5), notes="defender un ataque organizado 4v4"),
    Task("defend_restart", "S4", 4, 4, "defend_restart", timeout=720, timeout_outcome="success", success="regain",
         concede_outcome=-1.0, opp_levels=(3, 5), notes="córner/lateral/saque rival en campo propio"),
    Task("match_2v2", "S4", 2, 2, "kickoff", match_ticks=3600 * 3, opp_levels=(1, 4), notes="2v2 a cancha completa"),
    # ------------------------------------------------------------- S5 cooperación 4v4 y situaciones
    Task("match_4v4", "S5", 4, 4, "kickoff", match_ticks=3600 * 3, opp_levels=(1, 5)),
    Task("match_4v3", "S5", 4, 3, "kickoff", match_ticks=3600 * 3, opp_levels=(2, 5), notes="superioridad"),
    Task("match_4v2", "S5", 4, 2, "kickoff", match_ticks=3600 * 2, opp_levels=(3, 5), notes="superioridad amplia"),
    Task("match_3v4", "S5", 3, 4, "kickoff", match_ticks=3600 * 3, opp_levels=(0, 4), notes="inferioridad"),
    Task("situation_open", "S5", 4, 4, "recorded_open", timeout=720, timeout_outcome="truncate", success="goal",
         end_on_own_out=False, concede_outcome=-1.0, opp_levels=(2, 5), notes="estado humano de juego abierto"),
    Task("situation_attack", "S5", 4, 4, "recorded_attack", timeout=720, timeout_outcome="fail", success="goal",
         end_on_own_out=False, concede_outcome=-1.0, opp_levels=(2, 5), notes="estado humano de ataque"),
    Task("restart_attack", "S5", 4, 4, "restart_attack", timeout=720, timeout_outcome="fail", success="goal",
         end_on_own_out=False, concede_outcome=-1.0, opp_levels=(2, 5), notes="saque propio en campo rival"),
    Task("transition", "S5", 4, 4, "transition", timeout=600, timeout_outcome="truncate", success="goal",
         end_on_own_out=False, concede_outcome=-1.0, opp_levels=(2, 5), notes="pelota recién recuperada"),
    # ------------------------------------------------------------- S6 liga (rivales del pool, no RS-Pro sólo)
    Task("league_4v4", "S6", 4, 4, "kickoff", match_ticks=3600 * 3, opp_levels=(3, 5)),
]}


def team_slots(team, n):
    """Lugares de un equipo (0..3 rojo, 4..7 azul) para n jugadores."""
    base = 0 if team == 0 else 4
    return np.arange(base, base + n)


@dataclass
class DrillState:
    n: int
    task: np.ndarray = None          # índice de tarea por fila
    learner_team: np.ndarray = None
    start_tick: np.ndarray = None
    ticks: np.ndarray = None
    regain_since: np.ndarray = None  # ticks desde el último toque propio sin toque rival (-1 inactivo)
    lost_since: np.ndarray = None    # ticks desde el último toque rival sin toque propio (-1 inactivo)
    last_touch: np.ndarray = None    # último jugador propio que tocó la pelota (-1 ninguno, -2 la tocó el rival)
    passed: np.ndarray = None        # hubo un pase entre aprendices en el episodio
    difficulty: np.ndarray = None
    opp_level: np.ndarray = None

    def __post_init__(self):
        n = self.n
        self.task = np.zeros(n, dtype=np.int64)
        self.learner_team = np.zeros(n, dtype=np.int64)
        self.start_tick = np.zeros(n, dtype=np.int64)
        self.ticks = np.zeros(n, dtype=np.int64)
        self.regain_since = np.full(n, -1, dtype=np.int64)
        self.lost_since = np.full(n, -1, dtype=np.int64)
        self.last_touch = np.full(n, -1, dtype=np.int64)
        self.passed = np.zeros(n, dtype=bool)
        self.difficulty = np.zeros(n)
        self.opp_level = np.zeros(n, dtype=np.int64)


TASK_NAMES = list(TASKS)
TASK_INDEX = {name: i for i, name in enumerate(TASK_NAMES)}
_TASK_PARAMS = dict(
    timeout=np.array([TASKS[t].timeout for t in TASK_NAMES], dtype=np.int64),
    success_goal=np.array([TASKS[t].success == "goal" for t in TASK_NAMES]),
    require_pass=np.array([TASKS[t].require_pass for t in TASK_NAMES]),
    success_touch=np.array([TASKS[t].success == "touch" for t in TASK_NAMES]),
    success_regain=np.array([TASKS[t].success == "regain" for t in TASK_NAMES]),
    rival_end=np.array([TASKS[t].end_on_rival_touch for t in TASK_NAMES]),
    own_out_end=np.array([TASKS[t].end_on_own_out for t in TASK_NAMES]),
    concede=np.array([TASKS[t].concede_outcome for t in TASK_NAMES], dtype=np.float64),
    timeout_code=np.array([{"fail": 0, "success": 1, "truncate": 2}[TASKS[t].timeout_outcome] for t in TASK_NAMES]),
)


class Drills:
    """Asigna tareas a filas de un `RS4ZEnv`, coloca estados y detecta finales."""

    def __init__(self, env, rng, state_bank=None):
        self.env = env
        self.rng = rng
        self.bank = state_bank
        self.st = DrillState(env.N)

    # ------------------------------------------------------------------ inicio
    def start(self, rows, task_names, difficulty, learner_team=None, match_ticks=None):
        """match_ticks: duración de reloj de los partidos (por fila); por defecto, la de la tarea."""
        env, rng = self.env, self.rng
        rows = np.atleast_1d(np.asarray(rows, dtype=np.int64))
        if np.isscalar(task_names) or isinstance(task_names, str):
            task_names = [task_names] * len(rows)
        difficulty = np.broadcast_to(np.asarray(difficulty, dtype=np.float64), rows.shape)
        length = None if match_ticks is None else np.broadcast_to(np.asarray(match_ticks, dtype=np.int64), rows.shape)
        teams = rng.integers(0, 2, len(rows)) if learner_team is None else np.broadcast_to(learner_team, rows.shape)
        for i, n in enumerate(rows):
            task = TASKS[task_names[i]]
            lt = int(teams[i])
            active = np.zeros(8, dtype=bool)
            active[team_slots(lt, task.n_own)] = True
            active[team_slots(1 - lt, task.n_opp)] = True
            lo, hi = task.opp_levels
            level = int(np.clip(np.round(lo + difficulty[i] * (hi - lo) + rng.normal(0, 0.6)), lo, hi))
            ticks = task.match_ticks if length is None or not task.match_ticks else int(length[i])
            env.start_match([n], active=active[None], kickoff_team=lt if task.start == "kickoff" else 0,
                            match_ticks=ticks if ticks else 10 ** 9)
            if task.start != "kickoff":
                PLACERS[task.start](self, n, lt, float(difficulty[i]))
            st = self.st
            st.task[n] = TASK_INDEX[task.name]
            st.learner_team[n] = lt
            st.ticks[n] = 0
            st.regain_since[n] = -1
            st.lost_since[n] = -1
            st.last_touch[n] = -1
            st.passed[n] = False
            st.difficulty[n] = difficulty[i]
            st.opp_level[n] = level

    # ------------------------------------------------------------------ colocación (marco del aprendiz)
    def _world(self, lt, x, y):
        s = 1.0 if lt == 0 else -1.0
        return np.array([x * s, y])

    def _place(self, n, lt, ball, own, opp, ball_vel=(0.0, 0.0), last_touch=None):
        """ball/own/opp en el marco del aprendiz (ataca +x). own/opp: listas de (x, y)."""
        env = self.env
        s = 1.0 if lt == 0 else -1.0
        pp = np.zeros((8, 2))
        pv = np.zeros((8, 2))
        for slot, xy in zip(team_slots(lt, len(own)), own):
            pp[slot] = (xy[0] * s, xy[1])
        for slot, xy in zip(team_slots(1 - lt, len(opp)), opp):
            pp[slot] = (xy[0] * s, xy[1])
        lt_touch = -1 if last_touch is None else (lt if last_touch == "own" else 1 - lt)
        env.place(n, ball_pos=(ball[0] * s, ball[1]), ball_vel=(ball_vel[0] * s, ball_vel[1]), player_pos=pp,
                  player_vel=pv, last_touch=lt_touch, mass_phase=int(self.rng.random() < 0.65))

    def _u(self, a, b):
        return float(self.rng.uniform(a, b))

    # ------------------------------------------------------------------ comprobación
    def check(self, ev):
        """Después de cada `env.step`: devuelve (done, outcome, truncated) por fila (sólo ejercicios).

        done: el episodio de ejercicio terminó; outcome: +1/0/-1; truncated: corte por tiempo con bootstrap.
        Para partidos, el fin lo marca ev['match_end']; aquí sólo aparece el corte por partido trabado
        (truncación). Vectorizado.
        """
        env, st = self.env, self.st
        N = env.N
        P = _TASK_PARAMS
        task = st.task
        st.ticks += ev["ticks"]
        team = np.where(np.arange(8) < 4, 0, 1)
        lt = st.learner_team
        own_mask = team[None, :] == lt[:, None]
        own_touch = (ev["touched"] & own_mask).any(axis=1)
        rival_touch = (ev["touched"] & ~own_mask).any(axis=1)
        g = ev["goal"]
        scored = ((g == 1) & (lt == 0)) | ((g == -1) & (lt == 1))
        conceded = (g != 0) & ~scored
        drill = P["timeout"][task] > 0
        dt = ev["ticks"]
        # posesión recuperada / perdida (toque de un equipo sin respuesta del otro)
        only_own = own_touch & ~rival_touch
        only_rival = rival_touch & ~own_touch
        both = own_touch & rival_touch
        neither = ~own_touch & ~rival_touch
        st.regain_since = np.where(only_own, np.where(st.regain_since < 0, 0, st.regain_since + dt),
                                   np.where(only_rival | both, -1,
                                            np.where(st.regain_since >= 0, st.regain_since + dt, -1)))
        st.lost_since = np.where(only_rival, np.where(st.lost_since < 0, 0, st.lost_since + dt),
                                 np.where(only_own | both, -1,
                                          np.where(st.lost_since >= 0, st.lost_since + dt, -1)))
        # pase: toque de un propio distinto del último que la tocó, sin toque rival en el medio
        own_t = ev["touched"] & own_mask
        q = np.argmax(own_t, axis=1)
        clean_own = own_touch & ~rival_touch
        st.passed |= clean_own & (st.last_touch >= 0) & (st.last_touch != q)
        st.last_touch = np.where(clean_own, q, np.where(rival_touch, -2, st.last_touch))
        done = np.zeros(N, dtype=bool)
        outcome = np.zeros(N)
        truncated = np.zeros(N, dtype=bool)
        open_ = drill.copy()

        def resolve(mask, value):
            nonlocal open_
            m = mask & open_
            done[m] = True
            outcome[m] = value[m] if np.ndim(value) else value
            open_ &= ~m

        resolve(scored, np.where(P["success_goal"][task] & (~P["require_pass"][task] | st.passed), 1.0, 0.0))
        resolve(conceded, P["concede"][task])
        resolve(P["success_touch"][task] & own_touch, 1.0)
        resolve(P["success_regain"][task] & (st.regain_since >= REGAIN_TICKS), 1.0)
        resolve(P["rival_end"][task] & (st.lost_since >= LOSS_TICKS), 0.0)
        # pérdida limpia: el rival llega a la pelota sin disputa (ningún aprendiz cerca). El ataque terminó,
        # aunque después la recupere: el RL aprendía a soltarla para que el defensor saliera y robársela (2026-10-04)
        if (rival_touch & ~own_touch).any():
            d = np.hypot(env.player_pos[..., 0] - env.ball_pos[:, None, 0], env.player_pos[..., 1] - env.ball_pos[:, None, 1])
            d = np.where(own_mask & env.active, d, np.inf).min(axis=1)
            resolve(P["rival_end"][task] & only_rival & (d > CLEAN_LOSS_PX), 0.0)
        started = ev["restart_start"] > 0
        owner = env.ri[:, K.RI_TEAM]
        resolve(started & (owner == lt) & P["success_regain"][task], 1.0)
        resolve(started & (owner != lt) & P["own_out_end"][task], 0.0)
        timeout = st.ticks >= P["timeout"][task]
        code = P["timeout_code"][task]
        resolve(timeout & (code == 1), 1.0)
        resolve(timeout & (code == 0), 0.0)
        tr = timeout & (code == 2) & open_
        done[tr] = True
        truncated[tr] = True
        # partidos trabados (reloj congelado porque nadie saca el inicial): corte con bootstrap
        stalled = ~drill & (st.ticks >= MATCH_STALL_FACTOR * env.ri[:, K.RI_LEN] + MATCH_STALL_EXTRA)
        done[stalled] = True
        truncated[stalled] = True
        return done, outcome, truncated

    def controllers(self, rows=None):
        """Máscaras (N, 8): lugares del aprendiz y lugares de RS-Pro."""
        env, st = self.env, self.st
        learner = np.zeros((env.N, 8), dtype=bool)
        for n in range(env.N):
            task = TASKS[TASK_NAMES[st.task[n]]]
            slots = team_slots(st.learner_team[n], task.n_own)
            learner[n, slots[:task.n_own - task.scripted_mates]] = True
        scripted = env.active & ~learner
        return learner, scripted


# ============================================================================ colocadores
def _touch(d, n, lt, diff):
    ball = (d._u(-1000, 1000), d._u(-600, 600))
    ang = d._u(0, 2 * np.pi)
    dist = d._u(80, 250 + 900 * diff)
    own = [(np.clip(ball[0] + dist * np.cos(ang), -1100, 1100), np.clip(ball[1] + dist * np.sin(ang), -620, 620))]
    vel = (0.0, 0.0) if d.rng.random() < 0.5 else (d._u(-3, 3), d._u(-3, 3))
    d._place(n, lt, ball, own, [], ball_vel=vel)


def _empty_goal(d, n, lt, diff):
    bx = d._u(700 - 600 * diff, 1000)
    by = d._u(-200 - 350 * diff, 200 + 350 * diff)
    ang = d._u(0, 2 * np.pi)
    dist = d._u(40, 120 + 380 * diff)
    own = [(np.clip(bx + dist * np.cos(ang), -1100, 1100), np.clip(by + dist * np.sin(ang), -620, 620))]
    d._place(n, lt, (bx, by), own, [])


def _carry_goal(d, n, lt, diff):
    bx = d._u(-600 - 400 * diff, 200)
    by = d._u(-500, 500)
    own = [(bx - d._u(30, 200), by + d._u(-120, 120))]
    d._place(n, lt, (bx, by), own, [])


def _receive_shot(d, n, lt, diff):
    # pelota rodando hacia la zona de tiro desde un costado
    ty = d._u(-250, 250)
    tx = d._u(600, 950)
    sx = tx - d._u(100, 400)
    sy = np.sign(d._u(-1, 1)) * d._u(350, 600)
    sp = d._u(2.5, 4.5 + 2.0 * diff)
    v = np.array([tx - sx, ty - sy])
    v = v / np.linalg.norm(v) * sp
    own = [(tx - d._u(100, 350), ty + d._u(-200, 200))]
    d._place(n, lt, (sx, sy), own, [], ball_vel=tuple(v))


def _shot_vs_last(d, n, lt, diff):
    bx, by = d._u(650, 900), d._u(-280, 280)
    own = [(bx - 30, by + d._u(-15, 15))]
    opp = [(d._u(1000, 1080), by * 0.3 + d._u(-40, 40))]
    d._place(n, lt, (bx, by), own, opp, last_touch="own")


def _attack_1v1(d, n, lt, diff):
    bx, by = d._u(-100, 250), d._u(-350, 350)
    own = [(bx - 30, by)]
    opp = [(d._u(600, 950), by * 0.5 + d._u(-100, 100))]
    d._place(n, lt, (bx, by), own, opp, last_touch="own")


def _defend_1v1(d, n, lt, diff):
    bx, by = d._u(-250, 150), d._u(-350, 350)
    opp = [(bx + 30, by)]
    own = [(d._u(-950, -600), by * 0.5 + d._u(-100, 100))]
    d._place(n, lt, (bx, by), own, opp, last_touch="opp")


def _loose_ball(d, n, lt, diff):
    bx, by = d._u(-700, 700), d._u(-500, 500)
    da = d._u(80, 300)
    db = da * d._u(0.75 - 0.25 * diff, 1.3)
    a1, a2 = d._u(0, 2 * np.pi), d._u(0, 2 * np.pi)
    own = [(np.clip(bx + da * np.cos(a1), -1100, 1100), np.clip(by + da * np.sin(a1), -620, 620))]
    opp = [(np.clip(bx + db * np.cos(a2), -1100, 1100), np.clip(by + db * np.sin(a2), -620, 620))]
    d._place(n, lt, (bx, by), own, opp, ball_vel=(d._u(-1.5, 1.5), d._u(-1.5, 1.5)))


def _attack_2v1(d, n, lt, diff):
    bx, by = d._u(200, 500), d._u(-250, 250)
    side = 1.0 if by < 0 else -1.0
    own = [(bx - 30, by), (bx + d._u(-50, 150), by + side * d._u(180, 320))]
    opp = [(bx + d._u(120, 260 - 80 * diff), by + d._u(-40, 40))]
    d._place(n, lt, (bx, by), own, opp, last_touch="own")


def _one_two(d, n, lt, diff):
    bx, by = d._u(250, 550), d._u(-200, 200)
    side = 1.0 if by < 0 else -1.0
    own = [(bx - 30, by), (bx + d._u(40, 120), by + side * d._u(110, 180))]
    opp = [(bx + d._u(70, 120), by + d._u(-20, 20))]
    d._place(n, lt, (bx, by), own, opp, last_touch="own")


def _through_ball(d, n, lt, diff):
    bx, by = d._u(0, 300), d._u(-250, 250)
    run_y = d._u(-300, 300)
    own = [(bx - 30, by), (bx + d._u(150, 300), run_y)]
    line = d._u(600, 780)
    opp = [(line, run_y + d._u(-120, 120)), (line + d._u(150, 300), d._u(-150, 150))]
    d._place(n, lt, (bx, by), own, opp, last_touch="own")


def _redirect(d, n, lt, diff):
    sy = np.sign(d._u(-1, 1))
    sx = d._u(850, 1050)
    by = sy * d._u(380, 560)
    tx, ty = d._u(850, 1000), d._u(-120, 120)
    v = np.array([tx - sx, ty - by])
    v = v / np.linalg.norm(v) * d._u(4.0, 6.5)
    own = [(d._u(650, 800), d._u(-150, 150)), (sx - 40, by - sy * 20)]
    opp = [(d._u(1030, 1100), d._u(-60, 60))]
    d._place(n, lt, (sx, by), own, opp, ball_vel=tuple(v), last_touch="own")


def _attack_2v2(d, n, lt, diff):
    bx, by = d._u(150, 450), d._u(-300, 300)
    own = [(bx - 30, by), (bx + d._u(0, 200), -by * 0.5 + d._u(-150, 150))]
    opp = [(bx + d._u(150, 300), by + d._u(-80, 80)), (d._u(850, 1050), d._u(-120, 120))]
    d._place(n, lt, (bx, by), own, opp, last_touch="own")


def _defend_2v2(d, n, lt, diff):
    bx, by = d._u(-350, 0), d._u(-300, 300)
    opp = [(bx + 30, by), (bx - d._u(0, 200), -by * 0.5 + d._u(-150, 150))]
    own = [(bx - d._u(150, 300), by + d._u(-80, 80)), (d._u(-1050, -850), d._u(-120, 120))]
    d._place(n, lt, (bx, by), own, opp, last_touch="opp")


def _defend_4v4(d, n, lt, diff):
    if d.bank is not None and d.rng.random() < 0.7:
        if _recorded(d, n, lt, "attack", attacker="opp"):
            return
    bx, by = d._u(-400, 50), d._u(-350, 350)
    opp = [(bx + 30, by), (bx - d._u(50, 200), by + d._u(150, 300)), (bx + d._u(-100, 100), by - d._u(150, 300)),
           (bx + d._u(250, 450), by * 0.3)]
    own = [(bx - d._u(80, 200), by * 0.8), (bx - d._u(150, 350), by * 0.3 + d._u(-100, 100)),
           (d._u(-900, -700), d._u(-200, 200)), (d._u(-1050, -900), d._u(-80, 80))]
    d._place(n, lt, (bx, by), own, opp, last_touch="opp")


def _defend_restart(d, n, lt, diff):
    env, rng = d.env, d.rng
    kind = int(rng.choice([C.LATERAL, C.CORNER, C.GOAL_KICK], p=[0.5, 0.35, 0.15]))
    own = [(d._u(-900, -500), d._u(-400, 400)) for _ in range(4)]
    opp = [(d._u(-900, -300), d._u(-400, 400)) for _ in range(4)]
    sy = float(np.sign(d._u(-1, 1)))
    d._place(n, lt, (0.0, 0.0), own, opp)
    s = 1.0 if lt == 0 else -1.0
    if kind == C.LATERAL:
        spot = (d._u(-1000, -300) * s, sy * 688.0)
        env.start_restart(n, C.LATERAL, 1 - lt, spot)
    elif kind == C.CORNER:
        env.start_restart(n, C.CORNER, 1 - lt, (-1140.0 * s, sy * 660.0))
    else:
        # saque de arco del rival en SU campo con nosotros presionando arriba
        env.start_restart(n, C.GOAL_KICK, 1 - lt, (1030.0 * s, sy * 180.0))


def _restart_attack(d, n, lt, diff):
    env, rng = d.env, d.rng
    kind = int(rng.choice([C.LATERAL, C.CORNER], p=[0.55, 0.45]))
    own = [(d._u(400, 1000), d._u(-400, 400)) for _ in range(4)]
    opp = [(d._u(600, 1080), d._u(-350, 350)) for _ in range(4)]
    sy = float(np.sign(d._u(-1, 1)))
    d._place(n, lt, (0.0, 0.0), own, opp)
    s = 1.0 if lt == 0 else -1.0
    if kind == C.LATERAL:
        env.start_restart(n, C.LATERAL, lt, (d._u(300, 1000) * s, sy * 688.0))
    else:
        env.start_restart(n, C.CORNER, lt, (1140.0 * s, sy * 660.0))


def _transition(d, n, lt, diff):
    # pelota recién ganada en campo propio con el rival desordenado (adelantado)
    bx, by = d._u(-500, 0), d._u(-400, 400)
    own = [(bx - 25, by), (bx + d._u(-200, 200), d._u(-400, 400)), (d._u(-700, -300), d._u(-300, 300)),
           (d._u(-200, 300), d._u(-450, 450))]
    opp = [(bx + d._u(40, 120), by + d._u(-60, 60)), (d._u(-400, 200), d._u(-400, 400)),
           (d._u(-300, 300), d._u(-400, 400)), (d._u(200, 700), d._u(-300, 300))]
    d._place(n, lt, (bx, by), own, opp, last_touch="own")


def _recorded(d, n, lt, pool, attacker="own"):
    """Estado humano de la partición de entrenamiento, espejado para que el equipo indicado ataque."""
    from env.rs4_states import KIND_OPEN
    bank = d.bank
    if bank is None:
        return False
    members = bank.pools()[pool]
    if len(members) == 0:
        return False
    i = int(d.rng.choice(members))
    a = bank.arrays
    if a["kind"][i] != KIND_OPEN:
        return False
    last = int(a["last_touch"][i])
    # equipo atacante en la grabación: el del último toque
    want = lt if attacker == "own" else 1 - lt
    flip = last >= 0 and last != want
    bp, bv = a["ball_pos"][i].astype(float), a["ball_vel"][i].astype(float)
    pp, pv = a["player_pos"][i].astype(float), a["player_vel"][i].astype(float)
    held = a["kick_held"][i].astype(bool)
    if flip:
        r = np.array([-1.0, 1.0])
        bp, bv = bp * r, bv * r
        pp = np.concatenate([pp[4:], pp[:4]]) * r
        pv = np.concatenate([pv[4:], pv[:4]]) * r
        held = np.concatenate([held[4:], held[:4]])
        last = 1 - last if last >= 0 else -1
    if d.rng.random() < 0.5:  # espejo en y (la cancha es simétrica en y)
        r = np.array([1.0, -1.0])
        bp, bv, pp, pv = bp * r, bv * r, pp * r, pv * r
    mass = a["mass_phase"][i] if "mass_phase" in a else 1
    d.env.place(n, ball_pos=bp, ball_vel=bv, player_pos=pp, player_vel=pv, kick_held=held, last_touch=last,
                mass_phase=int(mass))
    return True


def _recorded_open(d, n, lt, diff):
    if not _recorded(d, n, lt, "open", attacker="own" if d.rng.random() < 0.5 else "opp"):
        _transition(d, n, lt, diff)


def _recorded_attack(d, n, lt, diff):
    if not _recorded(d, n, lt, "attack", attacker="own"):
        _attack_2v2(d, n, lt, diff)


PLACERS = dict(touch=_touch, empty_goal=_empty_goal, carry_goal=_carry_goal, receive_shot=_receive_shot,
               shot_vs_last=_shot_vs_last, attack_1v1=_attack_1v1, defend_1v1=_defend_1v1, loose_ball=_loose_ball,
               attack_2v1=_attack_2v1, one_two=_one_two, through_ball=_through_ball, redirect=_redirect,
               attack_2v2=_attack_2v2, defend_2v2=_defend_2v2, defend_4v4=_defend_4v4,
               defend_restart=_defend_restart, restart_attack=_restart_attack, transition=_transition,
               recorded_open=_recorded_open, recorded_attack=_recorded_attack)
