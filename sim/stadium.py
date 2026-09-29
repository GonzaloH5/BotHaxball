"""Carga de estadios HaxBall (.hbs) a arrays planos para el motor numba.

Coordenadas nativas de HaxBall (y hacia abajo). No se espeja nada aquí.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

STADIUM_DIR = Path(__file__).resolve().parent.parent / "stadiums"

# Flags de colisión de HaxBall
BALL, RED, BLUE, REDKO, BLUEKO, WALL, ALL, KICK, SCORE = 1, 2, 4, 8, 16, 32, 63, 64, 128
C0, C1, C2, C3 = 1 << 28, 1 << 29, 1 << 30, 1 << 31
FLAGS = {
    "": 0, "none": 0, "ball": BALL, "red": RED, "blue": BLUE, "redKO": REDKO, "blueKO": BLUEKO,
    "wall": WALL, "all": ALL, "kick": KICK, "score": SCORE,
    "c0": C0, "c1": C1, "c2": C2, "c3": C3,
}
PLAYER_MASK = BALL | RED | BLUE | WALL  # 39

# Valores por defecto de HaxBall
DEFAULT_PLAYER = dict(radius=15.0, invMass=0.5, bCoef=0.5, damping=0.96, acceleration=0.1,
                      kickingAcceleration=0.07, kickingDamping=0.96, kickStrength=5.0, kickback=0.0)
DEFAULT_DISC = dict(radius=10.0, invMass=1.0, bCoef=0.5, damping=0.99, cGroup=ALL, cMask=ALL)


def _flags(v, default: int) -> int:
    if v is None:
        return default
    out = 0
    for name in v:
        out |= FLAGS[name]
    return out


def _load_json(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    # los .hbs a veces traen comas colgantes o comentarios
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return json.loads(text)


@dataclass
class Stadium:
    name: str
    width: float
    height: float
    spawn_distance: float
    kickoff_reset_full: bool
    # vértices: (V, 2) pos, (V,) bcoef, cgroup, cmask
    v_pos: np.ndarray
    v_bcoef: np.ndarray
    v_group: np.ndarray
    v_mask: np.ndarray
    # segmentos: (S, 2) p0, p1; curva: (S,) curve (cot(ángulo/2) o 0), bias
    # centro (S,2), radio (S,), tangentes t0/t1 (S,2)
    s_p0: np.ndarray
    s_p1: np.ndarray
    s_curved: np.ndarray
    s_center: np.ndarray
    s_radius: np.ndarray
    s_t0: np.ndarray
    s_t1: np.ndarray
    s_bias: np.ndarray
    s_bcoef: np.ndarray
    s_group: np.ndarray
    s_mask: np.ndarray
    s_vis: np.ndarray
    # planos
    p_normal: np.ndarray
    p_dist: np.ndarray
    p_bcoef: np.ndarray
    p_group: np.ndarray
    p_mask: np.ndarray
    # discos fijos del estadio (sin la pelota): pos, radius, invMass, bCoef, damping, group, mask
    d_pos: np.ndarray
    d_radius: np.ndarray
    d_invmass: np.ndarray
    d_bcoef: np.ndarray
    d_damping: np.ndarray
    d_group: np.ndarray
    d_mask: np.ndarray
    # pelota
    ball: dict
    player: dict
    # arcos: (G,2) p0, p1, team (0 = rojo, 1 = azul): pelota que cruza el arco de `team` => gol en contra de `team`
    g_p0: np.ndarray
    g_p1: np.ndarray
    g_team: np.ndarray
    # para normalizar observaciones / recompensas
    goal_x: float
    goal_half_height: float
    field_half_w: float
    field_half_h: float
    kickoff_radius: float = 75.0
    # índice original de cada disco del estadio (sin la pelota); los decorativos se descartan
    d_src: np.ndarray | None = None
    # redSpawnPoints / blueSpawnPoints (mapas con script, p. ej. HaxEleven): (k, 2); vacíos = spawnDistance
    red_spawn: np.ndarray | None = None
    blue_spawn: np.ndarray | None = None


def _arc(p0: np.ndarray, p1: np.ndarray, curve_deg: float, bias: float):
    """Devuelve (p0, p1, curved, center, radius, t0, t1, bias) con la convención de HaxBall.

    Un disco colisiona con el arco si está del lado de la cuerda donde está el arco:
    dot(pos - center, t0) > 0 y dot(pos - center, t1) > 0.
    """
    a = math.radians(curve_deg)
    if a < 0:
        a = -a
        p0, p1 = p1, p0
        bias = -bias
    # HaxBall trata ángulos muy chicos como recta
    if not (0.17435839227423353 < a < math.radians(340)):
        return p0, p1, 0, np.zeros(2), 0.0, np.zeros(2), np.zeros(2), bias
    cot = 1.0 / math.tan(a / 2)
    h = (p1 - p0) / 2
    perp = np.array([-h[1], h[0]])
    center = p0 + h + perp * cot
    radius = float(np.linalg.norm(p1 - center))
    # tangentes que apuntan "hacia dentro" del arco
    r0 = p0 - center
    r1 = p1 - center
    t0 = np.array([-r0[1], r0[0]])
    t1 = np.array([r1[1], -r1[0]])
    # punto medio del arco: centro - dirección(perp) * R
    mid = center - perp / np.linalg.norm(perp) * radius
    if np.dot(mid - center, t0) < 0:
        t0, t1 = -t0, -t1
    # 1: arco <= 180° (test AND de las tangentes); 2: arco > 180° (test OR)
    return p0, p1, (1 if a <= math.pi + 1e-9 else 2), center, radius, t0, t1, bias


def load_stadium(name_or_path: str = "classic") -> Stadium:
    path = Path(name_or_path)
    if not path.exists():
        path = STADIUM_DIR / f"{name_or_path}.hbs"
    data = _load_json(path)
    traits = data.get("traits", {})

    def resolve(obj: dict) -> dict:
        base = dict(traits.get(obj.get("trait"), {})) if obj.get("trait") else {}
        base.update(obj)
        return base

    verts_raw = [resolve(v) for v in data.get("vertexes", [])]
    v_pos = np.array([[v["x"], v["y"]] for v in verts_raw], dtype=np.float64).reshape(-1, 2)
    v_bcoef = np.array([v.get("bCoef", 1.0) for v in verts_raw], dtype=np.float64)
    v_group = np.array([_flags(v.get("cGroup"), WALL) for v in verts_raw], dtype=np.int64)
    v_mask = np.array([_flags(v.get("cMask"), ALL) for v in verts_raw], dtype=np.int64)

    segs = []
    for s in (resolve(s) for s in data.get("segments", [])):
        p0 = v_pos[s["v0"]].copy()
        p1 = v_pos[s["v1"]].copy()
        if "curve" in s:
            deg = s["curve"]
        elif "curveF" in s:  # curveF = cot(a/2)
            deg = math.degrees(2 * math.atan2(1.0, s["curveF"]))
        else:
            deg = 0.0
        arc = _arc(p0, p1, deg, s.get("bias", 0.0))
        if arc[2] == 0 and arc[7] != 0:
            # En segmentos rectos el motor usa la normal (-sy, sx) y HaxBall mide el bias del lado
            # contrario: sin invertirlo, una pared de un solo lado empuja la pelota hacia afuera desde
            # adentro de la cancha. Verificado contra un partido real en HaxArg Big (líneas de fondo con
            # bias ±40): error de 49 unidades por tick sin esto.
            arc = arc[:7] + (-arc[7],)
        segs.append((arc, s.get("bCoef", 1.0), _flags(s.get("cGroup"), WALL), _flags(s.get("cMask"), ALL),
                     bool(s.get("vis", True))))

    def col(i, dtype=np.float64, shape=None):
        arr = np.array([x[0][i] for x in segs], dtype=dtype)
        return arr.reshape(shape) if shape else arr

    planes = [resolve(p) for p in data.get("planes", [])]
    p_normal = np.array([p["normal"] for p in planes], dtype=np.float64).reshape(-1, 2)
    if len(planes):
        p_normal = p_normal / np.linalg.norm(p_normal, axis=1, keepdims=True)

    discs = [resolve(d) for d in data.get("discs", [])]
    bp = data.get("ballPhysics")
    if bp == "disc0":  # formato exportado por las salas: la pelota es discs[0]
        bp, discs = discs[0], discs[1:]
    # discos que no pueden chocar con nada (decoración, anclas de joints): se descartan.
    # d_src guarda el índice original (sin contar la pelota) para alinear con grabaciones reales.
    # También los de masa casi nula (invMass ~1e250): son dibujos pegados a la pelota con joints
    # (p. ej. Real Futsal x7) que en el juego no empujan a nada; el sim no simula joints.
    # Y los que no chocan con la pelota y están del otro lado de un plano que frena a los jugadores
    # (tribunas, carteles, animaciones fuera de la cancha: p. ej. Real Soccer x4 GLH trae ~200): nadie
    # puede tocarlos y harían la física mucho más lenta (el choque disco-disco es por pares).
    pr = float(data.get("playerPhysics", {}).get("radius", DEFAULT_PLAYER["radius"]))
    player_planes = [(p_normal[k], float(planes[k]["dist"])) for k in range(len(planes))
                     if (_flags(planes[k].get("cMask"), ALL) & (RED | BLUE))
                     and (_flags(planes[k].get("cGroup"), WALL) & PLAYER_MASK)]

    # discos del script que no existen en el juego (se usan para ubicar jugadores y no se ven ni chocan):
    # "haxballrl": {"ignore_discs": [índices en discs, sin contar la pelota]}
    ignore = set(data.get("haxballrl", {}).get("ignore_discs", []))

    def reachable(d) -> bool:
        if _flags(d.get("cMask"), ALL) & BALL:
            return True
        q = np.asarray(d.get("pos", [0.0, 0.0]), dtype=np.float64)
        rd = float(d.get("radius", DEFAULT_DISC["radius"]))
        # un jugador cumple dot(c, n) >= dist + pr; para tocar el disco hace falta dot(q, n) > dist - rd
        return all(float(q @ n) > dist - rd for n, dist in player_planes)

    kept = [(i, d) for i, d in enumerate(discs)
            if _flags(d.get("cGroup"), ALL) != 0 and _flags(d.get("cMask"), ALL) != 0
            and d.get("radius", DEFAULT_DISC["radius"]) > 0
            and d.get("invMass", DEFAULT_DISC["invMass"]) < 1e100
            and reachable(d) and i not in ignore]
    d_src = np.array([i for i, _ in kept], dtype=np.int64)
    discs = [d for _, d in kept]

    def dget(k):
        return np.array([d.get(k, DEFAULT_DISC[k]) for d in discs], dtype=np.float64)

    ball = dict(DEFAULT_DISC)
    if isinstance(bp, dict):
        bp = resolve(bp)
        for k in ("radius", "invMass", "bCoef", "damping"):
            if k in bp:
                ball[k] = bp[k]
        ball["cGroup"] = _flags(bp.get("cGroup"), BALL)
        ball["cMask"] = _flags(bp.get("cMask"), ALL)
    else:
        ball["cGroup"] = BALL
        ball["cMask"] = ALL
    ball["cGroup"] |= KICK | SCORE
    ball["cMask"] &= ~(REDKO | BLUEKO)

    player = dict(DEFAULT_PLAYER)
    player.update({k: v for k, v in data.get("playerPhysics", {}).items() if k in DEFAULT_PLAYER})

    goals = data.get("goals", [])
    g_p0 = np.array([g["p0"] for g in goals], dtype=np.float64).reshape(-1, 2)
    g_p1 = np.array([g["p1"] for g in goals], dtype=np.float64).reshape(-1, 2)
    g_team = np.array([0 if g["team"] == "red" else 1 for g in goals], dtype=np.int64)
    goal_x = float(np.abs(g_p0[:, 0]).max()) if len(goals) else data["width"]
    goal_hh = float(np.abs(g_p0[:, 1]).max()) if len(goals) else 64.0

    bg = data.get("bg", {})
    # Medidas de la cancha: bg.width/height si el mapa las declara bien; si no (mapas con script,
    # donde bg es el fondo dibujado) se fijan en la clave propia "haxballrl" del .hbs.
    ov = data.get("haxballrl", {})
    field_w = float(ov.get("field_half_w", bg.get("width", data["width"])))
    field_h = float(ov.get("field_half_h", bg.get("height", data["height"])))
    return Stadium(
        name=data.get("name", path.stem),
        width=float(data["width"]), height=float(data["height"]),
        spawn_distance=float(data.get("spawnDistance", 170)),
        kickoff_reset_full=data.get("kickOffReset", "partial") == "full",
        v_pos=v_pos, v_bcoef=v_bcoef, v_group=v_group, v_mask=v_mask,
        s_p0=col(0, shape=(-1, 2)), s_p1=col(1, shape=(-1, 2)), s_curved=col(2, np.int64),
        s_center=col(3, shape=(-1, 2)), s_radius=col(4), s_t0=col(5, shape=(-1, 2)),
        s_t1=col(6, shape=(-1, 2)), s_bias=col(7),
        s_bcoef=np.array([x[1] for x in segs], dtype=np.float64),
        s_group=np.array([x[2] for x in segs], dtype=np.int64),
        s_mask=np.array([x[3] for x in segs], dtype=np.int64),
        s_vis=np.array([x[4] for x in segs], dtype=bool),
        p_normal=p_normal,
        p_dist=np.array([p["dist"] for p in planes], dtype=np.float64),
        p_bcoef=np.array([p.get("bCoef", 1.0) for p in planes], dtype=np.float64),
        p_group=np.array([_flags(p.get("cGroup"), WALL) for p in planes], dtype=np.int64),
        p_mask=np.array([_flags(p.get("cMask"), ALL) for p in planes], dtype=np.int64),
        d_pos=np.array([d.get("pos", [0.0, 0.0]) for d in discs], dtype=np.float64).reshape(-1, 2),
        d_src=d_src,
        # "haxballrl" puede fijar los puntos de saque reales cuando el script de la sala mueve a los jugadores
        # (p. ej. Sanguchito X1: el mapa los declara en una zona de espera y el script los lleva al arco)
        red_spawn=np.array(ov.get("red_spawn", data.get("redSpawnPoints")) or [], dtype=np.float64).reshape(-1, 2),
        blue_spawn=np.array(ov.get("blue_spawn", data.get("blueSpawnPoints")) or [], dtype=np.float64).reshape(-1, 2),
        d_radius=dget("radius"), d_invmass=dget("invMass"), d_bcoef=dget("bCoef"),
        d_damping=dget("damping"),
        d_group=np.array([_flags(d.get("cGroup"), ALL) for d in discs], dtype=np.int64),
        d_mask=np.array([_flags(d.get("cMask"), ALL) for d in discs], dtype=np.int64),
        ball=ball, player=player,
        g_p0=g_p0, g_p1=g_p1, g_team=g_team,
        goal_x=goal_x, goal_half_height=goal_hh,
        field_half_w=field_w,
        field_half_h=field_h,
        kickoff_radius=float(bg.get("kickOffRadius", 75.0)),
    )
