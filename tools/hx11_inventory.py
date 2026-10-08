"""Inventario del banco HaxEleven 11v11 (docs/PLAN_HAXELEVEN.md, S0).

Lee los metadatos de `bridge/hx11_meta.js` y arma:

* Duplicados exactos (mismo SHA-256; se conserva el primero por nombre) y del mismo partido grabado por dos
  personas (misma secuencia de goles no vacía y ≥ 8 nombres de jugadores en común).
* Minutos jugados por estadio y por plantel: 11v11 exacto, ambos equipos con ≥ 10 y con ≥ 9.
* Catálogo de estadios (exportados una vez por hash a `data/hx11/stadiums`).
* Actividad del script de la sala: `setDiscProperties` por disco y campo (jugadores agrupados), por minuto jugado.
* Anuncios del host como plantillas (nombres → <J>, IDs de Discord → <ID>, números → #); el chat que la sala
  reenvía como anuncio ("[x] nombre: texto") se cuenta aparte y no se copia. Valores de los anuncios con
  porcentaje (powershot utilizado, potencia de córner, tiro libre, saque de arco...).
* Sesiones y particiones, como en `tools.x4_index`: sesión = grabaciones con ≥ 50% de jugadores en común y
  ≤ 45 min entre una y la siguiente (con cualquiera de las lecturas de la fecha del nombre, que mezcla
  mes-día y día-mes), en bloques de hasta 2 h. Partición por sesión, estable por hash: ~15% prueba,
  ~10% desarrollo; además, `test_grupos` reserva todas las sesiones de los grupos elegidos (siglas del
  archivo o de la sala) para medir la generalización a técnicos no vistos.

Los nombres, el chat, los IDs y los títulos de sala sólo quedan en `data/hx11/` (no versionado). Lo versionado
(`reports/hx11/`) usa claves `p_<sha1[:10]>` de los nombres.

  python -m tools.hx11_inventory --meta 'data/hx11/meta_*.jsonl' --out reports/hx11
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import glob
import hashlib
import json
import re
from pathlib import Path

from tools.x4_index import player_key

ROOT = Path(__file__).resolve().parent.parent
SESSION_GAP_MIN = 45
SESSION_MAX_MIN = 120
SESSION_JACCARD = 0.5
DUP_MATCH_NAMES = 8
TEST_FRAC, DEV_FRAC = 0.15, 0.10
TEST_GROUP_FRAC = 0.10
FPS = 60.0

FILE_TIME = re.compile(r"HaxEleven-(\d{1,2})-(\d{1,2})-(\d{4})-(\d{1,2})h(\d{2})m")
GROUPS_FILE = re.compile(r"^([A-Z]{2,4})_([A-Z]{2,4})_")
GROUPS_ROOM = re.compile(r"\b([A-Z]{2,4}) vs ([A-Z]{2,4})\b")
DISCORD_ID = re.compile(r"\[?\b\d{15,21}\b\]?")
NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
PERCENT = re.compile(r"^\[[^\]]{1,6}\]\s*(Powershot utilizado\.|Potencia del [^:\d]{1,30}:)\s*(\d+(?:[.,]\d+)?)\s*%")
GK_DECLARED = re.compile(r"^\[✅\]\s+(.+?)\s+es el Arquero del")
POSITION_WORDS = re.compile(r"\b(gk|arq\w*|def\w*|central\w*|lateral\w*|lat|carriler\w*|volante\w*|medio\w*|mc|mcd|mco|"
                            r"enganche|extrem\w*|wing\w*|punta|delanter\w*|dc|li|ld|lb|rb|cb|cm|st)\b", re.I)
MODE_TOGGLE = re.compile(r"(offside|falta|powershot|power shot|gambeta)\w*\s+ha\w*\s+sido\s+(des)?activad", re.I)
CHAT_PREFIX = re.compile(r"^(⟮[^⟯]*⟯|\[👔\])")
LOGIN_AS = re.compile(r"(se logeó como )[^.]+\.")
MIN_TEMPLATE_FILES = 3


def time_candidates(name, latest=None):
    """Lecturas posibles de la fecha y hora del nombre: mes-día y día-mes (una sola si no es ambigua).

    Se descartan las lecturas posteriores a `latest` (una grabación no puede ser del futuro)."""
    m = FILE_TIME.search(name)
    if not m:
        return []
    a, b, y, h, mi = (int(x) for x in m.groups())
    out = []
    for mo, d in ((a, b), (b, a)):
        try:
            t = dt.datetime(y, mo, d, h, mi)
        except ValueError:
            continue
        if t not in out and (latest is None or t <= latest):
            out.append(t)
    return out


def stadium_family(name):
    """x11 (cualquier tamaño de RSx11), x7 u otro (entrenamiento, penales...)."""
    if "RSx11" in name:
        return "x11"
    if "RSx7" in name:
        return "x7"
    return "otro"


def groups_of(file, room):
    """Siglas de los dos grupos: del nombre del archivo (AV_KJ_...) o del título de la sala ("AV vs KJ")."""
    m = GROUPS_FILE.match(file) or GROUPS_ROOM.search(room or "")
    return tuple(sorted(m.groups())) if m else ()


def has_name(text, names):
    return any(len(n) >= 2 and re.search(r"(?<!\w)" + re.escape(n) + r"(?!\w)", text) for n in names)


def chat_split(msg, names):
    """Si el anuncio es chat reenviado por la sala ("[👔] nombre: texto", "⟮ 👤 ᵀᴼᴾ 26 ⟯ nombre (alias): texto"),
    devuelve el texto del jugador; si no, None."""
    if ": " not in msg:
        return None
    prefix, text = msg.split(": ", 1)
    if CHAT_PREFIX.match(prefix):  # rango del jugador: siempre es chat, aunque el nombre no esté en la sala
        return text
    return text if len(prefix) <= 80 and has_name(prefix, names) else None


def names_regex(names, min_len=2):
    """Una sola expresión con todos los nombres (enteros: un jugador "st" no debe tocar "ESTE")."""
    ns = sorted({n for n in names if len(n) >= min_len}, key=len, reverse=True)
    return re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, ns)) + r")(?!\w)") if ns else None


def template(msg, names, global_names=None):
    """Plantilla de un anuncio sin datos personales, o None si es chat reenviado.

    `names` son los del archivo; `global_names` (regex de `names_regex`) los de todo el banco, porque los anuncios
    también nombran a jugadores que no están en la sala (cuentas, rankings)."""
    if chat_split(msg, names) is not None:
        return None
    s = DISCORD_ID.sub("<ID>", msg)
    s = LOGIN_AS.sub(r"\1<J>.", s)
    for rx in (names_regex(names), global_names):
        if rx is not None:
            s = rx.sub("<J>", s)
    return NUMBER.sub("#", s).strip()


def is_rule_text(msg):
    """Consejos y modos que anuncia la sala (sin nombres): sirven como hipótesis para S2."""
    return msg.startswith("[🐐]") or "Modos:" in msg


def percent_value(msg):
    """(tipo, valor) de los anuncios de la sala con porcentaje: "Powershot utilizado. 33%",
    "Potencia del corner: 91%" (también saque y tiro libre)."""
    m = PERCENT.match(msg)
    if not m:
        return None
    prefix = m.group(1).strip().rstrip(":").strip()
    return prefix, float(m.group(2).replace(",", "."))


def public_stadium(name):
    """Nombre de estadio para lo versionado: los de x11 y x7 tal cual; los demás pueden nombrar a un jugador."""
    return name if stadium_family(name) in ("x11", "x7") else "otro"


def avatar_kind(v):
    v = (v or "").strip()
    if not v:
        return "vacío"
    if v.isascii() and v.isdigit():
        return "número"
    if v.isascii() and v.isalpha():
        return "letras"
    return "otro"


def team_names(m):
    """Nombres de los que estuvieron en un equipo (rojo o azul) en algún momento."""
    ids = {p["id"] for p in m.get("initial_players", []) if p["team"] in (1, 2)}
    ids |= {c["id"] for c in m.get("team_changes", []) if c["team"] in (1, 2)}
    return sorted({m["names"][str(i)].strip() for i in ids if str(i) in m["names"]})


def roster_minutes(played):
    """Minutos jugados por estadio y por nivel de plantel."""
    out = collections.defaultdict(collections.Counter)
    for k, v in played.items():
        st, fmt = k.rsplit("|", 1)
        r, b = (int(x) for x in fmt.split("v"))
        c = out[st]
        c["total"] += v
        if r == 11 and b == 11:
            c["11v11"] += v
        if min(r, b) >= 10:
            c["min10"] += v
        if min(r, b) >= 9:
            c["min9"] += v
    return {st: {k: round(v / FPS / 60, 2) for k, v in c.items()} for st, c in out.items()}


def hist(values, edges):
    c = collections.Counter()
    for v in values:
        for lo, hi in zip(edges[:-1], edges[1:]):
            if lo <= v < hi or (hi == edges[-1] and v == hi):
                c[f"{lo:g}-{hi:g}"] += 1
                break
    return dict(c)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta", default=str(ROOT / "data" / "hx11" / "meta_*.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "hx11"))
    ap.add_argument("--private", default=str(ROOT / "data" / "hx11"))
    ap.add_argument("--latest", default=dt.date.today().isoformat(),
                    help="fecha máxima posible de una grabación (descarta lecturas día-mes del futuro)")
    a = ap.parse_args()
    latest = dt.datetime.fromisoformat(a.latest) + dt.timedelta(days=1)
    metas = {}
    for path in sorted(glob.glob(a.meta)):
        for line in open(path, encoding="utf-8"):
            m = json.loads(line)
            metas[m["file"]] = m

    # nombres de todo el banco (para tapar también a los que no están en la sala de cada anuncio)
    global_rx = names_regex({v.strip() for m in metas.values() for v in m.get("names", {}).values()}, min_len=4)
    rows, seen_sha = {}, {}
    ann_tpl, ann_files, chat_n = collections.Counter(), collections.Counter(), 0
    rule_texts, mode_files = collections.Counter(), collections.Counter()
    avatar_kinds, avatar_numbers, gk_avatars, chat_lines = (collections.Counter() for _ in range(4))
    pct = collections.defaultdict(list)
    sdp, sdp_player = collections.Counter(), collections.Counter()
    stadiums = collections.defaultdict(lambda: dict(files=0, minutes=0.0, exports=set()))
    for name in sorted(metas):
        m = metas[name]
        r = dict(file=name, sha256=m["sha256"], bytes=m["bytes"], error=m["error"], end=m.get("end"))
        if m["error"]:
            r["status"] = "error"
            rows[name] = r
            continue
        if m["sha256"] in seen_sha:
            r.update(status="duplicate", duplicate_of=seen_sha[m["sha256"]])
            rows[name] = r
            continue
        seen_sha[m["sha256"]] = name
        names = team_names(m)
        all_names = {v.strip() for v in m["names"].values()}
        mins = roster_minutes(m["played"])
        fam_mins = collections.defaultdict(collections.Counter)
        for st_name, c in mins.items():
            fam_mins[stadium_family(st_name)].update(c)
        goals = [(round(e["time"] or 0, 1), tuple(e["score"] or ())) for e in m["events"] if e["t"] == "goal"]
        r.update(status="ok", room=m.get("room"), groups=groups_of(name, m.get("room")),
                 times=[t.isoformat() for t in time_candidates(name, latest)], frames=m["frames"],
                 minutes=mins, family_minutes={f: {k: round(v, 2) for k, v in c.items()} for f, c in fam_mins.items()},
                 goals=len(goals), _goals=goals, _names=names,
                 players=sorted(player_key(n) for n in names), n_players=len(names),
                 joins=len(m["joins"]), leaves=len(m["leaves"]), team_changes=len(m["team_changes"]),
                 kicks=m["kicks"], announcements=m["announcements_total"], sdp_total=m["sdp_total"],
                 gk_declared=sum(1 for x in m["announcements"] if "Arquero" in x["m"]))
        rows[name] = r
        # estadios
        for s in m["stadiums"]:
            st = stadiums[s["name"]]
            if s.get("file"):
                st["exports"].add(s["file"])
        for st_name, c in mins.items():
            stadiums[st_name]["files"] += 1
            stadiums[st_name]["minutes"] += c.get("total", 0.0)
        # anuncios
        file_tpls = set()
        modes = set()
        for x in m["announcements"]:
            t = template(x["m"], all_names, global_rx)
            if t is None:
                chat_n += x["n"]
                continue
            if "→" in t:  # rankings de la sala: listas de jugadores
                continue
            ann_tpl[t] += x["n"]
            file_tpls.add(t)
            if is_rule_text(x["m"]) and "<J>" not in t:
                rule_texts[x["m"].strip()] += x["n"]
            if "Modos:" in x["m"] or MODE_TOGGLE.search(x["m"]):
                modes.add(t if "<J>" in t else x["m"].strip())
        r["modes"] = sorted(modes)
        for md in modes:
            mode_files[md] += 1
        # avatares (último de cada jugador de equipo) y arqueros declarados con !gk
        ids_by_name = collections.defaultdict(set)
        for i, nm in m["names"].items():
            ids_by_name[nm.strip()].add(i)
        last_avatar = {i: v[-1]["v"] for i, v in m["avatars"].items() if v}
        team_ids = {i for nm in names for i in ids_by_name[nm]}
        for i in team_ids:
            av = last_avatar.get(i)
            avatar_kinds[avatar_kind(av)] += 1
            if avatar_kind(av) == "número":
                avatar_numbers[int(av)] += 1
        for x in m["announcements"]:
            g = GK_DECLARED.match(x["m"])
            if g:
                for i in ids_by_name.get(g.group(1).strip(), ()):
                    av = last_avatar.get(i)
                    gk_avatars[av if avatar_kind(av) == "número" else avatar_kind(av)] += 1
            text = chat_split(x["m"], all_names)
            if text is not None:
                chat_lines["total"] += x["n"]
                if POSITION_WORDS.search(text):
                    chat_lines["menciona_puesto"] += x["n"]
            pv = percent_value(x["m"])
            if pv:
                pct[pv[0]].extend([pv[1]] * x["n"])
        for t in file_tpls:
            ann_files[t] += 1
        # setDiscProperties
        for k, n in m["sdp"].items():
            kind, did, fields = k.split(":", 2)
            if kind in ("true", "1"):
                sdp_player[f"jugador:{fields}"] += n
            else:
                sdp[f"disco {did}:{fields}"] += n

    ok = [n for n, r in rows.items() if r["status"] == "ok"]
    # mismo partido grabado por dos personas
    by_goals = collections.defaultdict(list)
    for n in ok:
        if rows[n]["_goals"]:
            by_goals[tuple(rows[n]["_goals"])].append(n)
    for names in by_goals.values():
        names.sort()
        for other in names[1:]:
            if len(set(rows[other]["_names"]) & set(rows[names[0]]["_names"])) >= DUP_MATCH_NAMES:
                rows[other].update(status="duplicate_match", duplicate_of=names[0])
    ok = [n for n in ok if rows[n]["status"] == "ok"]

    # sesiones: jugadores en común y cercanía en el tiempo (cualquier lectura de la fecha)
    parent = {n: n for n in ok}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def close(t1, t2):
        return any(abs((x - y).total_seconds()) <= SESSION_GAP_MIN * 60
                   for x in map(dt.datetime.fromisoformat, t1) for y in map(dt.datetime.fromisoformat, t2))

    for i, n1 in enumerate(ok):
        s1 = set(rows[n1]["_names"])
        for n2 in ok[i + 1:]:
            s2 = set(rows[n2]["_names"])
            if not s1 or not s2 or len(s1 & s2) / len(s1 | s2) < SESSION_JACCARD:
                continue
            if close(rows[n1]["times"], rows[n2]["times"]):
                parent[find(n2)] = find(n1)
    comps = collections.defaultdict(list)
    for n in ok:
        comps[find(n)].append(n)
    sessions = {}
    for root, items in comps.items():
        # bloques de hasta 2 h, ordenados por la primera lectura de la fecha
        items.sort(key=lambda n: (rows[n]["times"][0] if rows[n]["times"] else "", n))
        k, first = 0, None
        for n in items:
            t = dt.datetime.fromisoformat(rows[n]["times"][0]) if rows[n]["times"] else None
            if first is not None and t is not None and (t - first).total_seconds() > SESSION_MAX_MIN * 60:
                k, first = k + 1, None
            first = first or t
            sid = "s_" + hashlib.sha1(f"{root}|{k}".encode()).hexdigest()[:10]
            rows[n]["session"] = sid
            sessions.setdefault(sid, []).append(n)

    # grupos reservados: los de frecuencia intermedia hasta ~10% de las grabaciones con siglas
    group_files = collections.Counter(g for n in ok for g in rows[n]["groups"])
    with_groups = sum(1 for n in ok if rows[n]["groups"])
    reserved, covered = [], set()
    for g, c in sorted(group_files.items(), key=lambda x: (abs(x[1] - 6), hashlib.sha256(x[0].encode()).hexdigest())):
        if len(covered) >= TEST_GROUP_FRAC * with_groups:
            break
        reserved.append(g)
        covered |= {n for n in ok if g in rows[n]["groups"]}
    split_of = {}
    for s, items in sessions.items():
        if any(set(rows[n]["groups"]) & set(reserved) for n in items):
            split_of[s] = "test_grupos"
    rest = sorted((s for s in sessions if s not in split_of), key=lambda s: hashlib.sha256(s.encode()).hexdigest())
    total = sum(len(sessions[s]) for s in rest)
    cnt = collections.Counter()
    for s in rest:
        sp = "test" if cnt["test"] < TEST_FRAC * total else "dev" if cnt["dev"] < DEV_FRAC * total else "train"
        split_of[s] = sp
        cnt[sp] += len(sessions[s])
    for n in ok:
        rows[n]["split"] = split_of[rows[n]["session"]]

    # resúmenes
    def fam_minutes(ns, fam, key):
        return round(sum(rows[n]["family_minutes"].get(fam, {}).get(key, 0.0) for n in ns), 1)

    status = collections.Counter(r["status"] for r in rows.values())
    by_split = {}
    players_by_split = collections.defaultdict(set)
    for sp in sorted(set(split_of.values())):
        ns = [n for n in ok if rows[n]["split"] == sp]
        by_split[sp] = dict(recordings=len(ns), sessions=sum(1 for s in sessions if split_of[s] == sp),
                            minutes_x11_11v11=fam_minutes(ns, "x11", "11v11"),
                            minutes_x11_min9=fam_minutes(ns, "x11", "min9"))
        for n in ns:
            players_by_split[sp] |= set(rows[n]["players"])
    overlap = {sp: dict(players=len(p), also_in_train=len(p & players_by_split.get("train", set())))
               for sp, p in players_by_split.items() if sp != "train"}
    played_total = sum(c.get("total", 0.0) for n in ok for c in rows[n]["minutes"].values())
    per_min = (lambda v: round(v / played_total, 3)) if played_total else (lambda v: None)
    pct_summary = {}
    for k, vs in sorted(pct.items(), key=lambda x: -len(x[1])):
        vs.sort()
        pct_summary[k] = dict(n=len(vs), min=vs[0], p10=vs[len(vs) // 10], p50=vs[len(vs) // 2],
                              p90=vs[(9 * len(vs)) // 10], max=vs[-1], hist=hist(vs, list(range(0, 101, 10))))
    inventory = dict(
        version="hx11-inventario-1", rules=__doc__.strip().splitlines()[2:17],
        files=dict(total=len(rows), status=dict(status),
                   errors={n: r["error"] for n, r in rows.items() if r["status"] == "error"},
                   duplicates={n: r["duplicate_of"] for n, r in rows.items() if r.get("duplicate_of")}),
        minutes={fam: {k: fam_minutes(ok, fam, k) for k in ("total", "11v11", "min10", "min9")}
                 for fam in ("x11", "x7", "otro")},
        recordings_by_x11_min9=hist([rows[n]["family_minutes"].get("x11", {}).get("min9", 0.0) for n in ok],
                                    [0, 1, 5, 10, 15, 20, 30, 60, 1e9]),
        x11_sizes=dict(
            distinct=sum(1 for s in stadiums if stadium_family(s) == "x11"),
            single_file=sum(1 for s, v in stadiums.items() if stadium_family(s) == "x11" and v["files"] == 1),
            minutes_single_file=round(sum(v["minutes"] for s, v in stadiums.items()
                                          if stadium_family(s) == "x11" and v["files"] == 1), 1)),
        stadiums={s: dict(family=stadium_family(s), files=v["files"], minutes=round(v["minutes"], 1),
                          exports=sorted(v["exports"]))
                  for s, v in sorted(stadiums.items(), key=lambda x: -x[1]["minutes"]) if v["files"] > 1},
        groups=dict(files_with_groups=with_groups, files_by_group=dict(group_files.most_common()),
                    reserved_for_test=sorted(reserved)),
        sessions=len(sessions), splits=by_split, player_overlap=overlap,
        players_total=len({p for n in ok for p in rows[n]["players"]}),
        script=dict(sdp_per_played_min={k: per_min(v) for k, v in (sdp + sdp_player).most_common(80)},
                    sdp_total=sum(sdp.values()) + sum(sdp_player.values())),
        room_rules=dict(texts=dict(rule_texts.most_common()), modes_by_files=dict(mode_files.most_common())),
        position_hints=dict(avatar_kinds=dict(avatar_kinds), avatar_numbers=dict(sorted(avatar_numbers.items())),
                            gk_declared_avatars=dict(gk_avatars.most_common(20)), chat_lines=dict(chat_lines)),
        announcements=dict(
            chat_relayed=chat_n,
            templates=[dict(t=t, n=n, files=ann_files[t]) for t, n in ann_tpl.most_common()
                       if ann_files[t] >= MIN_TEMPLATE_FILES][:200],
            percent_values=pct_summary),
    )
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "inventario.json").write_text(json.dumps(inventory, indent=1, ensure_ascii=False), encoding="utf-8")
    splits = dict(version="hx11-splits-1", reserved_groups=sorted(reserved),
                  sessions={s: dict(split=split_of[s], recordings=sorted(v)) for s, v in sorted(sessions.items())},
                  recordings={n: dict(split=rows[n]["split"], session=rows[n]["session"], groups=rows[n]["groups"],
                                      times=rows[n]["times"],
                                      stadium=public_stadium(max(rows[n]["minutes"],
                                                                 key=lambda s: rows[n]["minutes"][s]["total"], default="")),
                                      minutes=rows[n]["family_minutes"])
                              for n in ok})
    (out / "splits.json").write_text(json.dumps(splits, indent=1, ensure_ascii=False), encoding="utf-8")
    priv = Path(a.private)
    priv.mkdir(parents=True, exist_ok=True)
    (priv / "index.json").write_text(json.dumps({n: {k: v for k, v in r.items()} for n, r in rows.items()},
                                                indent=1, ensure_ascii=False, default=list), encoding="utf-8")
    (priv / "player_names.json").write_text(
        json.dumps({player_key(nm): nm for n in ok for nm in rows[n]["_names"]}, indent=0, ensure_ascii=False),
        encoding="utf-8")
    print(json.dumps(dict(files=inventory["files"]["status"], minutes=inventory["minutes"], sessions=len(sessions),
                          splits=by_split, reserved=sorted(reserved)), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
