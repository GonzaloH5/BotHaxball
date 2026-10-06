"""Índice y particiones del dataset X4 (grabaciones humanas 4v4 de Real Soccer).

Une los metadatos de `bridge/replay_meta.js` (sala, estadios, planteles con nombres, goles), el índice del
caché JSONL (`tools.rs4_jsonl_cache`: hash y estado de conversión) y el manifiesto de `tools.x4_ticks`
(ticks por mapa). Reglas:

* Duplicados exactos: mismo SHA-256 (los marca el caché JSONL).
* Duplicados del mismo partido grabado por dos personas: misma secuencia de goles (tiempo de reloj y
  marcador) y ≥ 6 nombres en común. Se conserva el primero por nombre.
* Grabaciones con bots (nombres `RL-*`): no son datos humanos; quedan fuera del dataset de imitación y de
  la referencia humana, pero sirven para conformidad del script y para medir latencia.
* Sesión: grabaciones del mismo día y la misma sala con menos de 45 min entre una y la siguiente. En una
  sala pública los jugadores habituales se repiten entre sesiones: las particiones separan partidos y
  sesiones, no jugadores (se informa el solapamiento de jugadores).
* Partición por sesión, estable (hash del nombre de la sesión), por familia de sala: ~15% prueba, ~10%
  desarrollo, resto entrenamiento.

Los nombres de jugadores sólo se guardan en `data/` (no versionado). El índice versionado usa claves
`p_<sha1[:10]>` del nombre.

  python -m tools.x4_index --meta 'data/meta/shard_*.jsonl' --out reports/x4
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

ROOT = Path(__file__).resolve().parent.parent
BOT_NAME = re.compile(r"^RL-", re.I)
SESSION_GAP_MIN = 45
TEST_FRAC, DEV_FRAC = 0.15, 0.10


def player_key(name):
    return "p_" + hashlib.sha1(name.strip().lower().encode("utf-8")).hexdigest()[:10]


def recording_time(name):
    """Fecha y hora del nombre del archivo (formatos vistos en el dataset) o None."""
    pats = [
        (r"SanguREC-(\d{1,2})-(\d{1,2})-(\d{4})-(\d{1,2})h(\d{2})m", lambda g: (g[2], g[1], g[0], g[3], g[4])),
        (r"(\d{4})-(\d{2})-(\d{2})-(\d{1,2})h(\d{2})m", lambda g: (g[0], g[1], g[2], g[3], g[4])),
        (r"(\d{4})-(\d{2})-(\d{2})T(\d{2})_(\d{2})", lambda g: (g[0], g[1], g[2], g[3], g[4])),
        (r"(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-\d{2}", lambda g: (g[0], g[1], g[2], g[3], g[4])),
        (r"HBReplay-(\d{2})-(\d{2})-(\d{4})-(\d{1,2})h(\d{2})m", lambda g: (g[2], g[1], g[0], g[3], g[4])),
    ]
    for pat, order in pats:
        m = re.search(pat, name)
        if m:
            y, mo, d, h, mi = (int(x) for x in order(m.groups()))
            try:
                return dt.datetime(y, mo, d, h, mi)
            except ValueError:
                return None
    return None


def room_family(room):
    room = room or ""
    if "SANGUCHITO" in room.upper():
        return "sanguchito"
    if "HAXARG" in room.upper():
        return "haxarg"
    if "MRHOST" in room.upper().replace(" ", ""):
        return "mrhost"
    return "otra"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta", default=str(ROOT / "data" / "meta" / "shard_*.jsonl"))
    ap.add_argument("--cache", default=str(ROOT / "data" / "rs4_jsonl" / "index.json"))
    ap.add_argument("--cache2k23", default=str(ROOT / "data" / "haxarg_jsonl" / "index.json"))
    ap.add_argument("--ticks", default=str(ROOT / "data" / "x4_ticks" / "manifest.json"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "x4"))
    a = ap.parse_args()
    metas = {}
    for path in sorted(glob.glob(a.meta)):
        for line in open(path, encoding="utf-8"):
            m = json.loads(line)
            metas[m["file"]] = m
    cache = {}
    for p in (a.cache, a.cache2k23):
        if Path(p).exists():
            cache.update(json.loads(Path(p).read_text(encoding="utf-8"))["recordings"])
    ticks = json.loads(Path(a.ticks).read_text(encoding="utf-8")) if Path(a.ticks).exists() else {}

    rows = {}
    for name, m in sorted(metas.items()):
        names = sorted({v.strip() for v in m["names"].values()})
        c = cache.get(name, {})
        goals = [(round(e["time"] or 0, 1), tuple(e["score"] or ())) for e in m["events"] if e["t"] == "goal"]
        played = collections.Counter()
        for k, v in m["played"].items():
            st, fmt = k.split("|")
            if fmt == "4v4":
                played[st] += v
        when = recording_time(name)
        rows[name] = dict(
            sha256=c.get("sha256"), status=c.get("status", "sin_convertir"), duplicate_of=c.get("duplicate_of"),
            room=m["room"], family=room_family(m["room"]), time=when.isoformat() if when else None,
            frames=m["frames"], minutes_4v4={k: round(v / 3600, 2) for k, v in played.items()},
            goals=len(goals), players=sorted(player_key(n) for n in names), n_players=len(names),
            bots=any(BOT_NAME.match(n) for n in names), _names=names, _goals=goals,
            ticks=ticks.get(name, {}).get("maps"))
    # duplicados del mismo partido (otro grabador)
    by_goals = collections.defaultdict(list)
    for name, r in rows.items():
        if r["status"] == "ok" and len(r["_goals"]) >= 2:
            by_goals[tuple(r["_goals"])].append(name)
    for names in by_goals.values():
        names.sort()
        for other in names[1:]:
            if len(set(rows[other]["_names"]) & set(rows[names[0]]["_names"])) >= 6:
                rows[other]["status"] = "duplicate_match"
                rows[other]["duplicate_of"] = names[0]
    # sesiones
    usable = [n for n, r in rows.items() if r["status"] == "ok"]
    by_room_day = collections.defaultdict(list)
    for n in usable:
        r = rows[n]
        t = dt.datetime.fromisoformat(r["time"]) if r["time"] else None
        day = t.date().isoformat() if t else "sin_fecha"
        by_room_day[(r["family"], r["room"] if r["family"] != "sanguchito" else "sangu", day)].append((t, n))
    for (fam, room, day), items in by_room_day.items():
        items.sort(key=lambda x: (x[0] or dt.datetime.min, x[1]))
        k, last = 0, None
        for t, n in items:
            if last is not None and (t is None or last is None or (t - last).total_seconds() > SESSION_GAP_MIN * 60):
                k += 1
            last = t
            rows[n]["session"] = f"{fam}:{day}:{k}" if t else f"{fam}:{re.sub(r'[^a-z0-9]+', '_', (room or n).lower())[:40]}"
    # particiones por sesión dentro de cada familia, estables por hash
    sessions = collections.defaultdict(list)
    for n in usable:
        sessions[rows[n]["session"]].append(n)
    split_of = {}
    for fam in sorted({rows[n]["family"] for n in usable}):
        ss = sorted((s for s in sessions if s.startswith(fam + ":")), key=lambda s: hashlib.sha256(s.encode()).hexdigest())
        total = sum(len(sessions[s]) for s in ss if not all(rows[n]["bots"] for n in sessions[s]))
        acc = collections.Counter()
        for s in ss:
            if all(rows[n]["bots"] for n in sessions[s]):
                split_of[s] = "bots"
                continue
            if acc["test"] < TEST_FRAC * total:
                split_of[s] = "test"
            elif acc["dev"] < DEV_FRAC * total:
                split_of[s] = "dev"
            else:
                split_of[s] = "train"
            acc[split_of[s]] += len(sessions[s])
    for n in usable:
        rows[n]["split"] = "bots" if rows[n]["bots"] else split_of[rows[n]["session"]]
    # resumen y solapamiento de jugadores
    summary = collections.defaultdict(lambda: collections.Counter())
    players_by_split = collections.defaultdict(set)
    for n in usable:
        r = rows[n]
        key = (r["family"], r["split"])
        summary[key]["recordings"] += 1
        for st, mins in r["minutes_4v4"].items():
            summary[key][f"min4v4:{st}"] += mins
        players_by_split[r["split"]] |= set(r["players"])
    overlap = {}
    for sp in ("dev", "test"):
        p = players_by_split.get(sp, set())
        overlap[sp] = dict(players=len(p), also_in_train=len(p & players_by_split.get("train", set())))
    status = collections.Counter(r["status"] for r in rows.values())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    public = {n: {k: v for k, v in r.items() if not k.startswith("_")} for n, r in rows.items()}
    splits = dict(version="x4-splits-1", rules=__doc__.strip().splitlines()[5:18], status=dict(status),
                  summary={f"{f}/{s}": {k: round(v, 1) for k, v in c.items()} for (f, s), c in sorted(summary.items())},
                  player_overlap=overlap,
                  sessions={s: dict(split=split_of.get(s), recordings=sorted(v)) for s, v in sorted(sessions.items())},
                  recordings={n: dict(split=r["split"], session=r["session"], family=r["family"], maps=r["ticks"],
                                      minutes_4v4=r["minutes_4v4"])
                              for n, r in public.items() if r["status"] == "ok"})
    (out / "index.json").write_text(json.dumps(public, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "splits.json").write_text(json.dumps(splits, indent=1, ensure_ascii=False), encoding="utf-8")
    # nombres reales sólo en data/
    names_out = ROOT / "data" / "x4_player_names.json"
    names_out.write_text(json.dumps({player_key(nm): nm for r in rows.values() for nm in r["_names"]}, indent=0,
                                    ensure_ascii=False), encoding="utf-8")
    print(json.dumps(dict(status=status, summary=splits["summary"], overlap=overlap, sessions=len(sessions)),
                     indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
