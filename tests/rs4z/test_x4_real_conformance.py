"""Regresión con una grabación real de SANGUCHITO RS X4 (versionada en replays_real/).

`SanguREC-4-10-2026-22h05m` tiene un córner con cuatro patadas del ejecutor hacia la cancha en x pero hacia
afuera en y, que la sala ignoró, y laterales cedidos al rival por corrida. Se convierte con el motor original
(node-haxball) y se corre la conformidad de saques: el simulador tiene que reproducir tipo, ejecutor, punto,
liberación y cesión de todos los saques (`reports/x4/conformance_x4.md`).
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
REPLAY = ROOT / "replays_real" / "stadiums" / "rsx4" / "SanguREC-4-10-2026-22h05m.hbr2"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None or not (ROOT / "bridge" / "node_modules" / "node-haxball").exists()
    or not REPLAY.exists(), reason="hace falta node, bridge/node_modules y la grabación")


@pytest.fixture(scope="module")
def jsonl(tmp_path_factory):
    src = tmp_path_factory.mktemp("src")
    out = tmp_path_factory.mktemp("cache")
    shutil.copy(REPLAY, src / REPLAY.name)
    subprocess.run([sys.executable, "-m", "tools.rs4_jsonl_cache", "--source", str(src), "--out", str(out)],
                   cwd=ROOT, check=True, capture_output=True, timeout=600)
    files = list(out.glob("*/*.jsonl.gz"))
    assert len(files) == 1
    return files[0]


def test_restarts_match_the_room(jsonl):
    from tools.rs4z_restart_conformance import audit_file
    res = audit_file(str(jsonl), "sanguchito_rs_x4")
    rows = res["rows"]
    assert len(rows) >= 15
    started = [r for r in rows if r["sim_start_off"] is not None]
    assert all(r["sim_kind"] == r["real_kind"] and r["taker_ok"] for r in started)
    executed = [r for r in rows if not r.get("skipped_execute")]
    ends = [r["sim_end"] == r["real_end"] for r in executed]
    assert sum(ends) == len(ends), [(r["start"], r["real_kind"], r["real_end"], r["sim_end"]) for r in executed
                                    if r["sim_end"] != r["real_end"]]
    # el córner de las cuatro patadas ignoradas (19859) se libera en el mismo tick que en la sala (+1 por
    # la convención del evento de patada), no en la primera patada
    corner = next(r for r in rows if r["start"] == 19859)
    assert corner["real_kind"] == 2 and corner["end_off"] == 1
    # rombo del saque inicial
    f = sorted(res["formation"])
    assert f and f[len(f) // 2] < 1e-3
