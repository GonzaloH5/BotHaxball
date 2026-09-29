"""Descarga replays públicos de MrREPLAY y los valida antes de incorporarlos.

Ejemplo:
    python -m tools.fetch_mrhost_replays --query 3v3 --team-size 3 --folder futsalx3

El catálogo público usa acciones internas de la web de MrREPLAY, no la API v1
autenticada. Los identificadores de esas acciones se descubren en cada ejecución.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import subprocess
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, build_opener

ROOT = Path(__file__).resolve().parent.parent
REPLAY_ROOT = ROOT / "replays_real" / "stadiums"
BASE_URL = "https://replay.mrhosthaxball.com"
MANIFEST_NAME = "_mrhost_manifest.jsonl"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 Chrome/140.0 Safari/537.36"
)
SORTS = ("newest", "oldest", "longest", "shortest", "most-players", "most-liked", "most-viewed")

# Las carpetas son una decisión del dataset, no un dato que exponga MrREPLAY.
FOLDER_STADIUM_KEYS = {
    "bigx3": ("aha_big", "haxarg_big"),
    "futsalx1": ("sanguchito_x1", "futsal_x1x2"),
    "futsalx3": ("futsalx3", "af_futsalx3"),
    "futsalx4": ("futsal_x4",),
    "futsalx7": ("liga_x7", "lhl_x7", "futsalx7"),
    "rfx7": ("real_futsal_x7",),
    "rsx4": ("rs_one", "rsx4"),
    "rsx6": ("jjrs_x6", "x6"),
}
STADIUM_ALIASES = {
    "futsalx3": ("Futsal x3 by Bazinga", "Futsal X3 by Bazinga"),
    # node-haxball puede reemplazar caracteres no ASCII del nombre embebido.
    "af_futsalx3": ("AF Official 3v3 by Vitão ®", "AF Official 3v3 by Vit�o �"),
    "futsal_x4": ("Futsal x4 ; By Bazinga!", "Futsal X4 by Bazinga"),
}


class MrReplayError(RuntimeError):
    pass


def normalize_stadium(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(value.split())


def stadium_names_for_folder(folder: str, root: Path = ROOT, stadium: str | None = None) -> set[str]:
    keys = FOLDER_STADIUM_KEYS.get(folder)
    if not keys:
        choices = ", ".join(sorted(FOLDER_STADIUM_KEYS))
        raise ValueError(f"modalidad desconocida '{folder}'; opciones: {choices}")
    if stadium is not None:
        if stadium not in keys:
            choices = ", ".join(keys)
            raise ValueError(f"el estadio '{stadium}' no pertenece a {folder}; opciones: {choices}")
        keys = (stadium,)
    names: set[str] = set()
    for key in keys:
        path = root / "stadiums" / f"{key}.hbs"
        try:
            stadium = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"no se pudo leer el estadio del catálogo: {path}") from exc
        names.add(normalize_stadium(stadium.get("name", key)))
        names.update(normalize_stadium(alias) for alias in STADIUM_ALIASES.get(key, ()))
    return names


def safe_replay_name(original: str | None, replay_id: str) -> str:
    # El nombre remoto nunca se usa como ruta. Se aplanan también ambos tipos
    # de separador para que un nombre malicioso no pueda elegir el directorio.
    flat = (original or "replay").replace("/", "_").replace("\\", "_")
    stem = Path(flat).stem
    stem = unicodedata.normalize("NFKC", stem)
    stem = re.sub(r"[^\w.-]+", "_", stem, flags=re.UNICODE).strip("._") or "replay"
    return f"{stem[:120]}__{replay_id}.hbr2"


def discover_action_ids_from_scripts(scripts: Iterable[str]) -> dict[str, str]:
    wanted = ("fetchPublicReplays", "downloadReplayFileAction")
    found: dict[str, str] = {}
    for script in scripts:
        for name in wanted:
            if name in found:
                continue
            # Next no garantiza una longitud pública para los IDs de acción
            # (el despliegue actual usa 42 caracteres).
            pattern = rf'"([0-9a-f]{{32,64}})"[^;]{{0,350}}"{re.escape(name)}"'
            match = re.search(pattern, script)
            if match:
                found[name] = match.group(1)
    return found


def parse_flight_json(payload: bytes) -> dict:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise MrReplayError("respuesta JSON de React Flight inválida") from exc
    for line in text.splitlines():
        _, separator, value = line.partition(":")
        if not separator or not value.startswith("{"):
            continue
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, dict) and "replays" in decoded:
            return decoded
        if isinstance(decoded, dict) and decoded.get("error"):
            raise MrReplayError(str(decoded["error"]))
    raise MrReplayError("MrREPLAY no devolvió un listado reconocible")


def parse_flight_binary(payload: bytes) -> bytes:
    match = re.search(rb"(?:^|\n)[0-9a-f]+:o([0-9a-f]+),", payload)
    if not match:
        raise MrReplayError("MrREPLAY no devolvió un bloque binario reconocible")
    size = int(match.group(1), 16)
    start = match.end()
    data = payload[start:start + size]
    if len(data) != size:
        raise MrReplayError(f"descarga truncada: se esperaban {size} bytes y llegaron {len(data)}")
    if not data.startswith(b"HBR2"):
        raise MrReplayError("la descarga no tiene la cabecera HBR2")
    return data


@dataclass
class Filters:
    query: str
    sort: str = "newest"
    min_duration: float = 0
    max_duration: float = 0
    country: str = ""
    continent: str = ""

    def request(self, page: int) -> dict:
        return {
            "page": page,
            "sort": self.sort,
            "search": self.query,
            "minDuration": self.min_duration,
            "maxDuration": self.max_duration,
            "country": self.country.upper(),
            "continent": self.continent.upper(),
        }


class MrReplayClient:
    def __init__(self, base_url: str = BASE_URL, retries: int = 3, timeout: float = 60):
        self.base_url = base_url.rstrip("/")
        self.retries = retries
        self.timeout = timeout
        self.opener = build_opener()
        self.actions: dict[str, str] = {}

    def _request(self, path: str, *, data: bytes | None = None, headers: dict | None = None) -> bytes:
        url = urljoin(self.base_url + "/", path.lstrip("/"))
        request_headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
        request_headers.update(headers or {})
        request = Request(url, data=data, headers=request_headers, method="POST" if data is not None else "GET")
        for attempt in range(self.retries + 1):
            try:
                with self.opener.open(request, timeout=self.timeout) as response:
                    return response.read()
            except HTTPError as exc:
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if not retryable or attempt == self.retries:
                    detail = exc.read(500).decode("utf-8", "replace")
                    raise MrReplayError(f"HTTP {exc.code} en {url}: {detail}") from exc
            except (URLError, TimeoutError, OSError) as exc:
                if attempt == self.retries:
                    raise MrReplayError(f"falló la conexión con {url}: {exc}") from exc
            time.sleep(min(2 ** attempt, 8))
        raise AssertionError("bucle de reintentos incompleto")

    def discover_actions(self, force: bool = False) -> dict[str, str]:
        if self.actions and not force:
            return self.actions
        page = self._request("/replays").decode("utf-8", "replace")
        sources = [html.unescape(src) for src in re.findall(r'<script[^>]+src="([^"]+)"', page)]
        scripts = []
        for source in sources:
            if "/_next/static/chunks/" not in source:
                continue
            scripts.append(self._request(urljoin(self.base_url, source)).decode("utf-8", "replace"))
            found = discover_action_ids_from_scripts(scripts)
            if len(found) == 2:
                self.actions = found
                return found
        found = discover_action_ids_from_scripts(scripts)
        missing = sorted({"fetchPublicReplays", "downloadReplayFileAction"} - set(found))
        raise MrReplayError("no se encontraron las acciones públicas: " + ", ".join(missing))

    def _action(self, name: str, arguments: list) -> bytes:
        last_error: Exception | None = None
        for refresh in (False, True):
            try:
                action = self.discover_actions(force=refresh)
                return self._request(
                    "/replays",
                    data=json.dumps(arguments, separators=(",", ":")).encode(),
                    headers={
                        "Accept": "text/x-component",
                        "Content-Type": "text/plain;charset=UTF-8",
                        "Next-Action": action[name],
                        "Origin": self.base_url,
                        "Referer": self.base_url + "/replays",
                    },
                )
            except (KeyError, MrReplayError) as exc:
                last_error = exc
                self.actions = {}
        raise MrReplayError(f"la acción {name} dejó de ser válida: {last_error}") from last_error

    def _parsed_action(self, name: str, arguments: list, parser: Callable[[bytes], dict | bytes]):
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                return parser(self._action(name, arguments))
            except MrReplayError as exc:
                last_error = exc
                self.actions = {}
                if attempt == 0:
                    continue
        raise MrReplayError(f"respuesta inválida de {name} después de redescubrir la acción: {last_error}")

    def list_page(self, filters: Filters, page: int) -> dict:
        return self._parsed_action("fetchPublicReplays", [filters.request(page)], parse_flight_json)

    def download(self, replay_id: str) -> bytes:
        return self._parsed_action("downloadReplayFileAction", [replay_id], parse_flight_binary)


def collect_replays(client: MrReplayClient, filters: Filters, max_results: int | None = None) -> list[dict]:
    replays: list[dict] = []
    page = 1
    total_pages = 1
    while page <= total_pages and (max_results is None or len(replays) < max_results):
        result = client.list_page(filters, page)
        total_pages = max(1, int(result.get("totalPages", 1)))
        rows = result.get("replays")
        if not isinstance(rows, list):
            raise MrReplayError(f"la página {page} no contiene una lista de replays")
        remaining = None if max_results is None else max_results - len(replays)
        replays.extend(rows if remaining is None else rows[:remaining])
        page += 1
    return replays


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def emit(message: str, *, error: bool = False) -> None:
    """Imprime incluso si una consola Windows heredó una página sin emoji."""
    stream = sys.stderr if error else sys.stdout
    encoding = stream.encoding or "utf-8"
    safe = str(message).encode(encoding, "replace").decode(encoding)
    print(safe, file=stream)


def existing_hashes(destination: Path) -> dict[str, Path]:
    hashes: dict[str, Path] = {}
    for path in destination.glob("*.hbr2"):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes.setdefault(digest.hexdigest(), path)
    return hashes


def inspect_replay(path: Path, team_size: int | None) -> dict:
    command = ["node", str(ROOT / "bridge" / "inspect_replay.js"), str(path)]
    if team_size is not None:
        command += ["--team-size", str(team_size)]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=180)
    if result.returncode:
        raise MrReplayError(result.stderr.strip()[-2000:] or "el inspector rechazó el replay")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise MrReplayError("el inspector no devolvió JSON válido") from exc


class Manifest:
    TERMINAL = {"downloaded", "existing", "duplicate", "wrong_stadium", "wrong_team_size"}

    def __init__(self, path: Path):
        self.path = path
        self.rows: dict[tuple[str, str, int | None, str | None], dict] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                    key = (row["replayId"], row.get("query", ""), row.get("teamSize"),
                           row.get("stadiumFilter"))
                    self.rows[key] = row
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue

    def completed(self, replay_id: str, query: str, team_size: int | None, stadium: str | None) -> bool:
        row = self.rows.get((replay_id, query, team_size, stadium))
        return bool(row and row.get("status") in self.TERMINAL)

    def append(self, row: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        key = (row["replayId"], row.get("query", ""), row.get("teamSize"), row.get("stadiumFilter"))
        self.rows[key] = row


def manifest_row(candidate: dict, filters: Filters, team_size: int | None, status: str,
                 stadium_filter: str | None = None, **extra) -> dict:
    keep = ("title", "originalFileName", "roomName", "duration", "playerCount", "matchCount", "createdAt")
    row = {key: candidate.get(key) for key in keep}
    row.update({
        "replayId": candidate["replayId"],
        "query": filters.query,
        "teamSize": team_size,
        "stadiumFilter": stadium_filter,
        "status": status,
        "recordedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    row.update(extra)
    return row


def process_candidates(
    client: MrReplayClient,
    candidates: list[dict],
    filters: Filters,
    destination: Path,
    allowed_stadiums: set[str],
    team_size: int | None,
    inspector: Callable[[Path, int | None], dict] = inspect_replay,
    stadium_filter: str | None = None,
) -> dict[str, int]:
    destination.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(destination / MANIFEST_NAME)
    hashes = existing_hashes(destination)
    counts: dict[str, int] = {}
    for index, candidate in enumerate(candidates, 1):
        replay_id = candidate.get("replayId")
        if not isinstance(replay_id, str) or not re.fullmatch(r"[0-9a-f]{12}", replay_id):
            emit(f"[{index}/{len(candidates)}] candidato sin replayId válido", error=True)
            continue
        if manifest.completed(replay_id, filters.query, team_size, stadium_filter):
            counts["existing"] = counts.get("existing", 0) + 1
            emit(f"[{index}/{len(candidates)}] {replay_id}: ya procesado")
            continue
        name = safe_replay_name(candidate.get("originalFileName"), replay_id)
        final_path = destination / name
        if final_path.exists():
            digest = hashlib.sha256(final_path.read_bytes()).hexdigest()
            manifest.append(manifest_row(candidate, filters, team_size, "existing", stadium_filter,
                                         file=name, sha256=digest))
            hashes.setdefault(digest, final_path)
            counts["existing"] = counts.get("existing", 0) + 1
            emit(f"[{index}/{len(candidates)}] {name}: ya existe")
            continue
        part = destination / f".{replay_id}.part"
        status = "failed"
        try:
            data = client.download(replay_id)
            with part.open("wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            digest = sha256_bytes(data)
            duplicate = hashes.get(digest)
            if duplicate is not None:
                status = "duplicate"
                manifest.append(manifest_row(candidate, filters, team_size, status, stadium_filter, sha256=digest,
                                             duplicateOf=duplicate.name))
                emit(f"[{index}/{len(candidates)}] {name}: duplicado de {duplicate.name}")
                counts[status] = counts.get(status, 0) + 1
                continue
            inspection = inspector(part, team_size)
            stadiums = [str(value) for value in inspection.get("stadiums", [])]
            relevant = inspection.get("teamSizeStadiums", []) if team_size is not None else stadiums
            if team_size is not None and not inspection.get("teamSizeMatched", False):
                status = "wrong_team_size"
                manifest.append(manifest_row(candidate, filters, team_size, status, stadium_filter, sha256=digest,
                                             stadiums=stadiums))
                emit(f"[{index}/{len(candidates)}] {name}: sin tramo {team_size}v{team_size}")
                counts[status] = counts.get(status, 0) + 1
                continue
            if not any(normalize_stadium(str(value)) in allowed_stadiums for value in relevant):
                status = "wrong_stadium"
                manifest.append(manifest_row(candidate, filters, team_size, status, stadium_filter, sha256=digest,
                                             stadiums=stadiums))
                emit(f"[{index}/{len(candidates)}] {name}: estadio incompatible ({', '.join(stadiums)})")
                counts[status] = counts.get(status, 0) + 1
                continue
            os.replace(part, final_path)
            hashes[digest] = final_path
            status = "downloaded"
            manifest.append(manifest_row(candidate, filters, team_size, status, stadium_filter, file=name, sha256=digest,
                                         stadiums=stadiums))
            emit(f"[{index}/{len(candidates)}] {name}: guardado")
        except Exception as exc:
            manifest.append(manifest_row(candidate, filters, team_size, "failed", stadium_filter, error=str(exc)))
            emit(f"[{index}/{len(candidates)}] {name}: ERROR: {exc}", error=True)
        finally:
            part.unlink(missing_ok=True)
        counts[status] = counts.get(status, 0) + 1
    return counts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Descarga replays públicos de MrREPLAY")
    parser.add_argument("--query", required=True, help="texto de la búsqueda pública, por ejemplo 3v3")
    parser.add_argument("--folder", required=True, choices=sorted(FOLDER_STADIUM_KEYS), help="modalidad destino")
    parser.add_argument("--stadium", default=None,
                        help="limita la descarga a una clave de estadio de la modalidad, por ejemplo af_futsalx3")
    parser.add_argument("--team-size", type=int, default=None, help="exige al menos un tramo NvN")
    parser.add_argument("--min-duration", type=float, default=0, help="duración mínima en segundos")
    parser.add_argument("--max-duration", type=float, default=0, help="duración máxima en segundos; 0 sin límite")
    parser.add_argument("--country", default="", help="código ISO de país, por ejemplo UY")
    parser.add_argument("--continent", default="", help="código de continente, por ejemplo SA")
    parser.add_argument("--sort", choices=SORTS, default="newest")
    parser.add_argument("--max-results", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="lista candidatos sin descargar")
    args = parser.parse_args(argv)
    if args.team_size is not None and args.team_size < 1:
        parser.error("--team-size debe ser mayor que cero")
    if args.max_results is not None and args.max_results < 1:
        parser.error("--max-results debe ser mayor que cero")
    if args.min_duration < 0 or args.max_duration < 0:
        parser.error("las duraciones no pueden ser negativas")
    if args.max_duration and args.max_duration < args.min_duration:
        parser.error("--max-duration no puede ser menor que --min-duration")
    return args


def main(argv: list[str] | None = None) -> int:
    # Windows puede heredar cp1252 aunque títulos y salas contengan emoji.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)
    filters = Filters(args.query, args.sort, args.min_duration, args.max_duration, args.country, args.continent)
    try:
        allowed = stadium_names_for_folder(args.folder, stadium=args.stadium)
        client = MrReplayClient()
        candidates = collect_replays(client, filters, args.max_results)
        print(f"MrREPLAY devolvió {len(candidates)} candidato(s) para {args.query!r}")
        if args.dry_run:
            for row in candidates:
                print(f"{row.get('replayId')}  {row.get('duration', 0):7.1f}s  "
                      f"{row.get('roomName') or row.get('title') or ''}")
            return 0
        destination = REPLAY_ROOT / args.folder
        counts = process_candidates(client, candidates, filters, destination, allowed, args.team_size,
                                    stadium_filter=args.stadium)
        summary = ", ".join(f"{key}={value}" for key, value in sorted(counts.items())) or "sin cambios"
        print(f"Resumen: {summary}")
        print(f"Destino: {destination}")
        return 1 if counts.get("failed") else 0
    except (MrReplayError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
