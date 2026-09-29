import json
import subprocess
from pathlib import Path

import pytest

from tools import fetch_mrhost_replays as fetch


def flight_json(value):
    return b'0:{"a":"$@1"}\n1:' + json.dumps(value).encode() + b"\n"


def flight_binary(data):
    return b'0:{"a":"$@1"}\n2:o' + f"{len(data):x}".encode() + b"," + data + b'1:"$2"\n'


def test_discovers_both_action_ids_from_minified_script():
    listing = "a" * 42
    download = "b" * 42
    script = (
        f'createServerReference)("{listing}",x,y,"fetchPublicReplays");'
        f'createServerReference)("{download}",x,y,"downloadReplayFileAction");'
    )
    assert fetch.discover_action_ids_from_scripts([script]) == {
        "fetchPublicReplays": listing,
        "downloadReplayFileAction": download,
    }


def test_parses_json_and_length_prefixed_binary():
    page = {"replays": [{"replayId": "123456789abc"}], "totalPages": 2}
    assert fetch.parse_flight_json(flight_json(page)) == page
    replay = b"HBR2\x00\x01\xff\n1:not-a-flight-row"
    assert fetch.parse_flight_binary(flight_binary(replay)) == replay


@pytest.mark.parametrize("payload", [b"missing", flight_binary(b"NOPE")])
def test_rejects_invalid_binary(payload):
    with pytest.raises(fetch.MrReplayError):
        fetch.parse_flight_binary(payload)


def test_pagination_forwards_filters_and_honors_limit():
    class Client:
        def __init__(self):
            self.calls = []

        def list_page(self, filters, page):
            self.calls.append(filters.request(page))
            return {"replays": [{"replayId": f"{page:012x}"}] * 2, "totalPages": 3}

    client = Client()
    filters = fetch.Filters("3v3", "oldest", 60, 600, "uy", "sa")
    rows = fetch.collect_replays(client, filters, max_results=3)
    assert len(rows) == 3
    assert [call["page"] for call in client.calls] == [1, 2]
    assert client.calls[0] == {
        "page": 1, "sort": "oldest", "search": "3v3", "minDuration": 60,
        "maxDuration": 600, "country": "UY", "continent": "SA",
    }


def test_safe_name_and_catalog_aliases():
    name = fetch.safe_replay_name("../Final: A/B?.hbr2", "abcdef123456")
    assert name == "Final_A_B__abcdef123456.hbr2"
    names = fetch.stadium_names_for_folder("futsalx3")
    assert fetch.normalize_stadium("Futsal X3 by Bazinga") in names
    af = fetch.stadium_names_for_folder("futsalx3", stadium="af_futsalx3")
    assert fetch.normalize_stadium("AF Official 3v3 by Vitão ®") in af
    assert fetch.normalize_stadium("AF Official 3v3 by Vit�o �") in af
    with pytest.raises(ValueError):
        fetch.stadium_names_for_folder("futsalx3", stadium="futsal_x4")
    with pytest.raises(ValueError):
        fetch.stadium_names_for_folder("unknown")


def test_process_is_atomic_deduplicated_and_resumable(tmp_path):
    replay = b"HBR2 data"
    candidates = [
        {"replayId": "aaaaaaaaaaaa", "originalFileName": "same.hbr2"},
        {"replayId": "bbbbbbbbbbbb", "originalFileName": "copy.hbr2"},
    ]

    class Client:
        calls = 0

        def download(self, replay_id):
            self.calls += 1
            return replay

    client = Client()
    inspector = lambda path, size: {
        "stadiums": ["Futsal X3 by Bazinga"], "teamSizeMatched": True,
        "teamSizeStadiums": ["Futsal X3 by Bazinga"],
    }
    filters = fetch.Filters("3v3")
    allowed = {fetch.normalize_stadium("Futsal X3 by Bazinga")}
    counts = fetch.process_candidates(client, candidates, filters, tmp_path, allowed, 3, inspector)
    assert counts == {"downloaded": 1, "duplicate": 1}
    assert len(list(tmp_path.glob("*.hbr2"))) == 1
    assert not list(tmp_path.glob("*.part"))
    assert client.calls == 2

    counts = fetch.process_candidates(client, candidates, filters, tmp_path, allowed, 3, inspector)
    assert counts == {"existing": 2}
    assert client.calls == 2
    rows = [json.loads(line) for line in (tmp_path / fetch.MANIFEST_NAME).read_text(encoding="utf-8").splitlines()]
    assert [row["status"] for row in rows] == ["downloaded", "duplicate"]


@pytest.mark.parametrize(
    ("inspection", "status"),
    [
        ({"stadiums": ["Other"], "teamSizeMatched": True, "teamSizeStadiums": ["Other"]},
         "wrong_stadium"),
        ({"stadiums": ["Futsal X3 by Bazinga"], "teamSizeMatched": False, "teamSizeStadiums": []},
         "wrong_team_size"),
    ],
)
def test_rejects_wrong_stadium_or_team_size(tmp_path, inspection, status):
    class Client:
        def download(self, replay_id):
            return b"HBR2 data"

    candidate = {"replayId": "aaaaaaaaaaaa", "originalFileName": "match.hbr2"}
    allowed = {fetch.normalize_stadium("Futsal X3 by Bazinga")}
    counts = fetch.process_candidates(Client(), [candidate], fetch.Filters("3v3"), tmp_path, allowed, 3,
                                      lambda path, size: inspection)
    assert counts == {status: 1}
    assert not list(tmp_path.glob("*.hbr2"))


def test_node_inspection_tracks_team_size_per_stadium():
    script = Path(fetch.ROOT / "bridge" / "inspect_replay.js")
    js = f"""
const {{ Inspection }} = require({json.dumps(str(script))});
const x = new Inspection(3);
x.stadium('Training');
x.players([{{team: 1}}, {{team: 1}}, {{team: 2}}, {{team: 2}}]);
x.stadium('Futsal X3 by Bazinga');
x.players([1,1,1,2,2,2].map(team => ({{team}})));
process.stdout.write(JSON.stringify(x.result()));
"""
    result = subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True)
    inspection = json.loads(result.stdout)
    assert inspection["teamSizeMatched"] is True
    assert inspection["teamSizeStadiums"] == ["Futsal X3 by Bazinga"]
