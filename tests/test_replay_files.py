from tools.replay_files import duplicate_recordings
from tools import audit_bc_dataset as audit


def test_byte_identical_recs_use_stable_shorter_name(tmp_path):
    records = []
    for name, data in [("match (1).hbr2", b"same"), ("match.hbr2", b"same"), ("other.hbr2", b"different")]:
        path = tmp_path / name
        path.write_bytes(data)
        records.append({"name": name, "file": str(path), "stadium": "RS", "minutes": 2, "names": []})
    expected = {"match (1).hbr2": "match.hbr2"}
    assert duplicate_recordings(records) == expected
    assert duplicate_recordings(list(reversed(records))) == expected
    records.append({"name": "bad.hbr2", "error": "broken"})
    assert duplicate_recordings(records) == expected


def test_audit_distinguishes_duplicates_from_pending(tmp_path, monkeypatch):
    records = []
    for name in ("match (1).hbr2", "match.hbr2"):
        path = tmp_path / name
        path.write_bytes(b"same")
        records.append({"name": name, "file": str(path), "stadium": "RS", "minutes": 2, "names": []})
    monkeypatch.setattr(audit, "catalog_by_name", lambda: {"RS": {"script": None}})
    report = audit.summarize_folder(tmp_path, records)
    assert report["pending"] == ["match.hbr2"]
    assert len(report["excluded"]) == 1
    assert "duplicate" in report["excluded"][0]["reason"]
