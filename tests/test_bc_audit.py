import hashlib

import numpy as np
import pytest

from tools import audit_bc_dataset as audit


def test_coverage_distinguishes_pending_excluded_and_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "catalog_by_name", lambda: {"Futsal": {"script": None}})
    name = "match.npz"
    np.savez(tmp_path / name, act=np.array([0, 9, 3]), act_lag6=np.array([1, 9, 4]),
             T=np.array([3, 3, 4]), obs_0=np.zeros((3, 111)))
    rows = [{"name": "match.hbr2", "stadium": "Futsal", "minutes": 8, "names": []},
            {"name": "new.hbr2", "stadium": "Futsal", "minutes": 9, "names": []},
            {"name": "bot.hbr2", "stadium": "Futsal", "minutes": 2, "names": ["[BOT] ours"]},
            {"name": "other.hbr2", "stadium": "Other", "minutes": 4, "names": []},
            {"name": "invalid.hbr2", "error": "bad data"}]
    report = audit.summarize_folder(tmp_path, rows)
    assert report["minutes"] == 23 and report["recordings"] == 5
    assert report["pending"] == ["new.hbr2"] and len(report["excluded"]) == 3
    assert report["datasets"] == 1 and report["samples"] == 3
    assert report["team_sizes"] == {3: 2, 4: 1}
    val = int(hashlib.md5(name.encode()).hexdigest(), 16) % 1000 < 120
    assert report["val_replays"] == int(val)
    assert report["val_samples"] == 3 * int(val)


@pytest.mark.parametrize("failure", ["nan", "bad_action", "missing_lag"])
def test_audit_rejects_invalid_shards(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(audit, "catalog_by_name", lambda: {})
    fields = {"act": np.array([0]), "act_lag6": np.array([0]), "T": np.array([3]),
              "obs_0": np.zeros((1, 111))}
    if failure == "nan":
        fields["obs_0"][0, 0] = np.nan
    elif failure == "bad_action":
        fields["act"][0] = 18
    else:
        del fields["act_lag6"]
    np.savez(tmp_path / "bad.npz", **fields)
    with pytest.raises(ValueError):
        audit.summarize_folder(tmp_path, [])
