import json

from tools.x3futsal_audit import STADIUM, is_complete_recording, make_splits, stadium_identity, target_spans


def test_stadium_identity_ignores_name_and_key_order(tmp_path):
    source = json.loads(STADIUM.read_text(encoding="utf-8"))
    renamed = {key: value for key, value in reversed(list(source.items()))}
    renamed["name"] = "renamed"
    first = tmp_path / "first.hbs"
    second = tmp_path / "second.hbs"
    first.write_text(json.dumps(source), encoding="utf-8")
    second.write_text(json.dumps(renamed), encoding="utf-8")
    assert stadium_identity(first) == stadium_identity(second)


def test_target_spans_match_content_and_stop_at_next_stadium(tmp_path):
    target_id = stadium_identity(STADIUM)
    other = json.loads(STADIUM.read_text(encoding="utf-8"))
    other["width"] += 1
    (tmp_path / "other.hbs").write_text(json.dumps(other), encoding="utf-8")
    spans = target_spans(
        [{"frame": 0, "name": "Futsal", "file": "target.hbs", "identity": target_id},
         {"frame": 10, "name": "Other", "file": "other.hbs", "identity": stadium_identity(tmp_path / "other.hbs")},
         {"frame": 20, "name": "renamed Futsal", "file": "target.hbs", "identity": target_id}],
        target_id,
        max_frame=29,
        last_frame=29,
    )
    assert [(span["start"], span["end"]) for span in spans] == [(0, 10), (20, 30)]


def test_splits_are_deterministic_and_keep_recordings_whole():
    recordings = {
        f"r{i}.hbr2": {"status": "ok", "complete": True, "targetTicks": 100,
                       "teamCounts": {"3v3": 100}, "room": f"room-{i}"}
        for i in range(20)
    }
    recordings["duplicate.hbr2"] = {"status": "duplicate", "complete": True, "targetTicks": 100}
    recordings["room-a-2.hbr2"] = {"status": "ok", "complete": True, "targetTicks": 100,
                                    "teamCounts": {"3v3": 100}, "room": "ROOM-0"}
    first = make_splits(recordings, seed=7)
    assert first == make_splits(recordings, seed=7)
    assert len(first["assignments"]) == 21
    assert set(first["assignments"].values()) == {"train", "dev", "test"}
    assert first["assignments"]["r0.hbr2"] == first["assignments"]["room-a-2.hbr2"]
    assert first["unit"] == "normalized room name (all recordings in a room stay together)"
    assert sum(first["recordingCounts"].values()) == 21


def test_end_callback_marks_complete_even_if_header_frame_count_differs():
    assert is_complete_recording("replay.hbr2: 100 ticks (fin del replay)", 100, 150)
    assert not is_complete_recording("replay.hbr2: 100 ticks (timeout de seguridad)", 100, 100)
    assert is_complete_recording("replay.hbr2: 100 ticks (el replay dejó de avanzar)", 98, 100)
