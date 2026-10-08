import datetime as dt

from tools.hx11_inventory import groups_of, percent_value, roster_minutes, template, time_candidates


def test_time_candidates_reads_both_date_orders():
    assert time_candidates("HaxEleven-9-14-2026-10h29m.hbr2") == [dt.datetime(2026, 9, 14, 10, 29)]
    assert time_candidates("BFC_BOL_HaxEleven-14-9-2026-22h52m.hbr2") == [dt.datetime(2026, 9, 14, 22, 52)]
    assert time_candidates("HaxEleven-5-10-2026-23h35m.hbr2") == [dt.datetime(2026, 5, 10, 23, 35),
                                                                 dt.datetime(2026, 10, 5, 23, 35)]
    assert time_candidates("otro.hbr2") == []


def test_groups_from_file_or_room():
    assert groups_of("AV_KJ_HaxEleven-1-10-2026-23h32m.hbr2", "") == ("AV", "KJ")
    assert groups_of("HaxEleven-1-10-2026-1h51m.hbr2", "⪦ 🏆 [H11] 🔴 X VS Y 🟡 ZFC vs BOV ⪧") == ("BOV", "ZFC")
    assert groups_of("HaxEleven-1-10-2026-1h51m.hbr2", "HaxEleven pública") == ()


def test_time_candidates_drop_future_readings():
    assert time_candidates("HaxEleven-5-11-2026-23h35m.hbr2", dt.datetime(2026, 10, 8)) == [
        dt.datetime(2026, 5, 11, 23, 35)]


def test_template_hides_names_ids_numbers_and_skips_chat():
    names = {"deku", "st"}
    assert template("[👔] deku: go gordo", names) is None
    assert template("⟮ 👤 ᵀᴼᴾ 148 ⟯ deku (alias): tipo saben 100%", names) is None
    assert template("[✅] deku firmó correctamente [837206287422980116].", names) == "[✅] <J> firmó correctamente <ID>."
    # un nombre corto no se reemplaza dentro de otra palabra
    assert template("[🎤] SERÁ ESTE UNO DE LOS MEJORES GOLES", names) == "[🎤] SERÁ ESTE UNO DE LOS MEJORES GOLES"
    assert template("[⚽] Potencia del corner: 91%", names) == "[⚽] Potencia del corner: #%"


def test_percent_value():
    assert percent_value("[⚽] Potencia del corner: 91%") == ("Potencia del corner", 91.0)
    assert percent_value("[🎉] Powershot utilizado. 33%") == ("Powershot utilizado.", 33.0)
    assert percent_value("[⚽] Potencia del tiro libre: 89%") == ("Potencia del tiro libre", 89.0)
    assert percent_value("[🐐] sin porcentaje") is None
    assert percent_value("[👔] alguien: le pegué al 100%") is None


def test_roster_minutes_levels():
    m = roster_minutes({"Mapa|11v11": 3600, "Mapa|11v10": 3600, "Mapa|10v9": 3600, "Mapa|8v9": 3600})
    assert m["Mapa"] == {"total": 4.0, "11v11": 1.0, "min10": 2.0, "min9": 3.0}
