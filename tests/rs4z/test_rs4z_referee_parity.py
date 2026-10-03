"""El kernel RS4-Z en modo v1 reproduce el árbitro rs_one_v1 (numpy) tick a tick."""
from env.rs4z.parity import run


def test_v1_one_tick_parity_covers_every_restart():
    result = run(64, 2500, seed=11, tol=1e-9, resync=True)
    assert result["ok"], result
    ev = result["events"]
    assert ev["goals"] > 0 and ev["outs"] > 50
    assert all(count > 0 for count in ev["restarts"].values()), ev


def test_v1_free_run_keeps_discrete_events_until_chaos():
    # Sin resincronizar, el redondeo numpy/numba (~1e-17) crece por el caos de las colisiones;
    # los eventos discretos deben coincidir mientras el estado continuo siga a <1e-6.
    result = run(32, 800, seed=12, tol=1e-6, resync=False)
    if not result["ok"]:
        assert not result["discrete"], result
        assert result["tick"] > 300, result
