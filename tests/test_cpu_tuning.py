from tools.tune_cpu import tune


def test_sweep_changes_one_axis_then_confirms_in_reverse_order():
    calls = []

    def measure(t, n, phase):
        calls.append((t, n, phase))
        return dict(torch_threads=t, numba_threads=n,
                    steps_per_second=1000 - 10 * abs(n - 6) - abs(t - 4))

    baseline, best, confirmed, repeated = tune(measure, [4, 6, 8, 10, 12], 8, 12)
    assert (best["torch_threads"], best["numba_threads"]) == (4, 6)
    assert confirmed == best and repeated == baseline
    assert calls[0] == (8, 12, "baseline")
    assert all(t == 8 for t, _, phase in calls if phase == "physics")
    assert all(n == 6 for _, n, phase in calls if phase == "torch")
    assert calls[-2:] == [(4, 6, "confirm_best"), (8, 12, "confirm_baseline")]
