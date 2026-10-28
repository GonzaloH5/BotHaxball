"""Paired count bootstrap, the fixed acceptance gate and real recorder timing."""
import copy
import json

import numpy as np
import pytest

from env.rs4z import kernel as K
from env.rs4z.core import RS4ZEnv
from learn import x4_coop_ab_eval as AB
from learn.x4_eval import Policy
from tools import x4_metrics as XM, x4_pass_chain as PC
from tools.x4_html_replay import write_html


def _row(seed=73001, side=0, index=0, **overrides):
    counts = AB.numeric_counts(PC.unit_counts([]))
    counts.update(rec_press=10, rec_press_ret=4, rec=20, rec_ret=16,
                  passes=20, failed=5, dec_press=20, dec_press_lost=6,
                  open_minutes=3.0, minutes=3.0, shots=2, goals=0)
    counts.update(overrides)
    return dict(seed=seed, side=side, index=index, completed=True,
                goals_for=0, goals_against=0, points=.5, counts=counts)


def _panels():
    result = {p: {} for p in AB.POLICIES}
    for policy in AB.POLICIES:
        for rival in AB.RIVALS:
            result[policy][rival] = [
                _row(seed, side, index, rec_press_ret=6 if policy == 'B' else 4)
                for seed in AB.SEEDS for side in (0, 1) for index in range(AB.MATCHES_PER_SIDE)
            ]
    return result


def test_paired_bootstrap_preserves_matching_and_common_variation():
    rows = [_row(73001, side, index, rec_press=20 + index,
                 rec_press_ret=3 + index) for side in (0, 1) for index in range(8)]
    paired = AB.paired_bootstrap(rows, list(reversed(copy.deepcopy(rows))))
    assert paired['valid']
    assert paired['strata'] == [[73001, 0], [73001, 1]]
    for metric in paired['metrics'].values():
        assert metric['difference']['valor'] == 0
        assert metric['difference']['ic90'] == [0.0, 0.0]
    assert paired['point_difference']['difference']['ic90'] == [0.0, 0.0]


def test_ratio_uses_pooled_counts_instead_of_average_match_rates():
    reference = [_row(index=0, rec_press=2, rec_press_ret=0),
                 _row(index=1, rec_press=18, rec_press_ret=9)]
    candidate = [_row(index=0, rec_press=2, rec_press_ret=2),
                 _row(index=1, rec_press=18, rec_press_ret=0)]
    metric = AB.paired_bootstrap(candidate, reference)['metrics']['retencion_bajo_presion']
    assert metric['candidate'] == pytest.approx(.1)
    assert metric['reference'] == pytest.approx(.45)
    assert metric['difference']['valor'] == pytest.approx(-.35)
    assert metric['enough_support']


def test_bootstrap_keeps_seed_side_strata_fixed():
    reference = [_row(73001, 0, rec_press=20, rec_press_ret=0),
                 _row(73002, 1, rec_press=200, rec_press_ret=200)]
    candidate = [_row(73001, 0, rec_press=20, rec_press_ret=10),
                 _row(73002, 1, rec_press=200, rec_press_ret=180)]
    metric = AB.paired_bootstrap(candidate, reference)['metrics']['retencion_bajo_presion']
    # One match per stratum: a stratified draw cannot change seed/side weights.
    np.testing.assert_allclose(metric['difference']['ic90'], [-10/220, -10/220])


@pytest.mark.parametrize('denominator,enough', [(19, False), (20, True)])
def test_twenty_pressure_receptions_are_required_per_comparator(denominator, enough):
    b = [_row(rec_press=denominator, rec_press_ret=10)]
    reference = [_row(rec_press=20, rec_press_ret=5)]
    metric = AB.paired_bootstrap(b, reference)['metrics']['retencion_bajo_presion']
    assert metric['enough_support'] is enough


@pytest.mark.parametrize('corruption', ['missing', 'duplicate', 'incomplete', 'nonfinite'])
def test_invalid_pairs_cannot_create_improvement_evidence(corruption):
    reference = [_row(index=0), _row(index=1)]
    candidate = copy.deepcopy(reference)
    if corruption == 'missing':
        candidate.pop()
    elif corruption == 'duplicate':
        candidate[1]['index'] = 0
    elif corruption == 'incomplete':
        candidate[0]['completed'] = False
    else:
        candidate[0]['counts']['rec_press'] = float('nan')
    result = AB.paired_bootstrap(candidate, reference)
    assert not result['valid'] and not result['metrics']


def test_registered_panel_can_accept_clear_supported_cooperative_gain():
    report = AB.build_report(_panels())
    assert report['decision']['status'] == 'accepted'
    assert report['decision']['accepted']
    assert len(report['matches']['parent800']['parent800']['rows']) == 64
    # A self-play baseline must not be skipped merely because names are equal.
    assert report['matches']['parent800']['parent800']['valid_for_strength']
    json.dumps(report, allow_nan=False)


def test_pressure_primary_is_only_ref200_and_secondary_rival_is_diagnostic():
    panels = _panels()
    for policy in AB.POLICIES:
        for row in panels[policy]['parent800']:
            row['counts']['rec_press'] = row['counts']['rec_press_ret'] = 0
    report = AB.build_report(panels)
    assert report['decision']['status'] == 'accepted'
    assert 'pressure_gain' not in report['decision']['checks']['B_vs_A']['parent800']
    assert 'support_retencion_bajo_presion' not in report['decision']['checks']['B_vs_A']['parent800']


@pytest.mark.parametrize('baseline', ['A', 'parent800'])
def test_pressure_gain_must_exceed_both_comparators(baseline):
    panels = _panels()
    for row in panels[baseline]['ref200']:
        row['counts']['rec_press_ret'] = 6
    decision = AB.build_report(panels)['decision']
    assert decision['status'] == 'not_demonstrated'
    assert not decision['checks']['B_vs_' + baseline]['ref200']['pressure_gain']


def test_pointwise_pressure_gain_with_ci_crossing_zero_is_not_demonstrated():
    report = AB.build_report(_panels())
    pressure = report['comparisons']['B_vs_A']['ref200']['metrics']['retencion_bajo_presion']
    pressure['difference'] = dict(valor=.10, ic90=[-.01, .21])
    assert AB.acceptance_decision(report)['status'] == 'not_demonstrated'


@pytest.mark.parametrize('metric,field,value,failed_check', [
    ('precision_pase', 'difference', -.031, 'precision_preserved'),
    ('perdida_sin_pase_bajo_presion', 'difference', .031, 'pressure_losses_preserved'),
    ('pases_por_min', 'ratio', .899, 'pass_volume_preserved'),
])
@pytest.mark.parametrize('rival', AB.RIVALS)
@pytest.mark.parametrize('baseline', ['A', 'parent800'])
def test_no_deterioration_guards_apply_to_both_rivals_and_both_baselines(metric, field, value,
                                                                       failed_check, rival, baseline):
    report = AB.build_report(_panels())
    report['comparisons']['B_vs_' + baseline][rival]['metrics'][metric][field]['valor'] = value
    decision = AB.acceptance_decision(report)
    assert decision['status'] == 'not_demonstrated'
    assert not decision['checks']['B_vs_' + baseline][rival][failed_check]


def test_exact_declared_boundaries_are_accepted():
    report = AB.build_report(_panels())
    for comparison in report['comparisons'].values():
        for rival, result in comparison.items():
            result['metrics']['precision_pase']['difference']['valor'] = -.03
            result['metrics']['perdida_sin_pase_bajo_presion']['difference']['valor'] = .03
            result['metrics']['pases_por_min']['ratio']['valor'] = .9
            if rival == 'ref200':
                result['metrics']['retencion_bajo_presion']['difference'] = dict(valor=.05, ic90=[.01, .09])
    for result in report['comparisons']['B_vs_A'].values():
        result['point_difference']['difference']['valor'] = -.05
    assert AB.acceptance_decision(report)['status'] == 'accepted'


@pytest.mark.parametrize('rival', AB.RIVALS)
def test_point_fraction_guard_compares_b_with_a_at_each_rival(rival):
    report = AB.build_report(_panels())
    report['comparisons']['B_vs_A'][rival]['point_difference']['difference']['valor'] = -.051
    decision = AB.acceptance_decision(report)
    assert decision['status'] == 'not_demonstrated'
    assert not decision['checks']['B_vs_A'][rival]['point_fraction_preserved']


@pytest.mark.parametrize('corruption', ['missing_rival', 'partial_match', 'missing_index', 'changed_panel'])
def test_incomplete_or_different_registered_panels_cannot_accept(corruption):
    panels = _panels()
    if corruption == 'missing_rival':
        del panels['parent800']['parent800']
    elif corruption == 'partial_match':
        panels['A']['ref200'][0]['completed'] = False
    elif corruption == 'missing_index':
        panels['B']['parent800'].pop()
    report = AB.build_report(panels)
    if corruption == 'changed_panel':
        report['panel']['seeds'] = [1, 2, 3, 4]
        report['decision'] = AB.acceptance_decision(report)
    assert report['decision']['status'] == 'not_demonstrated'


def test_low_pressure_support_blocks_acceptance_even_with_large_gain():
    panels = _panels()
    for row in panels['B']['ref200']:
        row['counts']['rec_press'] = row['counts']['rec_press_ret'] = 0
    panels['B']['ref200'][0]['counts'].update(rec_press=19, rec_press_ret=19)
    report = AB.build_report(panels)
    assert report['decision']['status'] == 'not_demonstrated'
    assert not report['decision']['checks']['B_vs_A']['ref200']['support_retencion_bajo_presion']


@pytest.mark.parametrize('goals,points', [((0, 0), .5), ((3, 3), .5), ((2, 1), 1.), ((0, 2), 0.)])
def test_draws_and_goals_keep_distinct_point_and_cooperation_metrics(goals, points):
    assert AB.points_for_goals(*goals) == points
    row = _row()
    row.update(goals_for=goals[0], goals_against=goals[1], points=points)
    summary = AB.match_summary([row])
    assert summary['point_fraction'] == points
    assert summary['draws'] == int(goals[0] == goals[1])
    assert summary['goals_for'] == goals[0]


def _open_env(match_ticks=3, ball_pos=(0., 0.), ball_vel=(1., 0.)):
    env = RS4ZEnv(1, map=AB.MAP, frame_skip=3, max_delay=15, seed=7)
    env.start_match([0], match_ticks=match_ticks, delay=np.zeros((1, 8), np.int64))
    pos = np.array([[-400., 0], [-600, 200], [-600, -200], [-900, 0],
                    [400, 300], [600, 200], [600, -200], [900, 0]])
    env.place(0, ball_pos=ball_pos, ball_vel=ball_vel, player_pos=pos, player_vel=np.zeros((8, 2)))
    return env


def test_replay_has_initial_and_terminal_frames_without_changing_metric_pre_state(tmp_path):
    env = _open_env()
    rec = AB.ReplayRecorder(env, capture=True)
    acts = np.zeros((1, 8), np.int64)
    rec.pre(env)
    event = env.step(acts)
    rec.record(env, event, acts)
    ep = rec.episode()
    assert rec.finished and ep.meta['ab_record']['completed']
    assert len(rec.frames) == len(ep.ball) + 1 == 2
    assert len(rec.frames[0]) == len(rec.frames[1]) == 24
    assert ep.ball[0, 0] == rec.frames[0][6] == 0
    assert rec.frames[1][6] == env.ball_pos[0, 0] > 0
    assert rec.frames[0][0] == 0 and rec.frames[1][0] == pytest.approx(3/60)
    assert rec.simulation_ticks == 3
    payload = dict(version='x4-replay-1', sample_hz=20, pitch=rec.pitch,
                   matches=[dict(frames=rec.frames, finished=rec.finished)])
    path = write_html(payload, tmp_path / 'replay.html')
    assert path.is_file()
    assert 'x4-replay-1' in path.read_text(encoding='utf-8')


def test_goal_frame_score_and_metric_event_remain_aligned():
    env = _open_env(match_ticks=90, ball_pos=(1120., 0.), ball_vel=(30., 0.))
    rec = AB.ReplayRecorder(env, capture=True)
    acts = np.zeros((1, 8), np.int64)
    rec.pre(env)
    event = env.step(acts)
    rec.record(env, event, acts)
    assert event['goal'][0] == 1
    ep = rec.episode()
    assert ep.goal_ev[0] == 1 and ep.ball[0, 0] == 1120
    assert rec.frames[0][1:3] == [0, 0]
    assert rec.frames[-1][1:3] == [1, 0]
    assert rec.frames[-1][3] == 1  # Actual post-goal kickoff state.
    counts = AB.numeric_counts(PC.unit_counts([ep], team=0), physical_minutes=True)
    # A single-frame Episode does not provide a valid pass-chain window.
    assert counts['goals'] == 0


def test_original_recorder_global_is_unchanged_by_private_play_adapter():
    original = XM.EpisodeRecorder
    result, recs = AB.recorded_play(Policy('still'), Policy('still'), capture=True,
                                   map_name=AB.MAP, matches=2, minutes=1,
                                   delays=AB.DELAYS, seed=7, record=2, max_decisions=2)
    assert XM.EpisodeRecorder is original
    assert len(result['episodes']) == len(recs) == 2
    assert len(recs[0].frames) == 3 and recs[1].frames is None
    assert all(not ep.meta['ab_record']['completed'] for ep in result['episodes'])


def test_physical_open_minutes_are_explicitly_twice_legacy_team_exposure():
    unit = PC.unit_counts([])
    unit.update(minutes=1.5, open_minutes=1.0, passes=8, failed=2)
    counts = AB.numeric_counts(unit, physical_minutes=True)
    assert counts['minutes'] == 3 and counts['open_minutes'] == 2
    row = _row()
    row['counts'].update(counts)
    rates = AB.pooled_rates([row])
    assert rates['pases_por_min'] == 4
    assert rates['precision_pase'] == .8


def test_master_seed_role_derivations_do_not_collide_across_sides():
    derived = [value for seed in AB.SEEDS for side in (0, 1) for value in AB.random_seeds(seed, side)]
    assert len(set(derived)) == len(derived)
    assert AB.random_seeds(73001, 0) == AB.random_seeds(73001, 0)


def test_tiny_cpu_panel_has_all_rows_and_four_replays_but_cannot_accept():
    policies = {name: Policy('still') for name in (*AB.POLICIES, 'ref200')}
    report, replay = AB.evaluate_panel(policies, seeds=(73001,), matches_per_side=1,
                                       minutes=1/1200, draws=32)
    assert report['decision']['status'] == 'not_demonstrated'
    assert all(len(report['matches'][p][r]['rows']) == 2 for p in AB.POLICIES for r in AB.RIVALS)
    assert len(replay['matches']) == 4
    assert {(m['policy'], m['side']) for m in replay['matches']} == {('A', 0), ('A', 1), ('B', 0), ('B', 1)}
    assert all(m['seed'] == 73001 for m in replay['matches'])
    json.dumps(report, allow_nan=False)
    json.dumps(replay, allow_nan=False)
