"""A fixed, paired 4v4 panel for one cooperative A/B experiment.

A, B and the frozen parent800 each play ref200 and parent800. The registered
panel is four master seeds (73001..73004), eight matches per side per seed,
three match-clock minutes and delays 8..11. No human split is read or trained.
Parent800 self-play is intentional: it supplies the same-rival baseline.

Counts are saved per match, not per tick. Rates use pooled numerators and
denominators. Unlike the legacy pass-chain half-match convention, open_minutes
here is the physical open-play exposure of ONE team. The 2,000 paired bootstrap
draws resample match indices within each fixed seed/side stratum. They quantify
match variability conditional on these four seeds, not human competitiveness.

Acceptance is predeclared for B against BOTH A and parent800: pressure retention
against ref200 improves >=.05 with a positive lower CI90 bound. At BOTH rivals,
precision does not fall >.03, pressure losses do not rise >.03 and pass-volume
ratio is >=.9. B's point fraction cannot fall >.05 relative to A at either rival.
All matches and >=20 relevant observations per policy/comparison are required.
An uncertain result is not_demonstrated; the panel is not extended after seeing
its result. Goals and direct play never require an obligatory pass.

Example: python -m learn.x4_coop_ab_eval --a A.pt --b B.pt --parent800 800.pt
    --ref200 200.pt --device cuda --threads 5 --out ab.json --replay-out ab.html
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import types

import numpy as np
import torch

from env.rs4z import contract as C, kernel as K
from learn.x4_coop_eval import checkpoint_fingerprint, seeded_policy
from learn.x4_eval import Policy, play
from tools import x4_metrics as XM, x4_pass_chain as PC

MAP = 'sanguchito_rs_x4'
SEEDS = (73001, 73002, 73003, 73004)
MATCHES_PER_SIDE = 8
MINUTES = 3.0
DELAYS = (8, 9, 10, 11)
BOOTSTRAP_DRAWS = 2000
MIN_SUPPORT = 20
POLICIES = ('A', 'B', 'parent800')
RIVALS = ('ref200', 'parent800')
METRICS = ('retencion_bajo_presion', 'retencion_tras_recibir', 'precision_pase',
           'perdida_sin_pase_bajo_presion', 'pases_por_min', 'tiros_por_min', 'goles_por_min')
COUNT_COLUMNS = ('rec_press_ret', 'rec_press', 'rec_ret', 'rec', 'passes', 'failed',
                 'dec_press_lost', 'dec_press', 'open_minutes', 'shots', 'goals')


def finite(value):
    return value is not None and math.isfinite(float(value))


def points_for_goals(goals_for, goals_against):
    return 1.0 if goals_for > goals_against else (0.5 if goals_for == goals_against else 0.0)


def numeric_counts(unit, *, physical_minutes=False):
    """Export scalar counts; list-valued distribution samples are not gate inputs."""
    counts = {k: (int(v) if isinstance(v, (int, np.integer)) else float(v))
              for k, v in unit.items() if np.isscalar(v)}
    if not all(math.isfinite(v) for v in counts.values()):
        raise ValueError('nonfinite pass-chain counts')
    if physical_minutes:
        # PC.unit_counts(team=side) divides exposure by two for its legacy gate.
        counts['minutes'] *= 2.0
        counts['open_minutes'] *= 2.0
    return counts


def denominators(rows):
    total = lambda key: sum(row['counts'][key] for row in rows)
    return dict(rec_press=total('rec_press'), rec=total('rec'),
                attempts=total('passes') + total('failed'),
                dec_press=total('dec_press'), passes=total('passes'),
                open_minutes=total('open_minutes'))


def pooled_rates(rows):
    # PC.rates requires all keys, including its list-valued distributions.
    # Empty lists are safe because only the scalar gate/context rates are used.
    units = []
    for row in rows:
        unit = PC.unit_counts([])
        unit.update(row['counts'])
        units.append(unit)
    values = PC.rates(units)
    return {key: float(values[key]) if finite(values[key]) else None for key in METRICS}


def _key(row):
    return int(row['seed']), int(row['side']), int(row['index'])


def _matrix(rows):
    return np.asarray([[row['counts'][key] for key in COUNT_COLUMNS] for row in rows], dtype=float)


def _vector_rates(totals):
    def ratio(a, b):
        return np.divide(a, b, out=np.full_like(a, np.nan), where=b > 0)
    return np.stack((ratio(totals[..., 0], totals[..., 1]),
                     ratio(totals[..., 2], totals[..., 3]),
                     ratio(totals[..., 4], totals[..., 4] + totals[..., 5]),
                     ratio(totals[..., 6], totals[..., 7]),
                     ratio(totals[..., 4], totals[..., 8]),
                     ratio(totals[..., 9], totals[..., 8]),
                     ratio(totals[..., 10], totals[..., 8])), axis=-1)


def _interval(samples):
    defined = np.asarray(samples)[np.isfinite(samples)]
    # Do not silently hide an unidentifiable denominator in many draws.
    if len(defined) < .95 * len(samples):
        return [None, None], len(defined)
    return [float(x) for x in np.percentile(defined, (5, 95))], len(defined)


def paired_bootstrap(rows_b, rows_reference, *, draws=BOOTSTRAP_DRAWS, seed=73000,
                     min_support=MIN_SUPPORT):
    """Pool count ratios after paired resampling inside each seed/side stratum."""
    if draws < 1:
        raise ValueError('bootstrap draws must be positive')
    keys_b, keys_ref = [_key(r) for r in rows_b], [_key(r) for r in rows_reference]
    if (not keys_b or len(set(keys_b)) != len(keys_b) or len(set(keys_ref)) != len(keys_ref)
            or set(keys_b) != set(keys_ref)):
        return dict(valid=False, reason='missing, duplicate or unmatched match keys', metrics={})
    if not all(r.get('completed', False) for r in rows_b + rows_reference):
        return dict(valid=False, reason='incomplete matches', metrics={})
    b = sorted(rows_b, key=_key)
    reference = sorted(rows_reference, key=_key)
    bm, rm = _matrix(b), _matrix(reference)
    if not (np.isfinite(bm).all() and np.isfinite(rm).all()):
        return dict(valid=False, reason='nonfinite match counts', metrics={})
    bp, rp = np.asarray([r['points'] for r in b]), np.asarray([r['points'] for r in reference])
    rng = np.random.default_rng(seed)
    btotal, rtotal = np.zeros((draws, len(COUNT_COLUMNS))), np.zeros((draws, len(COUNT_COLUMNS)))
    point_draw = np.zeros(draws)
    strata = sorted({k[:2] for k in keys_b})
    for stratum in strata:
        indices = np.asarray([i for i, row in enumerate(b) if _key(row)[:2] == stratum])
        picked = indices[rng.integers(0, len(indices), size=(draws, len(indices)))]
        btotal += bm[picked].sum(axis=1)
        rtotal += rm[picked].sum(axis=1)
        point_draw += (bp[picked] - rp[picked]).sum(axis=1)
    point_draw /= len(b)
    bd, rd = denominators(b), denominators(reference)
    rates_b, rates_ref = pooled_rates(b), pooled_rates(reference)
    draws_b, draws_ref = _vector_rates(btotal), _vector_rates(rtotal)
    den_keys = ('rec_press', 'rec', 'attempts', 'dec_press', 'passes', None, None)
    metrics = {}
    for col, (metric, den_key) in enumerate(zip(METRICS, den_keys)):
        value_b, value_ref = rates_b[metric], rates_ref[metric]
        diff = value_b - value_ref if finite(value_b) and finite(value_ref) else None
        ci, defined = _interval(draws_b[:, col] - draws_ref[:, col])
        enough = bool(den_key is None or (bd[den_key] >= min_support and rd[den_key] >= min_support))
        if metric == 'pases_por_min':
            enough &= bd['open_minutes'] > 0 and rd['open_minutes'] > 0
        entry = dict(candidate=value_b, reference=value_ref,
                     difference=dict(valor=diff, ic90=ci), enough_support=enough,
                     defined_bootstrap_draws=defined)
        if metric == 'pases_por_min':
            ratios = np.divide(draws_b[:, col], draws_ref[:, col],
                               out=np.full(draws, np.nan), where=draws_ref[:, col] > 0)
            ratio_ci, ratio_defined = _interval(ratios)
            entry['ratio'] = dict(valor=value_b / value_ref if finite(value_b) and finite(value_ref)
                                 and value_ref > 0 else None, ic90=ratio_ci)
            entry['defined_ratio_bootstrap_draws'] = ratio_defined
        metrics[metric] = entry
    point_ci, _ = _interval(point_draw)
    return dict(valid=True, matches=len(b), strata=[list(s) for s in strata],
                support=dict(candidate=bd, reference=rd), metrics=metrics,
                point_difference=dict(candidate=float(bp.mean()), reference=float(rp.mean()),
                                      difference=dict(valor=float((bp-rp).mean()), ic90=point_ci)))


def match_summary(rows):
    completed = all(row.get('completed', False) for row in rows) and bool(rows)
    diff = [row['goals_for'] - row['goals_against'] for row in rows]
    return dict(matches=len(rows), finished=sum(row.get('completed', False) for row in rows),
                valid_for_strength=completed, wins=sum(d > 0 for d in diff) if completed else None,
                draws=sum(d == 0 for d in diff) if completed else None,
                losses=sum(d < 0 for d in diff) if completed else None,
                goals_for=sum(row['goals_for'] for row in rows),
                goals_against=sum(row['goals_against'] for row in rows),
                point_fraction=float(np.mean([r['points'] for r in rows])) if completed else None,
                natural_metrics=pooled_rates(rows), support=denominators(rows), rows=rows)


def acceptance_decision(report):
    """Apply the registered gate; missing panels/observations cannot accept B."""
    reasons, checks = [], {}
    expected = {(seed, side, index) for seed in SEEDS for side in (0, 1)
                for index in range(MATCHES_PER_SIDE)}
    panel = report.get('panel', {})
    if (tuple(panel.get('seeds', ())) != SEEDS or panel.get('matches_per_side') != MATCHES_PER_SIDE
            or panel.get('minutes') != MINUTES or tuple(panel.get('delays', ())) != DELAYS
            or panel.get('bootstrap_draws') != BOOTSTRAP_DRAWS):
        reasons.append('evaluation differs from the registered 64-match panel')
    for policy in POLICIES:
        for rival in RIVALS:
            row = report.get('matches', {}).get(policy, {}).get(rival, {})
            rows = row.get('rows', [])
            if (len(rows) != len(expected) or {_key(r) for r in rows} != expected
                    or not row.get('valid_for_strength', False)):
                reasons.append(f'{policy}/{rival}: incomplete registered matches')
    for comparison in ('B_vs_A', 'B_vs_parent800'):
        checks[comparison] = {}
        for rival in RIVALS:
            result = report.get('comparisons', {}).get(comparison, {}).get(rival, {})
            metrics = result.get('metrics', {})
            test = {}
            required = ['precision_pase', 'perdida_sin_pase_bajo_presion', 'pases_por_min']
            if rival == 'ref200':
                required.append('retencion_bajo_presion')
            for name in required:
                entry = metrics.get(name, {})
                defined = entry.get('defined_bootstrap_draws', 0)
                test['support_' + name] = bool(entry.get('enough_support') and defined >= .95 * BOOTSTRAP_DRAWS)
            pressure = metrics.get('retencion_bajo_presion', {}).get('difference', {})
            precision = metrics.get('precision_pase', {}).get('difference', {}).get('valor')
            loss = metrics.get('perdida_sin_pase_bajo_presion', {}).get('difference', {}).get('valor')
            volume = metrics.get('pases_por_min', {}).get('ratio', {}).get('valor')
            increase, lower = pressure.get('valor'), pressure.get('ic90', [None, None])[0]
            # Tiny floating roundoff should not reject an exact declared margin.
            test.update(valid_pair=bool(result.get('valid')),
                        precision_preserved=finite(precision) and precision >= -.03 - 1e-12,
                        pressure_losses_preserved=finite(loss) and loss <= .03 + 1e-12,
                        pass_volume_preserved=finite(volume) and volume >= .9 - 1e-12)
            if rival == 'ref200':
                test.update(pressure_gain=finite(increase) and increase >= .05 - 1e-12,
                            pressure_ci_positive=finite(lower) and lower > 0)
            if comparison == 'B_vs_A':
                point = result.get('point_difference', {}).get('difference', {}).get('valor')
                test['point_fraction_preserved'] = finite(point) and point >= -.05 - 1e-12
            checks[comparison][rival] = test
            reasons.extend(f'{comparison}/{rival}: {key}' for key, ok in test.items() if not ok)
    return dict(status='accepted' if not reasons else 'not_demonstrated',
                accepted=not reasons, checks=checks, reasons=reasons,
                human_competitiveness='unverified',
                limitation='Acceptance is conditional on the registered simulator panel; it does not certify human play.')


def build_report(panels, *, seeds=SEEDS, matches_per_side=MATCHES_PER_SIDE, minutes=MINUTES,
                 draws=BOOTSTRAP_DRAWS):
    report = dict(version='x4-coop-ab-eval-1', map=MAP, training_access=False,
                  evaluation_kind='registered_simulator_panel',
                  panel=dict(seeds=list(seeds), matches_per_side=matches_per_side, minutes=minutes,
                             delays=list(DELAYS), bootstrap_draws=draws,
                             min_support=MIN_SUPPORT, bootstrap_strata=['seed', 'side']),
                  matches={}, comparisons={})
    for policy, rivals in panels.items():
        report['matches'][policy] = {rival: match_summary(rows) for rival, rows in rivals.items()}
    for baseline in ('A', 'parent800'):
        report['comparisons']['B_vs_' + baseline] = {}
        for rival in RIVALS:
            report['comparisons']['B_vs_' + baseline][rival] = paired_bootstrap(
                panels.get('B', {}).get(rival, []), panels.get(baseline, {}).get(rival, []), draws=draws)
    report['decision'] = acceptance_decision(report)
    return report


def pitch_payload(env):
    st = env.st
    segments = [dict(p0=p0.tolist(), p1=p1.tolist(), curved=int(curved),
                     center=center.tolist(), radius=float(radius), color='#d3e1d1')
                for p0, p1, curved, center, radius, visible in zip(
                    st.s_p0, st.s_p1, st.s_curved, st.s_center, st.s_radius, st.s_vis) if visible]
    discs = [dict(pos=pos.tolist(), r=float(radius), color='#f5cd54')
             for pos, radius in zip(st.d_pos, st.d_radius) if radius > 0]
    return dict(line_w=float(C.LINE_W), line_h=float(env.prm[C.PI['line_half_h']]),
                goal_x=float(st.goal_x), goal_half_h=float(st.goal_half_height),
                bound_w=float(st.width), bound_h=float(st.height),
                player_r=float(st.player['radius']), ball_r=float(st.ball['radius']),
                kickoff_r=float(st.kickoff_radius), segments=segments, discs=discs)


def replay_frame(env, row, kicked=None):
    state = env.ri[row]
    if state[K.RI_KO]:
        phase, team = 1, int(state[K.RI_KO_TEAM])
    elif state[K.RI_PEND]:
        phase, team = 5, int(state[K.RI_PEND_TEAM])
    elif state[K.RI_TEAM] >= 0 and state[K.RI_KIND]:
        phase, team = int(state[K.RI_KIND]) + 1, int(state[K.RI_TEAM])
    else:
        phase, team = 0, -1
    kick_mask = sum(1 << i for i, hit in enumerate(kicked if kicked is not None else []) if hit)
    return [float(state[K.RI_CLOCK]) / 60.0, int(state[K.RI_SCORE0]), int(state[K.RI_SCORE1]),
            phase, team, kick_mask, *env.ball_pos[row].tolist(), *env.player_pos[row].ravel().tolist()]


class ReplayRecorder(XM.EpisodeRecorder):
    """Keep original metric timing; viewer has initial and post-step frames."""
    def __init__(self, env, row=0, *, capture=False):
        super().__init__(env, row=row)
        self.finished = False
        self.delay = env.delay[row].tolist()
        self.simulation_ticks = 0
        self.frames = [replay_frame(env, row)] if capture else None
        self.pitch = pitch_payload(env) if capture else None

    def record(self, env, ev, actions):
        super().record(env, ev, actions)
        self.finished |= bool(ev['match_end'][self.row])
        self.simulation_ticks += int(ev.get('ticks', np.full(env.N, env.frame_skip))[self.row])
        if self.frames is not None:
            self.frames.append(replay_frame(env, self.row, ev['kicked'][self.row]))

    def episode(self):
        ep = super().episode()
        ep.meta['ab_record'] = dict(completed=self.finished, delay=self.delay,
                                    simulation_ticks=self.simulation_ticks)
        return ep


def recorded_play(red, blue, *, capture=False, **kwargs):
    """Inject a recorder into a private function-global copy, never XM globally."""
    recorders = []

    def factory(env, row=0):
        rec = ReplayRecorder(env, row, capture=capture and row == 0)
        recorders.append(rec)
        return rec

    adapter = types.SimpleNamespace(**vars(XM))
    adapter.EpisodeRecorder = factory
    namespace = dict(play.__globals__, XM=adapter)
    local_play = types.FunctionType(play.__code__, namespace, play.__name__, play.__defaults__, play.__closure__)
    local_play.__kwdefaults__ = dict(play.__kwdefaults__ or {})
    result = local_play(red, blue, **kwargs)
    return result, recorders


def random_seeds(seed, side):
    return [int(np.random.SeedSequence([seed, side, role]).generate_state(1)[0]) for role in range(3)]


def evaluate_panel(policies, *, seeds=SEEDS, matches_per_side=MATCHES_PER_SIDE, minutes=MINUTES,
                   draws=BOOTSTRAP_DRAWS, capture_replays=True, progress=None):
    """Evaluate loaded policies; expose small panels only for local smoke tests."""
    panels = {name: {rival: [] for rival in RIVALS} for name in POLICIES}
    replay = dict(version='x4-replay-1', title='A/B · Cooperación contra el checkpoint200',
                  sample_hz=20, matches=[])
    for name in POLICIES:
        for rival in RIVALS:
            for seed in seeds:
                for side in (0, 1):
                    simulator_seed, own_seed, rival_seed = random_seeds(seed, side)
                    own = seeded_policy(policies[name], own_seed)
                    other = seeded_policy(policies[rival], rival_seed)
                    red, blue = (own, other) if side == 0 else (other, own)
                    capture = capture_replays and name in ('A', 'B') and rival == 'ref200' and seed == SEEDS[0]
                    result, recorders = recorded_play(red, blue, capture=capture, map_name=MAP,
                        matches=matches_per_side, minutes=minutes, delays=DELAYS,
                        seed=simulator_seed, record=matches_per_side)
                    if len(result['episodes']) != matches_per_side:
                        raise RuntimeError('missing recorded match episodes')
                    goals = result['goals'] if side == 0 else result['goals'][:, ::-1]
                    for index, ep in enumerate(result['episodes']):
                        counts = numeric_counts(PC.unit_counts([ep], team=side), physical_minutes=True)
                        gf, ga = map(int, goals[index])
                        panels[name][rival].append(dict(policy=name, rival=rival, seed=int(seed),
                            side=side, index=index, simulator_seed=simulator_seed,
                            policy_seed=own_seed, rival_seed=rival_seed,
                            completed=bool(ep.meta['ab_record']['completed']),
                            delays=ep.meta['ab_record']['delay'], goals_for=gf, goals_against=ga,
                            points=points_for_goals(gf, ga), counts=counts,
                            denominators=denominators([dict(counts=counts)]),
                            safety=int(result['safety'][index]),
                            safety_kickoff=int(result['safety_kickoff'][index])))
                    if capture:
                        rec = recorders[0]
                        replay['pitch'] = rec.pitch
                        replay['matches'].append(dict(label=f'{name} vs200 · seed{seed} · ' + ('rojo' if side == 0 else 'azul'),
                            policy=name, rival=rival, seed=int(seed), side=side, index=0,
                            red_label=name if side == 0 else 'Checkpoint200',
                            blue_label='Checkpoint200' if side == 0 else name,
                            finished=rec.finished, frames=rec.frames,
                            delays=rec.delay, simulation_ticks=rec.simulation_ticks))
                    if progress is not None:
                        progress(dict(policy=name, rival=rival, seed=int(seed), side=side,
                                      finished=result['finished'], matches=matches_per_side))
    report = build_report(panels, seeds=seeds, matches_per_side=matches_per_side, minutes=minutes, draws=draws)
    return report, replay


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('a', 'b', 'parent800', 'ref200', 'out'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--replay-out', help='HTML; also saves its x4-replay-1 JSON payload')
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error('threads must be positive')
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    specs = dict(A=args.a, B=args.b, parent800=args.parent800, ref200=args.ref200)
    policies = {name: Policy(spec, device=args.device) for name, spec in specs.items()}
    report, replay = evaluate_panel(policies, progress=lambda row: print(json.dumps(row), flush=True))
    report['checkpoints'] = {name: checkpoint_fingerprint(spec) for name, spec in specs.items()}
    report['runtime'] = dict(torch=torch.__version__, numpy=np.__version__, device=args.device, threads=args.threads)
    report['limitations'] = ['One preregistered simulator experiment; repeated tuning on this panel invalidates confirmation.',
                             'Fixed seed/side strata; CI reflects match variability conditional on four master seeds.',
                             'No human competitiveness claim; the simulator panel does not read human train/dev/test states.',
                             'Open-play minutes are physical exposure per team; legacy half-match exposure is corrected.']
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    replay_out = Path(args.replay_out) if args.replay_out else output.with_name(output.stem + '_replays.html')
    from tools.x4_html_replay import write_html
    write_html(replay, replay_out)
    replay_out.with_suffix('.json').write_text(json.dumps(replay, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(json.dumps(report['decision'], ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
