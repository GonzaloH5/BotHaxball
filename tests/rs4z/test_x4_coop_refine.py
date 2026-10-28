from types import SimpleNamespace

from learn.x4_coop_refine import training_command, regression_checks, evaluation_command


def args(tmp_path):
    return SimpleNamespace(root=str(tmp_path), init='200.pt', bc='bc.pt', original='original.pt',
                           control='control200.pt', device='cpu', envs=512, rollout=128, threads=5,
                           recordings=80, eval_every=100, seed=7)


def test_refinement_keeps_skills_and_budget_distinct_from_parent(tmp_path):
    command = training_command(args(tmp_path), 100)
    assert command[command.index('--init')+1] == '200.pt'
    assert command[command.index('--updates')+1] == '100'
    assert command[command.index('--coop-transfer-frac')+1] == '0.5'
    assert command[command.index('--coop-start-stage')+1] == '1'
    assert command[command.index('--coop-mastery-windows')+1] == '2'
    assert '--coop-possession-retention' in command and '--pass-ready' in command


def test_guard_checks_cooperation_instead_of_bc_wins():
    def ev(passes, precision, retention, pressure):
        return dict(cadena_pase=dict(metricas={k:[v,1] for k,v in zip(
            ['pases_por_min','precision_pase','retencion_tras_recibir','retencion_bajo_presion'],
            [passes,precision,retention,pressure])}))
    baseline = ev(8,.55,.45,.45)
    assert not regression_checks(baseline,ev(8,.56,.40,.40))['material_regression']
    result = regression_checks(baseline,ev(6,.45,.44,.30))
    assert result['material_regression'] and result['regressed_metrics']['retencion_bajo_presion']


def test_frozen_parent_and_dev_separation(tmp_path):
    command = evaluation_command(args(tmp_path),100)
    assert '200.pt' in command and 'original.pt' in command and 'control200.pt' in command
    assert command[command.index('--split')+1] == 'dev'
    final = evaluation_command(args(tmp_path),600,'test')
    assert final[final.index('--split')+1] == 'test'
