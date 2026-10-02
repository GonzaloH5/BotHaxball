"""Map-discovered public joints: parity, no dynamics changes, safe run activation."""
import copy
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from env.public_joints import discover_joint_barriers
from tools import configure_rs4_public_joints as tool
from tools import prepare_rs4_v3 as preparer
from tools import upgrade_rs4_public_signals as upgrade
from train.runtime import load_config
from test_rs4_program import source
from test_public_signals import rs4

ROOT=Path(__file__).resolve().parents[1]


def test_discovery_color_topology_and_index_independence():
    stadium=json.loads((ROOT/'stadiums/rs_one.hbs').read_text())
    assert discover_joint_barriers(stadium,1150,600)==[0,1,2,3]
    stadium['joints'].reverse()
    assert discover_joint_barriers(stadium,1150,600)==[2,3,4,5]
    stadium['joints'].append(copy.deepcopy(stadium['joints'][-1]))
    assert discover_joint_barriers(stadium,1150,600)==[]
    assert discover_joint_barriers({'joints':[]},1150,600)==[]


def test_client_joint_suite():
    if not shutil.which('node'):
        pytest.skip('Node required')
    subprocess.run(['node','deploy/test_joint_barriers.js'],cwd=ROOT,check=True,capture_output=True)


def test_lateral_joint_renderer_by_color_without_physics_changes():
    legacy, public=rs4(),rs4()
    legacy.reset();public.reset()
    public.configure_public_signals({'auto_joints':True})
    public.setpiece_team[:]=[0,1]
    public.setpiece_kind[:]=1
    ball,barrier=public.public_colors()
    np.testing.assert_array_equal(ball,[0xFF3F34,0x0FBCF9])
    np.testing.assert_array_equal(barrier,[0xEC7458,0x48BEF9])
    public.setpiece_kind[:]=[2,3]
    assert np.all(public.public_colors()[1]==0xFFFFFF),'corners/goal kicks have ball cues, no invented line'
    public.setpiece_team[:]=-1
    assert np.all(public.public_colors()[0]==0xFFFFFF)
    actions=np.zeros((2,8),dtype=np.int64)
    for _ in range(5):
        a,ar,ad,_=legacy.step(actions)
        b,br,bd,_=public.step(actions)
        np.testing.assert_array_equal(ar,br)
        np.testing.assert_array_equal(ad,bd)
        np.testing.assert_array_equal(legacy.sim.pos,public.sim.pos)
        np.testing.assert_array_equal(a[...,:56],b[...,:56])
        np.testing.assert_array_equal(a[...,71:],b[...,71:])


def test_configure_preserves_checkpoint_ledger_and_calendars(source,monkeypatch):
    path,_,_=source
    ck=torch.load(path,weights_only=False)
    ck['model_config']['rule_observation']='masked'
    torch.save(ck,path)
    original=preparer.prepare(path)
    ledger=json.loads((original/'ledger.json').read_text())
    ledger['selected']='control'
    (original/'ledger.json').write_text(json.dumps(ledger))
    monkeypatch.setattr(upgrade,'ROOT',path.parents[2])
    upgrade.prepare()
    monkeypatch.setattr(tool,'ROOT',path.parents[2])
    directory=path.parents[1]/'rs4_v3_public'
    before={p.relative_to(directory):p.read_bytes() for p in directory.rglob('*') if p.is_file()}
    result=tool.configure(dry_run=True)
    assert not result['checkpoint_modified'] and result['budget_expansion']==0
    assert before=={p.relative_to(directory):p.read_bytes() for p in directory.rglob('*') if p.is_file()}
    config_before=load_config(directory/'control/config.yaml')
    tool.configure()
    current=load_config(directory/'control/config.yaml')
    assert current['public_signals']['auto_joints']
    assert current['run_name']=='rs4_v3_public/control'
    assert 0xFF3F34 in current['public_signals']['red_colors']
    assert 0x0FBCF9 in current['public_signals']['blue_colors']
    for field in ('rs4_program','model','ppo','bc_reference'):
        assert current.get(field)==config_before.get(field)
    assert (directory/'control/config_before_public_joints.yaml').read_bytes()==before[Path('control/config.yaml')]
    assert json.loads((directory/'specialization.json').read_text())['evaluation_directory']=='evaluations_public_joints_v1'
    for p in ('control/latest.pt','parent.pt','ledger.json','public_source.pt'):
        assert (directory/p).read_bytes()==before[Path(p)]
    tool.configure()
    assert (directory/'control/config_before_public_joints.yaml').read_bytes()==before[Path('control/config.yaml')]
    with tool.stopped(directory):
        with pytest.raises(ValueError,match='Stop the runner'):
            tool.configure(dry_run=True)
    with pytest.raises(ValueError,match='simple name'):
        tool.configure('../rs4_v3_public')
    with pytest.raises(ValueError,match='Missing public run'):
        tool.configure('missing')


def test_evaluation_baseline_cache_changes_with_source(source,monkeypatch):
    from tools import run_rs4_v3 as runner
    from eval import rs4_v3 as evaluator
    path,_,_=source
    directory=preparer.prepare(path)
    commands=[]
    def invoke(command):
        commands.append(command)
        Path(command[command.index('--out')+1]).write_text('{}')
    monkeypatch.setattr(runner,'_invoke',invoke)
    monkeypatch.setattr(evaluator,'evaluation_source_fingerprint',lambda:{'sha256':'a'*64})
    kwargs=dict(full=False,games=2,full_games=2,minutes=.1,seeds=(51,73,91),label='test')
    runner._evaluate(directory,'control',**kwargs)
    runner._evaluate(directory,'control',**kwargs)
    assert len(commands)==1
    first=commands[0][commands[0].index('--reference-report')+1]
    monkeypatch.setattr(evaluator,'evaluation_source_fingerprint',lambda:{'sha256':'b'*64})
    runner._evaluate(directory,'control',**kwargs)
    second=commands[1][commands[1].index('--reference-report')+1]
    assert first!=second and 'aaaaaaaaaaaa' in first and 'bbbbbbbbbbbb' in second
