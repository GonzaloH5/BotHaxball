"""Local native-control/inference checks; never connects a browser or user run."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]


def test_hybrid_node_contracts():
    if not shutil.which('node'):
        pytest.skip('Node required')
    subprocess.run(['node','deploy/hybrid/build_extension.js'],cwd=ROOT,check=True,capture_output=True)
    subprocess.run(['node','deploy/test_hybrid.js'],cwd=ROOT,check=True,capture_output=True,timeout=60)


@pytest.mark.parametrize('memory',[False,True])
def test_hybrid_loads_exported_onnx_without_touching_user_models(tmp_path,memory):
    pytest.importorskip('onnx');pytest.importorskip('onnxruntime')
    if not shutil.which('node'):
        pytest.skip('Node required')
    from test_rs4_runtime import source_checkpoint,memory_checkpoint
    from tools.upgrade_rs4_public_signals import migrate_public_checkpoint
    checkpoint=migrate_public_checkpoint(memory_checkpoint() if memory else source_checkpoint()[0])
    checkpoint['rs4_program_state']={'relative_steps':0}
    checkpoint['env']={'frame_skip':3,'public_signal_config':{'version':1,'auto_joints':True}}
    checkpoint['model']['public_proj.weight'].normal_(std=.05)
    source=tmp_path/'source.pt';torch.save(checkpoint,source)
    export=tmp_path/'public'/'model'
    subprocess.run([sys.executable,'-m','export.to_onnx',str(source),'--out',str(export)],cwd=ROOT,check=True,capture_output=True,timeout=90)
    metadata=json.loads(export.with_suffix('.json').read_text())
    assert metadata['public_signals']['auto_joints']
    assert metadata['recurrent']==memory
    subprocess.run(['node','deploy/test_hybrid.js','--model',str(export.with_suffix('.onnx'))],cwd=ROOT,check=True,capture_output=True,timeout=60)
    subprocess.run(['node','deploy/test_obs.js',str(export.parent)],cwd=ROOT,check=True,capture_output=True,timeout=60)
