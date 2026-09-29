import copy

import numpy as np
import torch

from env.tasks import load_catalog, make_env
from train import multitask
from train.multitask import MultiTrainer
from train.runtime import load_config


def test_only_target_soccer_formats_are_active_in_stage_one():
    cfg = load_config(multitask.ROOT / "train/config_runpod.yaml")
    trainer = MultiTrainer.__new__(MultiTrainer)
    trainer.stages, trainer.stage = cfg["stages"], 1
    names = trainer.active_tasks()
    assert {"jjrs_6v6", "rs4_4v4", "big_3v3", "futsal_3v3", "x1_1v1"} <= set(names)
    assert not {"rs4_3v3", "rs_2v2", "rs_3v3", "rs_4v4", "rs_6v6"} & set(names)
    trainer.stage = 2
    assert not {"rs4_3v3", "rs_2v2", "rs_3v3", "rs_4v4", "rs_6v6"} & set(trainer.active_tasks())
    # Retain legacy tasks for loading old replays and explicit evaluations.
    assert {"rs4_3v3", "rs_2v2", "rs_3v3", "rs_4v4", "rs_6v6"} <= set(load_catalog())


def test_target_soccer_has_setpieces_without_special_mechanics():
    catalog = load_catalog()
    for name, players in [("rs4_4v4", 4), ("jjrs_6v6", 6)]:
        task = catalog[name]
        assert task.n_per_team == players
        assert task.out_of_bounds and not task.powershot and task.rules is None
        env = make_env(task, 1, 11, seed=0)
        assert not env.sim.ps_on and env.rules is None and env.out_of_bounds
        obs = env.reset()
        assert obs.shape == (1, players * 2, env.obs_dim)
        assert np.isfinite(obs).all()


def test_old_stage_one_resumes_with_new_tasks_without_resetting_policy(tmp_path, monkeypatch):
    source = multitask.ROOT
    cfg = load_config(source / "train/config_multi.yaml")
    cfg["tasks_file"] = str(source / "train/tasks.yaml")
    cfg["env"]["agents"] = 96
    cfg["model"].update(hidden=16, ent_hidden=8, layers=1, ent_layers=1)
    cfg["ppo"].update(torch_threads=2, numba_threads=1, rollout_len=2, epochs=1, minibatch=512)
    cfg["log"].update(every=100, checkpoint_every=100, replay_every=0)
    old = copy.deepcopy(cfg)
    old.pop("task_metric_versions")
    old["stages"][1]["tasks"] = ["big_4v4", "futsal_4v4", "rs4_3v3", "rs4_4v4",
                                "rs_2v2", "rs_3v3", "rs_4v4"]
    monkeypatch.setattr(multitask, "ROOT", tmp_path)
    trainers = []
    try:
        trainer = MultiTrainer(old, "saved", False)
        trainers.append(trainer)
        trainer.stage = 1
        trainer.build_envs()
        trainer.iterate()
        trainer.steps, trainer.iteration, trainer.stage_steps = 300_000_000, 2975, 50_000_000
        trainer.league.add_snapshot(trainer.model, "existing")
        for slot in trainer.slots:
            slot.opp_stage = 1
            slot.set_regression_context(.1)
            slot.best_wr, slot.wr_window = .9, [(150, 50)]
        trainer.save(trainer.run_dir / "latest.pt")
        weights = {k: v.clone() for k, v in trainer.model.state_dict().items()}
        optimizer = copy.deepcopy(trainer.opt.state_dict())
        resumed = MultiTrainer(cfg, "saved", True)
        trainers.append(resumed)
        assert (resumed.steps, resumed.iteration, resumed.stage, resumed.stage_steps) == (300_000_000, 2975, 1, 50_000_000)
        assert resumed.league.members[0].name == "existing"
        assert resumed.max_entities == 11
        for key, value in weights.items():
            torch.testing.assert_close(resumed.model.state_dict()[key], value, rtol=0, atol=0)
        for parameter, state in optimizer["state"].items():
            for key, value in state.items():
                torch.testing.assert_close(resumed.opt.state_dict()["state"][parameter][key], value, rtol=0, atol=0)
        slots = {s.task.name: s for s in resumed.slots}
        assert slots["jjrs_6v6"].intro_step == resumed.steps
        assert slots["jjrs_6v6"].opp_stage == 0
        assert slots["big_3v3"].wr_window == [(150, 50)]
        assert slots["rs4_4v4"].wr_window == [] and slots["rs4_4v4"].metric_version == 1
        resumed.save(resumed.run_dir / "latest.pt")
        # The metric migration runs once, not on every resume.
        state = torch.load(resumed.run_dir / "latest.pt", weights_only=False)
        assert state["task_state"]["rs4_4v4"]["metric_version"] == 1
        resumed.iterate()  # exercise mixed 4v4/6v6 observations and optimizer update
    finally:
        for trainer in trainers:
            trainer.writer.close()
