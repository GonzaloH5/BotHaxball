"""Explicit, non-destructive RS4 v3 architecture and named-Adam migration.

The returned checkpoint owns its tensors; the source is never modified.  Optimizer
parameter numbers are not stable across architectures, so they are resolved through
names (or the legacy model registration order) before copying any Adam moments.
"""
from __future__ import annotations

import copy

import torch

from .model import build_model


_NEW_PREFIXES = ("gru.", "memory_pi.", "memory_v.", "q.", "residual_attn.",
                 "mate_attn_proj.", "opp_attn_proj.", "attn_gate")
_SHAPE_FIELDS = ("self_dim", "ent_dim", "n_actions", "hidden", "layers", "ent_hidden", "ent_layers")


def optimizer_parameter_names(model, optimizer=None):
    """Persist names alongside optimizer groups in all new v3 checkpoints."""
    named = dict(model.named_parameters())
    if optimizer is None:
        return [list(named)]
    by_identity = {id(parameter): name for name, parameter in named.items()}
    return [[by_identity[id(parameter)] for parameter in group["params"]]
            for group in optimizer.param_groups]


def _source_names(checkpoint, model):
    groups = checkpoint["opt"]["param_groups"]
    names = checkpoint.get("optimizer_param_names")
    if names is None:
        flat = list(dict(model.named_parameters()))
        if sum(len(group["params"]) for group in groups) != len(flat):
            raise ValueError("Legacy optimizer order cannot be resolved unambiguously")
        names, offset = [], 0
        for group in groups:
            names.append(flat[offset:offset + len(group["params"])])
            offset += len(group["params"])
    if (len(names) != len(groups) or any(len(n) != len(g["params"]) for n, g in zip(names, groups))
            or len({name for group in names for name in group}) != sum(map(len, names))):
        raise ValueError("Invalid optimizer_param_names manifest")
    if any(name not in dict(model.named_parameters()) for group in names for name in group):
        raise ValueError("Optimizer manifest contains an unknown model parameter")
    return names


def migrate_adam(checkpoint, source_model, target_model):
    """Return a standard single-group Adam state with compatible moments by name.

    Legacy trainers construct one Adam group. Different source group options are
    rejected instead of silently changing per-parameter optimizer semantics.
    New parameters have no optimizer state; Adam initializes them on their first
    gradient, while unchanged parameters retain their own update counter.
    """
    source_opt = checkpoint.get("opt")
    if source_opt is None:
        raise ValueError("RS4 v3 continuation requires the source Adam state")
    groups = source_opt.get("param_groups", [])
    if not groups:
        raise ValueError("Source Adam has no parameter groups")
    options = [{key: value for key, value in group.items() if key != "params"} for group in groups]
    if any(option != options[0] for option in options[1:]):
        raise ValueError("Different source Adam group options require an explicit optimizer migration")
    source_names = _source_names(checkpoint, source_model)
    named_states = {}
    for group, names in zip(groups, source_names):
        for number, name in zip(group["params"], names):
            if number in source_opt["state"]:
                named_states[name] = source_opt["state"][number]
    state, transferred, new_parameters = {}, [], []
    source_parameters = dict(source_model.named_parameters())
    target_parameters = dict(target_model.named_parameters())
    for number, (name, parameter) in enumerate(target_parameters.items()):
        old_parameter = source_parameters.get(name)
        old_state = named_states.get(name)
        if old_parameter is None or old_parameter.shape != parameter.shape:
            new_parameters.append(name)
            continue
        if old_state is None:
            continue  # Adam may never have seen a gradient for this parameter.
        for field in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            if field in old_state and tuple(old_state[field].shape) != tuple(parameter.shape):
                raise ValueError(f"Invalid Adam {field} shape for {name}")
        state[number] = copy.deepcopy(old_state)
        transferred.append(name)
    group = copy.deepcopy(options[0])
    group["params"] = list(range(len(target_parameters)))
    return dict(state=state, param_groups=[group]), transferred, new_parameters


def migrate_checkpoint(source_ck, target_config):
    """Migrate a set checkpoint to control or residual memory/attention candidate.

    ``target_config`` can be a model config or a full training config containing
    ``model``. The complete source checkpoint is copied, including league and
    counters. Program anchors are the preparer's responsibility, not recalculated
    here on resume.
    """
    source_config = dict(source_ck["model_config"])
    if source_config.get("type") != "set":
        raise ValueError("RS4 v3 pilots require a feedforward universal set source")
    target = dict(target_config.get("model", target_config))
    merged = {**source_config, **target}
    if merged.get("type") not in ("set", "recurrent_set"):
        raise ValueError("RS4 v3 candidates must be set or recurrent_set")
    for field in _SHAPE_FIELDS:
        if merged.get(field) != source_config.get(field):
            raise ValueError(f"Migration cannot change model.{field}")
    if merged.get("rule_observation", "full") != source_config.get("rule_observation", "full"):
        raise ValueError("Migration cannot silently change the observation contract")
    if merged.get("type") == "recurrent_set":
        if merged.get("memory_size", 32) != 32 or merged.get("pooling") != "attentive_meanmax":
            raise ValueError("RS4 v3 memory candidate requires GRU32 and attentive_meanmax")
        merged["memory_size"] = 32
    elif merged.get("pooling") != source_config.get("pooling"):
        raise ValueError("The feedforward control must preserve source pooling")
    source_model, target_model = build_model(source_config), build_model(merged)
    source_model.load_state_dict(source_ck["model"])
    target_state = target_model.state_dict()
    missing = []
    for name, value in source_ck["model"].items():
        if name not in target_state or target_state[name].shape != value.shape:
            raise ValueError(f"Incompatible source tensor {name}")
        target_state[name] = value.detach().clone()
    for name in target_state:
        if name not in source_ck["model"]:
            if not name.startswith(_NEW_PREFIXES):
                raise ValueError(f"Unexpected new model tensor {name}")
            missing.append(name)
    target_model.load_state_dict(target_state)
    # Only genuinely new residual contributions are zeroed; existing learned
    # attention branches must remain unchanged when migrating an attentive source.
    for name in ("attn_gate", "memory_pi.weight", "memory_v.weight"):
        if name in missing:
            target_state[name].zero_()
    target_model.load_state_dict(target_state)
    opt, transferred, new_parameters = migrate_adam(source_ck, source_model, target_model)
    result = copy.deepcopy(source_ck)
    result.update(model=copy.deepcopy(target_model.state_dict()), model_config=target_model.config(),
                  opt=opt, optimizer_param_names=optimizer_parameter_names(target_model),
                  frozen_normalizers=True)
    result["rs4_migration"] = dict(version=3, source_model_config=source_config,
                                    transferred_adam=transferred, new_parameters=new_parameters,
                                    source_steps=int(source_ck.get("steps", 0)))
    return result
