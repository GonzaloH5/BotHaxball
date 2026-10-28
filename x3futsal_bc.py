"""Train and evaluate the Futsal x3 behavior-cloning baseline (v2 and v3 tactical).

  python -m learn.x3futsal_bc --out runs/x3futsal_bc/tactical_v3 --steps 20000 --tactical
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from learn.x3futsal_data import (
    DATA_VERSION,
    DEFAULT_OUT,
    FEATURE_CONTRACT,
    TACTICAL_DATA_VERSION,
    TACTICAL_ENTITY_DIM,
    TACTICAL_FEATURE_CONTRACT,
    TACTICAL_SELF_DIM,
    X3Data,
)
from learn.x3futsal_policy import (
    POLICY_VERSION,
    TACTICAL_POLICY_VERSION,
    X3SetPolicy,
    X3TacticalPolicy,
)

CHANGE_ACTION_WEIGHT = 3.0
KICK_ACTION_WEIGHT = 2.5
TRAINING_OBJECTIVE = "change-and-kick-weighted-max8-v1"
TACTICAL_TRAINING_OBJECTIVE = "tactical-stride3-nearball-mirror-v2"


def _call(model, observation, prior, has_prior, device):
    own, mates, rivals = (torch.from_numpy(value).to(device) for value in observation)
    previous_action = torch.from_numpy(prior).to(device)
    has_previous = torch.from_numpy(has_prior).to(device)
    return model(own, mates, rivals, previous_action, has_previous)


@torch.no_grad()
def evaluate(model, data: X3Data, sample_indices: np.ndarray, slots: np.ndarray,
             majority_action: int, batch_size: int, device: torch.device,
             *, tactical: bool = False) -> dict:
    model.eval()
    losses, exact, moves, kicks, changes = [], [], [], [], []
    switch_margins = (-2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0)
    switch_correct = dict.fromkeys(switch_margins, 0)
    switch_prior_correct = dict.fromkeys(switch_margins, 0)
    switch_changed_correct = dict.fromkeys(switch_margins, 0)
    switch_gaps = []
    changed_total = 0
    prior_correct = 0
    true_kick = 0
    pred_kick = 0
    true_positive = 0
    majority_correct = 0
    repeated_correct = 0
    repeated_total = 0
    for start in range(0, len(sample_indices), batch_size):
        stop = start + batch_size
        observation, labels, prior, has_prior, changed = data.batch(
            sample_indices[start:stop], slots[start:stop], tactical=tactical)
        logits = _call(model, observation, prior, has_prior, device)
        target = torch.from_numpy(labels).to(device)
        prior_tensor = torch.from_numpy(prior).to(device)
        has_prior_tensor = torch.from_numpy(has_prior).to(device)
        changed_tensor = torch.from_numpy(changed).to(device)
        losses.append(F.cross_entropy(logits, target, reduction="none").cpu().numpy())
        raw_prediction = logits.argmax(-1)
        prediction = raw_prediction.cpu().numpy()
        majority_correct += int((labels == majority_action).sum())
        exact.append(prediction == labels)
        moves.append((prediction % 9) == (labels % 9))
        predicted_kick = prediction >= 9
        actual_kick = labels >= 9
        kicks.append(predicted_kick == actual_kick)
        true_kick += int(actual_kick.sum())
        pred_kick += int(predicted_kick.sum())
        true_positive += int((actual_kick & predicted_kick).sum())
        active_prior = has_prior
        if active_prior.any():
            repeated_correct += int((prior[active_prior] == labels[active_prior]).sum())
            repeated_total += int(active_prior.sum())
            prior_correct += int((raw_prediction[has_prior_tensor] == target[has_prior_tensor]).sum())
        changed_mask = active_prior & changed
        changed_total += int(changed_mask.sum())
        if changed_mask.any():
            changes.append((prediction[changed_mask] == labels[changed_mask]))
        rows = torch.arange(len(labels), device=device)
        prior_logits = logits[rows, prior_tensor]
        alternatives = logits.clone()
        alternatives[rows, prior_tensor] = -torch.inf
        alternative_logits, alternative_actions = alternatives.max(-1)
        if has_prior_tensor.any():
            switch_gaps.append((alternative_logits[has_prior_tensor] - prior_logits[has_prior_tensor]).cpu().numpy())
        for margin in switch_margins:
            should_switch = has_prior_tensor & (alternative_logits > prior_logits + margin)
            held_action = torch.where(should_switch, alternative_actions, prior_tensor)
            adjusted = torch.where(has_prior_tensor, held_action, raw_prediction)
            switch_correct[margin] += int((adjusted == target).sum())
            if has_prior_tensor.any():
                switch_prior_correct[margin] += int((adjusted[has_prior_tensor] == target[has_prior_tensor]).sum())
            if changed_tensor.any():
                switch_changed_correct[margin] += int((adjusted[changed_tensor] == target[changed_tensor]).sum())
    loss = np.concatenate(losses)
    result = {
        "nll": float(loss.mean()),
        "accuracy": float(np.concatenate(exact).mean()),
        "moveAccuracy": float(np.concatenate(moves).mean()),
        "kickAccuracy": float(np.concatenate(kicks).mean()),
        "kickPrecision": true_positive / max(1, pred_kick),
        "kickRecall": true_positive / max(1, true_kick),
        "changedActionAccuracy": float(np.concatenate(changes).mean()) if changes else None,
        "changedActionSamples": int(sum(len(values) for values in changes)),
        "majorityAccuracy": majority_correct / max(1, len(sample_indices)),
        "repeatAccuracy": repeated_correct / repeated_total if repeated_total else None,
        "accuracyWithPrior": prior_correct / repeated_total if repeated_total else None,
        "switchMarginMetrics": {
            f"{margin:g}": {
                "accuracy": correct / max(1, len(sample_indices)),
                "accuracyWithPrior": switch_prior_correct[margin] / max(1, repeated_total),
                "changedActionAccuracy": switch_changed_correct[margin] / max(1, changed_total),
            }
            for margin, correct in switch_correct.items()
        },
        "switchGapQuantiles": {
            str(q): float(np.quantile(np.concatenate(switch_gaps), q))
            for q in (0.5, 0.9, 0.95, 0.99, 1.0)
        } if switch_gaps else None,
        "samples": len(sample_indices),
    }
    model.train()
    return result


def _atomic_torch_save(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(value, temp)
        temp.replace(path)
    finally:
        if temp.exists():
            temp.unlink()


@torch.no_grad()
def _closed_loop_probe(checkpoint_path: Path, matches: int, minutes: float, seed: int) -> dict:
    """Run a fast closed-loop probe against ScriptedPolicy to select checkpoints by real gameplay."""
    from learn.x3futsal_eval import ModelPolicy, ScriptedPolicy, STADIUM, _aggregate, play_batch
    from sim.stadium import load_stadium

    stadium = load_stadium(STADIUM)
    candidate = ModelPolicy(checkpoint_path, torch.device("cpu"))
    scripted = ScriptedPolicy(stadium)
    games = []
    for red, blue, policy_index in ((candidate, scripted, 0), (scripted, candidate, 1)):
        side_games = play_batch(red, blue, matches=matches, minutes=minutes, seed=seed)
        games.extend({**game, "firstPolicyTeam": policy_index} for game in side_games)
    summary = _aggregate(games, 0)
    competitive_index = (
        summary["goalDifferencePerMatch"]
        + 0.05 * summary["shotsPerMatch"]
        + 0.02 * summary["completedPassesPerMatch"]
        + 0.01 * summary["kicksPerMatch"]
        - 0.50 * summary["safetyKickoffsPerMatch"]
    )
    summary["competitiveIndex"] = float(competitive_index)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--eval-checkpoint", type=Path, help="evaluar un checkpoint existente sin entrenar")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-n", type=int, default=20000)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--entity-hidden", type=int, default=96)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tactical", action="store_true", default=False,
                        help="usar observación táctica v3 (28/10 dims) y X3TacticalPolicy sin copycat")
    parser.add_argument("--use-history", action="store_true", default=False,
                        help="incluir acción previa en modo táctico (con --history-dropout)")
    parser.add_argument("--history-dropout", type=float, default=0.5,
                        help="probabilidad de borrar la acción previa durante entrenamiento")
    parser.add_argument("--near-ball-frac", type=float, default=0.35,
                        help="fracción del lote muestreada con el jugador a <= 140 px de la pelota")
    parser.add_argument("--no-mirror", action="store_true",
                        help="desactivar aumento de datos por reflexión vertical (y -> -y)")
    parser.add_argument("--change-weight", type=float, default=CHANGE_ACTION_WEIGHT)
    parser.add_argument("--kick-weight", type=float, default=KICK_ACTION_WEIGHT)
    parser.add_argument("--closed-loop-every", type=int, default=2000,
                        help="cada N pasos evaluar en lazo cerrado contra scripted (0 = desactivar)")
    parser.add_argument("--closed-loop-matches", type=int, default=2)
    parser.add_argument("--closed-loop-minutes", type=float, default=0.5)
    parser.add_argument("--resume", type=Path, help="continuar desde un checkpoint X3 compatible")
    args = parser.parse_args()
    if min(args.steps, args.batch, args.eval_every, args.eval_n, args.hidden, args.threads) < 1:
        parser.error("steps, batch, eval-every, eval-n, hidden y threads deben ser positivos")
    if args.eval_checkpoint and args.resume:
        parser.error("--eval-checkpoint y --resume no se pueden usar juntos")
    if not args.eval_checkpoint and not args.out:
        parser.error("--out es obligatorio al entrenar")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    tactical = bool(args.tactical)
    train = X3Data(args.dataset, "train", tactical=tactical)
    dev = X3Data(args.dataset, "dev", tactical=tactical)
    labels_count = train.label_counts()
    majority_action = int(labels_count.argmax())
    dev_indices, dev_slots = dev.fixed_indices(args.eval_n, args.seed + 11)
    if tactical:
        model = X3TacticalPolicy(
            hidden=args.hidden,
            entity_hidden=args.entity_hidden,
            use_history=args.use_history,
            history_dropout=args.history_dropout,
        ).to(device)
        policy_version = TACTICAL_POLICY_VERSION
        feature_contract = TACTICAL_FEATURE_CONTRACT
        training_objective = TACTICAL_TRAINING_OBJECTIVE
        obs_dims = [TACTICAL_SELF_DIM, TACTICAL_ENTITY_DIM, TACTICAL_ENTITY_DIM,
                    19 if args.use_history else 0]
    else:
        model = X3SetPolicy(hidden=args.hidden, history_dropout=args.history_dropout).to(device)
        policy_version = POLICY_VERSION
        feature_contract = FEATURE_CONTRACT
        training_objective = TRAINING_OBJECTIVE
        obs_dims = [12, 6, 6, 19]

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    out = args.out
    dataset_manifest = json.loads((args.dataset / "manifest.json").read_text(encoding="utf-8"))
    dataset_signature = {
        "dataVersion": dataset_manifest["dataVersion"],
        "targetStadiumIdentity": dataset_manifest["targetStadiumIdentity"],
        "splitSeed": dataset_manifest["split"]["seed"],
        "shards": [(row["recording"], row["split"], row["ticks"]) for row in dataset_manifest["shards"]],
    }
    if args.eval_checkpoint:
        checkpoint = torch.load(args.eval_checkpoint, map_location="cpu", weights_only=False)
        ckpt_tactical = checkpoint.get("policyVersion") == TACTICAL_POLICY_VERSION
        if ckpt_tactical != tactical:
            raise ValueError(f"checkpoint incompatible con modo --tactical={tactical}: {args.eval_checkpoint}")
        if (checkpoint.get("obs_version") not in (DATA_VERSION, TACTICAL_DATA_VERSION)
                or checkpoint.get("policyVersion") != policy_version
                or checkpoint.get("featureContract") != feature_contract
                or checkpoint.get("datasetSignature") != dataset_signature
                or checkpoint.get("hidden") != args.hidden):
            raise ValueError(f"checkpoint incompatible con el contrato o dataset: {args.eval_checkpoint}")
        model.load_state_dict(checkpoint["model"])
        print(json.dumps(evaluate(model, dev, dev_indices, dev_slots, majority_action,
                                  args.batch, device, tactical=tactical)), flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    start_step = 0
    best_nll = None
    best_cl_index = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if (checkpoint.get("obs_version") not in (DATA_VERSION, TACTICAL_DATA_VERSION)
                or checkpoint.get("policyVersion") != policy_version
                or checkpoint.get("featureContract") != feature_contract
                or checkpoint.get("datasetSignature") != dataset_signature):
            raise ValueError(f"checkpoint incompatible con el contrato o dataset: {args.resume}")
        if checkpoint.get("hidden") != args.hidden or "optimizer" not in checkpoint:
            raise ValueError("el checkpoint requiere el mismo --hidden y estado de optimizador")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
        if start_step >= args.steps:
            parser.error("--steps debe ser mayor que el paso del checkpoint")
        if "numpyRng" not in checkpoint or "torchRng" not in checkpoint:
            raise ValueError("checkpoint sin estado RNG; no se puede reanudar de forma reproducible")
        rng.bit_generator.state = checkpoint["numpyRng"]
        torch.set_rng_state(checkpoint["torchRng"])
        best_nll = checkpoint.get("bestNll")
        best_cl_index = checkpoint.get("bestCompetitiveIndex")
    elif (out / "last.pt").exists():
        parser.error(f"ya existe {out / 'last.pt'}; usa --resume para continuar o elige otro --out")

    metrics_path = out / "metrics.json"
    log = {
        "args": vars(args) | {"dataset": str(args.dataset), "out": str(args.out),
                              "resume": str(args.resume) if args.resume else None},
        "dataVersion": TACTICAL_DATA_VERSION if tactical else DATA_VERSION,
        "policyVersion": policy_version,
        "trainingObjective": training_objective,
        "changeActionWeight": args.change_weight,
        "kickActionWeight": args.kick_weight,
        "datasetSignature": dataset_signature,
        "trainTicks": train.ticks,
        "devTicks": dev.ticks,
        "trainLabelCounts": labels_count.tolist(),
        "majorityAction": majority_action,
        "evaluations": [],
    }
    if args.resume and metrics_path.is_file():
        previous_log = json.loads(metrics_path.read_text(encoding="utf-8"))
        if previous_log.get("datasetSignature") != dataset_signature:
            raise ValueError("metrics existentes pertenecen a otro dataset")
        log = previous_log
        log["args"] = vars(args) | {"dataset": str(args.dataset), "out": str(args.out),
                                    "resume": str(args.resume)}
    if args.resume and not best_nll:
        best_nll = log.get("bestNll")
    t0 = time.time()
    for step in range(start_step + 1, args.steps + 1):
        indices, slots = train.sample_indices(args.batch, rng, near_ball_frac=args.near_ball_frac)
        mirror_mask = None if args.no_mirror else (rng.random(args.batch) < 0.5)
        observation, labels, prior, has_prior, changed = train.batch(
            indices, slots, tactical=tactical, mirror_mask=mirror_mask
        )
        logits = _call(model, observation, prior, has_prior, device)
        target = torch.from_numpy(labels).to(device)
        per_sample_loss = F.cross_entropy(logits, target, reduction="none")
        changed_mask = torch.from_numpy(changed).to(device)
        kick_mask = target >= 9
        sample_weights = torch.maximum(
            torch.where(changed_mask, float(args.change_weight), 1.0),
            torch.where(kick_mask, float(args.kick_weight), 1.0),
        )
        loss = (per_sample_loss * sample_weights).sum() / sample_weights.sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(model, dev, dev_indices, dev_slots, majority_action,
                               args.batch, device, tactical=tactical)
            metrics.update(step=step, trainWeightedLoss=float(loss.detach().cpu()),
                           seconds=round(time.time() - t0, 1))
            checkpoint = {
                "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "hidden": args.hidden,
                "entityHidden": args.entity_hidden,
                "useHistory": args.use_history,
                "obs_version": TACTICAL_DATA_VERSION if tactical else DATA_VERSION,
                "policyVersion": policy_version,
                "trainingObjective": training_objective,
                "action_version": "directions9-kick-v1",
                "obs_dims": obs_dims,
                "featureContract": feature_contract,
                "step": step,
                "metrics": metrics,
                "optimizer": optimizer.state_dict(),
                "numpyRng": rng.bit_generator.state,
                "torchRng": torch.get_rng_state(),
                "bestNll": min(metrics["nll"], best_nll) if best_nll is not None else metrics["nll"],
                "bestCompetitiveIndex": best_cl_index,
                "datasetSignature": dataset_signature,
            }
            _atomic_torch_save(out / "last.pt", checkpoint)
            if best_nll is None or metrics["nll"] < best_nll:
                best_nll = metrics["nll"]
                log["bestNll"] = best_nll
                _atomic_torch_save(out / "best.pt", checkpoint)
            if args.closed_loop_every > 0 and (step % args.closed_loop_every == 0 or step == args.steps):
                cl = _closed_loop_probe(
                    out / "last.pt",
                    matches=args.closed_loop_matches,
                    minutes=args.closed_loop_minutes,
                    seed=args.seed + step,
                )
                metrics["closedLoopVsScripted"] = cl
                if best_cl_index is None or cl["competitiveIndex"] > best_cl_index:
                    best_cl_index = cl["competitiveIndex"]
                    log["bestCompetitiveIndex"] = best_cl_index
                    checkpoint["bestCompetitiveIndex"] = best_cl_index
                    _atomic_torch_save(out / "best_competitive.pt", checkpoint)
            log["evaluations"].append(metrics)
            print(json.dumps(metrics), flush=True)
            log["elapsedSeconds"] = round(time.time() - t0, 1)
            temp_log = metrics_path.with_suffix(".json.tmp")
            temp_log.write_text(json.dumps(log, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            temp_log.replace(metrics_path)


if __name__ == "__main__":
    main()