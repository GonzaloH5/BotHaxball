"""Train and independently validate a weak RS4 imitation candidate from shards.

Never replaces PPO weights or the active BC reference. A validation report is
required before a user deliberately activates the candidate as a weak teacher.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from train.model import build_model
from tools.build_rs4_sequences import iter_shards, ROOT
from tools.audit_rs4_simulator import digest


def batches(manifest, split, batch_sequences, *, shuffle=False, seed=0):
    """Only one shard is materialized in RAM; padding stays masked."""
    rng = np.random.default_rng(seed)
    for shard in iter_shards(manifest, split):
        index = np.arange(len(shard["obs"]))
        if shuffle:
            rng.shuffle(index)
        for start in range(0, len(index), batch_sequences):
            ix = index[start:start + batch_sequences]
            yield {key: shard[key][ix] for key in
                   ("obs", "act", "previous_action", "episode_start", "valid", "labels")}


def batch_logits(model, batch, device):
    obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=device)
    valid = torch.as_tensor(batch["valid"], dtype=torch.bool, device=device)
    labels = torch.as_tensor(batch["act"], dtype=torch.long, device=device)
    if getattr(model, "is_recurrent", False):
        previous = torch.as_tensor(batch["previous_action"], dtype=torch.long, device=device)
        starts = torch.as_tensor(batch["episode_start"], dtype=torch.bool, device=device)
        logits, _, _ = model.sequence(obs.transpose(0, 1), model.initial_state(len(obs)),
                                      previous.transpose(0, 1), starts.transpose(0, 1))
        logits = logits.transpose(0, 1)
    else:
        logits = model.logits(obs.reshape(-1, obs.shape[-1])).reshape(*labels.shape, -1)
    return logits, labels, valid


@torch.no_grad()
def validate(model, manifest, batch_sequences, device):
    model.eval()
    totals = np.zeros(3, dtype=np.float64)  # sample, correct, summed CE
    recognized = np.zeros(3, dtype=np.float64)
    by_label = {}
    for batch in batches(manifest, "validation", batch_sequences):
        logits, labels, valid = batch_logits(model, batch, device)
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none").reshape_as(labels)
        correct = logits.argmax(-1) == labels
        scenario = torch.as_tensor(batch["labels"], device=device)
        for mask, accumulator in ((valid, totals), (valid & (scenario >= 0), recognized)):
            accumulator += np.array([mask.sum().item(), (correct & mask).sum().item(), ce[mask].sum().item()])
        for label in (1, 2, 4):
            mask = valid & (scenario == label)
            count, successes = int(mask.sum()), int((correct & mask).sum())
            prior = by_label.setdefault(str(label), {"samples": 0, "correct": 0})
            prior["samples"] += count
            prior["correct"] += successes
    def result(array):
        return {"samples": int(array[0]), "accuracy": float(array[1] / array[0]) if array[0] else None,
                "cross_entropy": float(array[2] / array[0]) if array[0] else None}
    return {"general": result(totals), "recognized_restarts": result(recognized), "by_label": by_label}


def validation_gate(candidate, reference):
    candidate_general, reference_general = candidate["general"], reference["general"]
    candidate_specific, reference_specific = candidate["recognized_restarts"], reference["recognized_restarts"]
    enough = candidate_general["samples"] > 0 and candidate_specific["samples"] >= 128
    approved = bool(enough and candidate_general["accuracy"] >= reference_general["accuracy"]
                    and candidate_specific["accuracy"] > reference_specific["accuracy"]
                    and candidate_general["cross_entropy"] <= reference_general["cross_entropy"])
    return {"approved_as_weak_reference": approved, "minimum_specific_samples": 128,
            "policy": "specific accuracy improves; general accuracy and CE do not degrade",
            "activated": False}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--source", default=str(ROOT / "runs" / "bc_rs4_v2_20261001" / "bc.pt"))
    ap.add_argument("--run", default="bc_rs4_v3_candidate")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-sequences", type=int, default=128)
    ap.add_argument("--architecture", choices=("feedforward", "memory"), default="feedforward")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()
    run = ROOT / "runs" / args.run
    if args.run in ("", ".", "..") or Path(args.run).name != args.run or run.exists():
        ap.error("Use a new simple run name; existing runs are never overwritten")
    if args.epochs < 1 or not 1 <= args.batch_sequences <= 256 or args.lr <= 0:
        ap.error("Invalid epochs, batch size or LR")
    if not Path(args.source).is_file() or not Path(args.manifest).is_file():
        ap.error("Source and shard manifest must exist")
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if manifest.get("cadence_ticks") != 3 or manifest.get("version") != 3:
        ap.error("RS4 v3 decision-cadence sequence manifest required")
    if not all(any(r["split"] == split and r["useful_samples"] > 0 for r in manifest["replays"])
               for split in ("train", "validation")):
        ap.error("Both replay-disjoint training and validation samples required")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = "cpu" if device == "auto" else device
    torch.set_num_threads(2)
    torch.manual_seed(51)
    source = torch.load(args.source, map_location="cpu", weights_only=False)
    reference = build_model(source["model_config"])
    reference.load_state_dict(source["model"])
    reference.to(device)
    config = copy.deepcopy(source["model_config"])
    if args.architecture == "memory":
        config.update(type="recurrent_set", memory_size=32, pooling="attentive_meanmax")
    candidate = build_model(config)
    if config == source["model_config"]:
        candidate.load_state_dict(source["model"])
    else:
        candidate.initialize_from(source)
    candidate.to(device)
    # RunningNorm has no automatic update in forward. Never call update_norm:
    # current teacher and candidate share frozen source normalization.
    reference_metrics = validate(reference, args.manifest, args.batch_sequences, device)
    optimizer = torch.optim.Adam(candidate.parameters(), lr=args.lr)
    run.mkdir(parents=True)
    best = None
    history = []
    for epoch in range(args.epochs):
        candidate.train()
        total_loss, samples = 0.0, 0
        for batch in batches(args.manifest, "train", args.batch_sequences, shuffle=True, seed=51 + epoch):
            logits, labels, valid = batch_logits(candidate, batch, device)
            if not valid.any():
                continue
            loss = F.cross_entropy(logits[valid], labels[valid])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(candidate.parameters(), 1.0)
            optimizer.step()
            n = int(valid.sum())
            total_loss += float(loss.detach()) * n
            samples += n
        metrics = validate(candidate, args.manifest, args.batch_sequences, device)
        history.append({"epoch": epoch + 1, "train_ce": total_loss / max(samples, 1), "validation": metrics})
        print(f"epoch {epoch + 1}: CE {history[-1]['train_ce']:.4f}; validation {metrics['general']['accuracy']:.4f}", flush=True)
        if best is None or metrics["general"]["cross_entropy"] < best["general"]["cross_entropy"]:
            best = metrics
            torch.save({"model": candidate.state_dict(), "model_config": candidate.config(),
                        "steps": 0, "iteration": 0, "bc": {"source": args.source, "source_sha256": digest(args.source),
                        "sequence_manifest": args.manifest, "validation": metrics, "normalizers_frozen": True}}, run / "bc.pt")
    report = {"version": 3, "reference": reference_metrics, "candidate": best,
              "gate": validation_gate(best, reference_metrics), "history": history,
              "source_sha256": digest(args.source), "candidate_sha256": digest(run / "bc.pt"),
              "manifest_sha256": digest(args.manifest),
              "limitations": ["Accuracy is imitation validation, not proof of football skill.",
                               "A memory teacher requires compatible recurrent BC training support before activation."]}
    (run / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Candidate saved at {run / 'bc.pt'}; validation gate {report['gate']['approved_as_weak_reference']}; NOT activated")


if __name__ == "__main__":
    main()
