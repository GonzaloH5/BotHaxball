"""Entrena por hitos de 25M, evalúa gates y para tras dos aprobaciones seguidas."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from eval.coordination import run as run_evaluation
from .multitask import MultiTrainer, ROOT
from .runtime import load_config


def _path(value):
    value = Path(str(value).replace("\\", "/"))
    return value if value.is_absolute() else ROOT / value


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--run")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init-from")
    ap.add_argument("--skip-evaluation", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    trainer_class = MultiTrainer
    if cfg["model"].get("type") == "recurrent_set":
        from .recurrent_ppo import RecurrentTrainer
        trainer_class = RecurrentTrainer
    trainer = trainer_class(cfg, args.run or cfg["run_name"], args.resume, args.init_from)
    evaluation = cfg.get("evaluation", {})
    every = int(evaluation.get("every_steps", 25_000_000))
    next_step = ((trainer.steps // every) + 1) * every
    streak = 0
    last_report = None
    try:
        while trainer.steps < cfg["ppo"]["total_steps"]:
            trainer.iterate()
            if args.skip_evaluation or trainer.steps < next_step:
                continue
            checkpoint = trainer.run_dir / f"step_{next_step:09d}.pt"
            trainer.save(checkpoint)
            suite = yaml.safe_load(_path(evaluation["suite"]).read_text(encoding="utf-8"))
            report = run_evaluation(checkpoint, _path(evaluation["baseline"]), suite,
                                    [_path(p) for p in evaluation.get("historical", [])])
            streak = streak + 1 if report["gate"]["passed"] else 0
            report["gate"]["consecutive_passes"] = streak
            report["gate"]["early_stop"] = streak >= int(evaluation.get("stop_after_consecutive", 2))
            reports = _path(evaluation["reports_dir"])
            reports.mkdir(parents=True, exist_ok=True)
            last_report = reports / f"step_{next_step:09d}.json"
            last_report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            print(json.dumps(report["gate"], ensure_ascii=False), flush=True)
            next_step += every
            if report["gate"]["early_stop"]:
                print("*** parada anticipada: dos evaluaciones consecutivas aprobaron ***", flush=True)
                break
    except KeyboardInterrupt:
        print("interrumpido: guardando...", flush=True)
    finally:
        trainer.save(trainer.run_dir / "latest.pt")
        trainer.writer.close()
    print(f"guardado en {trainer.run_dir / 'latest.pt'}")
    if last_report:
        print(f"última evaluación: {last_report}")


if __name__ == "__main__":
    main()

