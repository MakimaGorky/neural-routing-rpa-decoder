"""Evaluate a saved experiment checkpoint with the run's final_eval config."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from routing_rpa.experiments.build import build_experiment
from routing_rpa.experiments.config import load_experiment_config
from routing_rpa.experiments.runner import find_best_checkpoint
from routing_rpa.utils.serialization import dumps_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--layer-id", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config_path = args.run_dir / "config.json"
    config = load_experiment_config(config_path)
    components = build_experiment(config)

    checkpoint_path = args.checkpoint
    checkpoint_decision = None
    if checkpoint_path is None:
        checkpoint = find_best_checkpoint(args.run_dir / "checkpoints", layer_id=args.layer_id)
        checkpoint_path = checkpoint.path
        checkpoint_decision = checkpoint.decision

    payload = torch.load(checkpoint_path, map_location=components.device, weights_only=True)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise TypeError(f"Checkpoint does not contain model_state_dict: {checkpoint_path}")
    components.model.load_state_dict(payload["model_state_dict"])
    if checkpoint_decision is None:
        decision = payload.get("checkpoint_decision")
        checkpoint_decision = decision if isinstance(decision, dict) else None

    metrics = dict(
        components.evaluator.evaluate(
            components.evaluation_mode,
            components.final_eval_config,
        )
    )
    result = {
        "run_dir": str(args.run_dir),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_decision": checkpoint_decision,
        "metrics": metrics,
    }

    if args.output is not None:
        write_json(args.output, result)
    print(dumps_json(result))


if __name__ == "__main__":
    main()
