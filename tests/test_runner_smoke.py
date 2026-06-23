import json
from dataclasses import replace
from pathlib import Path

import torch

from routing_rpa.experiments.config import (
    ChannelConfig,
    CodeConfig,
    DecoderConfig,
    EvalConfig,
    ExperimentConfig,
    LoggingConfig,
    PathsConfig,
    ProjectionConfig,
    RoutingConfig,
    TrainingConfig,
)
from routing_rpa.experiments.runner import run_experiment


def write_toy_artifacts(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    torch.save(torch.eye(4, dtype=torch.float32), artifacts / "G.pt")
    torch.save(
        {
            "m": 2,
            "n": 4,
            "subspace_dim": 1,
            "directions": torch.tensor([1, 2, 3], dtype=torch.long),
            "coset_indices": torch.tensor(
                [
                    [[0, 1], [2, 3]],
                    [[0, 2], [1, 3]],
                    [[0, 3], [1, 2]],
                ],
                dtype=torch.long,
            ),
            "metadata": {"name": "toy", "family": "unit_test"},
        },
        artifacts / "projections.pt",
    )


def make_training_config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        seed=13,
        device="cpu",
        paths=PathsConfig(project_root=str(tmp_path), runs_dir="runs"),
        code=CodeConfig(
            family="rm",
            m=2,
            r=2,
            generator_artifact="artifacts/G.pt",
        ),
        projections=ProjectionConfig(
            artifact="artifacts/projections.pt",
            expected_count=3,
        ),
        decoder=DecoderConfig(num_unfolded_steps=1),
        routing=RoutingConfig(
            policy="input_dependent_router",
            router="mlp",
            hidden_size=5,
            selection_scope="full",
            execution_mode="compute_all_mask",
            top_k=None,
        ),
        channel_train=ChannelConfig(name="awgn", snr=1.0),
        validation=EvalConfig(phase="validation", num_batches=1, batch_size=2, snr=1.0),
        final_eval=EvalConfig(phase="final_eval", num_batches=1, batch_size=2, snr=1.25),
        training=TrainingConfig(
            enabled=True,
            epochs=1,
            steps_per_epoch=1,
            batch_size=2,
            optimizer="sgd",
            lr=0.01,
            train_layers=[0],
        ),
        logging=LoggingConfig(experiment_name="runner-smoke"),
    )


def test_runner_mini_flow_writes_structured_outputs(tmp_path):
    write_toy_artifacts(tmp_path)
    config = make_training_config(tmp_path)

    result = run_experiment(config, timestamp="2026-06-15--12-00-00")

    run = result.run_directory
    assert run.config_json.is_file()
    assert run.metrics_jsonl.is_file()
    assert run.summary_json.is_file()
    assert run.artifacts_used_json.is_file()

    config_payload = json.loads(run.config_json.read_text(encoding="utf-8"))
    assert config_payload["logging"]["experiment_name"] == "runner-smoke"

    records = [
        json.loads(line)
        for line in run.metrics_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    phases = [record["phase"] for record in records]
    assert phases == ["baseline", "train", "validation", "final_eval"]
    assert all("ber" in record and "bler" in record for record in records if record["phase"] != "train")

    summary = json.loads(run.summary_json.read_text(encoding="utf-8"))
    assert summary["training_records"] == 1
    assert summary["baseline"]["phase"] == "baseline"
    assert summary["final_eval"]["phase"] == "final_eval"

    checkpoint_files = list(run.checkpoints_dir.glob("*.pt"))
    assert len(checkpoint_files) == 1
    checkpoint = torch.load(checkpoint_files[0], map_location="cpu", weights_only=True)
    assert "model_state_dict" in checkpoint
    assert checkpoint["checkpoint_decision"]["should_save"] is True
    assert checkpoint["checkpoint_decision"]["metadata"]["policy_name"] == (
        "bler_primary_with_ber_tiebreaker"
    )

    artifacts_used = json.loads(run.artifacts_used_json.read_text(encoding="utf-8"))
    assert artifacts_used["projection_metadata"]["name"] == "toy"
    assert result.training_result is not None


def test_runner_can_use_best_checkpoint_for_final_eval(tmp_path):
    write_toy_artifacts(tmp_path)
    config = make_training_config(tmp_path)
    config = replace(
        config,
        final_eval=replace(config.final_eval, model_source="best_checkpoint"),
    )

    result = run_experiment(config, timestamp="2026-06-15--12-30-00")

    summary = json.loads(result.run_directory.summary_json.read_text(encoding="utf-8"))
    final_eval_model = summary["final_eval_model"]
    assert final_eval_model["source"] == "best_checkpoint"
    assert final_eval_model["layer_id"] == 0
    assert Path(final_eval_model["checkpoint_path"]).is_file()
    assert summary["final_eval"]["model_source"] == "best_checkpoint"
    assert Path(summary["final_eval"]["checkpoint_path"]).is_file()
