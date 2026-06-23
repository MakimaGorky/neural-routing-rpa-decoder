import json
from pathlib import Path

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
    TrainingDiagnosticsConfig,
    load_experiment_config,
    save_experiment_config,
)


def make_config(project_root: Path) -> ExperimentConfig:
    return ExperimentConfig(
        seed=7,
        device="cpu",
        paths=PathsConfig(project_root=str(project_root), runs_dir="runs"),
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
            policy="uniform_full",
            selection_scope="full",
            execution_mode="compute_all_mask",
            top_k=None,
        ),
        channel_train=ChannelConfig(name="awgn", snr=1.5),
        validation=EvalConfig(phase="validation", num_batches=1, batch_size=2, snr=1.5),
        final_eval=EvalConfig(phase="final_eval", num_batches=2, batch_size=3, snr=2.5),
        training=TrainingConfig(enabled=False),
        logging=LoggingConfig(experiment_name="unit-test"),
    )


def test_experiment_config_dataclasses_serialize_and_deserialize(tmp_path):
    config = make_config(tmp_path)
    payload = config.to_dict()

    loaded = ExperimentConfig.from_dict(payload)

    assert loaded.to_dict() == payload
    assert loaded.validation.phase == "validation"
    assert loaded.final_eval.phase == "final_eval"
    assert loaded.validation.snr != loaded.final_eval.snr


def test_experiment_config_json_round_trip(tmp_path):
    config = make_config(tmp_path)
    path = tmp_path / "experiment.json"

    save_experiment_config(path, config)
    loaded = load_experiment_config(path)

    assert loaded.to_dict() == config.to_dict()
    assert json.loads(path.read_text(encoding="utf-8")) == config.to_dict()


def test_experiment_config_accepts_checkpoint_source_and_training_diagnostics(tmp_path):
    config = make_config(tmp_path)
    payload = config.to_dict()
    payload["final_eval"]["model_source"] = "best_checkpoint"
    payload["training"]["full_cascade_validation"] = {
        "phase": "validation_full",
        "num_batches": 1,
        "batch_size": 2,
        "snr": 1.5,
    }
    payload["training"]["diagnostics"] = {
        "enabled": True,
        "phase": "diagnostics",
        "num_batches": 1,
        "batch_size": 2,
        "snr": 1.5,
    }

    loaded = ExperimentConfig.from_dict(payload)

    assert loaded.final_eval.model_source == "best_checkpoint"
    assert loaded.training.full_cascade_validation is not None
    assert loaded.training.full_cascade_validation.phase == "validation_full"
    assert isinstance(loaded.training.diagnostics, TrainingDiagnosticsConfig)
    assert loaded.training.diagnostics.enabled is True


def test_required_sample_configs_are_valid_json_objects():
    root = Path(__file__).resolve().parents[1]
    sample_names = {
        "uniform_full.json",
        "full_masked_topk.json",
        "selected_static_topk.json",
        "input_dependent_compute_all_mask.json",
    }

    for name in sample_names:
        config = load_experiment_config(root / "configs" / name)

        assert config.paths.project_root == "."
        assert config.code.family == "rm"
        assert config.projections.expected_count == 512
        assert "src_old" not in config.code.generator_artifact
        assert "src_old" not in config.projections.artifact
        assert (root / config.code.generator_artifact).is_file()
        assert (root / config.projections.artifact).is_file()
