from pathlib import Path

import torch

from routing_rpa.decoders.routing import (
    InputDependentRouterRouting,
    SelectedStaticRouting,
    UniformRouting,
)
from routing_rpa.experiments.build import build_experiment, select_device
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


def make_config(
    tmp_path: Path,
    *,
    routing: RoutingConfig,
    training: TrainingConfig | None = None,
) -> ExperimentConfig:
    return ExperimentConfig(
        seed=11,
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
        routing=routing,
        channel_train=ChannelConfig(name="awgn", snr=1.0),
        validation=EvalConfig(phase="validation", num_batches=1, batch_size=2, snr=1.0),
        final_eval=EvalConfig(phase="final_eval", num_batches=1, batch_size=2, snr=1.0),
        training=training if training is not None else TrainingConfig(enabled=False),
        logging=LoggingConfig(experiment_name="build-test"),
    )


def test_build_creates_uniform_full_without_router(tmp_path):
    write_toy_artifacts(tmp_path)
    config = make_config(
        tmp_path,
        routing=RoutingConfig(
            policy="uniform_full",
            selection_scope="full",
            execution_mode="compute_all_mask",
            top_k=None,
        ),
    )

    components = build_experiment(config)

    assert isinstance(components.routing_policy, UniformRouting)
    assert not hasattr(components.routing_policy, "router")
    assert list(components.routing_policy.parameters()) == []
    assert components.trainer is None


def test_build_creates_input_dependent_router_when_configured(tmp_path):
    write_toy_artifacts(tmp_path)
    config = make_config(
        tmp_path,
        routing=RoutingConfig(
            policy="input_dependent_router",
            router="mlp",
            hidden_size=5,
            selection_scope="full",
            execution_mode="compute_all_mask",
            top_k=None,
        ),
    )

    components = build_experiment(config)

    assert isinstance(components.routing_policy, InputDependentRouterRouting)
    assert components.routing_policy.router.num_steps == 1
    assert components.routing_policy.router(torch.zeros(2, 4), step=0).shape == (2, 3)


def test_build_creates_selected_static_pruning_mode(tmp_path):
    write_toy_artifacts(tmp_path)
    config = make_config(
        tmp_path,
        routing=RoutingConfig(
            policy="selected_static_topk",
            selection_scope="static",
            execution_mode="compute_selected",
            top_k=2,
            selected_indices=[2, 0],
        ),
    )

    components = build_experiment(config)
    output = components.model(
        torch.zeros(1, components.code.n),
        components.evaluation_mode.with_channel_context(snr=1.0),
    )

    assert isinstance(components.routing_policy, SelectedStaticRouting)
    assert output.stats["candidate_projections"] == 3
    assert output.stats["executed_projections"] == 2
    assert output.stats["aggregated_projections"] == 2
    assert output.stats["selection_scope"] == "static"
    assert output.stats["execution_mode"] == "compute_selected"


def test_select_device_canonicalizes_bare_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    assert select_device("cuda") == torch.device("cuda:0")
    assert select_device("auto") == torch.device("cuda:0")


def test_select_device_preserves_explicit_cuda_index(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    assert select_device("cuda:1") == torch.device("cuda:1")
