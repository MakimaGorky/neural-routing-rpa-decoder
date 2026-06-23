import torch
from torch import nn

from routing_rpa.decoders.outputs import DecoderOutput
from routing_rpa.decoders.routing import InputDependentRouterRouting, StepwiseRouter
from routing_rpa.eval.evaluator import EvaluationConfig
from routing_rpa.training.checkpoints import BlerPrimaryWithBERTieBreakerPolicy
from routing_rpa.training.losses import LossComposer, LossConfig
from routing_rpa.training.strategies import LayerwiseTraining
from routing_rpa.training.trainer import DiagnosticsConfig, Trainer, TrainerConfig


class TinyStream:
    def __init__(self) -> None:
        self.generator = torch.Generator().manual_seed(0)

    def next_batch(self, batch_size: int, snr: float):
        target = torch.randint(
            0,
            2,
            (batch_size, 4),
            generator=self.generator,
            dtype=torch.float32,
        )
        channel_output = target.mul(2.0).sub(1.0) + snr * 0.0
        return target, channel_output


class RecordingLayeredModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_unfolded_steps = 2
        routers = [nn.Linear(4, 4), nn.Linear(4, 4)]
        self.routing_policy = InputDependentRouterRouting(StepwiseRouter(routers))
        self.modes = []

    def set_training_phase(self, *args, **kwargs):
        raise AssertionError("trainer must not mutate decoder phase")

    def forward(self, channel_output, mode):
        self.modes.append(mode)
        layer_id = mode.target_layer if mode.target_layer is not None else 0
        logits = self.routing_policy.router.routers[layer_id](channel_output)
        return DecoderOutput(
            logits=logits,
            aux={},
            stats={
                "candidate_projections": 1,
                "executed_projections": 1,
                "aggregated_projections": 1,
                "selection_scope": mode.selection_scope,
                "execution_mode": mode.execution_mode,
            },
        )


class InMemoryLogger:
    def __init__(self) -> None:
        self.records = []

    def write(self, record):
        self.records.append(dict(record))


def make_loss_composer():
    return LossComposer(LossConfig(components=[{"name": "bce_logits", "weight": 1.0}]))


def test_tiny_cpu_layerwise_training_smoke_logs_validation_and_checkpoint():
    torch.manual_seed(0)
    model = RecordingLayeredModel()
    stream = TinyStream()
    logger = InMemoryLogger()
    checkpoint_policy = BlerPrimaryWithBERTieBreakerPolicy()
    checkpoint_calls = []

    trainer = Trainer(
        model=model,
        stream=stream,
        strategy=LayerwiseTraining(train_layers=[0], forward_depth_policy="local_depth"),
        loss_composer=make_loss_composer(),
        optimizer_factory=lambda params: torch.optim.SGD(params, lr=0.1),
        config=TrainerConfig(
            epochs=1,
            steps_per_epoch=2,
            batch_size=3,
            train_snr=1.25,
            validation=EvaluationConfig(
                phase="validation",
                num_batches=1,
                batch_size=2,
                snr=2.5,
            ),
        ),
        checkpoint_policy=checkpoint_policy,
        metrics_logger=logger,
        checkpoint_callback=lambda model, decision: checkpoint_calls.append(decision),
    )

    result = trainer.train()

    assert len(result.records) == 1
    assert result.records[0].train_metrics["phase"] == "train"
    assert result.records[0].validation_metrics["phase"] == "validation"
    assert result.records[0].checkpoint_decision is not None
    assert checkpoint_calls
    assert any(record["phase"] == "validation" for record in logger.records)
    assert logger.records[-1]["candidate_projections"] == 1
    assert any(dict(mode.channel_context) == {"snr": 1.25} for mode in model.modes)
    assert any(dict(mode.channel_context) == {"snr": 2.5} for mode in model.modes)
    assert not any(
        parameter.requires_grad
        for parameter in model.routing_policy.router.routers[1].parameters()
    )


def test_model_outputs_depend_on_explicit_mode_not_previous_phase_calls():
    torch.manual_seed(1)
    model = RecordingLayeredModel()
    strategy = LayerwiseTraining()
    x = torch.ones(2, 4)
    mode0 = strategy.mode_for_step(0, phase="train")
    mode1 = strategy.mode_for_step(1, phase="train")

    first = model(x, mode0).logits.detach().clone()
    _ = model(x, mode1)
    second = model(x, mode0).logits.detach().clone()

    torch.testing.assert_close(first, second)


def test_training_diagnostics_log_full_cascade_and_active_parameter_delta():
    torch.manual_seed(2)
    model = RecordingLayeredModel()
    stream = TinyStream()
    logger = InMemoryLogger()
    layer0_before = [
        parameter.detach().clone()
        for parameter in model.routing_policy.router.routers[0].parameters()
    ]
    layer1_before = [
        parameter.detach().clone()
        for parameter in model.routing_policy.router.routers[1].parameters()
    ]

    trainer = Trainer(
        model=model,
        stream=stream,
        strategy=LayerwiseTraining(train_layers=[1], forward_depth_policy="local_depth"),
        loss_composer=make_loss_composer(),
        optimizer_factory=lambda params: torch.optim.SGD(params, lr=0.1),
        config=TrainerConfig(
            epochs=1,
            steps_per_epoch=1,
            batch_size=3,
            train_snr=1.0,
            validation=EvaluationConfig(
                phase="validation",
                num_batches=1,
                batch_size=2,
                snr=1.0,
            ),
            full_cascade_validation=EvaluationConfig(
                phase="validation_full",
                num_batches=1,
                batch_size=2,
                snr=1.0,
            ),
            diagnostics=DiagnosticsConfig(
                phase="diagnostics",
                num_batches=1,
                batch_size=2,
                snr=1.0,
            ),
        ),
        metrics_logger=logger,
    )

    result = trainer.train()

    phases = [record["phase"] for record in logger.records]
    assert phases == [
        "train_layer_start",
        "train",
        "validation",
        "validation_full",
        "diagnostics",
    ]
    train_metrics = result.records[0].train_metrics
    assert train_metrics["diagnostics/grad_norm"] > 0.0
    assert train_metrics["diagnostics/parameter_delta_norm"] > 0.0
    assert result.records[0].full_cascade_validation_metrics["validation_scope"] == (
        "full_cascade"
    )
    assert result.records[0].diagnostics_metrics["diagnostics_batches"] == 1

    for before, after in zip(
        layer0_before,
        model.routing_policy.router.routers[0].parameters(),
        strict=True,
    ):
        torch.testing.assert_close(before, after.detach())
    assert any(
        not torch.allclose(before, after.detach())
        for before, after in zip(
            layer1_before,
            model.routing_policy.router.routers[1].parameters(),
            strict=True,
        )
    )
    assert any(mode.target_layer is None for mode in model.modes)
    assert any(mode.collect_debug for mode in model.modes)
