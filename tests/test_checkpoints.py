import pytest

from routing_rpa.training.checkpoints import (
    BlerPrimaryWithBERTieBreakerPolicy,
    BothImproveCheckpointPolicy,
    build_checkpoint_metadata,
    build_checkpoint_policy,
)


def test_both_improve_policy_decision_table():
    policy = BothImproveCheckpointPolicy()

    first = policy.decide({"bler": 0.5, "ber": 0.1}, epoch=0)
    only_bler = policy.decide({"bler": 0.4, "ber": 0.12}, epoch=1)
    only_ber = policy.decide({"bler": 0.55, "ber": 0.08}, epoch=2)
    tie = policy.decide({"bler": 0.5, "ber": 0.1}, epoch=3)
    both = policy.decide({"bler": 0.4, "ber": 0.08}, epoch=4)

    assert first.should_save
    assert not only_bler.should_save
    assert not only_ber.should_save
    assert not tie.should_save
    assert both.should_save
    assert policy.best_metrics == {"bler": 0.4, "ber": 0.08}


def test_bler_primary_with_ber_tiebreaker_decision_table():
    policy = BlerPrimaryWithBERTieBreakerPolicy()

    first = policy.decide({"bler": 0.5, "ber": 0.1})
    worse_bler = policy.decide({"bler": 0.6, "ber": 0.01})
    tie_better_ber = policy.decide({"bler": 0.5, "ber": 0.08})
    tie_worse_ber = policy.decide({"bler": 0.5, "ber": 0.09})
    better_bler = policy.decide({"bler": 0.45, "ber": 0.2})

    assert first.should_save
    assert not worse_bler.should_save
    assert tie_better_ber.should_save
    assert not tie_worse_ber.should_save
    assert better_bler.should_save
    assert policy.best_metrics == {"bler": 0.45, "ber": 0.2}


def test_checkpoint_metadata_includes_policy_config_layer_and_metrics_fields():
    policy = BothImproveCheckpointPolicy()
    decision = policy.decide(
        {"bler": 0.25, "ber": 0.031, "loss": 1.2},
        epoch=3,
        step=40,
        layer_id=2,
        config_snapshot_path="runs/example/config.json",
        config_snapshot_hash="abc123",
    )

    assert decision.metadata["policy_name"] == "both_improve"
    assert decision.metadata["monitored_metrics"] == {"bler": 0.25, "ber": 0.031}
    assert decision.metadata["epoch"] == 3
    assert decision.metadata["step"] == 40
    assert decision.metadata["layer_id"] == 2
    assert decision.metadata["config_snapshot_path"] == "runs/example/config.json"
    assert decision.metadata["config_snapshot_hash"] == "abc123"


def test_checkpoint_metadata_builder_rejects_missing_metrics():
    with pytest.raises(ValueError, match="missing required keys"):
        build_checkpoint_metadata(policy_name="both_improve", metrics={"ber": 0.1})


def test_checkpoint_policy_factory():
    assert isinstance(build_checkpoint_policy("both_improve"), BothImproveCheckpointPolicy)
    assert isinstance(
        build_checkpoint_policy("bler_primary_with_ber_tiebreaker"),
        BlerPrimaryWithBERTieBreakerPolicy,
    )

    with pytest.raises(ValueError, match="Unsupported"):
        build_checkpoint_policy("unknown_policy")
