# neural-routing-rpa-decoder

Research code for experiments with a routed, unfolded Recursive Projection-Aggregation
(RPA) decoder for Reed-Muller codes.

The project accompanies a study of whether the aggregation step of an RPA decoder can
benefit from learned projection weights. The main experimental setting is the full
Reed-Muller code `RM(10, 2)` transmitted through a BPSK AWGN channel. Classical RPA uses
uniform aggregation over a fixed projection set; this codebase keeps the algebraic RPA
structure and replaces only the projection weighting/selection rule with configurable
routing policies.

## Research Context

The paper in studies two routed variants of RPA:

- Dense learned aggregation: all `P = 512` one-dimensional projections are computed and
  aggregated, but an MLP router assigns sigmoid weights to projections at each unfolded
  RPA step.
- MoE-style top-k aggregation: projections are treated as fixed experts, and the router
  selects or masks the top-k projections by score.

The experiments reported in the paper found no convincing input-dependent routing gain
for `RM(10, 2)` in an i.i.d. AWGN channel. The strongest learned dense setting improved
BLER modestly, but diagnostics showed nearly static routing behavior: high top-k overlap
between samples and very small per-input projection-quality variance. The top-k/MoE
variant selected better subsets than random at small `k`, but preserving baseline quality
required keeping roughly `384-448` projections out of `512`, so the practical complexity
reduction was small.

The implementation is therefore best understood as an experimental framework for
measuring routed RPA behavior, not as a polished production decoder.

## What Is Implemented

- Binary linear-code utilities and a Reed-Muller wrapper backed by generator-matrix
  artifacts.
- AWGN channel simulation with BPSK modulation and LLR conversion.
- One-dimensional projection geometry loading and validation.
- An order-2 unfolded RPA decoder backend:
  - projection of length-n LLRs to `P` projected words,
  - Hadamard/FHT-style first-order bottom decoding,
  - differentiable straight-through sign in aggregation,
  - repeated unfolded projection-decode-aggregate steps.
- Routing policies:
  - `uniform_full`: classical equal-weight baseline,
  - `input_dependent_router`: per-step MLP router with dense sigmoid weights,
  - `full_masked_topk`: diagnostic top-k masking while still computing all projections,
  - `random_static`: fixed random top-k baseline,
  - `selected_static_topk`: fixed explicit subset with real `compute_selected` pruning.
- Layerwise router training, checkpointing, structured JSONL metrics, and final
  evaluation.
- Tests covering GF(2) operations, code/channel utilities, projection loading,
  decoder modes, routing baselines, pruning equivalence, training smoke tests, and
  experiment assembly.

## Repository Layout

```text
src/routing_rpa/
  channels/        AWGN and channel/reliability variants
  codes/           GF(2), linear-code, and Reed-Muller utilities
  data/            synthetic random-codeword streams
  decoders/        unfolded RPA kernels, routing policies, modes, outputs
  eval/            evaluation loop and BER/BLER accumulation
  experiments/     JSON config loading, registries, component builder, runner
  projections/     projection artifact loaders and validation
  training/        losses, metrics, layerwise strategy, trainer, checkpoints
  utils/           structured logging and JSON serialization

configs/           ready experiment configurations
artifacts/rm_10_2/ generator matrix and half-size projection set for RM(10,2)
scripts/           helper scripts for checkpoint evaluation and run summaries
tests/             pytest suite
paper/             research write-up
```

## Setup

The checked-in Pixi environment is the recommended way to run the project. It targets
Python 3.11 on Windows and includes PyTorch with CUDA support.

```powershell
pixi install
pixi run smoke
```

Run the full test suite with:

```powershell
pixi run python -m pytest -p no:cacheprovider tests
```

For CPU-only quick checks, prefer the CPU configs such as `configs/uniform_full.json`,
`configs/full_masked_topk.json`, or `configs/selected_static_topk.json`. The paper-scale
training configs use `device: "cuda"` and are substantially heavier.

## Running Experiments

Experiments are driven by JSON configs and the Python API:

```powershell
pixi run python -c "from routing_rpa.experiments.runner import run_experiment; r = run_experiment('configs/uniform_full.json'); print(r.run_directory.root); print(r.final_metrics)"
```

Each run creates:

```text
runs/<experiment_name>/<YYYY-MM-DD--HH-MM-SS>/
  config.json
  artifacts_used.json
  metrics.jsonl
  summary.json
  checkpoints/
  plots/
  report.txt
```

Summarize a completed run:

```powershell
pixi run python scripts/summarize_run.py runs/<experiment_name>/<timestamp>
```

Evaluate a saved checkpoint from a run directory:

```powershell
pixi run python scripts/evaluate_checkpoint.py runs/<experiment_name>/<timestamp>
```

Pass `--checkpoint path/to/checkpoint.pt` or `--layer-id N` to override automatic
best-checkpoint selection.

## Important Configs

- `configs/uniform_full.json`: compact CPU baseline with uniform aggregation.
- `configs/uniform_full_gpu.json`: CUDA baseline at the paper's `SNR = 1.0 dB`.
- `configs/adaptive_default.json`: dense input-dependent MLP routing with layerwise
  training.
- `configs/adaptive_default_diagnostic.json`: dense routing plus additional full-cascade
  validation and diagnostics.
- `configs/adaptive_topk.json` and `configs/random_top_k.json`: top-k routing variants
  used for comparison.
- `configs/full_masked_topk.json`: CPU diagnostic top-k mask while still computing all
  candidate projections.
- `configs/selected_static_topk.json`: CPU static subset with `compute_selected`, useful
  for checking real projection pruning behavior.

Common fields:

- `code`: Reed-Muller parameters and generator artifact.
- `projections`: projection artifact and expected projection count.
- `decoder`: backend, unfolded depth, bottom decoder, and debug collection flags.
- `routing`: routing policy, top-k value, router shape, selection scope, execution mode.
- `channel_train`, `validation`, `final_eval`: SNRs and batch counts.
- `training`: layerwise schedule, loss components, optimizer, checkpoint policy, and
  frozen-step behavior.
- `logging`: experiment name and optional timestamp.

## Routing and Execution Modes

The decoder separates projection selection from projection execution:

- `selection_scope = "full"` keeps the full candidate set visible to the policy.
- `selection_scope = "static"` uses a fixed subset.
- `execution_mode = "compute_all_mask"` computes every candidate projection and applies
  weights or masks during aggregation. This is useful for diagnostics and for dense
  learned routing.
- `execution_mode = "compute_selected"` slices the projection set before the hot kernels.
  This is the mode to use when measuring actual static pruning.

Input-dependent top-k currently uses `compute_all_mask`, so it measures routed
aggregation quality but does not by itself provide an optimized sparse kernel for
per-sample top-k execution.

## Programmatic Use

```python
from routing_rpa.experiments.runner import run_experiment

result = run_experiment("configs/uniform_full.json")

print(result.run_directory.root)
print(result.baseline_metrics)
print(result.final_metrics)
```

Lower-level components can be built directly:

```python
from routing_rpa.experiments.config import load_experiment_config
from routing_rpa.experiments.build import build_experiment

config = load_experiment_config("configs/uniform_full.json")
components = build_experiment(config)
```

## Metrics

The evaluator reports:

- `ber`: bit error rate,
- `bler`: block error rate,
- raw bit/block error counts,
- decoder execution stats such as candidate, executed, and aggregated projection counts,
- optional router diagnostics when `collect_debug` or diagnostic configs are enabled.

Training logs include per-layer records, loss components, validation metrics, checkpoint
decisions, and optional gradient/debug statistics.

## Artifacts

The main bundled artifacts are:

- `artifacts/rm_10_2/G.pt`: generator matrix for `RM(10, 2)`.
- `artifacts/rm_10_2/projections_half512.txt`: fixed set of `512` one-dimensional
  projections used by the experiments.

Configs refer to these paths relative to `paths.project_root`, which defaults to the
repository root.

## Notes and Limitations

- The active decoder backend is `order2_unfolded`; the provided experiments target
  second-order Reed-Muller codes, especially `RM(10, 2)`.
- `rm_subcode` and `soft_map_subcode` entries are reserved for future subcode work and
  intentionally raise `NotImplementedError`.
- The code is deterministic where seeds are configured, but GPU kernels and large Monte
  Carlo evaluations can still vary slightly across environments.
- The README describes the current `src/routing_rpa` package;
