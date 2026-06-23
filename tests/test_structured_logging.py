import json

from routing_rpa.utils.logging import JsonlWriter, create_run_directory
from routing_rpa.utils.serialization import stable_json_hash, write_json


def test_jsonl_writer_writes_valid_json_per_line(tmp_path):
    path = tmp_path / "metrics.jsonl"
    writer = JsonlWriter(path)
    writer.write(
        {
            "phase": "validation",
            "ber": 0.1,
            "bler": 0.5,
            "candidate_projections": 512,
            "executed_projections": 128,
            "aggregated_projections": 128,
            "selection_scope": "static",
            "execution_mode": "compute_selected",
        }
    )
    writer.write({"phase": "final", "ber": 0.05, "bler": 0.25})

    lines = path.read_text(encoding="utf-8").strip().splitlines()

    assert len(lines) == 2
    assert [json.loads(line) for line in lines] == writer.read_records()
    first = json.loads(lines[0])
    assert first["candidate_projections"] == 512
    assert first["executed_projections"] == 128
    assert first["aggregated_projections"] == 128
    assert first["selection_scope"] == "static"
    assert first["execution_mode"] == "compute_selected"


def test_run_directory_creator_creates_required_files_and_subdirs(tmp_path):
    run = create_run_directory(
        tmp_path,
        "experiment-a",
        timestamp="2026-06-15--12-00-00",
        config={"seed": 7},
        artifacts_used={"G": "artifact/G.pt"},
    )

    assert run.root == tmp_path / "experiment-a" / "2026-06-15--12-00-00"
    assert run.root.is_dir()
    assert run.checkpoints_dir.is_dir()
    assert run.plots_dir.is_dir()
    assert run.config_json.is_file()
    assert run.metrics_jsonl.is_file()
    assert run.summary_json.is_file()
    assert run.artifacts_used_json.is_file()
    assert run.report_txt.is_file()
    assert json.loads(run.config_json.read_text(encoding="utf-8")) == {"seed": 7}
    assert json.loads(run.summary_json.read_text(encoding="utf-8")) == {}
    assert json.loads(run.artifacts_used_json.read_text(encoding="utf-8")) == {
        "G": "artifact/G.pt"
    }


def test_write_json_and_stable_hash_support_config_snapshots(tmp_path):
    payload = {"decoder": {"num_unfolded_steps": 5}, "seed": 123}
    path = tmp_path / "config.json"

    write_json(path, payload)

    assert json.loads(path.read_text(encoding="utf-8")) == payload
    assert stable_json_hash(payload) == stable_json_hash(
        {"seed": 123, "decoder": {"num_unfolded_steps": 5}}
    )
