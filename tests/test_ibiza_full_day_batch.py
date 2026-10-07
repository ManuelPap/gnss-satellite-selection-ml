import inspect
import json
from pathlib import Path

import numpy as np
import pytest

import validation.ibiza_generalization.full_day_batch as batch
import validation.ibiza_generalization.inference as inference
from validation.ibiza_generalization.checkpoints import (
    ARCHITECTURES,
    load_checkpoint_manifest,
    validate_checkpoint_inventory,
)
from validation.ibiza_generalization.inference import (
    IBIZA_NPZ_SHA256,
    load_ibiza_dataset,
)


@pytest.fixture(scope="module")
def ibiza_dataset() -> dict[str, np.ndarray]:
    return load_ibiza_dataset()


@pytest.fixture(scope="module")
def deterministic_sample(tmp_path_factory: pytest.TempPathFactory):
    plan = batch.discover_full_plan()
    jobs = tuple(
        next(record for record in plan if record.architecture == architecture)
        for architecture in ARCHITECTURES
    )
    first_dir = tmp_path_factory.mktemp("full_day_sample_first")
    second_dir = tmp_path_factory.mktemp("full_day_sample_second")
    arguments = {
        "jobs": jobs,
        "accepted_epoch_indices": (0, 1),
        "require_complete_plan": False,
        "progress_every": 0,
    }
    first = batch.run_batch(output_dir=first_dir, **arguments)
    second = batch.run_batch(output_dir=second_dir, **arguments)
    return jobs, first_dir, second_dir, first, second


def _read_json_lines(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_full_plan_discovers_all_30_checkpoints_exactly_once() -> None:
    plan = batch.discover_full_plan()
    inventory = validate_checkpoint_inventory()
    identities = [(record.architecture, record.seed) for record in plan]
    assert len(plan) == len(set(identities)) == 30
    assert identities == [
        (architecture, seed)
        for architecture in ARCHITECTURES
        for seed in range(10)
    ]
    assert plan == inventory.records
    assert plan == load_checkpoint_manifest()


def test_tracked_batch_spec_fixes_inputs_runtime_schema_and_controls() -> None:
    spec = batch.load_batch_spec()
    assert spec["status"] == "runner_ready_not_executed"
    assert spec["source_dataset"]["sha256"] == IBIZA_NPZ_SHA256
    assert spec["checkpoint_inventory"]["checkpoint_count"] == 30
    assert spec["common_epoch_policy"]["shared_across_models"] is True
    assert spec["runtime_versions"]["validated_execution_environment"] == {
        "python": "3.12.3",
        "numpy": "2.5.1",
        "torch": "2.14.0+cpu",
        "pymap3d": "3.2.0",
        "device": "cpu",
        "model_and_wls_dtype": "torch.float64",
    }
    schema = spec["result_schema"]
    for name in (
        "architecture",
        "seed",
        "accepted_epoch_index",
        "source_split_epoch_index",
        "timestamp_gpst_like_s",
        "satellite_count",
        "estimated_ecef_m",
        "active_state_dimension",
        "rank",
        "condition_number",
        "iteration_count",
        "solution_status",
        "convergence_status",
    ):
        assert name in schema
    source = Path(batch.__file__).read_text(encoding="utf-8")
    assert "torch.optim" not in source
    assert ".backward(" not in source
    assert "dataset[\"features\"].mean" not in source
    assert "dataset[\"features\"].std" not in source
    assert batch.evaluate_epoch is inference.evaluate_epoch


def test_ground_truth_is_rejected_and_has_no_runner_parameter(
    ibiza_dataset: dict[str, np.ndarray],
) -> None:
    parameters = inspect.signature(batch.run_batch).parameters
    assert not any(
        token in name.lower()
        for name in parameters
        for token in ("ground_truth", "reference_coordinate", "accuracy")
    )
    batch.reject_ground_truth_arrays(ibiza_dataset)
    contaminated = dict(ibiza_dataset)
    contaminated["future_ground_truth_ecef_m"] = np.asarray([1.0, 2.0, 3.0])
    with pytest.raises(RuntimeError, match="rejects ground-truth"):
        batch.reject_ground_truth_arrays(contaminated)


def test_one_verified_ibiza_load_is_shared_without_feature_replacement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    ibiza_dataset: dict[str, np.ndarray],
) -> None:
    load_count = 0
    original_features = ibiza_dataset["features"]
    original_evaluate = batch.evaluate_epoch

    def counted_load(*args, **kwargs):
        nonlocal load_count
        load_count += 1
        return ibiza_dataset

    def checked_evaluate(frozen, dataset, epoch_index):
        assert dataset is ibiza_dataset
        assert dataset["features"] is original_features
        return original_evaluate(frozen, dataset, epoch_index)

    monkeypatch.setattr(batch, "load_ibiza_dataset", counted_load)
    monkeypatch.setattr(batch, "evaluate_epoch", checked_evaluate)
    plan = batch.discover_full_plan()
    manifest = batch.run_batch(
        output_dir=tmp_path,
        jobs=(plan[0], plan[10]),
        accepted_epoch_indices=(0,),
        require_complete_plan=False,
        progress_every=0,
    )
    assert load_count == 1
    assert manifest["source_dataset"]["sha256"] == IBIZA_NPZ_SHA256
    assert manifest["source_dataset"]["loaded_once_and_shared_by_all_jobs"] is True
    for job in manifest["jobs"]:
        row = _read_json_lines(tmp_path / job["result_file"])[0]
        assert row["source_ibiza_npz_sha256"] == IBIZA_NPZ_SHA256


def test_failed_solution_is_retained_in_place(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original_evaluate = batch.evaluate_epoch

    def fail_middle_epoch(frozen, dataset, epoch_index):
        if epoch_index == 1:
            raise RuntimeError("synthetic retained WLS failure")
        return original_evaluate(frozen, dataset, epoch_index)

    monkeypatch.setattr(batch, "evaluate_epoch", fail_middle_epoch)
    job = batch.discover_full_plan()[0]
    manifest = batch.run_batch(
        output_dir=tmp_path,
        jobs=(job,),
        accepted_epoch_indices=(0, 1, 2),
        require_complete_plan=False,
        progress_every=0,
    )
    rows = _read_json_lines(tmp_path / batch.result_filename(job))
    assert [row["accepted_epoch_index"] for row in rows] == [0, 1, 2]
    assert len(rows) == 3
    failed = rows[1]
    assert failed["solution_status"] == "exception"
    assert failed["convergence_status"] == "not_available"
    assert failed["estimated_ecef_m"] is None
    assert failed["rank"] is None
    assert failed["condition_number"] is None
    assert failed["error"] == {
        "type": "RuntimeError",
        "message": "synthetic retained WLS failure",
    }
    assert manifest["status"] == "completed_with_failed_epochs"
    assert manifest["totals"]["failed_epoch_count"] == 1
    assert manifest["jobs"][0]["record_count"] == 3


def test_small_sample_rerun_is_byte_identical_and_counts_are_consistent(
    deterministic_sample,
    ibiza_dataset: dict[str, np.ndarray],
) -> None:
    jobs, first_dir, second_dir, first, second = deterministic_sample
    filenames = [batch.result_filename(job) for job in jobs]
    filenames.append(batch.RUN_MANIFEST_FILENAME)
    for filename in filenames:
        assert (first_dir / filename).read_bytes() == (second_dir / filename).read_bytes()
    assert first == second
    assert first["status"] == "completed"
    assert first["totals"] == {
        "result_record_count": 6,
        "satellite_row_count_across_models": 114,
        "failed_epoch_count": 0,
    }
    for job, summary in zip(jobs, first["jobs"], strict=True):
        assert summary["record_count"] == 2
        assert summary["satellite_row_count"] == 38
        assert summary["failed_epoch_count"] == 0
        validated = batch.validate_result_file(
            first_dir / batch.result_filename(job),
            job,
            ibiza_dataset,
            (0, 1),
        )
        assert validated["record_count"] == 2
        assert validated["satellite_row_count"] == 38


def test_architecture_specific_output_diagnostics_are_exported(
    deterministic_sample,
) -> None:
    _jobs, first_dir, _second_dir, manifest, _second = deterministic_sample
    summaries = {item["architecture"]: item for item in manifest["jobs"]}

    bias_row = _read_json_lines(first_dir / summaries["TDL-B"]["result_file"])[0]
    assert bias_row["model_output"]["bias_m"] is not None
    assert bias_row["model_output"]["weight"] is None
    assert bias_row["model_output"]["weight_fraction_lt_0_01"] is None

    weight_row = _read_json_lines(first_dir / summaries["TDL-W"]["result_file"])[0]
    assert weight_row["model_output"]["bias_m"] is None
    assert weight_row["model_output"]["weight"] is not None
    assert 0.0 <= weight_row["model_output"]["weight_fraction_lt_0_01"] <= 1.0

    hybrid_row = _read_json_lines(first_dir / summaries["TDL-BW"]["result_file"])[0]
    assert hybrid_row["model_output"]["bias_m"] is not None
    assert hybrid_row["model_output"]["weight"] is not None
    zero = hybrid_row["model_output"]["bias_fraction_eq_0"]
    nonzero = hybrid_row["model_output"]["bias_fraction_gt_0"]
    assert zero + nonzero == pytest.approx(1.0)
    for row in (bias_row, weight_row, hybrid_row):
        assert row["satellite_count"] == row["row_stop"] - row["row_start"]
        assert row["rank"] == row["active_state_dimension"]
        assert row["solution_status"] == "solved"
        assert row["convergence_status"] == "converged"


def test_cli_requires_explicit_full_run_acknowledgement() -> None:
    assert batch.parse_args([]).confirm_full_run is False
    with pytest.raises(SystemExit, match="--confirm-full-run"):
        batch.main([])
