import hashlib
import json
from pathlib import Path

import numpy as np
import pymap3d
import pytest

import validation.ibiza_generalization.evaluate_ground_truth as evaluation


@pytest.fixture(scope="module")
def frozen_validation() -> evaluation.FrozenResultValidation:
    return evaluation.validate_frozen_result_set()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_reference_manifest_records_supplied_coordinate_frame_source_and_epochs() -> None:
    manifest, reference = evaluation.load_reference_manifest()
    assert manifest["station_id"] == "IBIZ00ESP"
    assert np.array_equal(
        reference, np.asarray([4967979.25220, 125663.49615, 3984693.03705])
    )
    assert manifest["coordinate_reference_frame"] == {
        "system": "ITRS",
        "realization": "IGS20",
        "parent_frame": "ITRF2020",
    }
    assert manifest["source"]["sinex_identifier"] == (
        "EUR0OPSSNX_1996001_2026115_00U_SOL.SNX.gz"
    )
    assert manifest["validity"]["valid_from"].startswith("2009-06-24")
    assert manifest["validity"]["valid_to"].startswith("2026-04-25")
    assert manifest["coordinate_epoch"]["original_reference_epoch"].startswith(
        "2020-01-01"
    )
    assert manifest["coordinate_epoch"]["target_evaluation_epoch"].startswith(
        "2025-01-01"
    )
    assert manifest["coordinate_epoch"]["propagation_applied"] is True
    assert manifest["coordinate_epoch"]["elapsed_years"] == 5.0
    assert manifest["coordinate_epoch"]["velocity_m_per_year"] == {
        "vx": -0.01198,
        "vy": 0.01995,
        "vz": 0.01163,
    }
    published = manifest["published_reference_ecef_m"]
    original = np.asarray([published[axis] for axis in ("x", "y", "z")])
    velocity = manifest["coordinate_epoch"]["velocity_m_per_year"]
    velocity_xyz = np.asarray([velocity[axis] for axis in ("vx", "vy", "vz")])
    assert np.allclose(reference, original + 5.0 * velocity_xyz, rtol=0.0, atol=1e-9)


def test_completed_frozen_inventory_hashes_counts_and_common_epochs(
    frozen_validation: evaluation.FrozenResultValidation,
) -> None:
    identities = [(item.architecture, item.seed) for item in frozen_validation.files]
    assert identities == [
        (architecture, seed)
        for architecture in evaluation.ARCHITECTURES
        for seed in evaluation.SEEDS
    ]
    assert len(frozen_validation.files) == 30
    assert {item.record_count for item in frozen_validation.files} == {2856}
    assert {item.solved_count for item in frozen_validation.files} == {2856}
    assert frozen_validation.accepted_epoch_indices == tuple(range(2856))
    assert len(frozen_validation.timestamps_gpst_like_s) == 2856
    evaluation.assert_frozen_inputs_unchanged(frozen_validation)


def test_changing_ground_truth_changes_only_errors_and_not_stored_output(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "tdl_b_seed_0.jsonl"
    source_row = {
        "architecture": "TDL-B",
        "seed": 0,
        "accepted_epoch_index": 0,
        "source_split_epoch_index": 0,
        "timestamp_gpst_like_s": 1735689600.0,
        "solution_status": "solved",
        "estimated_ecef_m": [4968000.0, 125660.0, 3984700.0],
    }
    source_path.write_text(json.dumps(source_row) + "\n", encoding="utf-8")
    source_hash = _sha256(source_path)
    descriptor = evaluation.ValidatedResultFile(
        architecture="TDL-B",
        seed=0,
        path=source_path,
        sha256=source_hash,
        record_count=1,
        solved_count=1,
    )
    first_path = tmp_path / "first_errors.jsonl"
    second_path = tmp_path / "second_errors.jsonl"
    first_reference = np.asarray([4967979.25220, 125663.49615, 3984693.03705])
    second_reference = first_reference + np.asarray([10.0, -20.0, 30.0])
    evaluation.evaluate_model_result(descriptor, first_path, first_reference)
    evaluation.evaluate_model_result(descriptor, second_path, second_reference)
    first = json.loads(first_path.read_text(encoding="utf-8"))
    second = json.loads(second_path.read_text(encoding="utf-8"))

    assert _sha256(source_path) == source_hash
    assert first["estimated_ecef_m"] == second["estimated_ecef_m"]
    assert first["ecef_difference_m"] != second["ecef_difference_m"]
    assert first["enu_error_m"] != second["enu_error_m"]


def test_ecef_to_enu_matches_independent_pymap3d_implementation() -> None:
    _manifest, reference = evaluation.load_reference_manifest()
    latitude_deg, longitude_deg, height_m = pymap3d.ecef2geodetic(*reference)
    differences = np.asarray(
        [
            [12.5, -4.25, 9.0],
            [-31.0, 18.75, 2.5],
            [0.125, 0.5, -0.75],
        ]
    )
    actual = evaluation.ecef_differences_to_enu(differences, reference)
    independent = np.asarray(
        [
            pymap3d.ecef2enu(
                *(reference + difference),
                latitude_deg,
                longitude_deg,
                height_m,
            )
            for difference in differences
        ]
    )
    assert np.allclose(actual, independent, rtol=0.0, atol=1e-8)


def test_e2d_and_e3d_obey_required_norm_identities() -> None:
    _manifest, reference = evaluation.load_reference_manifest()
    estimates = reference + np.asarray(
        [[1.0, 2.0, 3.0], [-20.0, 4.0, 7.0], [0.5, -0.25, 0.125]]
    )
    errors = evaluation.evaluate_ecef_positions(estimates, reference)
    enu = errors["enu_error_m"]
    assert np.allclose(
        errors["e2d_m"], np.sqrt(np.square(enu[:, 0]) + np.square(enu[:, 1]))
    )
    assert np.allclose(
        errors["e3d_m"],
        np.linalg.norm(errors["ecef_difference_m"], axis=1),
        rtol=1e-12,
        atol=1e-12,
    )


def test_quantiles_are_deterministic_linear_numpy_quantiles() -> None:
    values = np.asarray([1.0, 2.0, 8.0, 10.0, 11.0])
    first = evaluation.linear_quantile(values, 0.68)
    second = evaluation.linear_quantile(values.copy(), 0.68)
    expected = float(np.quantile(values, 0.68, method="linear"))
    assert evaluation.QUANTILE_METHOD == "linear"
    assert first == second == expected


def test_across_seed_summary_aggregates_seed_metrics_not_epoch_rows() -> None:
    rows = []
    for seed in evaluation.SEEDS:
        row = {
            "architecture": "TEST",
            "seed": seed,
            "solved_epochs": 1 if seed == 0 else 1000,
            "availability": float(seed) / 10.0,
        }
        row.update({metric: float(seed) for metric in evaluation.ERROR_METRICS})
        rows.append(row)
    summary = evaluation.aggregate_seed_metrics(rows, expected_architectures=("TEST",))
    e2d_mean = next(row for row in summary if row["metric"] == "e2d_mean_m")
    assert e2d_mean["seed_count"] == 10
    assert e2d_mean["mean"] == pytest.approx(4.5)
    epoch_pooled_counterexample = (0.0 + sum(seed * 1000 for seed in range(1, 10))) / 9001
    assert e2d_mean["mean"] != pytest.approx(epoch_pooled_counterexample)


def test_stored_ols_baseline_is_aligned_to_identical_2856_epochs(
    frozen_validation: evaluation.FrozenResultValidation,
) -> None:
    positions, metadata = evaluation.load_stored_ols_baseline(
        evaluation.DEFAULT_IBIZA_NPZ, frozen_validation
    )
    assert positions.shape == (2856, 3)
    assert np.all(np.isfinite(positions))
    assert tuple(metadata["split"].tolist()) == frozen_validation.source_split_epoch_indices
    assert tuple(metadata["timestamp"].tolist()) == frozen_validation.timestamps_gpst_like_s


def test_rinex_approx_position_provenance_check() -> None:
    _manifest, reference = evaluation.load_reference_manifest()
    result = evaluation.reference_provenance_check(
        evaluation.DEFAULT_OBSERVATION_RINEX, reference
    )
    assert result["available"] is True
    assert result["rinex_approx_position_ecef_m"] == [
        4967979.688,
        125662.813,
        3984692.538,
    ]
    assert result["coordinate_difference_3d_m"] == pytest.approx(0.9516651012)


def test_evaluator_has_no_inference_training_normalization_or_wls_route() -> None:
    source = Path(evaluation.__file__).read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "inference import" not in source
    assert "load_frozen_model" not in source
    assert "optimizer" not in source.lower()
    assert ".backward(" not in source
    assert "features" not in source
