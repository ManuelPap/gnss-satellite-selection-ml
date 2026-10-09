from __future__ import annotations

import ast
import inspect
import os
from pathlib import Path

import numpy as np
import pytest

from validation.tdl_9feature_tasgnss_analysis import core
from validation.tdl_9feature_tasgnss_analysis import evaluate_ibiza as ibiza


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "validation/tdl_9feature_tasgnss_analysis/evaluate_ibiza.py"


def test_authoritative_inventory_contains_exactly_ten_matching_hashes() -> None:
    inventory = ibiza.load_checkpoint_inventory()
    assert inventory["seed_count"] == 10
    assert [entry["seed"] for entry in inventory["seeds"]] == list(range(10))
    for entry in inventory["seeds"]:
        assert core.sha256_file(entry["checkpoint"]) == entry["checkpoint_sha256"]


def test_checkpoint_hash_after_guard_detects_any_change(tmp_path: Path) -> None:
    checkpoint = tmp_path / "multinet_3d.pth"
    checkpoint.write_bytes(b"frozen")
    manifest = tmp_path / "frozen_checkpoint_manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    inventory = {
        "manifest_path": manifest,
        "manifest_sha256": core.sha256_file(manifest),
        "seeds": [
            {"checkpoint": checkpoint, "checkpoint_sha256": core.sha256_file(checkpoint)}
        ],
    }
    ibiza.assert_checkpoint_inventory_unchanged(inventory)
    checkpoint.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="frozen checkpoint changed"):
        ibiza.assert_checkpoint_inventory_unchanged(inventory)


def test_frozen_model_is_eval_only_and_uses_checkpoint_scaler() -> None:
    entry = ibiza.load_checkpoint_inventory()["seeds"][0]
    model, controls = ibiza.load_frozen_model(entry)
    state = model.state_dict()
    assert model.training is False
    assert all(module.training is False for module in model.modules())
    assert controls["scaler_source"].startswith("checkpoint state_dict")
    assert controls["scaler_refit_on_ibiza"] is False
    np.testing.assert_array_equal(
        state["seq.0.mean"].numpy(), np.asarray(entry["scaler_mean"], dtype=np.float64)
    )
    np.testing.assert_array_equal(
        state["seq.0.std"].numpy(), np.asarray(entry["scaler_std"], dtype=np.float64)
    )


def test_evaluator_has_no_training_optimizer_or_ibiza_scaler_fit_path() -> None:
    source = EVALUATOR.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            calls.append(ast.unparse(node.func))
    assert not any(name.endswith(".backward") for name in calls)
    assert not any("optimizer" in name.lower() for name in calls)
    assert "scaler_from_records" not in source
    assert "model.eval()" in source
    assert "torch.inference_mode()" in source


def _synthetic_position_arrays(seed: int = 0) -> dict[str, np.ndarray]:
    count = 4
    reference = np.array([4967979.2522, 125663.49615, 3984693.03705])
    return {
        "schema_version": np.array([1]),
        "seed": np.array([seed]),
        "candidate_epoch_id": np.arange(count, dtype=np.int64),
        "preprocessing_accepted": np.ones(count, dtype=np.uint8),
        "feature_accepted": np.array([1, 1, 0, 1], dtype=np.uint8),
        "neutral_solved": np.array([1, 1, 0, 1], dtype=np.uint8),
        "learned_solved": np.array([1, 0, 0, 1], dtype=np.uint8),
        "paired": np.array([1, 0, 0, 1], dtype=np.uint8),
        "neutral_position_ecef_m": np.vstack(
            (reference + [3, 0, 0], reference + [2, 0, 0], [np.nan] * 3, reference + [4, 0, 0])
        ),
        "learned_position_ecef_m": np.vstack(
            (reference + [1, 0, 0], [np.nan] * 3, [np.nan] * 3, reference + [2, 0, 0])
        ),
        "feature_hash": np.array(["a", "b", "", "d"], dtype="U64"),
        "neutral_solver_input_hash": np.array(["e", "f", "", "h"], dtype="U64"),
        "learned_solver_input_hash": np.array(["i", "j", "", "l"], dtype="U64"),
        "model_output_hash": np.array(["m", "n", "", "p"], dtype="U64"),
    }


def test_gt_perturbation_changes_metrics_only() -> None:
    arrays = _synthetic_position_arrays()
    reference = np.array([4967979.2522, 125663.49615, 3984693.03705])
    audit = ibiza.reference_perturbation_audit(
        arrays, reference, np.array([0, 1, 2], dtype=np.int64)
    )
    assert audit["feature_hashes_unchanged"] is True
    assert audit["neutral_solver_input_hashes_unchanged"] is True
    assert audit["learned_solver_input_hashes_unchanged"] is True
    assert audit["model_outputs_unchanged"] is True
    assert audit["neutral_estimated_ecef_unchanged"] is True
    assert audit["learned_estimated_ecef_unchanged"] is True
    assert audit["evaluation_metrics_changed"] is True
    assert audit["only_evaluation_metrics_changed"] is True


def test_metrics_use_exact_identical_paired_epoch_ids_and_no_gt_threshold() -> None:
    arrays = _synthetic_position_arrays()
    reference = np.array([4967979.2522, 125663.49615, 3984693.03705])
    result = ibiza.evaluate_seed_support(
        arrays,
        reference,
        support_name=ibiza.PRIMARY_SUPPORT,
        historical_epoch_ids=np.array([0, 1, 2], dtype=np.int64),
    )
    assert result["epoch_counts"]["candidate_epochs"] == 4
    assert result["epoch_counts"]["feature_rejected_epochs"] == 1
    assert result["epoch_counts"]["neutral_failed_epochs"] == 0
    assert result["epoch_counts"]["learned_failed_epochs"] == 1
    assert result["epoch_counts"]["exact_paired_epoch_count"] == 2
    assert result["pairing"]["epoch_ids"] == [0, 3]
    assert result["pairing"]["neutral_epoch_ids_sha256"] == result["pairing"]["learned_epoch_ids_sha256"]
    assert result["pairing"]["identical_epoch_ids"] is True
    assert "no ground-truth error threshold" in result["inclusion_policy"]
    source = inspect.getsource(ibiza.evaluate_seed_support)
    mask_source = source[: source.index("neutral_position =")]
    assert "reference_ecef" not in mask_source.replace(
        "    reference_ecef: np.ndarray,\n", ""
    )


def _fake_seed_summary(seed: int, support: str = ibiza.PRIMARY_SUPPORT) -> dict[str, object]:
    summary = {name: float(seed) for name in ("mean", "median", "rms", "p68", "p95", "max")}
    delta = {
        "mean_delta": float(seed),
        "median_delta": float(seed),
        "p05": float(seed),
        "p95": float(seed),
        "fraction_improved": 0.5,
        "fraction_worsened": 0.5,
    }
    return {
        "support": support,
        "seed": seed,
        "learned": {"2d": summary, "3d": summary, "east_rms": 1.0, "north_rms": 2.0, "up_rms": 3.0},
        "neutral_paired": {"2d": summary, "3d": summary, "east_rms": 1.0, "north_rms": 2.0, "up_rms": 3.0},
        "paired_delta": {"2d": delta, "3d": delta},
        "epoch_counts": {
            "candidate_epochs": 2880,
            "preprocessing_accepted_epochs": 2880,
            "preprocessing_failed_epochs": 0,
            "feature_accepted_epochs": 2849,
            "feature_rejected_epochs": 31,
            "neutral_solved_epochs": 2849,
            "neutral_failed_epochs": 0,
            "learned_solved_epochs": 2849,
            "learned_failed_epochs": 0,
            "exact_paired_epoch_count": 2849,
        },
    }


def test_across_seed_statistics_require_ten_seed_summaries() -> None:
    results = [_fake_seed_summary(seed) for seed in ibiza.SEEDS]
    summary = ibiza.across_seed_summary(results)
    assert summary["seed_count"] == 10
    assert "not pooled" in summary["aggregation_unit"]
    assert summary["metrics"]["learned_2d_mean"]["population_sd"] == pytest.approx(
        np.std(np.arange(10), ddof=0)
    )
    with pytest.raises(RuntimeError, match="all ten"):
        ibiza.across_seed_summary(results[:-1])


def test_feature_order_and_tasgnss_weight_bias_contract_are_explicit() -> None:
    assert ibiza.FEATURE_NAMES == (
        "SNR", "elevation", "azimuth", "neutral_residual", "G", "R", "E", "C", "J"
    )
    source = EVALUATOR.read_text(encoding="utf-8")
    assert "w=weight" in source
    assert "b=bias" in source
    assert "enable_torch=True" in source
    tas_source = (core.TASGNSS_REPOSITORY / "tasgnss/core.py").read_text(encoding="utf-8")
    assert "W_pr = backend.diag(w_tensor)" in tas_source
    assert "p_residual = pr_tensor - psr - b" in tas_source


def test_historical_support_is_the_frozen_2856_epoch_set() -> None:
    epoch_ids, timestamps = ibiza.historical_common_epoch_ids(ibiza.DEFAULT_HISTORICAL_DATASET)
    assert epoch_ids.shape == timestamps.shape == (2856,)
    assert np.array_equal(epoch_ids, np.unique(epoch_ids))
    assert set(range(2880)) - set(epoch_ids.tolist())


def test_external_repositories_are_clean_and_pinned() -> None:
    provenance = core.verify_provenance()
    assert all(not record["status_short"] for record in provenance["upstream"].values())


@pytest.mark.integration
def test_one_epoch_real_replay_feature_and_leakage_contract(tmp_path: Path) -> None:
    if os.environ.get("CURRENT_TDL_IBIZA_RUN_INTEGRATION") != "1":
        pytest.skip("set CURRENT_TDL_IBIZA_RUN_INTEGRATION=1 for the real Ibiza replay")
    inventory = ibiza.load_checkpoint_inventory()
    records, manifest = ibiza.load_or_preprocess_ibiza(
        tmp_path,
        ibiza.DEFAULT_OBSERVATION,
        ibiza.DEFAULT_NAVIGATION,
        ibiza.DEFAULT_NEUTRAL_AUDIT,
        smoke_epochs=1,
        progress_every=0,
    )
    assert manifest["ground_truth_available_to_preprocessing"] is False
    assert len(records) == 1
    assert not ({"gt", "ground_truth", "reference"} & records[0].keys())
    parts = core.record_feature_parts(records[0])
    assert parts is not None
    assert core.feature_tensor(parts).shape[1] == 9

    first, report_first = ibiza.infer_seed(
        records,
        candidate_count=1,
        checkpoint_entry=inventory["seeds"][0],
        progress_every=0,
    )
    replay, report_replay = ibiza.infer_seed(
        records,
        candidate_count=1,
        checkpoint_entry=inventory["seeds"][0],
        progress_every=0,
    )
    for name in first:
        np.testing.assert_array_equal(first[name], replay[name])
    assert report_first["exact_paired_epoch_count"] == report_replay["exact_paired_epoch_count"] == 1
    reference = trusted_reference()
    historical_ids, _times = ibiza.historical_common_epoch_ids(ibiza.DEFAULT_HISTORICAL_DATASET)
    metrics_first = ibiza.evaluate_seed_support(
        first,
        reference,
        support_name=ibiza.PRIMARY_SUPPORT,
        historical_epoch_ids=historical_ids,
    )
    metrics_replay = ibiza.evaluate_seed_support(
        replay,
        reference,
        support_name=ibiza.PRIMARY_SUPPORT,
        historical_epoch_ids=historical_ids,
    )
    assert metrics_first == metrics_replay
    assert ibiza.reference_perturbation_audit(first, reference, historical_ids)[
        "only_evaluation_metrics_changed"
    ] is True


def trusted_reference() -> np.ndarray:
    import json

    document = json.loads(ibiza.DEFAULT_REFERENCE.read_text(encoding="utf-8"))
    return trusted_reference_ecef_for_test(document)


def trusted_reference_ecef_for_test(document: dict[str, object]) -> np.ndarray:
    values = document["trusted_reference_ecef_m"]
    assert isinstance(values, dict)
    return np.asarray([values["x"], values["y"], values["z"]], dtype=np.float64)
