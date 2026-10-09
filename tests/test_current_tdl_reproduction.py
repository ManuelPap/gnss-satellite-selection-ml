from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from validation.tdl_9feature_tasgnss_analysis import core
from validation.tdl_9feature_tasgnss_analysis.evaluate_klt import (
    _assert_frozen_unchanged,
    across_seed_summary,
)


ROOT = Path(__file__).resolve().parents[1]
SMOKE_REPORT = core.DEFAULT_OUTPUT_ROOT / "smoke/smoke_report.json"


def test_required_branch_and_clean_pinned_upstreams() -> None:
    record = core.verify_provenance()
    assert record["project"]["branch"] == core.REQUIRED_BRANCH
    assert record["upstream"]["TDL-GNSS"]["head"] == core.TDL_COMMIT
    assert record["upstream"]["TASGNSS"]["head"] == core.TASGNSS_COMMIT
    assert record["upstream"]["pyrtklib"]["head"] == core.PYRTKLIB_COMMIT
    assert all(not item["status_short"] for item in record["upstream"].values())


def test_current_dataset_configs_and_feature_contract() -> None:
    assert core.DATASET_SPECS["KLT1"].expected_epochs == 203
    assert core.DATASET_SPECS["KLT2"].expected_epochs == 209
    assert core.DATASET_SPECS["KLT3"].expected_epochs == 404
    assert core.FEATURE_NAMES == (
        "SNR", "elevation", "azimuth", "neutral_residual", "G", "R", "E", "C", "J"
    )
    assert len(core.FEATURE_UNITS) == 9
    assert core.FEATURE_UNITS[1:4] == ("radian", "radian", "metre")


def test_exact_upstream_model_architecture_and_output_semantics() -> None:
    core.configure_determinism(0, threads=1)
    model = core.make_model(np.zeros(9), np.ones(9))
    state = model.state_dict()
    assert state["seq.0.mean"].shape == (9,)
    assert state["seq.0.std"].shape == (9,)
    assert state["seq.1.weight"].shape == (64, 9)
    assert state["seq.3.weight"].shape == (128, 64)
    assert state["seq.5.weight"].shape == (64, 128)
    assert state["seq.7.weight"].shape == (2, 64)
    weight, bias = model(torch.zeros((5, 9), dtype=torch.float64))
    assert weight.shape == (5,)
    assert bias.shape == (5,)
    assert torch.all((weight >= 0) & (weight <= 1))
    assert torch.all(bias >= 0)


def test_current_training_semantics_match_source() -> None:
    contract = core.training_source_contract()
    assert contract["epochs"] == 120
    assert contract["optimizer"] == "Adam"
    assert contract["learning_rate"] == 0.01
    assert contract["weight_decay"] == 0.0
    assert contract["scheduler"] is None
    assert contract["loss"] == "3D Euclidean ENU norm"
    assert contract["skip_position_loss_above_m"] == 200.0
    assert contract["accumulation_parameter"] == 3000
    assert contract["checkpoint_interval_epochs"] == 10
    assert contract["constructed_but_unused_loss"] == "MSELoss(reduction='sum')"

    train_source = (core.TDL_REPOSITORY / "train.py").read_text(encoding="utf-8")
    for fragment in (
        "np.random.shuffle(pres)",
        "SNR.astype(np.int8)",
        "pos_loss = torch.norm(enu[:loss_locate])",
        "if pos_loss > 200:",
        "torch.optim.Adam(net.parameters(), lr=0.01)",
        "batch_size = conf.get('batch', 3000)",
        "checkpoint_interval = 10",
        "multinet_3d.pth",
    ):
        assert fragment in train_source
    assert "lossFn = MSELoss(reduction='sum')" in train_source
    assert "lossFn(" not in train_source


def test_tasgnss_custom_weight_and_bias_sign_semantics() -> None:
    source = (core.TASGNSS_REPOSITORY / "tasgnss/core.py").read_text(encoding="utf-8")
    assert "W_pr = backend.diag(w_tensor)" in source
    assert "p_residual = pr_tensor - psr - b" in source
    assert "result = backend.linalg_lstsq(W_H, W_r, rcond=None)" in source


def test_seed_initialization_and_numpy_shuffle_controls() -> None:
    def initialized(seed: int) -> dict[str, torch.Tensor]:
        core.configure_determinism(seed, threads=1)
        model = core.make_model(np.zeros(9), np.ones(9))
        return core.clone_state_dict(model)

    first = initialized(0)
    repeat = initialized(0)
    different = initialized(1)
    assert core.state_dict_equal(first, repeat)
    assert core.state_dict_changed(first, different)

    def shuffled(seed: int) -> list[list[int]]:
        np.random.seed(seed)
        result = []
        for _ in range(3):
            values = np.arange(20)
            np.random.shuffle(values)
            result.append(values.tolist())
        return result

    assert shuffled(0) == shuffled(0)
    assert shuffled(0) != shuffled(1)


@pytest.mark.integration
def test_real_current_klt3_features_scaler_and_gt_leakage_boundary() -> None:
    if os.environ.get("CURRENT_TDL_RUN_INTEGRATION") != "1":
        pytest.skip("set CURRENT_TDL_RUN_INTEGRATION=1 for current-stack data checks")
    records, manifest = core.load_or_preprocess_dataset("KLT3")
    assert len(records) == 404
    assert manifest["neutral_solved_epochs"] == 404
    parts = core.record_feature_parts(records[0])
    assert parts is not None
    features = core.feature_tensor(parts)
    assert features.shape[1] == 9
    mean, std = core.scaler_from_records(records)
    assert mean.shape == std.shape == (9,)

    before = core.solver_input_fingerprint(records[0])
    altered = dict(records[0])
    altered["gt"] = records[0]["gt"].copy()
    altered["gt"][1] += 0.001
    after = core.solver_input_fingerprint(altered)
    assert before == after
    position = torch.tensor(records[0]["gnss"]["pos"], dtype=torch.float64)
    assert core.position_loss(records[0], position) != core.position_loss(altered, position)


@pytest.mark.integration
def test_deterministic_scientific_smoke_report() -> None:
    if not SMOKE_REPORT.is_file():
        pytest.skip("run current deterministic smoke test first")
    report = json.loads(SMOKE_REPORT.read_text(encoding="utf-8"))
    assert report["label"].startswith("NON-SCIENTIFIC")
    assert report["feature_shape_first_epoch"][1] == 9
    assert report["scaler_mean_shape"] == [9]
    assert report["scaler_std_shape"] == [9]
    assert report["tasgnss_differentiable_solve_status"] is True
    assert report["position_loss_has_autograd_graph"] is True
    assert report["nonzero_gradient_parameter_tensors"] > 0
    assert report["gradient_reached_network"] is True
    assert report["optimizer_changed_parameters"] is True
    assert report["checkpoint_saved"] is True
    assert report["checkpoint_reload_identical_outputs"] is True
    assert report["same_seed_training_replay_identical"] is True
    assert report["gt_leakage"]["only_loss_changed"] is True
    assert report["seed_protocol"]["same_seed_initial_parameters_identical"] is True
    assert report["seed_protocol"]["different_seed_initial_parameters_different"] is True
    assert report["seed_protocol"]["same_seed_numpy_shuffle_identical"] is True
    assert report["seed_protocol"]["model_construction_is_first_torch_rng_consumer"] is True


def test_frozen_checkpoint_hash_guard(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"frozen")
    manifest = {
        "seeds": [
            {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": core.sha256_file(checkpoint),
            }
        ]
    }
    _assert_frozen_unchanged(manifest)
    checkpoint.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="changed frozen checkpoint"):
        _assert_frozen_unchanged(manifest)


def _fake_seed_result(seed: int) -> dict[str, object]:
    metrics = {name: float(seed + index) for index, name in enumerate(("mean", "median", "rms", "p68", "p95", "max"))}
    delta = {
        "mean_delta": float(seed),
        "median_delta": float(seed),
        "p05": float(seed),
        "p95": float(seed),
        "fraction_improved": 0.5,
        "fraction_worsened": 0.5,
    }
    return {
        "seed": seed,
        "learned": {"2d": metrics, "3d": metrics, "east_rms": 1.0, "north_rms": 2.0, "up_rms": 3.0},
        "neutral_paired": {"2d": metrics, "3d": metrics, "east_rms": 1.0, "north_rms": 2.0, "up_rms": 3.0},
        "paired_delta": {"2d": delta, "3d": delta},
        "epoch_counts": {"solved_epochs": 203, "failed_epochs": 0, "exact_matched_epoch_count": 203},
    }


def test_across_seed_summary_requires_all_ten_and_does_not_pool_epochs() -> None:
    results = [_fake_seed_result(seed) for seed in core.SEEDS]
    summary = across_seed_summary(results)
    assert summary["seed_count"] == 10
    assert "epoch×seed rows are not pooled" in summary["aggregation_unit"]
    assert summary["metrics"]["learned_2d_mean"]["min"] == 0.0
    assert summary["metrics"]["learned_2d_mean"]["max"] == 9.0
    with pytest.raises(RuntimeError, match="all ten"):
        across_seed_summary(results[:-1])


def test_training_and_evaluation_code_enforce_dataset_separation_and_pairing() -> None:
    train_source = (ROOT / "validation/tdl_9feature_tasgnss_analysis/train.py").read_text()
    evaluate_source = (ROOT / "validation/tdl_9feature_tasgnss_analysis/evaluate_klt.py").read_text()
    assert '"training_datasets": ["KLT3"]' in train_source
    assert '"held_out_datasets": ["KLT1", "KLT2"]' in train_source
    assert 'for dataset in ("KLT1", "KLT2")' in evaluate_source
    assert "for seed in SEEDS" in evaluate_source
    assert "matched + failed + feature_rejected != total" in evaluate_source
    assert "trained TDL error - neutral TASGNSS error" in evaluate_source
    assert "_assert_frozen_unchanged(freeze_manifest)" in evaluate_source
    assert "best" not in evaluate_source.lower().replace("no best-seed selection", "")
