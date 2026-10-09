from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from validation.tdl_9feature_tasgnss_analysis import render_klt_results as results


ARTIFACT_ROOT = results.DEFAULT_ARTIFACT_ROOT
DOCUMENT = results.DEFAULT_DOCUMENT


@pytest.fixture(scope="module")
def bundle() -> dict[str, object]:
    if not ARTIFACT_ROOT.is_dir():
        pytest.skip(f"completed result archive not found: {ARTIFACT_ROOT}")
    return results.load_and_validate(ARTIFACT_ROOT)


def _fingerprint(path: Path) -> tuple[int, int, str]:
    stat = path.stat()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return stat.st_size, stat.st_mtime_ns, digest


def test_completed_results_have_full_paired_support_and_frozen_hashes(
    bundle: dict[str, object],
) -> None:
    frozen = bundle["frozen"]
    evaluation = bundle["evaluation"]
    per_seed = bundle["per_seed"]
    frozen_hashes = {row["seed"]: row["checkpoint_sha256"] for row in frozen["seeds"]}

    assert [row["seed"] for row in frozen["seeds"]] == list(range(10))
    assert evaluation["evaluated_seeds"] == list(range(10))
    for dataset, expected_epochs in results.EXPECTED_EPOCHS.items():
        assert [row["seed"] for row in per_seed[dataset]] == list(range(10))
        for row in per_seed[dataset]:
            assert row["checkpoint_sha256"] == frozen_hashes[row["seed"]]
            assert row["epoch_counts"] == {
                "preprocessed_epochs": expected_epochs,
                "solved_epochs": expected_epochs,
                "failed_epochs": 0,
                "feature_rejected_epochs": 0,
                "exact_matched_epoch_count": expected_epochs,
                "reconciled": True,
            }


def test_across_seed_results_are_seed_summary_aggregates(
    bundle: dict[str, object],
) -> None:
    for dataset in results.EXPECTED_EPOCHS:
        summary = bundle["summaries"][dataset]
        assert summary["seed_count"] == 10
        assert summary["aggregation_unit"] == results.AGGREGATION_UNIT
        assert len(summary["epoch_counts_by_seed"]) == 10


def test_markdown_is_current_and_formats_source_results_to_two_decimals(
    bundle: dict[str, object],
) -> None:
    rendered = results.render_markdown(bundle)
    assert DOCUMENT.read_text(encoding="utf-8") == rendered

    for dataset in results.EXPECTED_EPOCHS:
        metrics = bundle["summaries"][dataset]["metrics"]
        expected_values = (
            metrics["learned_2d_mean"]["mean"],
            metrics["learned_3d_mean"]["mean"],
            metrics["delta_2d_mean_delta"]["mean"],
            metrics["delta_3d_mean_delta"]["mean"],
            metrics["learned_up_rms"]["mean"],
            metrics["neutral_paired_up_rms"]["mean"],
        )
        for value in expected_values:
            assert f"{value:.2f}" in rendered
        assert f"{100.0 * metrics['delta_2d_fraction_improved']['mean']:.2f}" in rendered
        assert f"{100.0 * metrics['delta_3d_fraction_improved']['mean']:.2f}" in rendered


def test_renderer_does_not_mutate_json_or_frozen_models(
    bundle: dict[str, object],
) -> None:
    frozen = bundle["frozen"]
    paths = [ARTIFACT_ROOT / relative for relative in results.SOURCE_FILES]
    checkpoints = {
        Path(row["checkpoint"]): row["checkpoint_sha256"] for row in frozen["seeds"]
    }
    paths.extend(checkpoints)
    before = {path: _fingerprint(path) for path in paths}
    for checkpoint, expected_hash in checkpoints.items():
        assert before[checkpoint][2] == expected_hash

    reread = results.load_and_validate(ARTIFACT_ROOT)
    results.render_markdown(reread)

    after = {path: _fingerprint(path) for path in paths}
    assert after == before
