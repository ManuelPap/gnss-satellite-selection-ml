import importlib.metadata
import inspect
from pathlib import Path
import sys

import numpy as np
import pytest

from validation.ibiza_generalization.adapters import (
    ARCHITECTURES,
    raw_features_for_architecture,
)
from validation.ibiza_generalization.preprocess import (
    EXCLUSION_REASON_CODES,
    STATUS_ACCEPTED,
    canonical_content_sha256,
    preprocess_dataset,
    preprocess_epoch,
    parse_args as parse_preprocess_args,
    read_rinex_header,
    sha256_file,
    validate_dataset_arrays,
    write_deterministic_npz,
)
from validation.ibiza_generalization.prepare_runtime import prepare_runtime
from validation.ibiza_generalization.runtime_cache import (
    DEFAULT_PYRTKLIB_REPOSITORY,
    DEFAULT_RUNTIME_DIR,
    DEFAULT_TDL_REPOSITORY,
    import_pyrtklib,
    load_rtk_util,
    resolve_runtime,
    runtime_paths,
    validate_runtime_cache,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
IBIZA_RAW = REPOSITORY_ROOT.parent / "external_data/ibiza_2025_01_01/raw"
IBIZA_OBSERVATION = (
    IBIZA_RAW / "observation/IBIZ00ESP_R_20250010000_01D_30S_MO.rnx"
)
IBIZA_NPZ = REPOSITORY_ROOT.parent / "external_data/ibiza_2025_01_01/derived/ibiza_preprocessed.npz"
IBIZA_MANIFEST = (
    REPOSITORY_ROOT
    / "validation/ibiza_generalization/ibiza_preprocessed_manifest.json"
)
KLT_ROOT = Path("/tmp/gnss-weightnet-repro/extracted/data/0610_KLT")
PAPER_RUNTIME = runtime_paths(DEFAULT_RUNTIME_DIR)
TDL_DIR = PAPER_RUNTIME.tdl_dir
PYRTKLIB_SITE = PAPER_RUNTIME.pyrtklib_site
KLT_REFERENCE = REPOSITORY_ROOT / "validation/paper_weightnet/klt1_nn_weight_trace.npz"


def representative_arrays() -> dict[str, np.ndarray]:
    features = np.asarray([[41.25, 0.4, -2.0], [39.5, 0.8, 2.0]], dtype=np.float64)
    return {
        "features": features,
        "epoch_index": np.asarray([0, 0], dtype=np.int64),
        "row_epoch_time_gpst_like_s": np.asarray([1.0, 1.0], dtype=np.float64),
        "source_row_index": np.asarray([1, 3], dtype=np.int64),
        "source_observation_index": np.asarray([1, 3], dtype=np.int64),
        "rtklib_satellite_number": np.asarray([2, 61], dtype=np.int64),
        "satellite_prn": np.asarray([2, 2], dtype=np.int64),
        "constellation_code": np.asarray([1, 3], dtype=np.uint8),
        "source_signal_code": np.asarray([1, 1], dtype=np.int64),
        "system_clock_index": np.asarray([3, 5], dtype=np.int64),
        "raw_pseudorange_m": np.asarray([2.1e7, 2.2e7]),
        "raw_snr_units": np.asarray([41250.0, 39500.0]),
        "corrected_pseudorange_m": np.asarray([2.1e7 - 1.0, 2.2e7 + 1.0]),
        "satellite_position_ecef_m": np.asarray(
            [[2.0e7, 1.0e7, 1.5e7], [-1.0e7, 2.0e7, 1.4e7]]
        ),
        "satellite_clock_bias_s": np.asarray([1.0e-5, -2.0e-5]),
        "satellite_clock_correction_m": np.asarray([-2997.92458, 5995.84916]),
        "cn0_snr0_div_1000": features[:, 0].copy(),
        "elevation_rad": features[:, 1].copy(),
        "ols_residual_m": features[:, 2].copy(),
        "validity_code": np.full(2, STATUS_ACCEPTED, dtype=np.int8),
        "exclusion_reason_code": np.full(
            2, EXCLUSION_REASON_CODES["accepted"], dtype=np.int16
        ),
        "epoch_offsets": np.asarray([0, 2], dtype=np.int64),
        "epoch_time_gpst_like_s": np.asarray([1.0], dtype=np.float64),
        "epoch_ols_initial_state": np.asarray(
            [[1.0, 2.0, 3.0, 4.0, 0.0, 5.0, 0.0]]
        ),
        "epoch_design_rank": np.asarray([5], dtype=np.int64),
        "epoch_active_state_count": np.asarray([5], dtype=np.int64),
        "epoch_design_condition_number": np.asarray([10.0]),
        "epoch_normal_condition_number": np.asarray([100.0]),
        "epoch_ols_converged": np.asarray([1], dtype=np.uint8),
    }


def test_deterministic_npz_and_canonical_content_hash(tmp_path: Path) -> None:
    arrays = representative_arrays()
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    first_hash = write_deterministic_npz(first, arrays)
    second_hash = write_deterministic_npz(second, arrays)

    assert first.read_bytes() == second.read_bytes()
    assert first_hash == second_hash == sha256_file(first)
    assert canonical_content_sha256(arrays) == canonical_content_sha256(arrays)
    with np.load(first, allow_pickle=False) as loaded:
        assert set(loaded.files) == set(arrays)
        for name, expected in arrays.items():
            np.testing.assert_array_equal(loaded[name], expected)
            assert loaded[name].dtype.kind not in "OSU"


def test_ground_truth_has_no_preprocessing_input_or_data_route() -> None:
    parameters = inspect.signature(preprocess_dataset).parameters
    assert not any(
        token in name.lower()
        for name in parameters
        for token in ("ground_truth", "reference_coordinate", "evaluation_coordinate")
    )
    arrays = representative_arrays()
    first_reference_coordinate = np.asarray([38.9, 1.4, 100.0])
    changed_reference_coordinate = first_reference_coordinate + [1.0, -1.0, 1000.0]
    before = canonical_content_sha256(arrays)
    _ = first_reference_coordinate, changed_reference_coordinate
    after = canonical_content_sha256(arrays)
    assert before == after


def test_all_architectures_receive_one_identical_raw_feature_matrix() -> None:
    features = representative_arrays()["features"]
    dataset = {"features": features}
    adapted = [raw_features_for_architecture(dataset, name) for name in ARCHITECTURES]
    assert all(value is features for value in adapted)
    assert all(np.array_equal(value, features) for value in adapted)


def test_feature_units_row_alignment_and_full_rank_gate() -> None:
    arrays = representative_arrays()
    validate_dataset_arrays(arrays)
    np.testing.assert_array_equal(
        arrays["features"][:, 0], arrays["raw_snr_units"] / 1000.0
    )
    assert np.all(
        (arrays["features"][:, 1] >= 0.0)
        & (arrays["features"][:, 1] <= np.pi / 2)
    )
    assert arrays["features"][:, 2].dtype == np.float64
    assert arrays["corrected_pseudorange_m"].dtype == np.float64

    deficient = {name: value.copy() for name, value in arrays.items()}
    deficient["epoch_design_rank"][0] -= 1
    with pytest.raises(RuntimeError, match="rank deficient"):
        validate_dataset_arrays(deficient)

    misaligned = {name: value.copy() for name, value in arrays.items()}
    misaligned["satellite_prn"] = misaligned["satellite_prn"][:-1]
    with pytest.raises(RuntimeError, match="not aligned"):
        validate_dataset_arrays(misaligned)


@pytest.mark.skipif(not IBIZA_OBSERVATION.is_file(), reason="external Ibiza RINEX absent")
def test_ibiza_header_first_signal_semantics_and_units() -> None:
    header = read_rinex_header(IBIZA_OBSERVATION)
    assert header.version == "3.03"
    assert header.interval_s == 30.0
    assert header.first_observation == "2025-01-01T00:00:00.0000000"
    assert header.last_observation == "2025-01-01T23:59:30.0000000"
    assert header.time_system == "GPS"
    assert header.signal_strength_unit == "DBHZ"
    assert set(header.observation_types) == {"G", "R", "E"}
    for values in header.observation_types.values():
        assert values[:3] == ("C1C", "L1C", "S1C")


def test_generated_manifest_records_full_day_scientific_controls() -> None:
    import json

    manifest = json.loads(IBIZA_MANIFEST.read_text())
    assert manifest["status"] == "passed"
    assert manifest["epoch_counts"] == {
        "raw": 2880,
        "accepted": 2856,
        "rejected": 24,
        "rejected_by_reason": {"ols_max_iterations": 24},
    }
    assert manifest["satellite_epoch_observations"]["accepted"] == 73204
    assert manifest["features"]["shape"] == [73204, 3]
    assert manifest["features"]["ibiza_normalization_computed"] is False
    assert manifest["scientific_controls"]["ground_truth_used"] is False
    assert manifest["scientific_controls"]["network_inference_run"] is False
    assert manifest["scientific_controls"]["network_training_run"] is False
    assert manifest["provenance"]["paper_runtime"]["mode"] == (
        "persistent generated cache"
    )
    assert manifest["provenance"]["paper_runtime"]["cache_validated"] is True
    assert manifest["ols_diagnostics"]["all_accepted_epochs_converged"] is True
    assert manifest["ols_diagnostics"]["all_accepted_epochs_full_column_rank"] is True
    assert "/home/" not in IBIZA_MANIFEST.read_text()
    for mapping in manifest["rinex"]["first_processed_signal_mapping"].values():
        assert mapping["raw_pseudorange_observation"] == "C1C"
        assert mapping["raw_cn0_observation"] == "S1C"
        assert mapping["rtklib_code_labels_seen_in_accepted_rows"] == ["1C"]
        assert mapping["substitution_performed"] is False


@pytest.mark.skipif(
    not runtime_paths(DEFAULT_RUNTIME_DIR).manifest.is_file(),
    reason="persistent paper runtime cache absent",
)
def test_persistent_runtime_cache_integrity_and_readme() -> None:
    manifest = validate_runtime_cache(DEFAULT_RUNTIME_DIR)
    assert manifest["status"] == "ready"
    assert manifest["safe_to_delete_and_rebuild"] is True
    readme = PAPER_RUNTIME.readme.read_text()
    assert "persistent, generated compatibility cache" in readme
    assert "may disappear after a reboot" in readme
    assert "prepare_runtime" in readme

    action, second_manifest = prepare_runtime(
        runtime_dir=DEFAULT_RUNTIME_DIR,
        tdl_repository=DEFAULT_TDL_REPOSITORY,
        pyrtklib_repository=DEFAULT_PYRTKLIB_REPOSITORY,
    )
    assert action == "reused"
    assert second_manifest == manifest


def test_preprocessor_defaults_to_persistent_runtime() -> None:
    arguments = parse_preprocess_args([])
    assert arguments.runtime_dir.resolve() == DEFAULT_RUNTIME_DIR.resolve()
    assert not hasattr(arguments, "tdl_dir")
    assert not hasattr(arguments, "pyrtklib_site")


def test_runtime_import_rejects_ambient_wrong_origin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    selected_site = tmp_path / ".paper_runtime/pyrtklib-0.2.6-site"
    selected_site.mkdir(parents=True)
    ambient_site = tmp_path / "ambient-site"
    ambient_package = ambient_site / "pyrtklib"
    ambient_package.mkdir(parents=True)
    (ambient_package / "__init__.py").write_text(
        "ORIGIN = 'independent ambient installation'\n",
        encoding="utf-8",
    )
    ambient_metadata = ambient_site / "pyrtklib-0.2.6.dist-info"
    ambient_metadata.mkdir()
    (ambient_metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: pyrtklib\nVersion: 0.2.6\n",
        encoding="utf-8",
    )
    filtered_path = [
        item for item in sys.path if "pyrtklib-0.2.6-site" not in item
    ]
    monkeypatch.setattr(sys, "path", [str(ambient_site), *filtered_path])
    monkeypatch.delitem(sys.modules, "pyrtklib", raising=False)

    assert importlib.metadata.version("pyrtklib") == "0.2.6"

    try:
        with pytest.raises(RuntimeError, match="outside selected runtime site"):
            import_pyrtklib(selected_site)
    finally:
        sys.modules.pop("pyrtklib", None)


@pytest.mark.skipif(
    not runtime_paths(DEFAULT_RUNTIME_DIR).manifest.is_file(),
    reason="persistent paper runtime cache absent",
)
def test_resolved_runtime_import_has_selected_origin_and_version() -> None:
    runtime, manifest = resolve_runtime(DEFAULT_RUNTIME_DIR)
    prl = import_pyrtklib(runtime.pyrtklib_site)
    assert Path(prl.__file__).resolve().is_relative_to(runtime.pyrtklib_site)
    assert manifest["pyrtklib_version"] == "0.2.6"


@pytest.mark.skipif(not IBIZA_NPZ.is_file(), reason="generated external Ibiza NPZ absent")
def test_generated_npz_hash_schema_alignment_and_shared_adapter() -> None:
    import json

    manifest = json.loads(IBIZA_MANIFEST.read_text())
    assert sha256_file(IBIZA_NPZ) == manifest["output"]["npz_sha256"]
    with np.load(IBIZA_NPZ, allow_pickle=False) as dataset:
        arrays = {name: dataset[name] for name in dataset.files}
    assert all(value.dtype.kind not in "OSU" for value in arrays.values())
    assert not any(
        token in name.lower()
        for name in arrays
        for token in ("ground_truth", "reference_coordinate", "normalization")
    )
    validate_dataset_arrays(arrays)
    assert canonical_content_sha256(arrays) == manifest["output"][
        "canonical_content_sha256"
    ]
    raw_inputs = [
        raw_features_for_architecture(arrays, architecture)
        for architecture in ARCHITECTURES
    ]
    assert all(value is arrays["features"] for value in raw_inputs)


def _historical_modules():
    prl = import_pyrtklib(PYRTKLIB_SITE)
    util = load_rtk_util(TDL_DIR, module_name="ibiza_test_rtk_util")
    return prl, util


def _satellite_id(prl: object, satellite: int) -> str:
    value = prl.Arr1Dchar(4)
    prl.satno2id(satellite, value)
    return str(value[0])


def _gps_subset(prl: object, epoch: object) -> object:
    rows = [
        row
        for row in range(epoch.n)
        if _satellite_id(prl, epoch.data[row].sat).startswith("G")
    ]
    result = prl.obs_t()
    result.data = prl.Arr1Dobsd_t(len(rows))
    for target, source in enumerate(rows):
        result.data[target] = epoch.data[source]
    result.n = len(rows)
    result.nmax = len(rows)
    return result


@pytest.mark.integration
@pytest.mark.skipif(
    not all(
        path.exists()
        for path in (
            KLT_ROOT / "COM38_210610_025603.obs",
            KLT_ROOT / "sta/hksc161d.21n",
            TDL_DIR / "rtk_util.py",
            PYRTKLIB_SITE / "pyrtklib/pyrtklib.so",
            KLT_REFERENCE,
        )
    ),
    reason="validated disposable KLT/pyrtklib environment absent",
)
def test_paper_era_klt_epoch_compatibility_smoke() -> None:
    prl, util = _historical_modules()
    observation = KLT_ROOT / "COM38_210610_025603.obs"
    obs, nav, _station = util.read_obs(
        str(observation), str(KLT_ROOT / "sta/hksc161d.21*")
    )
    prl.sortobs(obs)
    epoch = util.split_obs(obs)[2382]
    gps = _gps_subset(prl, epoch)
    product, _audit, reason = preprocess_epoch(
        prl,
        util,
        gps,
        nav,
        split_epoch_index=2382,
        source_observation_offset=0,
    )
    assert reason == EXCLUSION_REASON_CODES["accepted"]
    assert product is not None
    with np.load(KLT_REFERENCE, allow_pickle=False) as reference:
        np.testing.assert_allclose(
            product.features, reference["features"], rtol=0.0, atol=1.0e-10
        )
        np.testing.assert_allclose(
            product.ols_state,
            reference["ols_initial_state"],
            rtol=0.0,
            atol=1.0e-7,
        )
