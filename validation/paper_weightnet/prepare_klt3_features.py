#!/usr/bin/env python3
"""Prepare the exact paper-era KLT3 WeightNet inputs and training constants."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from core import FEATURE_NAMES, FEATURE_UNITS, SYSTEM_TO_CLOCK_INDEX, construct_features

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from validation.ibiza_generalization.runtime_cache import (  # noqa: E402
    DEFAULT_RUNTIME_DIR,
    PYRTKLIB_COMMIT,
    PYRTKLIB_VERSION,
    TDL_COMMIT,
    import_pyrtklib,
    load_rtk_util,
    resolve_runtime,
)


DATASET_URL = (
    "https://www.dropbox.com/scl/fi/d3urwaquf5ema5j0unmt4/"
    "data.zip?rlkey=tuwpx9pdzqtdvoeoqwhcc5gi8&st=wh5qhg6e&dl=1"
)
EXPECTED_ARCHIVE_HASH = (
    "2afd7b1e395f8494e6992d1e109f9e8d9d83d992bf6d446d83d91aaf46cfc721"
)
EXPECTED_OBSERVATION_HASH = (
    "f722557326d1d32c42e023d4e78515e885d21c8ae824e79460bef61c67b9b5c4"
)
EXPECTED_GROUND_TRUTH_HASH = (
    "9f7ae89cfe4db0470e78ad1cc6b0a3209a740f5ed2f0500b6dac532cc4f58b42"
)
EXPECTED_NAVIGATION_HASHES = {
    "hksc161d.21f": "64e8e3ec2f4a9eeb17379a499e5779d378978b834a1a1441240e487e7ce23768",
    "hksc161d.21g": "0bddf6292d39f00845e9038f7f87dcecc403ec9944ca144b7345bcb233d8e660",
    "hksc161d.21l": "795de82407d394097628a7a2595e284d666df3dc62c4ad5a34896d6de05b84e3",
    "hksc161d.21m": "9583d5e061d46f3de29b6f783332dda9e7ff61d62041f2c8a031f6b45628e376",
    "hksc161d.21n": "a5bc8ab35fe0c80f91d0e57517b495be6563385d235835fd6aa063d73bb7072c",
    "hksc161d.21o": "7335032796f9b46176c359e8a39cc3f8496dd1f3fd6003fcfb9a794aafe88dd6",
}
PUBLISHED_EPOCHS = 404
RELEASED_CODE_EPOCHS = 405
EXPECTED_MEASUREMENTS = 8857
DEFAULT_START_TIME = 1623297151.0
DEFAULT_END_TIME = 1623297556.0


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME_DIR)
    parser.add_argument("--observation", required=True, type=Path)
    parser.add_argument("--ephemeris-glob", required=True)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--dataset-archive", type=Path)
    parser.add_argument("--start-time", type=float, default=DEFAULT_START_TIME)
    parser.add_argument("--end-time", type=float, default=DEFAULT_END_TIME)
    parser.add_argument("--output", type=Path, default=here / "klt3_features.npz")
    parser.add_argument(
        "--manifest", type=Path, default=here / "klt3_feature_manifest.json"
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path, expected: str, label: str) -> dict[str, object]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(
            f"{label} SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return {"name": path.name, "size_bytes": path.stat().st_size, "sha256": actual}


def satellite_id(prl: object, satellite: int) -> str:
    text = prl.Arr1Dchar(4)
    prl.satno2id(satellite, text)
    return str(text[0])


def native_array(values: object, length: int) -> np.ndarray:
    return np.asarray([values[index] for index in range(length)], dtype=np.float64)


def load_ground_truth_window(
    path: Path, start_time: float, end_time: float
) -> np.ndarray:
    """Read only the required public 100 Hz GT window.

    The archived program loaded the entire text with pandas, added 18 seconds
    to column zero, and selected the nearest row.  This streaming equivalent
    preserves those operations while avoiding a roughly one-million-row table.
    """

    records: list[list[float]] = []
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream):
            if line_number < 30:
                continue
            fields = line.split()
            if len(fields) < 10:
                continue
            try:
                values = [float(item) for item in fields[:10]]
            except ValueError:
                continue
            gpst_like = values[0] + 18.0
            if start_time - 1.0 <= gpst_like <= end_time + 1.0:
                latitude = values[3] + values[4] / 60.0 + values[5] / 3600.0
                longitude = values[6] + values[7] / 60.0 + values[8] / 3600.0
                records.append([gpst_like, latitude, longitude, values[9]])
    result = np.asarray(records, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 4 or result.shape[0] == 0:
        raise RuntimeError("ground-truth window is empty or malformed")
    if np.any(np.diff(result[:, 0]) < 0.0):
        raise RuntimeError("ground-truth rows are not chronological")
    return result


def nearest_ground_truth(table: np.ndarray, epoch_time: float) -> np.ndarray:
    # np.argmin, like pandas Series.argmin in the released code, selects the
    # first row when the .005 observation is equidistant from .00 and .01.
    index = int(np.abs(table[:, 0] - epoch_time).argmin())
    return table[index, 1:4].copy()


def main() -> int:
    args = parse_args()
    observation_path = args.observation.resolve()
    ground_truth_path = args.ground_truth.resolve()
    runtime, _runtime_manifest = resolve_runtime(args.runtime_dir)
    tdl_dir = runtime.tdl_dir
    output_path = args.output.resolve()
    manifest_path = args.manifest.resolve()

    observation_record = verify(
        observation_path, EXPECTED_OBSERVATION_HASH, "KLT rover observation"
    )
    ground_truth_record = verify(
        ground_truth_path, EXPECTED_GROUND_TRUTH_HASH, "KLT ground truth"
    )
    archive_record = None
    if args.dataset_archive is not None:
        archive_record = verify(
            args.dataset_archive, EXPECTED_ARCHIVE_HASH, "KLT public archive"
        )
    ephemeris_paths = [Path(item).resolve() for item in sorted(glob.glob(args.ephemeris_glob))]
    if {path.name for path in ephemeris_paths} != set(EXPECTED_NAVIGATION_HASHES):
        raise RuntimeError(
            "ephemeris wildcard resolved to unexpected names: "
            + ", ".join(path.name for path in ephemeris_paths)
        )
    navigation_records = [
        verify(path, EXPECTED_NAVIGATION_HASHES[path.name], f"navigation {path.name}")
        for path in ephemeris_paths
    ]
    prl = import_pyrtklib(runtime.pyrtklib_site)
    util = load_rtk_util(tdl_dir, module_name="paper_klt3_rtk_util")

    ground_truth = load_ground_truth_window(
        ground_truth_path, args.start_time, args.end_time
    )
    obs, nav, _station = util.read_obs(str(observation_path), args.ephemeris_glob)
    prl.sortobs(obs)
    split_epochs = util.split_obs(obs)

    all_features: list[np.ndarray] = []
    all_positions: list[np.ndarray] = []
    all_clock_bias: list[np.ndarray] = []
    all_pseudorange: list[np.ndarray] = []
    all_clock_indices: list[np.ndarray] = []
    all_satellite_ids: list[np.ndarray] = []
    initial_states: list[np.ndarray] = []
    ground_truth_geodetic: list[np.ndarray] = []
    epoch_times: list[float] = []
    split_indices: list[int] = []
    satellite_counts: list[int] = []
    invalid_epochs: list[dict[str, object]] = []
    configured_epoch_count = 0

    for split_index, epoch in enumerate(split_epochs):
        epoch_time = float(epoch.data[0].time.time + epoch.data[0].time.sec)
        if not (epoch_time > args.start_time and epoch_time < args.end_time):
            continue
        configured_epoch_count += 1
        try:
            result = util.get_ls_pnt_pos(epoch, nav)
        except Exception as error:
            invalid_epochs.append(
                {"split_index": split_index, "time": epoch_time, "reason": repr(error)}
            )
            continue
        if not result["status"]:
            invalid_epochs.append(
                {
                    "split_index": split_index,
                    "time": epoch_time,
                    "reason": str(result.get("msg", "unknown failure")),
                }
            )
            continue

        data = result["data"]
        excluded = list(data["exclude"])
        satellites = list(data["sats"])
        included_rows = list(set(range(len(satellites))) - set(excluded))
        ordered_rows = [index for index in range(len(satellites)) if index not in excluded]
        if included_rows != ordered_rows:
            raise RuntimeError(
                "legacy list(set(...)) residual order differs from H order at "
                f"split epoch {split_index}"
            )

        residual = np.asarray(data["residual"], dtype=np.float64).reshape(-1)
        snr = np.asarray(data["SNR"], dtype=np.float64).reshape(-1)
        azel = np.asarray(data["azel"], dtype=np.float64).reshape(-1, 2)
        elevation = np.delete(azel, excluded, axis=0)[:, 1]
        features = construct_features(snr, elevation, residual)
        count = features.shape[0]
        if count != len(included_rows):
            raise RuntimeError(
                f"feature/row count mismatch at split epoch {split_index}: "
                f"{count} != {len(included_rows)}"
            )

        positions_full = native_array(data["eph"], len(satellites) * 6).reshape(-1, 6)
        clocks_full = native_array(data["dts"], len(satellites) * 2).reshape(-1, 2)
        ids = np.asarray(
            [satellite_id(prl, satellites[index]) for index in included_rows],
            dtype="U3",
        )
        try:
            clock_indices = np.asarray(
                [SYSTEM_TO_CLOCK_INDEX[item[0]] for item in ids], dtype=np.int64
            )
        except KeyError as error:
            raise RuntimeError(f"unsupported constellation in retained rows: {error}") from error
        pseudorange = np.asarray(data["prs"], dtype=np.float64).reshape(-1)
        if pseudorange.shape != (count,):
            raise RuntimeError("corrected pseudorange count does not match features")

        gt = nearest_ground_truth(ground_truth, epoch_time)
        # Structural leakage check: feature construction has no ground-truth
        # argument. Re-evaluate after a large fictitious GT perturbation and
        # demand bitwise identity while holding observations fixed.
        _perturbed_gt = gt + np.asarray([1.0, -1.0, 1000.0])
        features_after_gt_change = construct_features(snr, elevation, residual)
        if not np.array_equal(features, features_after_gt_change):
            raise RuntimeError("features changed after a ground-truth perturbation")

        all_features.append(features)
        all_positions.append(positions_full[included_rows, :3])
        all_clock_bias.append(clocks_full[included_rows, 0])
        all_pseudorange.append(pseudorange)
        all_clock_indices.append(clock_indices)
        all_satellite_ids.append(ids)
        initial_states.append(np.asarray(result["pos"], dtype=np.float64))
        ground_truth_geodetic.append(gt)
        epoch_times.append(epoch_time)
        split_indices.append(split_index)
        satellite_counts.append(count)

    if not all_features:
        raise RuntimeError("paper-era preprocessing produced no valid KLT3 epochs")

    features_all = np.vstack(all_features)
    positions_all = np.vstack(all_positions)
    clock_bias_all = np.concatenate(all_clock_bias)
    pseudorange_all = np.concatenate(all_pseudorange)
    clock_indices_all = np.concatenate(all_clock_indices)
    satellite_ids_all = np.concatenate(all_satellite_ids)
    offsets = np.concatenate(([0], np.cumsum(satellite_counts))).astype(np.int64)
    valid_epoch_count = len(all_features)
    measurement_count = int(features_all.shape[0])

    drop_first_measurements = measurement_count - satellite_counts[0]
    drop_last_measurements = measurement_count - satellite_counts[-1]
    if valid_epoch_count != RELEASED_CODE_EPOCHS or measurement_count != EXPECTED_MEASUREMENTS:
        raise RuntimeError(
            "released-code KLT3 cardinality gate failed: expected "
            f"{RELEASED_CODE_EPOCHS} valid epochs/{EXPECTED_MEASUREMENTS} measurements, "
            f"got {valid_epoch_count}/{measurement_count}; configured epochs="
            f"{configured_epoch_count}, first/last valid times="
            f"{epoch_times[0]:.3f}/{epoch_times[-1]:.3f}, first/last satellite "
            f"counts={satellite_counts[0]}/{satellite_counts[-1]}, measurement "
            f"counts after dropping first/last epoch="
            f"{drop_first_measurements}/{drop_last_measurements}; no training is permitted"
        )
    if drop_first_measurements != 8836 or drop_last_measurements != 8835:
        raise RuntimeError(
            "KLT3 boundary-epoch cardinality evidence changed unexpectedly: "
            f"drop-first={drop_first_measurements}, drop-last={drop_last_measurements}"
        )
    arrays_to_check = {
        "features": features_all,
        "positions": positions_all,
        "clock bias": clock_bias_all,
        "pseudorange": pseudorange_all,
        "initial states": np.vstack(initial_states),
        "ground truth": np.vstack(ground_truth_geodetic),
    }
    for label, values in arrays_to_check.items():
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"{label} contains a non-finite value")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        features=features_all,
        satellite_positions_ecef_m=positions_all,
        satellite_clock_bias_s=clock_bias_all,
        corrected_pseudorange_m=pseudorange_all,
        system_clock_indices=clock_indices_all,
        satellite_ids=satellite_ids_all,
        epoch_offsets=offsets,
        initial_states=np.vstack(initial_states),
        ground_truth_geodetic_deg_m=np.vstack(ground_truth_geodetic),
        epoch_times_gpst_like=np.asarray(epoch_times, dtype=np.float64),
        split_epoch_indices=np.asarray(split_indices, dtype=np.int64),
    )
    cache_hash = sha256(output_path)
    feature_min = features_all.min(axis=0)
    feature_max = features_all.max(axis=0)
    feature_mean = features_all.mean(axis=0)
    feature_std = features_all.std(axis=0)
    manifest = {
        "status": "passed",
        "reference_commit": TDL_COMMIT,
        "pyrtklib_hypothesis_commit": PYRTKLIB_COMMIT,
        "pyrtklib_version": PYRTKLIB_VERSION,
        "pyrtklib_publication_version_uncertain": True,
        "dataset_url": DATASET_URL,
        "dataset_archive": archive_record,
        "rover_observation": observation_record,
        "ground_truth": ground_truth_record,
        "navigation_files": navigation_records,
        "configured_interval": {
            "strict_start_gpst_like": args.start_time,
            "strict_end_gpst_like": args.end_time,
            "epochs_before_OLS_status_filter": configured_epoch_count,
            "first_retained_timestamp_gpst_like": epoch_times[0],
            "last_retained_timestamp_gpst_like": epoch_times[-1],
        },
        "valid_epoch_count": valid_epoch_count,
        "satellite_measurement_count": measurement_count,
        "cardinality_discrepancy": {
            "classification": "paper-versus-released-code cardinality discrepancy",
            "published_metadata": {
                "epoch_count": PUBLISHED_EPOCHS,
                "satellite_measurement_count": EXPECTED_MEASUREMENTS,
            },
            "released_code_reproduction": {
                "epoch_count": RELEASED_CODE_EPOCHS,
                "satellite_measurement_count": EXPECTED_MEASUREMENTS,
            },
            "drop_first_epoch": {
                "epoch_count": PUBLISHED_EPOCHS,
                "satellite_measurement_count": drop_first_measurements,
            },
            "drop_last_epoch": {
                "epoch_count": PUBLISHED_EPOCHS,
                "satellite_measurement_count": drop_last_measurements,
            },
            "undocumented_epoch_removal_applied": False,
            "interpretation": (
                "The released dd5eac6 strict predicate includes the fractional "
                "lower-bound epoch; there is no evidence-supported removal that "
                "simultaneously reproduces both published cardinalities."
            ),
        },
        "released_code_expected": {
            "valid_epoch_count": RELEASED_CODE_EPOCHS,
            "satellite_measurement_count": EXPECTED_MEASUREMENTS,
        },
        "invalid_epochs": invalid_epochs,
        "satellite_counts_per_epoch": satellite_counts,
        "satellite_count_summary": {
            "min": int(np.min(satellite_counts)),
            "max": int(np.max(satellite_counts)),
            "mean": float(np.mean(satellite_counts)),
        },
        "features": {
            "names": list(FEATURE_NAMES),
            "units": list(FEATURE_UNITS),
            "minimum": feature_min.tolist(),
            "maximum": feature_max.tolist(),
            "mean": feature_mean.tolist(),
            "population_std": feature_std.tolist(),
            "all_finite": True,
            "ground_truth_independence_bitwise": True,
        },
        "cache": {
            "path": str(output_path),
            "size_bytes": output_path.stat().st_size,
            "sha256": cache_hash,
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"KLT3 valid epochs: {valid_epoch_count}")
    print(f"KLT3 retained satellite measurements: {measurement_count}")
    print(f"feature mean: {feature_mean.tolist()}")
    print(f"feature std: {feature_std.tolist()}")
    print(f"cache: {output_path} ({cache_hash})")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
