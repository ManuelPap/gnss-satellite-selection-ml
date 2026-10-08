"""Transparent held-out evaluation helpers for the paper-era WeightNet.

This module deliberately keeps data preparation, inference, positioning, and
metric aggregation as separate functions.  The command-line entry points in
this directory compose those functions; none of them retrains the network.
"""

from __future__ import annotations

import csv
import glob
import hashlib
import json
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pymap3d as p3d
import torch

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

try:
    from .core import (
        FEATURE_NAMES,
        FEATURE_UNITS,
        SPEED_OF_LIGHT_M_S,
        SYSTEM_TO_CLOCK_INDEX,
        WeightNet,
        construct_features,
        solve_paper_weighted_position,
    )
except ImportError:  # Direct execution from validation/paper_weightnet.
    from core import (
        FEATURE_NAMES,
        FEATURE_UNITS,
        SPEED_OF_LIGHT_M_S,
        SYSTEM_TO_CLOCK_INDEX,
        WeightNet,
        construct_features,
        solve_paper_weighted_position,
    )


CHECKPOINT_SHA256 = (
    "2ccb3f7efc17755499f38a2ccff6205a8a54816f4dbe984a941bf6cc4d26644c"
)
UPSTREAM_RTK_UTIL_SHA256 = (
    "25fa8e26ff5772960e0dd33763950868aaaf1d181ebae25b3b93051e7511692c"
)
CPU_PATCHED_RTK_UTIL_SHA256 = (
    "4d905e4334ec0451c89311d44e304ff9fdac426785edf2e9332e75182f1c48c2"
)

# These are the float64 KLT3 population statistics before the released
# program's intentional float32 tensor construction followed by net.double().
KLT3_FEATURE_MEAN = np.asarray(
    [29.084904595235407, 0.8471899979658503, -1.7873233713548635e-07],
    dtype=np.float64,
)
KLT3_FEATURE_POPULATION_STD = np.asarray(
    [5.89184205821295, 0.28798909935383876, 4.3018036109624145],
    dtype=np.float64,
)
FROZEN_MODEL_MEAN = KLT3_FEATURE_MEAN.astype(np.float32).astype(np.float64)
FROZEN_MODEL_STD = KLT3_FEATURE_POPULATION_STD.astype(np.float32).astype(np.float64)

KLT_OBSERVATION_SHA256 = (
    "f722557326d1d32c42e023d4e78515e885d21c8ae824e79460bef61c67b9b5c4"
)
KLT_GROUND_TRUTH_SHA256 = (
    "9f7ae89cfe4db0470e78ad1cc6b0a3209a740f5ed2f0500b6dac532cc4f58b42"
)
KLT_NAVIGATION_SHA256 = {
    "hksc161d.21f": "64e8e3ec2f4a9eeb17379a499e5779d378978b834a1a1441240e487e7ce23768",
    "hksc161d.21g": "0bddf6292d39f00845e9038f7f87dcecc403ec9944ca144b7345bcb233d8e660",
    "hksc161d.21l": "795de82407d394097628a7a2595e284d666df3dc62c4ad5a34896d6de05b84e3",
    "hksc161d.21m": "9583d5e061d46f3de29b6f783332dda9e7ff61d62041f2c8a031f6b45628e376",
    "hksc161d.21n": "a5bc8ab35fe0c80f91d0e57517b495be6563385d235835fd6aa063d73bb7072c",
    "hksc161d.21o": "7335032796f9b46176c359e8a39cc3f8496dd1f3fd6003fcfb9a794aafe88dd6",
}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    start_time: float
    end_time: float
    observation_relative: str
    ephemeris_relative: tuple[str, ...]
    ground_truth_relative: str
    published_epochs: int
    published_measurements: int
    paper_2d_mean_m: float
    paper_3d_mean_m: float


DATASET_SPECS = {
    "KLT1": DatasetSpec(
        name="KLT1",
        start_time=1623296154.0,
        end_time=1623296357.0,
        observation_relative="0610_KLT/COM38_210610_025603.obs",
        ephemeris_relative=("0610_KLT/sta/hksc161d.21*",),
        ground_truth_relative="0610_KLT/20210610_100.txt",
        published_epochs=203,
        published_measurements=4676,
        paper_2d_mean_m=2.57,
        paper_3d_mean_m=9.92,
    ),
    "KLT2": DatasetSpec(
        name="KLT2",
        start_time=1623296917.0,
        end_time=1623297126.0,
        observation_relative="0610_KLT/COM38_210610_025603.obs",
        ephemeris_relative=("0610_KLT/sta/hksc161d.21*",),
        ground_truth_relative="0610_KLT/20210610_100.txt",
        published_epochs=209,
        published_measurements=4914,
        paper_2d_mean_m=2.89,
        paper_3d_mean_m=7.75,
    ),
    "Whampoa": DatasetSpec(
        name="Whampoa",
        start_time=1626238258.0,
        end_time=1626239511.0,
        observation_relative="whampoa/20210714.2.whampoa.ublox.f9p.obs",
        ephemeris_relative=(
            "whampoa/hksc195e.21*",
            "whampoa/hksc195f.21*",
        ),
        ground_truth_relative="whampoa/20210714_2_100hz.txt",
        published_epochs=1205,
        published_measurements=12926,
        paper_2d_mean_m=16.11,
        paper_3d_mean_m=42.49,
    ),
}


@dataclass(frozen=True)
class FeatureInputPaths:
    observation: Path
    ephemeris_patterns: tuple[str, ...]
    tdl_dir: Path
    pyrtklib_site: Path


@dataclass(frozen=True)
class InputPaths(FeatureInputPaths):
    ground_truth: Path


@dataclass(frozen=True)
class PreparedFeatureEpoch:
    valid_epoch_index: int
    candidate_epoch_index: int
    split_epoch_index: int
    epoch_time: float
    satellite_ids: np.ndarray
    satellite_numbers: np.ndarray
    raw_pseudorange_m: np.ndarray
    raw_snr_units: np.ndarray
    features: np.ndarray
    satellite_positions_ecef_m: np.ndarray
    satellite_clock_bias_s: np.ndarray
    corrected_pseudorange_m: np.ndarray
    system_clock_indices: np.ndarray
    initial_ols_state: np.ndarray


@dataclass(frozen=True)
class PreparedEpoch(PreparedFeatureEpoch):
    gt_time: float
    gt_time_difference_s: float
    ground_truth_geodetic_deg_m: np.ndarray


@dataclass(frozen=True)
class PreparedFeatureDataset:
    spec: DatasetSpec
    inputs: FeatureInputPaths
    raw_split_epoch_count: int
    candidate_epoch_count: int
    epochs: tuple[PreparedFeatureEpoch, ...]
    invalid_epochs: tuple[dict[str, object], ...]
    input_provenance: dict[str, object]

    @property
    def measurement_count(self) -> int:
        return sum(epoch.features.shape[0] for epoch in self.epochs)


@dataclass(frozen=True)
class PreparedDataset:
    spec: DatasetSpec
    inputs: InputPaths
    raw_split_epoch_count: int
    candidate_epoch_count: int
    epochs: tuple[PreparedEpoch, ...]
    invalid_epochs: tuple[dict[str, object], ...]
    input_provenance: dict[str, object]

    @property
    def measurement_count(self) -> int:
        return sum(epoch.features.shape[0] for epoch in self.epochs)


@dataclass(frozen=True)
class EpochEvaluation:
    prepared: PreparedEpoch
    normalized_features: np.ndarray
    weights: np.ndarray
    solution: object
    estimated_ecef_m: np.ndarray
    estimated_geodetic_deg_m: np.ndarray
    ground_truth_ecef_m: np.ndarray
    enu_error_m: np.ndarray
    error_2d_m: float
    error_3d_m: float
    ols_enu_error_m: np.ndarray
    ols_error_2d_m: float
    ols_error_3d_m: float
    historical_wls_status: str
    historical_wls_residual_norm_m: float


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, object]:
    path = path.resolve()
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def verify_checkpoint(path: Path) -> str:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"frozen checkpoint not found: {path}")
    actual = sha256(path)
    if actual != CHECKPOINT_SHA256:
        raise RuntimeError(
            f"checkpoint SHA-256 mismatch: expected {CHECKPOINT_SHA256}, got {actual}"
        )
    return actual


def _first_existing(candidates: Iterable[Path], label: str) -> Path:
    candidate_list = tuple(candidates)
    for candidate in candidate_list:
        if candidate.exists():
            return candidate.resolve()
    rendered = "\n  ".join(str(item) for item in candidate_list)
    raise FileNotFoundError(f"could not locate {label}; checked:\n  {rendered}")


def _input_roots(data_root: Path | None) -> list[Path]:
    if data_root is not None:
        roots = [data_root.resolve()]
        if (roots[0] / "data").is_dir():
            roots.insert(0, roots[0] / "data")
        return roots
    return [
        Path("/tmp/gnss-weightnet-repro/extracted/data"),
        Path("/tmp/tdl_gnss_paper_data/data"),
        repository_root() / "validation/paper_weightnet/data",
    ]


def resolve_feature_input_paths(
    spec: DatasetSpec,
    *,
    data_root: Path | None = None,
    observation: Path | None = None,
    ephemeris_patterns: Sequence[str] | None = None,
    runtime_dir: Path = DEFAULT_RUNTIME_DIR,
) -> FeatureInputPaths:
    """Resolve only inputs consumed by historical preprocessing and OLS."""

    roots = _input_roots(data_root)

    observation_path = observation.resolve() if observation else _first_existing(
        (root / spec.observation_relative for root in roots),
        f"{spec.name} historical observation file",
    )
    if ephemeris_patterns:
        eph = tuple(str(item) for item in ephemeris_patterns)
    else:
        selected_root = None
        for root in roots:
            patterns = tuple(str(root / item) for item in spec.ephemeris_relative)
            if all(glob.glob(pattern) for pattern in patterns):
                selected_root = root
                eph = patterns
                break
        if selected_root is None:
            rendered = "\n  ".join(
                str(root / item) for root in roots for item in spec.ephemeris_relative
            )
            raise FileNotFoundError(
                f"could not locate {spec.name} historical navigation files; checked:\n  "
                + rendered
            )

    runtime, _runtime_manifest = resolve_runtime(runtime_dir)
    return FeatureInputPaths(
        observation=observation_path,
        ephemeris_patterns=eph,
        tdl_dir=runtime.tdl_dir,
        pyrtklib_site=runtime.pyrtklib_site,
    )


def resolve_input_paths(
    spec: DatasetSpec,
    *,
    data_root: Path | None = None,
    observation: Path | None = None,
    ephemeris_patterns: Sequence[str] | None = None,
    ground_truth: Path | None = None,
    runtime_dir: Path = DEFAULT_RUNTIME_DIR,
) -> InputPaths:
    """Resolve historical preprocessing plus later evaluation inputs."""

    feature_inputs = resolve_feature_input_paths(
        spec,
        data_root=data_root,
        observation=observation,
        ephemeris_patterns=ephemeris_patterns,
        runtime_dir=runtime_dir,
    )
    roots = _input_roots(data_root)
    ground_truth_path = ground_truth.resolve() if ground_truth else _first_existing(
        (root / spec.ground_truth_relative for root in roots),
        f"{spec.name} historical ground-truth file",
    )
    return InputPaths(
        observation=feature_inputs.observation,
        ephemeris_patterns=feature_inputs.ephemeris_patterns,
        tdl_dir=feature_inputs.tdl_dir,
        pyrtklib_site=feature_inputs.pyrtklib_site,
        ground_truth=ground_truth_path,
    )


def load_historical_modules(inputs: FeatureInputPaths) -> tuple[object, object]:
    """Import the pinned preprocessing dependency without writing to it."""

    source = inputs.tdl_dir / "rtk_util.py"
    if not source.is_file():
        raise FileNotFoundError(f"missing historical rtk_util.py: {source}")
    source_hash = sha256(source)
    if source_hash not in (UPSTREAM_RTK_UTIL_SHA256, CPU_PATCHED_RTK_UTIL_SHA256):
        raise RuntimeError(
            "rtk_util.py is not the audited dd5eac6 source (or its documented "
            f"CPU-only CUDA string substitution): {source_hash}"
        )

    prl = import_pyrtklib(inputs.pyrtklib_site)
    util = load_rtk_util(
        inputs.tdl_dir,
        module_name="paper_weightnet_rtk_util",
    )
    return prl, util


def _verify_feature_input_files(
    spec: DatasetSpec, inputs: FeatureInputPaths
) -> dict[str, object]:
    observation = file_record(inputs.observation)
    navigation_paths = [
        Path(item).resolve()
        for pattern in inputs.ephemeris_patterns
        for item in sorted(glob.glob(pattern))
    ]
    if not navigation_paths:
        raise FileNotFoundError("ephemeris patterns resolved to no files")
    navigation = [file_record(path) for path in navigation_paths]

    if spec.name.startswith("KLT"):
        if observation["sha256"] != KLT_OBSERVATION_SHA256:
            raise RuntimeError("KLT observation hash does not match the audited archive")
        actual_navigation = {
            Path(item["path"]).name: item["sha256"] for item in navigation
        }
        if actual_navigation != KLT_NAVIGATION_SHA256:
            raise RuntimeError(
                "KLT navigation hashes/names do not match the audited archive"
            )
    return {
        "observation": observation,
        "navigation": navigation,
        "tdl_rtk_util": file_record(inputs.tdl_dir / "rtk_util.py"),
        "pyrtklib_version": PYRTKLIB_VERSION,
        "pyrtklib_hypothesis_commit": PYRTKLIB_COMMIT,
        "pyrtklib_publication_version_uncertain": True,
    }


def _verify_input_files(spec: DatasetSpec, inputs: InputPaths) -> dict[str, object]:
    feature_provenance = _verify_feature_input_files(spec, inputs)
    ground_truth = file_record(inputs.ground_truth)
    if spec.name.startswith("KLT") and ground_truth["sha256"] != KLT_GROUND_TRUTH_SHA256:
        raise RuntimeError("KLT ground-truth hash does not match the audited archive")
    return {
        "observation": feature_provenance["observation"],
        "ground_truth": ground_truth,
        "navigation": feature_provenance["navigation"],
        "tdl_rtk_util": feature_provenance["tdl_rtk_util"],
        "pyrtklib_version": feature_provenance["pyrtklib_version"],
        "pyrtklib_hypothesis_commit": feature_provenance[
            "pyrtklib_hypothesis_commit"
        ],
        "pyrtklib_publication_version_uncertain": feature_provenance[
            "pyrtklib_publication_version_uncertain"
        ],
    }


def load_ground_truth_window(
    path: Path, start_time: float, end_time: float
) -> np.ndarray:
    """Reproduce pandas skiprows/+18 s/DMS conversion for the needed window."""

    records: list[list[float]] = []
    pending: deque[str] = deque()

    def process(line: str) -> None:
        fields = line.split()
        if len(fields) < 10:
            return
        try:
            values = [float(item) for item in fields[:10]]
        except ValueError:
            return
        aligned_time = values[0] + 18.0
        if start_time - 2.0 <= aligned_time <= end_time + 2.0:
            latitude = values[3] + values[4] / 60.0 + values[5] / 3600.0
            longitude = values[6] + values[7] / 60.0 + values[8] / 3600.0
            records.append([aligned_time, latitude, longitude, values[9]])

    with path.open(encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream):
            if line_number < 30:
                continue
            pending.append(line)
            if len(pending) > 4:
                process(pending.popleft())
    # The four still-pending lines reproduce pandas ``skipfooter=4``.
    result = np.asarray(records, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 4 or not result.shape[0]:
        raise RuntimeError("ground-truth window is empty or malformed")
    if np.any(np.diff(result[:, 0]) < 0.0):
        raise RuntimeError("ground-truth rows are not chronological")
    return result


def nearest_ground_truth(table: np.ndarray, epoch_time: float) -> np.ndarray:
    """Return [aligned timestamp, latitude, longitude, ellipsoidal height]."""

    # argmin selects the first of equal minima, matching pandas Series.argmin.
    index = int(np.abs(table[:, 0] - epoch_time).argmin())
    return table[index].copy()


def _satellite_id(prl: object, satellite: int) -> str:
    value = prl.Arr1Dchar(4)
    prl.satno2id(satellite, value)
    return str(value[0])


def _native_array(values: object, length: int) -> np.ndarray:
    return np.asarray([values[index] for index in range(length)], dtype=np.float64)


def _raw_observation_by_satellite(epoch: object) -> dict[int, tuple[float, float]]:
    result: dict[int, tuple[float, float]] = {}
    for index in range(epoch.n):
        observation = epoch.data[index]
        result[int(observation.sat)] = (
            float(observation.P[0]),
            float(observation.SNR[0]),
        )
    return result


def prepare_feature_dataset(
    spec: DatasetSpec, inputs: FeatureInputPaths
) -> PreparedFeatureDataset:
    """Stop after released preprocessing, equal-weight OLS, and raw features."""

    provenance = _verify_feature_input_files(spec, inputs)
    prl, util = load_historical_modules(inputs)
    ephemeris_arg: str | list[str]
    if len(inputs.ephemeris_patterns) == 1:
        ephemeris_arg = inputs.ephemeris_patterns[0]
    else:
        ephemeris_arg = list(inputs.ephemeris_patterns)
    obs, nav, _station = util.read_obs(str(inputs.observation), ephemeris_arg)
    prl.sortobs(obs)
    split_epochs = util.split_obs(obs)

    prepared: list[PreparedFeatureEpoch] = []
    invalid: list[dict[str, object]] = []
    candidate_index = 0
    for split_index, epoch in enumerate(split_epochs):
        epoch_time = float(epoch.data[0].time.time + epoch.data[0].time.sec)
        if not (epoch_time > spec.start_time and epoch_time < spec.end_time):
            continue
        current_candidate_index = candidate_index
        candidate_index += 1
        try:
            result = util.get_ls_pnt_pos(epoch, nav)
        except Exception as error:
            invalid.append(
                {
                    "candidate_epoch_index": current_candidate_index,
                    "split_epoch_index": split_index,
                    "timestamp": epoch_time,
                    "reason": f"exception: {error!r}",
                }
            )
            continue
        if not result["status"]:
            invalid.append(
                {
                    "candidate_epoch_index": current_candidate_index,
                    "split_epoch_index": split_index,
                    "timestamp": epoch_time,
                    "reason": str(result.get("msg", "unknown OLS failure")),
                }
            )
            continue

        data = result["data"]
        excluded = list(data["exclude"])
        satellites = [int(item) for item in data["sats"]]
        historical_rows = list(set(range(len(satellites))) - set(excluded))
        ordered_rows = [index for index in range(len(satellites)) if index not in excluded]
        if historical_rows != ordered_rows:
            raise RuntimeError(
                "legacy list(set(...)) row order differs from observation order at "
                f"split epoch {split_index}"
            )

        residual = np.asarray(data["residual"], dtype=np.float64).reshape(-1)
        snr = np.asarray(data["SNR"], dtype=np.float64).reshape(-1)
        azel = np.asarray(data["azel"], dtype=np.float64).reshape(-1, 2)
        elevation = np.delete(azel, excluded, axis=0)[:, 1]
        features = construct_features(snr, elevation, residual)
        if features.shape[0] != len(ordered_rows):
            raise RuntimeError("feature count does not match retained satellite rows")

        positions = _native_array(data["eph"], len(satellites) * 6).reshape(-1, 6)
        clocks = _native_array(data["dts"], len(satellites) * 2).reshape(-1, 2)
        ids = np.asarray(
            [_satellite_id(prl, satellites[index]) for index in ordered_rows],
            dtype="U3",
        )
        try:
            clock_indices = np.asarray(
                [SYSTEM_TO_CLOCK_INDEX[item[0]] for item in ids], dtype=np.int64
            )
        except KeyError as error:
            raise RuntimeError(f"unsupported retained constellation: {error}") from error
        corrected = np.asarray(data["prs"], dtype=np.float64).reshape(-1)
        if corrected.shape != (features.shape[0],):
            raise RuntimeError("corrected-pseudorange count does not match features")

        raw_by_satellite = _raw_observation_by_satellite(epoch)
        retained_satellites = np.asarray(
            [satellites[index] for index in ordered_rows], dtype=np.int64
        )
        raw = np.asarray(
            [raw_by_satellite[int(item)] for item in retained_satellites],
            dtype=np.float64,
        )
        prepared.append(
            PreparedFeatureEpoch(
                valid_epoch_index=len(prepared),
                candidate_epoch_index=current_candidate_index,
                split_epoch_index=split_index,
                epoch_time=epoch_time,
                satellite_ids=ids,
                satellite_numbers=retained_satellites,
                raw_pseudorange_m=raw[:, 0],
                raw_snr_units=raw[:, 1],
                features=features,
                satellite_positions_ecef_m=positions[ordered_rows, :3],
                satellite_clock_bias_s=clocks[ordered_rows, 0],
                corrected_pseudorange_m=corrected,
                system_clock_indices=clock_indices,
                initial_ols_state=np.asarray(result["pos"], dtype=np.float64),
            )
        )

    if not prepared:
        raise RuntimeError(f"{spec.name} produced no valid released-code epochs")
    return PreparedFeatureDataset(
        spec=spec,
        inputs=inputs,
        raw_split_epoch_count=len(split_epochs),
        candidate_epoch_count=candidate_index,
        epochs=tuple(prepared),
        invalid_epochs=tuple(invalid),
        input_provenance=provenance,
    )


def prepare_dataset(spec: DatasetSpec, inputs: InputPaths) -> PreparedDataset:
    """Attach historical ground truth after the unchanged feature stage."""

    provenance = _verify_input_files(spec, inputs)
    ground_truth = load_ground_truth_window(
        inputs.ground_truth, spec.start_time, spec.end_time
    )
    feature_inputs = FeatureInputPaths(
        observation=inputs.observation,
        ephemeris_patterns=inputs.ephemeris_patterns,
        tdl_dir=inputs.tdl_dir,
        pyrtklib_site=inputs.pyrtklib_site,
    )
    feature_dataset = prepare_feature_dataset(spec, feature_inputs)
    prepared: list[PreparedEpoch] = []
    for epoch in feature_dataset.epochs:
        gt = nearest_ground_truth(ground_truth, epoch.epoch_time)
        prepared.append(
            PreparedEpoch(
                valid_epoch_index=epoch.valid_epoch_index,
                candidate_epoch_index=epoch.candidate_epoch_index,
                split_epoch_index=epoch.split_epoch_index,
                epoch_time=epoch.epoch_time,
                satellite_ids=epoch.satellite_ids,
                satellite_numbers=epoch.satellite_numbers,
                raw_pseudorange_m=epoch.raw_pseudorange_m,
                raw_snr_units=epoch.raw_snr_units,
                features=epoch.features,
                satellite_positions_ecef_m=epoch.satellite_positions_ecef_m,
                satellite_clock_bias_s=epoch.satellite_clock_bias_s,
                corrected_pseudorange_m=epoch.corrected_pseudorange_m,
                system_clock_indices=epoch.system_clock_indices,
                initial_ols_state=epoch.initial_ols_state,
                gt_time=float(gt[0]),
                gt_time_difference_s=float(gt[0] - epoch.epoch_time),
                ground_truth_geodetic_deg_m=gt[1:4],
            )
        )
    return PreparedDataset(
        spec=spec,
        inputs=inputs,
        raw_split_epoch_count=feature_dataset.raw_split_epoch_count,
        candidate_epoch_count=feature_dataset.candidate_epoch_count,
        epochs=tuple(prepared),
        invalid_epochs=feature_dataset.invalid_epochs,
        input_provenance=provenance,
    )


def load_frozen_weightnet(checkpoint: Path) -> WeightNet:
    """Perform the exact released default-model/double/load/eval sequence."""

    verify_checkpoint(checkpoint)
    model = WeightNet()
    model.double()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    actual_mean = model.seq[0].mean.detach().cpu().numpy()
    actual_std = model.seq[0].std.detach().cpu().numpy()
    np.testing.assert_array_equal(actual_mean, FROZEN_MODEL_MEAN)
    np.testing.assert_array_equal(actual_std, FROZEN_MODEL_STD)
    return model


def normalize_with_frozen_klt3(features: np.ndarray) -> np.ndarray:
    """Expose the affine transform stored by the frozen checkpoint."""

    values = np.asarray(features, dtype=np.float32).astype(np.float64)
    return (values - FROZEN_MODEL_MEAN) / FROZEN_MODEL_STD


def infer_weights(model: WeightNet, features: np.ndarray) -> torch.Tensor:
    """Infer aligned per-row weights without constructing an autograd graph."""

    feature_tensor = torch.as_tensor(features, dtype=torch.float32)
    with torch.no_grad():
        weights = model(feature_tensor).squeeze(-1)
    if weights.shape != (features.shape[0],):
        raise RuntimeError("WeightNet output rows do not match satellite rows")
    if weights.requires_grad:
        raise RuntimeError("test-time weights unexpectedly require gradients")
    if not bool(torch.all(torch.isfinite(weights))):
        raise RuntimeError("WeightNet emitted a non-finite weight")
    if not bool(torch.all((weights > 0.0) & (weights < 10.0))):
        raise RuntimeError("finite sigmoid WeightNet outputs must be strictly in (0, 10)")
    return weights


def historical_position_error(
    estimated_ecef_m: np.ndarray, ground_truth_geodetic_deg_m: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Apply the released ECEF->geodetic->ENU evaluation path."""

    estimated_geodetic = np.asarray(
        p3d.ecef2geodetic(*estimated_ecef_m), dtype=np.float64
    )
    enu = np.asarray(
        p3d.geodetic2enu(*estimated_geodetic, *ground_truth_geodetic_deg_m),
        dtype=np.float64,
    )
    error_2d = float(np.sqrt(enu[0] ** 2 + enu[1] ** 2))
    error_3d = float(np.sqrt(enu[0] ** 2 + enu[1] ** 2 + enu[2] ** 2))
    return estimated_geodetic, enu, error_2d, error_3d


def evaluate_epoch(model: WeightNet, epoch: PreparedEpoch) -> EpochEvaluation:
    normalized = normalize_with_frozen_klt3(epoch.features)
    weights = infer_weights(model, epoch.features)
    solution = solve_paper_weighted_position(
        epoch.satellite_positions_ecef_m,
        epoch.satellite_clock_bias_s,
        epoch.corrected_pseudorange_m,
        epoch.system_clock_indices,
        weights,
        epoch.initial_ols_state,
        return_trace=True,
    )
    estimated_ecef = solution.state[:3].detach().cpu().numpy().copy()
    if not np.all(np.isfinite(estimated_ecef)):
        raise RuntimeError(
            f"non-finite WLS state at {epoch.epoch_time:.9f}; historical prediction "
            "would not produce a usable position"
        )
    estimated_geodetic, enu, error_2d, error_3d = historical_position_error(
        estimated_ecef, epoch.ground_truth_geodetic_deg_m
    )
    _, ols_enu, ols_2d, ols_3d = historical_position_error(
        epoch.initial_ols_state[:3], epoch.ground_truth_geodetic_deg_m
    )
    ground_truth_ecef = np.asarray(
        p3d.geodetic2ecef(*epoch.ground_truth_geodetic_deg_m), dtype=np.float64
    )
    residual_norm = (
        float(
            torch.linalg.vector_norm(solution.iterations[-1].residual_v_m)
            .detach()
            .cpu()
        )
        if solution.iterations
        else float("nan")
    )
    if not solution.converged:
        status = "failed_maximum_iterations_included_by_historical_predictor"
    elif residual_norm > 1000.0:
        status = "failed_residual_norm_included_by_historical_predictor"
    else:
        status = "converged"
    return EpochEvaluation(
        prepared=epoch,
        normalized_features=normalized,
        weights=weights.detach().cpu().numpy().copy(),
        solution=solution,
        estimated_ecef_m=estimated_ecef,
        estimated_geodetic_deg_m=estimated_geodetic,
        ground_truth_ecef_m=ground_truth_ecef,
        enu_error_m=enu,
        error_2d_m=error_2d,
        error_3d_m=error_3d,
        ols_enu_error_m=ols_enu,
        ols_error_2d_m=ols_2d,
        ols_error_3d_m=ols_3d,
        historical_wls_status=status,
        historical_wls_residual_norm_m=residual_norm,
    )


def evaluate_prepared_dataset(
    model: WeightNet, prepared: PreparedDataset
) -> list[EpochEvaluation]:
    return [evaluate_epoch(model, epoch) for epoch in prepared.epochs]


CSV_FIELDS = (
    "dataset",
    "epoch_index",
    "candidate_epoch_index",
    "split_epoch_index",
    "gnss_timestamp",
    "gt_timestamp",
    "gt_time_difference_s",
    "number_of_satellites",
    "estimated_ecef_x_m",
    "estimated_ecef_y_m",
    "estimated_ecef_z_m",
    "ground_truth_ecef_x_m",
    "ground_truth_ecef_y_m",
    "ground_truth_ecef_z_m",
    "east_error_m",
    "north_error_m",
    "up_error_m",
    "error_2d_m",
    "error_3d_m",
    "ols_error_2d_m",
    "ols_error_3d_m",
    "wls_convergence_status",
    "wls_iterations",
    "wls_last_residual_norm_m",
)


def evaluation_csv_row(item: EpochEvaluation) -> dict[str, object]:
    epoch = item.prepared
    return {
        "dataset": "" if epoch is None else "pending",
        "epoch_index": epoch.valid_epoch_index,
        "candidate_epoch_index": epoch.candidate_epoch_index,
        "split_epoch_index": epoch.split_epoch_index,
        "gnss_timestamp": f"{epoch.epoch_time:.9f}",
        "gt_timestamp": f"{epoch.gt_time:.9f}",
        "gt_time_difference_s": f"{epoch.gt_time_difference_s:.12g}",
        "number_of_satellites": epoch.features.shape[0],
        "estimated_ecef_x_m": f"{item.estimated_ecef_m[0]:.15g}",
        "estimated_ecef_y_m": f"{item.estimated_ecef_m[1]:.15g}",
        "estimated_ecef_z_m": f"{item.estimated_ecef_m[2]:.15g}",
        "ground_truth_ecef_x_m": f"{item.ground_truth_ecef_m[0]:.15g}",
        "ground_truth_ecef_y_m": f"{item.ground_truth_ecef_m[1]:.15g}",
        "ground_truth_ecef_z_m": f"{item.ground_truth_ecef_m[2]:.15g}",
        "east_error_m": f"{item.enu_error_m[0]:.15g}",
        "north_error_m": f"{item.enu_error_m[1]:.15g}",
        "up_error_m": f"{item.enu_error_m[2]:.15g}",
        "error_2d_m": f"{item.error_2d_m:.15g}",
        "error_3d_m": f"{item.error_3d_m:.15g}",
        "ols_error_2d_m": f"{item.ols_error_2d_m:.15g}",
        "ols_error_3d_m": f"{item.ols_error_3d_m:.15g}",
        "wls_convergence_status": item.historical_wls_status,
        "wls_iterations": len(item.solution.iterations),
        "wls_last_residual_norm_m": f"{item.historical_wls_residual_norm_m:.15g}",
    }


def write_results_csv(
    path: Path, dataset_name: str, evaluations: Sequence[EpochEvaluation]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for item in evaluations:
            row = evaluation_csv_row(item)
            row["dataset"] = dataset_name
            writer.writerow(row)


def _difference(reproduction: float, paper: float) -> dict[str, float]:
    difference = reproduction - paper
    return {
        "reproduction_minus_paper_m": difference,
        "paper_minus_reproduction_m": -difference,
        "absolute_difference_m": abs(difference),
        "absolute_relative_difference_percent": abs(difference) / paper * 100.0,
    }


def summarize_results(
    prepared: PreparedDataset,
    evaluations: Sequence[EpochEvaluation],
    *,
    csv_path: Path,
    checkpoint: Path,
) -> dict[str, object]:
    errors_2d = np.asarray([item.error_2d_m for item in evaluations], dtype=np.float64)
    errors_3d = np.asarray([item.error_3d_m for item in evaluations], dtype=np.float64)
    ols_2d = np.asarray([item.ols_error_2d_m for item in evaluations], dtype=np.float64)
    ols_3d = np.asarray([item.ols_error_3d_m for item in evaluations], dtype=np.float64)
    weights = np.concatenate([item.weights for item in evaluations])
    failures = [
        {
            "epoch_index": item.prepared.valid_epoch_index,
            "timestamp": item.prepared.epoch_time,
            "status": item.historical_wls_status,
            "iterations": len(item.solution.iterations),
            "last_residual_norm_m": item.historical_wls_residual_norm_m,
        }
        for item in evaluations
        if item.historical_wls_status != "converged"
    ]
    mean_2d = float(errors_2d.mean())
    mean_3d = float(errors_3d.mean())
    return {
        "status": "passed",
        "dataset": prepared.spec.name,
        "reference_commit": TDL_COMMIT,
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "sha256": verify_checkpoint(checkpoint),
        },
        "configured_interval": {
            "strict_start_gpst_like": prepared.spec.start_time,
            "strict_end_gpst_like": prepared.spec.end_time,
        },
        "cardinality": {
            "raw_split_epochs_in_observation_file": prepared.raw_split_epoch_count,
            "raw_candidate_epochs_in_strict_interval": prepared.candidate_epoch_count,
            "released_code_valid_evaluation_epochs": len(prepared.epochs),
            "retained_satellite_measurements": prepared.measurement_count,
            "first_retained_timestamp": prepared.epochs[0].epoch_time,
            "last_retained_timestamp": prepared.epochs[-1].epoch_time,
            "published_epochs": prepared.spec.published_epochs,
            "published_satellite_measurements": prepared.spec.published_measurements,
            "invalid_ols_epochs": list(prepared.invalid_epochs),
            "undocumented_epoch_removal_applied": False,
        },
        "normalization": {
            "klt3_float64_source_mean": KLT3_FEATURE_MEAN.tolist(),
            "klt3_float64_source_population_std": KLT3_FEATURE_POPULATION_STD.tolist(),
            "checkpoint_stored_mean_after_float32_then_double": FROZEN_MODEL_MEAN.tolist(),
            "checkpoint_stored_std_after_float32_then_double": FROZEN_MODEL_STD.tolist(),
            "test_dataset_statistics_used": False,
        },
        "weight_range": {
            "minimum": float(weights.min()),
            "maximum": float(weights.max()),
            "all_finite": bool(np.all(np.isfinite(weights))),
        },
        "tdl_w": {
            "mean_2d_error_m": mean_2d,
            "mean_3d_error_m": mean_3d,
            "paper_mean_2d_error_m": prepared.spec.paper_2d_mean_m,
            "paper_mean_3d_error_m": prepared.spec.paper_3d_mean_m,
            "difference_2d": _difference(mean_2d, prepared.spec.paper_2d_mean_m),
            "difference_3d": _difference(mean_3d, prepared.spec.paper_3d_mean_m),
        },
        "equal_weight_ols_sanity_baseline": {
            "mean_2d_error_m": float(ols_2d.mean()),
            "mean_3d_error_m": float(ols_3d.mean()),
            "interpretation": (
                "Same retained epochs and GT matching; this is the released "
                "equal-weight OLS initializer, not the paper RTKLIB/goGPS baseline."
            ),
        },
        "wls_failures_included_by_released_predictor": failures,
        "aggregation": {
            "formula_2d": "mean_i(sqrt(E_i^2 + N_i^2))",
            "formula_3d": "mean_i(sqrt(E_i^2 + N_i^2 + U_i^2))",
            "arithmetic_mean_of_per_epoch_euclidean_errors": True,
        },
        "input_provenance": prepared.input_provenance,
        "per_epoch_csv": {
            "path": str(csv_path.resolve()),
            "sha256": sha256(csv_path),
            "rows": len(evaluations),
        },
    }


def write_summary(path: Path, summary: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def cardinality_record(prepared: PreparedDataset) -> dict[str, object]:
    return {
        "dataset": prepared.spec.name,
        "configured_interval": {
            "strict_start": prepared.spec.start_time,
            "strict_end": prepared.spec.end_time,
        },
        "raw_split_epoch_count": prepared.raw_split_epoch_count,
        "raw_candidate_epoch_count": prepared.candidate_epoch_count,
        "valid_evaluation_epoch_count": len(prepared.epochs),
        "retained_satellite_measurement_count": prepared.measurement_count,
        "first_retained_timestamp": prepared.epochs[0].epoch_time,
        "last_retained_timestamp": prepared.epochs[-1].epoch_time,
        "published_epoch_count": prepared.spec.published_epochs,
        "published_satellite_measurement_count": prepared.spec.published_measurements,
        "invalid_epochs": list(prepared.invalid_epochs),
    }


def common_input_arguments(parser: object) -> None:
    parser.add_argument("--dataset", required=True, choices=tuple(DATASET_SPECS))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--observation", type=Path)
    parser.add_argument(
        "--ephemeris-glob",
        action="append",
        dest="ephemeris_patterns",
        help="Repeat for multiple historical wildcard patterns.",
    )
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME_DIR)


def inputs_from_args(args: object) -> tuple[DatasetSpec, InputPaths]:
    spec = DATASET_SPECS[args.dataset]
    inputs = resolve_input_paths(
        spec,
        data_root=args.data_root,
        observation=args.observation,
        ephemeris_patterns=args.ephemeris_patterns,
        ground_truth=args.ground_truth,
        runtime_dir=args.runtime_dir,
    )
    return spec, inputs


__all__ = [
    "CHECKPOINT_SHA256",
    "CSV_FIELDS",
    "DATASET_SPECS",
    "FEATURE_NAMES",
    "FEATURE_UNITS",
    "FROZEN_MODEL_MEAN",
    "FROZEN_MODEL_STD",
    "KLT3_FEATURE_MEAN",
    "KLT3_FEATURE_POPULATION_STD",
    "SPEED_OF_LIGHT_M_S",
    "DatasetSpec",
    "EpochEvaluation",
    "FeatureInputPaths",
    "InputPaths",
    "PreparedDataset",
    "PreparedEpoch",
    "PreparedFeatureDataset",
    "PreparedFeatureEpoch",
    "cardinality_record",
    "common_input_arguments",
    "evaluate_epoch",
    "evaluate_prepared_dataset",
    "historical_position_error",
    "infer_weights",
    "inputs_from_args",
    "load_frozen_weightnet",
    "normalize_with_frozen_klt3",
    "prepare_dataset",
    "prepare_feature_dataset",
    "repository_root",
    "resolve_feature_input_paths",
    "resolve_input_paths",
    "sha256",
    "summarize_results",
    "verify_checkpoint",
    "write_results_csv",
    "write_summary",
]
