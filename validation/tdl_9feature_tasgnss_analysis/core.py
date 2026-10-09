"""Shared current-stack data, model, provenance, and training primitives.

TASGNSS's solver is always imported from the pinned external checkout.  It is
not copied or reimplemented here.  The upstream HybridShareSysNet class is
loaded directly from the pinned TDL-GNSS source file; a tiny torchvision stub
is used only because that file has an unused resnet18 import and torchvision is
not part of this project's locked runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import pickle
import platform
import random
import re
import subprocess
import sys
import time
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pymap3d as p3d
import torch
import torch.nn.functional as F


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PHD_ROOT = REPOSITORY_ROOT.parent
TDL_REPOSITORY = PHD_ROOT / "external_references/TDL-GNSS"
TASGNSS_REPOSITORY = PHD_ROOT / "external_references/TASGNSS"
PYRTKLIB_REPOSITORY = PHD_ROOT / "external_references/pyrtklib"
BACKEND_SITE = PHD_ROOT / "external_data/tasgnss_comparison/runtime/pyrtklib-0.2.7-site"
KLT_DATA_ROOT = PHD_ROOT / "external_data/TDL-GNSS/data/0610_KLT"
DEFAULT_OUTPUT_ROOT = PHD_ROOT / "external_data/current_tdl_reproduction"

REQUIRED_BRANCH = "research/current-tdl-tasgnss-reproduction"
TDL_COMMIT = "a640b2832c90daeb1ce25644a1da91c8edeb13fc"
TASGNSS_COMMIT = "fdd7e8ebc0019ad9b7c73f31363de066290d057a"
PYRTKLIB_COMMIT = "1c468dbe14074f1b7b3276ce265fe0fa5d6bef8b"
PYRTKLIB_VERSION = "0.2.7"
TASGNSS_VERSION = "0.1.5"

SEEDS = tuple(range(10))
SYS_MAP = {"G": 0, "R": 1, "E": 2, "C": 3, "J": 4}
FEATURE_NAMES = (
    "SNR",
    "elevation",
    "azimuth",
    "neutral_residual",
    "G",
    "R",
    "E",
    "C",
    "J",
)
FEATURE_UNITS = (
    "dB-Hz-like RTKLIB SNR[0]/1000, cast through int8 for model input",
    "radian",
    "radian",
    "metre",
    "dimensionless one-hot",
    "dimensionless one-hot",
    "dimensionless one-hot",
    "dimensionless one-hot",
    "dimensionless one-hot",
)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    config_name: str
    start_utc: float
    end_utc: float
    expected_epochs: int


# Values are from the official KLTDataset current config files.  Current
# TASGNSS filter_obs converts RTKLIB GPST-like time to UTC by subtracting 18 s.
DATASET_SPECS = {
    "KLT1": DatasetSpec("KLT1", "0610_klt1_203.json", 1623296137.0, 1623296340.0, 203),
    "KLT2": DatasetSpec("KLT2", "0610_klt2_209.json", 1623296900.0, 1623297109.0, 209),
    "KLT3": DatasetSpec("KLT3", "0610_klt3_404.json", 1623297134.0, 1623297538.0, 404),
}


def _run_text(arguments: Sequence[str]) -> str:
    result = subprocess.run(arguments, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def git_record(path: Path) -> dict[str, object]:
    remotes: dict[str, dict[str, str]] = {}
    for line in _run_text(("git", "-C", str(path), "remote", "-v")).splitlines():
        fields = line.split()
        if len(fields) >= 3:
            remotes.setdefault(fields[0], {})[fields[2].strip("()")]=fields[1]
    status = _run_text(("git", "-C", str(path), "status", "--short"))
    return {
        "path": str(path.resolve()),
        "head": _run_text(("git", "-C", str(path), "rev-parse", "HEAD")),
        "status_short": status.splitlines() if status else [],
        "remotes": remotes,
    }


def verify_provenance(*, require_project_branch: bool = True) -> dict[str, object]:
    project_branch = _run_text(("git", "-C", str(REPOSITORY_ROOT), "branch", "--show-current"))
    if require_project_branch and project_branch != REQUIRED_BRANCH:
        raise RuntimeError(f"project branch is {project_branch!r}, expected {REQUIRED_BRANCH!r}")
    upstream = {
        "TDL-GNSS": git_record(TDL_REPOSITORY),
        "TASGNSS": git_record(TASGNSS_REPOSITORY),
        "pyrtklib": git_record(PYRTKLIB_REPOSITORY),
    }
    expected = {
        "TDL-GNSS": TDL_COMMIT,
        "TASGNSS": TASGNSS_COMMIT,
        "pyrtklib": PYRTKLIB_COMMIT,
    }
    for name, revision in expected.items():
        record = upstream[name]
        if record["head"] != revision:
            raise RuntimeError(f"{name} is at {record['head']}, expected {revision}")
        if record["status_short"]:
            raise RuntimeError(f"{name} is dirty: {record['status_short']}")
    return {
        "project": {**git_record(REPOSITORY_ROOT), "branch": project_branch},
        "upstream": upstream,
    }


def _setup_version(path: Path) -> str:
    match = re.search(r"\bversion\s*=\s*[\"']([^\"']+)", path.read_text(encoding="utf-8"))
    if match is None:
        raise RuntimeError(f"version is absent from {path}")
    return match.group(1)


def import_current_stack() -> tuple[ModuleType, ModuleType]:
    """Import pinned current TASGNSS and its pinned built pyrtklib backend."""

    os.environ["rtklib"] = "origin"
    for path in (BACKEND_SITE, TASGNSS_REPOSITORY):
        rendered = str(path.resolve())
        if rendered not in sys.path:
            sys.path.insert(0, rendered)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        tas = importlib.import_module("tasgnss")
        core = importlib.import_module("tasgnss.core")
    finally:
        sys.dont_write_bytecode = previous
    tas_file = Path(tas.__file__).resolve()
    backend_file = Path(core.prl.__file__).resolve()
    if not tas_file.is_relative_to(TASGNSS_REPOSITORY.resolve()):
        raise RuntimeError(f"tasgnss imported from unexpected path {tas_file}")
    if not backend_file.is_relative_to(BACKEND_SITE.resolve()):
        raise RuntimeError(f"pyrtklib imported from unexpected path {backend_file}")
    if core.rtklib_version != "origin" or core.prl.__name__ != "pyrtklib":
        raise RuntimeError("current stack requires TASGNSS's origin pyrtklib backend")
    if importlib.metadata.version("pyrtklib") != PYRTKLIB_VERSION:
        raise RuntimeError("the selected pyrtklib binary is not version 0.2.7")
    if _setup_version(TASGNSS_REPOSITORY / "setup.py") != TASGNSS_VERSION:
        raise RuntimeError("the selected TASGNSS source is not version 0.1.5")
    return tas, core


def load_hybrid_share_sys_net() -> type[torch.nn.Module]:
    """Load the exact upstream class without installing unused torchvision."""

    source = TDL_REPOSITORY / "model/model.py"
    module_name = "_pinned_current_tdl_model"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing.HybridShareSysNet
    inserted: list[str] = []
    try:
        importlib.import_module("torchvision.models.resnet")
    except ModuleNotFoundError:
        # resnet18 is imported by upstream but never referenced by
        # HybridShareSysNet.  The stub makes that unused import explicit.
        import types

        torchvision = types.ModuleType("torchvision")
        models = types.ModuleType("torchvision.models")
        resnet = types.ModuleType("torchvision.models.resnet")

        def unavailable_resnet18(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("resnet18 is outside the current reproduction")

        resnet.resnet18 = unavailable_resnet18
        torchvision.models = models
        models.resnet = resnet
        for name, module in (
            ("torchvision", torchvision),
            ("torchvision.models", models),
            ("torchvision.models.resnet", resnet),
        ):
            sys.modules[name] = module
            inserted.append(name)
    specification = importlib.util.spec_from_file_location(module_name, source)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load upstream model source {source}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        specification.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
        for name in reversed(inserted):
            sys.modules.pop(name, None)
    cls = module.HybridShareSysNet
    if cls.__name__ != "HybridShareSysNet":
        raise RuntimeError("failed to load current HybridShareSysNet")
    return cls


def configure_determinism(seed: int, *, threads: int = 1) -> dict[str, object]:
    if threads <= 0:
        raise ValueError("threads must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(threads)
    except RuntimeError:
        if torch.get_num_interop_threads() != threads:
            raise
    return {
        "seed": seed,
        "python_random_seed": seed,
        "numpy_seed": seed,
        "torch_seed": seed,
        "cuda_seed_all": seed,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
    }


def environment_manifest(seed_record: Mapping[str, object] | None = None) -> dict[str, object]:
    tas_version = _setup_version(TASGNSS_REPOSITORY / "setup.py")
    pyrtklib_version = _setup_version(PYRTKLIB_REPOSITORY / "setup.py")
    result: dict[str, object] = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "pymap3d": importlib.metadata.version("pymap3d"),
        "tasgnss_source_version": tas_version,
        "pyrtklib_source_version": pyrtklib_version,
        "pyrtklib_distribution_version": importlib.metadata.version("pyrtklib"),
        "cpu_count": os.cpu_count(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_count": torch.cuda.device_count(),
        "torch_parallel_info": torch.__config__.parallel_info(),
        "torch_build_config": torch.__config__.show(),
        "numpy_build_config": np.show_config(mode="dicts"),
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
    }
    if torch.cuda.is_available():
        result["gpus"] = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    if seed_record is not None:
        result["determinism"] = dict(seed_record)
    return result


def _time_of_epoch(epoch: Any) -> float:
    value = epoch.data[0].time
    return float(value.time + value.sec)


def load_ground_truth_window(path: Path, minimum_gpst: float, maximum_gpst: float) -> np.ndarray:
    """Return [GPST-like time, lat, lon, h, roll, pitch, heading]."""

    rows: list[list[float]] = []
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream):
            if line_number < 30:
                continue
            fields = line.split()
            if len(fields) < 19:
                continue
            try:
                values = [float(item) for item in fields[:19]]
            except ValueError:
                continue
            aligned = values[0] + 18.0
            if minimum_gpst - 1.0 <= aligned <= maximum_gpst + 1.0:
                latitude = values[3] + values[4] / 60.0 + values[5] / 3600.0
                longitude = values[6] + values[7] / 60.0 + values[8] / 3600.0
                rows.append(
                    [aligned, latitude, longitude, values[9], values[16], values[17], values[18]]
                )
    result = np.asarray(rows, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 7 or not result.shape[0]:
        raise RuntimeError("ground-truth window is empty or malformed")
    if np.any(np.diff(result[:, 0]) < 0):
        raise RuntimeError("ground-truth rows are not chronological")
    return result


def nearest_ground_truth(table: np.ndarray, epoch_time: float) -> np.ndarray:
    return table[int(np.abs(table[:, 0] - epoch_time).argmin())].copy()


def leverarm_correction(gt: np.ndarray) -> np.ndarray:
    """Exact current TDL-GNSS preprocess.py lever-arm transform."""

    roll, pitch, heading = gt[4:7]
    phi, theta, psi = np.radians((roll, pitch, -heading))
    c_phi, s_phi = np.cos(phi), np.sin(phi)
    c_theta, s_theta = np.cos(theta), np.sin(theta)
    c_psi, s_psi = np.cos(psi), np.sin(psi)
    body_to_enu = np.array(
        [
            [c_psi * c_phi - s_psi * s_theta * s_phi, -s_psi * c_theta, c_psi * s_phi + s_psi * s_theta * c_phi, 0],
            [s_psi * c_phi + c_psi * s_theta * s_phi, c_psi * c_theta, s_psi * s_phi - c_psi * s_theta * c_phi, 0],
            [-c_theta * s_phi, s_theta, c_theta * c_phi, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )
    local = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0.86], [0, 0, 1, -0.31], [0, 0, 0, 1]],
        dtype=np.float64,
    )
    arm = body_to_enu @ local
    return np.hstack((gt, arm[:3, 3]))


def dataset_input_records() -> dict[str, object]:
    observation = KLT_DATA_ROOT / "COM38_210610_025603.obs"
    ground_truth = KLT_DATA_ROOT / "20210610_100.txt"
    navigation = sorted((KLT_DATA_ROOT / "sta").glob("hksc161d.21*"))
    for path in (observation, ground_truth, *navigation):
        if not path.is_file():
            raise FileNotFoundError(path)
    return {
        "observation": {"path": str(observation), "sha256": sha256_file(observation), "bytes": observation.stat().st_size},
        "ground_truth": {"path": str(ground_truth), "sha256": sha256_file(ground_truth), "bytes": ground_truth.stat().st_size},
        "navigation": [
            {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in navigation
        ],
    }


def preprocess_dataset(spec: DatasetSpec) -> tuple[list[dict[str, Any]], dict[str, object]]:
    """Run current TASGNSS neutral preprocessing without passing ground truth."""

    tas, _core = import_current_stack()
    observation = KLT_DATA_ROOT / "COM38_210610_025603.obs"
    navigation_glob = KLT_DATA_ROOT / "sta/hksc161d.21*"
    ground_truth_path = KLT_DATA_ROOT / "20210610_100.txt"
    observations, navigation, _station = tas.read_obs(str(observation), str(navigation_glob))
    epochs = tas.filter_obs(tas.split_obs(observations), spec.start_utc, spec.end_utc)
    if len(epochs) != spec.expected_epochs:
        raise RuntimeError(
            f"{spec.name} current config produced {len(epochs)} epochs, expected {spec.expected_epochs}"
        )
    epoch_times = np.asarray([_time_of_epoch(epoch) for epoch in epochs], dtype=np.float64)
    ground_truth = load_ground_truth_window(ground_truth_path, epoch_times[0], epoch_times[-1])

    records: list[dict[str, Any]] = []
    failures: list[dict[str, object]] = []
    matches: list[float] = []
    start = time.perf_counter()
    for index, epoch in enumerate(epochs):
        # The neutral solve receives only observations/navigation.  Ground truth
        # is attached strictly after the position and residuals exist.
        result = tas.wls_pnt_pos(epoch, navigation, return_residual=True, w=1)
        if not result.get("status", False):
            failures.append({"index": index, "time": float(epoch_times[index]), "message": result.get("msg")})
            continue
        matched = nearest_ground_truth(ground_truth, epoch_times[index])
        matches.append(float(matched[0] - epoch_times[index]))
        records.append(
            {
                "dataset": spec.name,
                "candidate_index": index,
                "epoch_time_gpst_like": float(epoch_times[index]),
                "gnss": result,
                "gt": leverarm_correction(matched),
            }
        )
        tas.cache_data.pop(id(epoch), None)
    duration = time.perf_counter() - start
    if failures:
        raise RuntimeError(f"{spec.name} neutral preprocessing failures: {failures[:3]}")
    if len(records) != spec.expected_epochs:
        raise RuntimeError(f"{spec.name} retained {len(records)} records, expected {spec.expected_epochs}")
    manifest = {
        "schema_version": 1,
        "dataset": spec.name,
        "official_current_config": {
            "name": spec.config_name,
            "start_utc": spec.start_utc,
            "end_utc": spec.end_utc,
            "expected_epochs": spec.expected_epochs,
            "note": "official KLTDataset config is referenced by TDL-GNSS but not vendored there",
        },
        "candidate_epochs": len(epochs),
        "neutral_solved_epochs": len(records),
        "neutral_failed_epochs": len(failures),
        "first_epoch_gpst_like": float(epoch_times[0]),
        "last_epoch_gpst_like": float(epoch_times[-1]),
        "gt_minus_gnss_seconds": {
            "maximum_absolute": float(np.max(np.abs(matches))),
            "median_absolute": float(np.median(np.abs(matches))),
        },
        "duration_seconds": duration,
        "inputs": dataset_input_records(),
        "provenance": verify_provenance(),
    }
    return records, manifest


def cache_paths(output_root: Path, dataset: str) -> tuple[Path, Path]:
    directory = output_root.resolve() / "preprocessed"
    return directory / f"{dataset.lower()}_current.pkl", directory / f"{dataset.lower()}_manifest.json"


def load_or_preprocess_dataset(
    dataset: str, output_root: Path = DEFAULT_OUTPUT_ROOT, *, force: bool = False
) -> tuple[list[dict[str, Any]], dict[str, object]]:
    spec = DATASET_SPECS[dataset.upper()]
    cache_path, manifest_path = cache_paths(output_root, spec.name)
    if cache_path.is_file() and manifest_path.is_file() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("cache_sha256") != sha256_file(cache_path):
            raise RuntimeError(f"preprocessing cache hash mismatch: {cache_path}")
        expected = manifest["official_current_config"]
        if expected["start_utc"] != spec.start_utc or expected["end_utc"] != spec.end_utc:
            raise RuntimeError(f"cached {spec.name} config bounds do not match current harness")
        with cache_path.open("rb") as stream:
            records = pickle.load(stream)
        if len(records) != spec.expected_epochs:
            raise RuntimeError(f"cached {spec.name} cardinality is invalid")
        return records, manifest
    records, manifest = preprocess_dataset(spec)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(records, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(cache_path)
    manifest["cache_path"] = str(cache_path)
    manifest["cache_bytes"] = cache_path.stat().st_size
    manifest["cache_sha256"] = sha256_file(cache_path)
    write_json(manifest_path, manifest)
    return records, manifest


def one_hot(systems: Iterable[str]) -> torch.Tensor:
    indices = np.asarray([SYS_MAP[value] for value in systems])
    return F.one_hot(torch.tensor(indices), num_classes=len(SYS_MAP))


def record_feature_parts(record: Mapping[str, Any]) -> dict[str, Any] | None:
    gnss = record["gnss"]
    data = gnss["data"]
    systems = data[:, 2]
    snr = data[:, 9]
    residual = np.asarray(gnss["residual_info"]["residual"]).squeeze()
    elevation = data[:, 12].astype(np.float64)
    azimuth = data[:, 11].astype(np.float64)
    if np.linalg.norm(residual) > 500:
        return None
    if snr.shape[0] != elevation.shape[0] or snr.shape[0] != np.atleast_1d(residual).shape[0]:
        return None
    return {
        "systems": systems,
        "snr_source": np.asarray(snr, dtype=np.float64),
        "snr_model": torch.tensor(snr.astype(np.int8), dtype=torch.float64),
        "elevation": torch.tensor(elevation, dtype=torch.float64),
        "azimuth": torch.tensor(azimuth, dtype=torch.float64),
        "residual": torch.tensor(residual, dtype=torch.float64),
        "system_one_hot": one_hot(systems).double(),
    }


def feature_tensor(parts: Mapping[str, Any], *, device: str = "cpu") -> torch.Tensor:
    value = torch.hstack(
        (
            parts["snr_model"].reshape(-1, 1),
            parts["elevation"].reshape(-1, 1),
            parts["azimuth"].reshape(-1, 1),
            parts["residual"].reshape(-1, 1),
            parts["system_one_hot"],
        )
    )
    if value.ndim != 2 or value.shape[1] != len(FEATURE_NAMES):
        raise RuntimeError(f"feature tensor has invalid shape {tuple(value.shape)}")
    return value.to(device)


def scaler_from_records(records: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    snr: list[np.ndarray] = []
    residual: list[np.ndarray] = []
    elevation: list[np.ndarray] = []
    azimuth: list[np.ndarray] = []
    for record in records:
        parts = record_feature_parts(record)
        if parts is None:
            continue
        snr.append(parts["snr_source"])
        residual.append(parts["residual"].numpy())
        elevation.append(parts["elevation"].numpy())
        azimuth.append(parts["azimuth"].numpy())
    if not snr:
        raise RuntimeError("no records survived current TDL feature filters")
    mean = np.array(
        [
            np.hstack(snr).mean(),
            np.hstack(elevation).mean(),
            np.hstack(azimuth).mean(),
            np.hstack(residual).mean(),
            0,
            0,
            0,
            0,
            0,
        ],
        dtype=np.float64,
    )
    std = np.array(
        [
            np.hstack(snr).std(),
            np.hstack(elevation).std(),
            np.hstack(azimuth).std(),
            np.hstack(residual).std(),
            1,
            1,
            1,
            1,
            1,
        ],
        dtype=np.float64,
    )
    if mean.shape != (9,) or std.shape != (9,) or np.any(std == 0):
        raise RuntimeError("invalid current TDL scaler")
    return mean, std


def initialize_solver_cache(tas: ModuleType, records: Sequence[Mapping[str, Any]]) -> None:
    tas.cache_data.clear()
    for record in records:
        gnss = record["gnss"]
        tas.cache_data[id(record)] = [
            np.asarray(gnss["pos"]).copy(),
            np.asarray(gnss["cb"]).copy(),
            None,
            None,
            gnss["data"],
            gnss["solve_data"],
            gnss["raw_data"],
        ]


def make_model(mean: np.ndarray, std: np.ndarray, *, device: str = "cpu") -> torch.nn.Module:
    cls = load_hybrid_share_sys_net()
    # This is intentionally the first Torch RNG-consuming operation after the
    # caller invokes configure_determinism.
    model = cls(torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32))
    return model.double().to(device)


def model_tensor_shapes(model: torch.nn.Module) -> dict[str, list[int]]:
    return {name: list(value.shape) for name, value in model.state_dict().items()}


def position_enu(record: Mapping[str, Any], position: torch.Tensor, *, device: str = "cpu") -> torch.Tensor:
    gt = np.asarray(record["gt"], dtype=np.float64)
    return torch.hstack(p3d.ecef2enu(*position[:3], gt[1], gt[2], gt[3])) + torch.tensor(
        gt[7:10], dtype=torch.float64, device=device
    )


def position_loss(record: Mapping[str, Any], position: torch.Tensor, *, dimensions: int = 3, device: str = "cpu") -> torch.Tensor:
    return torch.norm(position_enu(record, position, device=device)[:dimensions])


def solver_input_fingerprint(record: Mapping[str, Any]) -> dict[str, str]:
    parts = record_feature_parts(record)
    if parts is None:
        raise RuntimeError("record is outside the current feature boundary")
    solve = record["gnss"]["solve_data"]
    combined = np.hstack(
        [
            np.asarray(solve["satpos"], dtype=np.float64).reshape(-1),
            np.asarray(solve["pr"], dtype=np.float64).reshape(-1),
            np.asarray(solve["sdt"], dtype=np.float64).reshape(-1),
            np.asarray(solve["sagnac"], dtype=np.float64).reshape(-1),
            np.asarray(solve["I"], dtype=np.float64).reshape(-1),
            np.asarray(solve["T"], dtype=np.float64).reshape(-1),
        ]
    )
    return {
        "features": sha256_array(feature_tensor(parts).numpy()),
        "neutral_position": sha256_array(np.asarray(record["gnss"]["pos"], dtype=np.float64)),
        "learned_solver_inputs": sha256_array(combined),
    }


def state_dict_equal(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> bool:
    return left.keys() == right.keys() and all(torch.equal(left[name], right[name]) for name in left)


def state_dict_changed(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> bool:
    return any(not torch.equal(left[name], right[name]) for name in left)


def clone_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def training_source_contract() -> dict[str, object]:
    return {
        "epochs": 120,
        "optimizer": "Adam",
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "scheduler": None,
        "loss": "3D Euclidean ENU norm",
        "skip_position_loss_above_m": 200.0,
        "accumulation_parameter": 3000,
        "shuffle": "numpy.random.shuffle each epoch",
        "device": "cpu",
        "checkpoint_interval_epochs": 10,
        "final_checkpoint_name": "multinet_3d.pth",
        "constructed_but_unused_loss": "MSELoss(reduction='sum')",
        "normalization": "NumPy population mean/std; one-hot mean 0 and std 1; constructor tensors float32 then model.double()",
        "snr_model_cast": "source SNR.astype(np.int8) then torch.float64",
        "custom_weight_semantics": "W = diag(w), objective ||W(H dx-r)||^2",
        "bias_semantics": "residual = corrected_pseudorange - predicted_pseudorange - b",
        "solver": "tasgnss.wls_pnt_pos(..., use_cache=True, w=weight, b=bias, enable_torch=True, device='cpu')",
    }


def percentile_metrics(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        raise ValueError("cannot summarize empty values")
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "rms": float(np.sqrt(np.mean(values**2))),
        "p68": float(np.percentile(values, 68)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def distribution_metrics(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "population_sd": float(array.std(ddof=0)),
        "iqr": float(np.percentile(array, 75) - np.percentile(array, 25)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def paired_delta_metrics(candidate: np.ndarray, neutral: np.ndarray) -> dict[str, float]:
    delta = np.asarray(candidate, dtype=np.float64) - np.asarray(neutral, dtype=np.float64)
    return {
        "mean_delta": float(delta.mean()),
        "median_delta": float(np.median(delta)),
        "p05": float(np.percentile(delta, 5)),
        "p95": float(np.percentile(delta, 95)),
        "fraction_improved": float(np.mean(delta < 0)),
        "fraction_worsened": float(np.mean(delta > 0)),
    }


__all__ = [name for name in globals() if not name.startswith("_")]
