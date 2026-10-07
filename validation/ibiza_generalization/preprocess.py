#!/usr/bin/env python3
"""Create the frozen, raw-feature Ibiza external-generalization dataset.

The runtime dependency on ``rtk_util`` is intentional. The default persistent
cache is prepared from pinned source commits by :mod:`prepare_runtime`; no
upstream reference repository is modified and no dependency is installed
globally.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import subprocess
from typing import Any, Iterable, Mapping, Sequence
import zipfile

import numpy as np

from .runtime_cache import (
    DEFAULT_RUNTIME_DIR,
    PYRTKLIB_COMMIT,
    PYRTKLIB_VERSION,
    TDL_COMMIT,
    import_pyrtklib,
    load_rtk_util,
    resolve_runtime,
)

SPEED_OF_LIGHT_M_S = 299_792_458.0
SYSTEM_TO_CLOCK_INDEX = {"G": 3, "C": 4, "E": 5, "R": 6}
CONSTELLATION_TO_CODE = {"G": 1, "R": 2, "E": 3, "C": 4}
CODE_TO_CONSTELLATION = {value: key for key, value in CONSTELLATION_TO_CODE.items()}

STATUS_ACCEPTED = 1
STATUS_EXCLUDED = 0
STATUS_EPOCH_REJECTED = -1

EXCLUSION_REASON_CODES = {
    "accepted": 0,
    "no_broadcast_ephemeris": 1,
    "zero_first_pseudorange": 2,
    "unsupported_constellation": 3,
    "below_horizon": 4,
    "insufficient_observations": 5,
    "ols_max_iterations": 6,
    "ols_residual_too_large": 7,
    "rank_deficient": 8,
    "ols_exception": 9,
    "unknown_ols_failure": 10,
}
REASON_CODE_TO_NAME = {value: key for key, value in EXCLUSION_REASON_CODES.items()}

FEATURE_NAMES = ("C/N0", "elevation", "equal-weight OLS residual")
FEATURE_UNITS = ("SNR[0]/1000", "radian", "metre")

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = REPOSITORY_ROOT.parent / "external_data/ibiza_2025_01_01"
DEFAULT_OBSERVATION = (
    DEFAULT_DATASET_DIR
    / "raw/observation/IBIZ00ESP_R_20250010000_01D_30S_MO.rnx"
)
DEFAULT_NAVIGATION = (
    DEFAULT_DATASET_DIR
    / "raw/navigation/BRDM00DLR_S_20250010000_01D_MN.rnx"
)
DEFAULT_OUTPUT = DEFAULT_DATASET_DIR / "derived/ibiza_preprocessed.npz"
DEFAULT_MANIFEST = Path(__file__).resolve().parent / "ibiza_preprocessed_manifest.json"


@dataclass(frozen=True)
class RinexHeader:
    version: str
    file_type: str
    satellite_system: str
    observation_types: dict[str, tuple[str, ...]]
    interval_s: float | None
    first_observation: str | None
    last_observation: str | None
    time_system: str | None
    signal_strength_unit: str | None


@dataclass(frozen=True)
class EpochProduct:
    split_epoch_index: int
    epoch_time: float
    raw_observation_count: int
    source_observation_offset: int
    source_rows: np.ndarray
    source_observation_indices: np.ndarray
    satellite_numbers: np.ndarray
    satellite_prns: np.ndarray
    constellation_codes: np.ndarray
    signal_codes: np.ndarray
    raw_pseudorange_m: np.ndarray
    raw_snr_units: np.ndarray
    corrected_pseudorange_m: np.ndarray
    satellite_positions_ecef_m: np.ndarray
    satellite_clock_bias_s: np.ndarray
    features: np.ndarray
    system_clock_indices: np.ndarray
    ols_state: np.ndarray
    active_state_mask: np.ndarray
    design_rank: int
    design_condition_number: float
    normal_condition_number: float
    audit_status: np.ndarray
    audit_reason: np.ndarray
    audit_satellite_numbers: np.ndarray
    audit_satellite_prns: np.ndarray
    audit_constellation_codes: np.ndarray
    audit_signal_codes: np.ndarray
    audit_raw_pseudorange_m: np.ndarray
    audit_raw_snr_units: np.ndarray


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=DEFAULT_RUNTIME_DIR,
        help="Persistent cache prepared by prepare_runtime (default: external_data/.paper_runtime).",
    )
    parser.add_argument("--observation", type=Path, default=DEFAULT_OBSERVATION)
    parser.add_argument("--navigation", type=Path, default=DEFAULT_NAVIGATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=200,
        help="Print deterministic progress every N raw epochs; zero disables it.",
    )
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_rinex_time(value: str) -> tuple[str, str] | None:
    fields = value.split()
    if len(fields) < 6:
        return None
    year, month, day, hour, minute = (int(item) for item in fields[:5])
    second = float(fields[5])
    whole_second = int(second)
    fraction = second - whole_second
    base = datetime(year, month, day, hour, minute, whole_second)
    precision = max(0, len(fields[5].partition(".")[2]))
    if precision:
        fraction_text = f"{fraction:.{precision}f}"[1:]
    else:
        fraction_text = ""
    rendered = base.strftime("%Y-%m-%dT%H:%M:%S") + fraction_text
    time_system = fields[6] if len(fields) >= 7 else ""
    return rendered, time_system


def read_rinex_header(path: Path) -> RinexHeader:
    version = ""
    file_type = ""
    satellite_system = ""
    observation_types: dict[str, list[str]] = defaultdict(list)
    observation_type_counts: dict[str, int] = {}
    interval: float | None = None
    first: str | None = None
    last: str | None = None
    time_system: str | None = None
    signal_strength_unit: str | None = None

    with path.open(encoding="ascii", errors="replace") as stream:
        for line in stream:
            label = line[60:80].strip()
            value = line[:60]
            if label == "RINEX VERSION / TYPE":
                version = value[:9].strip()
                file_type = value[20:40].strip()
                satellite_system = value[40:60].strip()
            elif label == "SYS / # / OBS TYPES":
                system = value[0].strip()
                if system:
                    observation_type_counts[system] = int(value[3:6])
                elif observation_types:
                    system = next(reversed(observation_types))
                if system:
                    observation_types[system].extend(value[7:60].split())
            elif label == "INTERVAL":
                interval = float(value.split()[0])
            elif label in {"TIME OF FIRST OBS", "TIME OF LAST OBS"}:
                parsed = _parse_rinex_time(value)
                if parsed is not None:
                    rendered, parsed_system = parsed
                    if label == "TIME OF FIRST OBS":
                        first = rendered
                    else:
                        last = rendered
                    time_system = parsed_system or time_system
            elif label == "SIGNAL STRENGTH UNIT":
                signal_strength_unit = value.strip()
            if label == "END OF HEADER":
                break
        else:
            raise RuntimeError(f"RINEX END OF HEADER not found in {path.name}")

    if not version:
        raise RuntimeError(f"RINEX version not found in {path.name}")
    finalized: dict[str, tuple[str, ...]] = {}
    for system, values in observation_types.items():
        expected = observation_type_counts[system]
        if len(values) != expected:
            raise RuntimeError(
                f"{path.name} {system} declares {expected} observation types, "
                f"parsed {len(values)}"
            )
        finalized[system] = tuple(values)
    return RinexHeader(
        version=version,
        file_type=file_type,
        satellite_system=satellite_system,
        observation_types=finalized,
        interval_s=interval,
        first_observation=first,
        last_observation=last,
        time_system=time_system,
        signal_strength_unit=signal_strength_unit,
    )


def navigation_record_counts(path: Path) -> dict[str, int]:
    counts: Counter[str] = Counter()
    in_header = True
    pattern = re.compile(r"^([A-Z])\d{2}\s+\d{4}\s")
    with path.open(encoding="ascii", errors="replace") as stream:
        for line in stream:
            if in_header:
                if line[60:80].strip() == "END OF HEADER":
                    in_header = False
                continue
            match = pattern.match(line)
            if match:
                counts[match.group(1)] += 1
    return dict(sorted(counts.items()))


def canonical_content_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(np.asarray(arrays[name]))
        if array.dtype.hasobject:
            raise TypeError(f"object arrays are forbidden: {name}")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    """Write a byte-reproducible compressed NPZ containing numeric arrays only."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with zipfile.ZipFile(
        temporary,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=True,
    ) as archive:
        for name in sorted(arrays):
            array = np.asarray(arrays[name])
            if array.dtype.hasobject or array.dtype.kind in "SU":
                raise TypeError(
                    f"NPZ arrays must be numerical; {name} has dtype {array.dtype}"
                )
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, array, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            info.create_system = 3
            archive.writestr(info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED)
    os.replace(temporary, path)
    return sha256_file(path)


def _satellite_id(prl: Any, satellite: int) -> str:
    value = prl.Arr1Dchar(4)
    prl.satno2id(satellite, value)
    return str(value[0])


def _native_array(values: Any, length: int) -> np.ndarray:
    return np.asarray([values[index] for index in range(length)], dtype=np.float64)


def _failure_reason(message: str) -> int:
    normalized = message.lower()
    if "no enough observations" in normalized:
        return EXCLUSION_REASON_CODES["insufficient_observations"]
    if "over max iteration" in normalized:
        return EXCLUSION_REASON_CODES["ols_max_iterations"]
    if "residual too large" in normalized:
        return EXCLUSION_REASON_CODES["ols_residual_too_large"]
    return EXCLUSION_REASON_CODES["unknown_ols_failure"]


def _raw_epoch_arrays(prl: Any, epoch: Any) -> dict[str, np.ndarray]:
    satellite_numbers = np.empty(epoch.n, dtype=np.int64)
    satellite_prns = np.empty(epoch.n, dtype=np.int64)
    constellation_codes = np.empty(epoch.n, dtype=np.uint8)
    signal_codes = np.empty(epoch.n, dtype=np.int64)
    pseudorange = np.empty(epoch.n, dtype=np.float64)
    snr = np.empty(epoch.n, dtype=np.float64)
    for row in range(epoch.n):
        observation = epoch.data[row]
        satellite = int(observation.sat)
        satellite_id = _satellite_id(prl, satellite)
        system = satellite_id[0]
        satellite_numbers[row] = satellite
        satellite_prns[row] = int(satellite_id[1:])
        constellation_codes[row] = CONSTELLATION_TO_CODE.get(system, 0)
        signal_codes[row] = int(observation.code[0])
        pseudorange[row] = float(observation.P[0])
        snr[row] = float(observation.SNR[0])
    return {
        "satellite_numbers": satellite_numbers,
        "satellite_prns": satellite_prns,
        "constellation_codes": constellation_codes,
        "signal_codes": signal_codes,
        "pseudorange": pseudorange,
        "snr": snr,
    }


def _rejected_audit(prl: Any, epoch: Any, reason: int) -> dict[str, np.ndarray]:
    raw = _raw_epoch_arrays(prl, epoch)
    raw["status"] = np.full(epoch.n, STATUS_EPOCH_REJECTED, dtype=np.int8)
    raw["reason"] = np.full(epoch.n, reason, dtype=np.int16)
    return raw


def preprocess_epoch(
    prl: Any,
    util: Any,
    epoch: Any,
    nav: Any,
    *,
    split_epoch_index: int,
    source_observation_offset: int,
) -> tuple[EpochProduct | None, dict[str, np.ndarray], int]:
    """Run the released equal-weight OLS path and preserve row provenance."""

    epoch_time = float(epoch.data[0].time.time + epoch.data[0].time.sec)
    raw = _raw_epoch_arrays(prl, epoch)
    try:
        result = util.get_ls_pnt_pos(epoch, nav)
    except Exception:
        reason = EXCLUSION_REASON_CODES["ols_exception"]
        return None, _rejected_audit(prl, epoch, reason), reason
    if not result["status"]:
        reason = _failure_reason(str(result.get("msg", "unknown OLS failure")))
        return None, _rejected_audit(prl, epoch, reason), reason

    data = result["data"]
    # Replay only the satellite-state call to recover its mapping back to the
    # original RINEX epoch rows.  The OLS values remain those returned by the
    # historical get_ls_pnt_pos() call above.
    _rs, no_ephemeris, _dts, _variance = util.get_sat_pos(epoch.data, epoch.n, nav)
    no_ephemeris_set = set(int(item) for item in no_ephemeris)
    compact_source_rows = [
        row for row in range(epoch.n) if row not in no_ephemeris_set
    ]
    satellites = [int(item) for item in data["sats"]]
    if len(satellites) != len(compact_source_rows):
        raise RuntimeError("satellite-state compaction changed between repeated calls")
    if any(
        satellites[index] != int(epoch.data[source].sat)
        for index, source in enumerate(compact_source_rows)
    ):
        raise RuntimeError("satellite-state rows no longer map to source observations")

    excluded = [int(item) for item in data["exclude"]]
    excluded_set = set(excluded)
    historical_rows = list(set(range(len(satellites))) - excluded_set)
    ordered_rows = [
        index for index in range(len(satellites)) if index not in excluded_set
    ]
    if historical_rows != ordered_rows:
        raise RuntimeError(
            "legacy list(set(...)) residual order differs from deterministic row order"
        )
    if not ordered_rows:
        reason = EXCLUSION_REASON_CODES["insufficient_observations"]
        return None, _rejected_audit(prl, epoch, reason), reason

    positions_full = _native_array(data["eph"], len(satellites) * 6).reshape(-1, 6)
    clocks_full = _native_array(data["dts"], len(satellites) * 2).reshape(-1, 2)
    corrected = np.asarray(data["prs"], dtype=np.float64).reshape(-1)
    residual = np.asarray(data["residual"], dtype=np.float64).reshape(-1)
    snr_scaled = np.asarray(data["SNR"], dtype=np.float64).reshape(-1)
    azel_full = np.asarray(data["azel"], dtype=np.float64).reshape(-1, 2)
    elevation = np.delete(azel_full, excluded, axis=0)[:, 1]
    features = np.column_stack((snr_scaled, elevation, residual))
    accepted_count = len(ordered_rows)
    if corrected.shape != (accepted_count,):
        raise RuntimeError("corrected-pseudorange rows do not match accepted rows")
    if features.shape != (accepted_count, 3):
        raise RuntimeError("feature rows do not match accepted rows")

    ols_state = np.asarray(result["pos"], dtype=np.float64).reshape(-1)
    if ols_state.shape != (7,) or not np.all(np.isfinite(ols_state)):
        raise RuntimeError("historical OLS returned an invalid seven-state solution")
    h, _predicted, _azel, final_excluded, active, *_rest = util.H_matrix_prl(
        data["eph"],
        ols_state,
        data["dts"],
        epoch.data[0].time,
        nav,
        satellites,
        excluded,
    )
    if [int(item) for item in final_excluded] != excluded:
        raise RuntimeError("final OLS exclusion rows changed during diagnostics")
    h = np.asarray(h, dtype=np.float64)
    active_indices = np.asarray(active, dtype=np.int64).reshape(-1)
    if h.shape != (accepted_count, active_indices.size):
        raise RuntimeError(
            f"unexpected final design shape {h.shape}; expected "
            f"({accepted_count}, {active_indices.size})"
        )
    rank = int(np.linalg.matrix_rank(h))
    if rank != active_indices.size:
        reason = EXCLUSION_REASON_CODES["rank_deficient"]
        return None, _rejected_audit(prl, epoch, reason), reason
    design_condition = float(np.linalg.cond(h))
    normal_condition = float(np.linalg.cond(h.T @ h))
    if not np.isfinite(design_condition) or not np.isfinite(normal_condition):
        reason = EXCLUSION_REASON_CODES["rank_deficient"]
        return None, _rejected_audit(prl, epoch, reason), reason

    source_rows = np.asarray(
        [compact_source_rows[index] for index in ordered_rows], dtype=np.int64
    )
    source_indices = source_observation_offset + source_rows
    accepted_satellites = np.asarray(
        [satellites[index] for index in ordered_rows], dtype=np.int64
    )
    accepted_ids = [_satellite_id(prl, item) for item in accepted_satellites]
    systems = [item[0] for item in accepted_ids]
    if any(system not in SYSTEM_TO_CLOCK_INDEX for system in systems):
        raise RuntimeError("accepted rows contain an unsupported constellation")
    satellite_prns = np.asarray([int(item[1:]) for item in accepted_ids], dtype=np.int64)
    constellation_codes = np.asarray(
        [CONSTELLATION_TO_CODE[item] for item in systems], dtype=np.uint8
    )
    system_clock_indices = np.asarray(
        [SYSTEM_TO_CLOCK_INDEX[item] for item in systems], dtype=np.int64
    )
    active_mask = np.zeros(7, dtype=np.uint8)
    active_mask[active_indices] = 1
    if not np.array_equal(np.flatnonzero(active_mask), active_indices):
        raise RuntimeError("active-state indices are not strictly increasing")

    audit_status = np.full(epoch.n, STATUS_EXCLUDED, dtype=np.int8)
    audit_reason = np.full(
        epoch.n, EXCLUSION_REASON_CODES["unknown_ols_failure"], dtype=np.int16
    )
    for source_row in no_ephemeris_set:
        audit_reason[source_row] = EXCLUSION_REASON_CODES["no_broadcast_ephemeris"]
    for compact_index, source_row in enumerate(compact_source_rows):
        system_code = int(raw["constellation_codes"][source_row])
        if compact_index not in excluded_set:
            audit_status[source_row] = STATUS_ACCEPTED
            audit_reason[source_row] = EXCLUSION_REASON_CODES["accepted"]
        elif raw["pseudorange"][source_row] == 0.0:
            audit_reason[source_row] = EXCLUSION_REASON_CODES[
                "zero_first_pseudorange"
            ]
        elif system_code not in {
            CONSTELLATION_TO_CODE[item] for item in SYSTEM_TO_CLOCK_INDEX
        }:
            audit_reason[source_row] = EXCLUSION_REASON_CODES[
                "unsupported_constellation"
            ]
        else:
            audit_reason[source_row] = EXCLUSION_REASON_CODES["below_horizon"]

    if not np.array_equal(np.flatnonzero(audit_status == STATUS_ACCEPTED), source_rows):
        raise RuntimeError("accepted source rows differ from exclusion audit")
    if not np.array_equal(features[:, 0], raw["snr"][source_rows] / 1000.0):
        raise RuntimeError("C/N0 feature does not equal historical SNR[0]/1000")
    if np.any(features[:, 1] < 0.0) or np.any(features[:, 1] > np.pi / 2.0 + 1e-12):
        raise RuntimeError("retained elevation is outside [0, pi/2] radians")

    product = EpochProduct(
        split_epoch_index=split_epoch_index,
        epoch_time=epoch_time,
        raw_observation_count=epoch.n,
        source_observation_offset=source_observation_offset,
        source_rows=source_rows,
        source_observation_indices=source_indices,
        satellite_numbers=accepted_satellites,
        satellite_prns=satellite_prns,
        constellation_codes=constellation_codes,
        signal_codes=raw["signal_codes"][source_rows],
        raw_pseudorange_m=raw["pseudorange"][source_rows],
        raw_snr_units=raw["snr"][source_rows],
        corrected_pseudorange_m=corrected,
        satellite_positions_ecef_m=positions_full[ordered_rows, :3],
        satellite_clock_bias_s=clocks_full[ordered_rows, 0],
        features=features,
        system_clock_indices=system_clock_indices,
        ols_state=ols_state,
        active_state_mask=active_mask,
        design_rank=rank,
        design_condition_number=design_condition,
        normal_condition_number=normal_condition,
        audit_status=audit_status,
        audit_reason=audit_reason,
        audit_satellite_numbers=raw["satellite_numbers"],
        audit_satellite_prns=raw["satellite_prns"],
        audit_constellation_codes=raw["constellation_codes"],
        audit_signal_codes=raw["signal_codes"],
        audit_raw_pseudorange_m=raw["pseudorange"],
        audit_raw_snr_units=raw["snr"],
    )
    return product, {
        **raw,
        "status": audit_status,
        "reason": audit_reason,
    }, EXCLUSION_REASON_CODES["accepted"]


def _concatenate(products: Sequence[EpochProduct], attribute: str) -> np.ndarray:
    return np.concatenate([np.asarray(getattr(item, attribute)) for item in products])


def validate_dataset_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    features = arrays["features"]
    row_count = features.shape[0]
    if features.shape != (row_count, 3) or features.dtype != np.float64:
        raise RuntimeError("features must be an (n, 3) float64 matrix")
    row_arrays = (
        "epoch_index",
        "row_epoch_time_gpst_like_s",
        "source_row_index",
        "source_observation_index",
        "rtklib_satellite_number",
        "satellite_prn",
        "constellation_code",
        "source_signal_code",
        "system_clock_index",
        "raw_pseudorange_m",
        "raw_snr_units",
        "corrected_pseudorange_m",
        "satellite_clock_bias_s",
        "satellite_clock_correction_m",
        "cn0_snr0_div_1000",
        "elevation_rad",
        "ols_residual_m",
        "validity_code",
        "exclusion_reason_code",
    )
    for name in row_arrays:
        if arrays[name].shape != (row_count,):
            raise RuntimeError(f"{name} is not aligned to the feature rows")
    if arrays["satellite_position_ecef_m"].shape != (row_count, 3):
        raise RuntimeError("satellite positions are not aligned to feature rows")
    if not np.array_equal(features[:, 0], arrays["cn0_snr0_div_1000"]):
        raise RuntimeError("feature C/N0 column lost row alignment")
    if not np.array_equal(features[:, 1], arrays["elevation_rad"]):
        raise RuntimeError("feature elevation column lost row alignment")
    if not np.array_equal(features[:, 2], arrays["ols_residual_m"]):
        raise RuntimeError("feature residual column lost row alignment")
    if not np.array_equal(
        arrays["cn0_snr0_div_1000"], arrays["raw_snr_units"] / 1000.0
    ):
        raise RuntimeError("C/N0 scaling differs from SNR[0]/1000")
    offsets = arrays["epoch_offsets"]
    if offsets[0] != 0 or offsets[-1] != row_count or np.any(np.diff(offsets) <= 0):
        raise RuntimeError("epoch offsets are not a strict partition of accepted rows")
    expected_epoch_index = np.repeat(
        np.arange(offsets.size - 1, dtype=np.int64), np.diff(offsets)
    )
    if not np.array_equal(arrays["epoch_index"], expected_epoch_index):
        raise RuntimeError("per-row epoch indices differ from epoch offsets")
    if not np.array_equal(
        arrays["row_epoch_time_gpst_like_s"],
        arrays["epoch_time_gpst_like_s"][arrays["epoch_index"]],
    ):
        raise RuntimeError("per-row timestamps differ from their accepted epoch")
    if not np.all(arrays["validity_code"] == STATUS_ACCEPTED):
        raise RuntimeError("a retained row is not marked accepted")
    if not np.all(
        arrays["exclusion_reason_code"] == EXCLUSION_REASON_CODES["accepted"]
    ):
        raise RuntimeError("a retained row carries an exclusion reason")
    numeric_to_check = (
        "features",
        "raw_pseudorange_m",
        "corrected_pseudorange_m",
        "satellite_position_ecef_m",
        "satellite_clock_bias_s",
        "satellite_clock_correction_m",
        "epoch_ols_initial_state",
        "epoch_design_condition_number",
        "epoch_normal_condition_number",
    )
    for name in numeric_to_check:
        if not np.all(np.isfinite(arrays[name])):
            raise RuntimeError(f"{name} contains a non-finite value")
    if np.any(arrays["epoch_design_rank"] != arrays["epoch_active_state_count"]):
        raise RuntimeError("an accepted epoch is rank deficient")
    if not np.all(arrays["epoch_ols_converged"] == 1):
        raise RuntimeError("an accepted epoch is not marked converged")


def _git_implementation_record() -> dict[str, Any]:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=REPOSITORY_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"repository_head": head, "working_tree_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"repository_head": None, "working_tree_dirty": None}


def _header_manifest(header: RinexHeader) -> dict[str, Any]:
    return {
        "version": header.version,
        "file_type": header.file_type,
        "satellite_system": header.satellite_system,
        "observation_types_by_constellation": {
            key: list(value) for key, value in header.observation_types.items()
        },
        "interval_s": header.interval_s,
        "first_observation": header.first_observation,
        "last_observation": header.last_observation,
        "time_system": header.time_system,
        "signal_strength_unit": header.signal_strength_unit,
    }


def _counter_names(values: Iterable[int], names: Mapping[int, str]) -> dict[str, int]:
    counts = Counter(int(value) for value in values)
    return {names.get(code, f"unknown_{code}"): counts[code] for code in sorted(counts)}


def preprocess_dataset(
    *,
    observation_path: Path,
    navigation_path: Path,
    runtime_dir: Path = DEFAULT_RUNTIME_DIR,
    progress_every: int = 0,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    observation_path = observation_path.resolve()
    navigation_path = navigation_path.resolve()
    runtime, runtime_cache_manifest = resolve_runtime(runtime_dir)
    tdl_dir = runtime.tdl_dir
    pyrtklib_site = runtime.pyrtklib_site
    for label, path in (
        ("observation", observation_path),
        ("navigation", navigation_path),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    prl = import_pyrtklib(pyrtklib_site)
    installed_version = PYRTKLIB_VERSION
    util = load_rtk_util(
        tdl_dir,
        module_name="ibiza_paper_rtk_util",
    )

    observation_header = read_rinex_header(observation_path)
    navigation_header = read_rinex_header(navigation_path)
    nav_text_counts = navigation_record_counts(navigation_path)
    obs, nav, _station = util.read_obs(str(observation_path), str(navigation_path))
    prl.sortobs(obs)
    split_epochs = util.split_obs(obs)
    if not split_epochs:
        raise RuntimeError("RINEX parser produced no observation epochs")

    products: list[EpochProduct] = []
    audit_epochs: list[np.ndarray] = []
    audit_source_rows: list[np.ndarray] = []
    audit_source_indices: list[np.ndarray] = []
    audit_satellite_numbers: list[np.ndarray] = []
    audit_satellite_prns: list[np.ndarray] = []
    audit_constellations: list[np.ndarray] = []
    audit_signal_codes: list[np.ndarray] = []
    audit_raw_pseudorange: list[np.ndarray] = []
    audit_raw_snr: list[np.ndarray] = []
    audit_status: list[np.ndarray] = []
    audit_reason: list[np.ndarray] = []
    raw_epoch_times: list[float] = []
    raw_epoch_counts: list[int] = []
    raw_epoch_status: list[int] = []
    raw_epoch_reason: list[int] = []
    raw_epoch_accepted_index: list[int] = []
    source_offset = 0

    for split_index, epoch in enumerate(split_epochs):
        epoch_time = float(epoch.data[0].time.time + epoch.data[0].time.sec)
        raw_epoch_times.append(epoch_time)
        raw_epoch_counts.append(int(epoch.n))
        product, audit, reason = preprocess_epoch(
            prl,
            util,
            epoch,
            nav,
            split_epoch_index=split_index,
            source_observation_offset=source_offset,
        )
        accepted_index = len(products) if product is not None else -1
        if product is not None:
            products.append(product)
            raw_epoch_status.append(STATUS_ACCEPTED)
            raw_epoch_reason.append(EXCLUSION_REASON_CODES["accepted"])
        else:
            raw_epoch_status.append(STATUS_EPOCH_REJECTED)
            raw_epoch_reason.append(reason)
        raw_epoch_accepted_index.append(accepted_index)
        audit_epochs.append(np.full(epoch.n, split_index, dtype=np.int64))
        audit_source_rows.append(np.arange(epoch.n, dtype=np.int64))
        audit_source_indices.append(source_offset + np.arange(epoch.n, dtype=np.int64))
        audit_satellite_numbers.append(audit["satellite_numbers"])
        audit_satellite_prns.append(audit["satellite_prns"])
        audit_constellations.append(audit["constellation_codes"])
        audit_signal_codes.append(audit["signal_codes"])
        audit_raw_pseudorange.append(audit["pseudorange"])
        audit_raw_snr.append(audit["snr"])
        audit_status.append(audit["status"])
        audit_reason.append(audit["reason"])
        source_offset += epoch.n
        if progress_every and (split_index + 1) % progress_every == 0:
            print(
                f"processed {split_index + 1}/{len(split_epochs)} raw epochs; "
                f"accepted {len(products)}",
                flush=True,
            )

    if source_offset != obs.n:
        raise RuntimeError(
            f"split epoch rows total {source_offset}, but sorted RINEX has {obs.n} rows"
        )
    if not products:
        raise RuntimeError("paper-era preprocessing accepted no Ibiza epochs")

    counts = np.asarray([item.features.shape[0] for item in products], dtype=np.int64)
    epoch_offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    epoch_index = np.repeat(np.arange(len(products), dtype=np.int64), counts)
    features = _concatenate(products, "features")
    positions = _concatenate(products, "satellite_positions_ecef_m").reshape(-1, 3)
    clock_bias = _concatenate(products, "satellite_clock_bias_s")
    accepted_epoch_times = np.asarray(
        [item.epoch_time for item in products], dtype=np.float64
    )
    arrays: dict[str, np.ndarray] = {
        "schema_version": np.asarray([1], dtype=np.int64),
        "features": features,
        "epoch_index": epoch_index,
        "row_epoch_time_gpst_like_s": np.repeat(accepted_epoch_times, counts),
        "source_row_index": _concatenate(products, "source_rows"),
        "source_observation_index": _concatenate(
            products, "source_observation_indices"
        ),
        "rtklib_satellite_number": _concatenate(products, "satellite_numbers"),
        "satellite_prn": _concatenate(products, "satellite_prns"),
        "constellation_code": _concatenate(products, "constellation_codes"),
        "source_signal_code": _concatenate(products, "signal_codes"),
        "system_clock_index": _concatenate(products, "system_clock_indices"),
        "raw_pseudorange_m": _concatenate(products, "raw_pseudorange_m"),
        "raw_snr_units": _concatenate(products, "raw_snr_units"),
        "corrected_pseudorange_m": _concatenate(
            products, "corrected_pseudorange_m"
        ),
        "satellite_position_ecef_m": positions,
        "satellite_clock_bias_s": clock_bias,
        "satellite_clock_correction_m": -SPEED_OF_LIGHT_M_S * clock_bias,
        "cn0_snr0_div_1000": features[:, 0].copy(),
        "elevation_rad": features[:, 1].copy(),
        "ols_residual_m": features[:, 2].copy(),
        "validity_code": np.full(features.shape[0], STATUS_ACCEPTED, dtype=np.int8),
        "exclusion_reason_code": np.full(
            features.shape[0], EXCLUSION_REASON_CODES["accepted"], dtype=np.int16
        ),
        "epoch_offsets": epoch_offsets,
        "epoch_time_gpst_like_s": accepted_epoch_times,
        "epoch_split_index": np.asarray(
            [item.split_epoch_index for item in products], dtype=np.int64
        ),
        "epoch_raw_observation_count": np.asarray(
            [item.raw_observation_count for item in products], dtype=np.int64
        ),
        "epoch_accepted_observation_count": counts,
        "epoch_ols_initial_state": np.vstack(
            [item.ols_state for item in products]
        ),
        "epoch_active_state_mask": np.vstack(
            [item.active_state_mask for item in products]
        ),
        "epoch_active_state_count": np.asarray(
            [int(item.active_state_mask.sum()) for item in products], dtype=np.int64
        ),
        "epoch_design_rank": np.asarray(
            [item.design_rank for item in products], dtype=np.int64
        ),
        "epoch_design_condition_number": np.asarray(
            [item.design_condition_number for item in products], dtype=np.float64
        ),
        "epoch_normal_condition_number": np.asarray(
            [item.normal_condition_number for item in products], dtype=np.float64
        ),
        "epoch_ols_converged": np.ones(len(products), dtype=np.uint8),
        "raw_epoch_time_gpst_like_s": np.asarray(raw_epoch_times, dtype=np.float64),
        "raw_epoch_observation_count": np.asarray(raw_epoch_counts, dtype=np.int64),
        "raw_epoch_status_code": np.asarray(raw_epoch_status, dtype=np.int8),
        "raw_epoch_rejection_reason_code": np.asarray(
            raw_epoch_reason, dtype=np.int16
        ),
        "raw_epoch_accepted_epoch_index": np.asarray(
            raw_epoch_accepted_index, dtype=np.int64
        ),
        "audit_split_epoch_index": np.concatenate(audit_epochs),
        "audit_source_row_index": np.concatenate(audit_source_rows),
        "audit_source_observation_index": np.concatenate(audit_source_indices),
        "audit_rtklib_satellite_number": np.concatenate(audit_satellite_numbers),
        "audit_satellite_prn": np.concatenate(audit_satellite_prns),
        "audit_constellation_code": np.concatenate(audit_constellations),
        "audit_source_signal_code": np.concatenate(audit_signal_codes),
        "audit_raw_pseudorange_m": np.concatenate(audit_raw_pseudorange),
        "audit_raw_snr_units": np.concatenate(audit_raw_snr),
        "audit_status_code": np.concatenate(audit_status),
        "audit_exclusion_reason_code": np.concatenate(audit_reason),
        "clock_state_index_by_constellation_code": np.asarray(
            [0, 3, 6, 5, 4], dtype=np.int64
        ),
    }
    validate_dataset_arrays(arrays)

    obs_system_counts = _counter_names(
        arrays["audit_constellation_code"], CODE_TO_CONSTELLATION
    )
    accepted_system_counts = _counter_names(
        arrays["constellation_code"], CODE_TO_CONSTELLATION
    )
    exclusion_counts = _counter_names(
        arrays["audit_exclusion_reason_code"], REASON_CODE_TO_NAME
    )
    rejected_epoch_counts = _counter_names(
        arrays["raw_epoch_rejection_reason_code"][
            arrays["raw_epoch_status_code"] != STATUS_ACCEPTED
        ],
        REASON_CODE_TO_NAME,
    )
    no_eph_code = EXCLUSION_REASON_CODES["no_broadcast_ephemeris"]
    no_eph_mask = arrays["audit_exclusion_reason_code"] == no_eph_code
    no_eph_system_counts = _counter_names(
        arrays["audit_constellation_code"][no_eph_mask], CODE_TO_CONSTELLATION
    )
    if runtime_cache_manifest is None:
        runtime_provenance: dict[str, Any] = {
            "mode": "explicit runtime path overrides",
            "cache_validated": False,
        }
    else:
        runtime_provenance = {
            "mode": "persistent generated cache",
            "cache_validated": True,
            "schema_version": runtime_cache_manifest["schema_version"],
            "status": runtime_cache_manifest["status"],
            "rtk_util_sha256": runtime_cache_manifest["rtk_util_sha256"],
            "pyrtklib_binary_sha256": runtime_cache_manifest[
                "pyrtklib_binary_sha256"
            ],
            "cpu_device_substitution": runtime_cache_manifest[
                "cpu_device_substitution"
            ],
        }

    first_signal_mapping: dict[str, dict[str, Any]] = {}
    for system, obs_types in observation_header.observation_types.items():
        first_code = next((item for item in obs_types if item.startswith("C")), None)
        first_snr = next((item for item in obs_types if item.startswith("S")), None)
        constellation_mask = (
            arrays["audit_constellation_code"]
            == CONSTELLATION_TO_CODE.get(system, 255)
        )
        accepted_mask = constellation_mask & (
            arrays["audit_status_code"] == STATUS_ACCEPTED
        )
        raw_codes = sorted(
            set(int(item) for item in arrays["audit_source_signal_code"][constellation_mask])
        )
        accepted_codes = sorted(
            set(int(item) for item in arrays["audit_source_signal_code"][accepted_mask])
        )
        first_signal_mapping[system] = {
            "raw_pseudorange_observation": first_code,
            "raw_cn0_observation": first_snr,
            "rtklib_slot": 0,
            "rtklib_code_values_seen_in_raw_rows": raw_codes,
            "rtklib_code_values_seen_in_accepted_rows": accepted_codes,
            "rtklib_code_labels_seen_in_accepted_rows": [
                str(prl.code2obs(code)) for code in accepted_codes
            ],
            "model_cn0_value": "SNR[0]/1000",
            "substitution_performed": False,
        }

    manifest: dict[str, Any] = {
        "status": "passed",
        "dataset": "Ibiza 2025-01-01 external generalization",
        "inputs": {
            "observation": {
                "filename": observation_path.name,
                "sha256": sha256_file(observation_path),
                "size_bytes": observation_path.stat().st_size,
            },
            "navigation": {
                "filename": navigation_path.name,
                "sha256": sha256_file(navigation_path),
                "size_bytes": navigation_path.stat().st_size,
            },
        },
        "rinex": {
            "observation": _header_manifest(observation_header),
            "navigation": _header_manifest(navigation_header),
            "navigation_record_counts_by_constellation": nav_text_counts,
            "first_processed_signal_mapping": first_signal_mapping,
            "pyrtklib_parse_audit": {
                "observation_parse_succeeded": True,
                "navigation_parse_succeeded": True,
                "sorted_observation_rows": int(obs.n),
                "split_epochs": len(split_epochs),
                "first_epoch_time_gpst_like_s": raw_epoch_times[0],
                "last_epoch_time_gpst_like_s": raw_epoch_times[-1],
            },
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pyrtklib": installed_version,
            "platform": platform.platform(),
        },
        "provenance": {
            "tdl_gnss_reference_commit": TDL_COMMIT,
            "pyrtklib_version_commit_hypothesis": {
                "version": PYRTKLIB_VERSION,
                "commit": PYRTKLIB_COMMIT,
                "publication_dependency_proven": False,
            },
            "preprocessing_implementation": _git_implementation_record(),
            "paper_runtime": runtime_provenance,
            "external_repository_modified": False,
        },
        "epoch_counts": {
            "raw": len(split_epochs),
            "accepted": len(products),
            "rejected": len(split_epochs) - len(products),
            "rejected_by_reason": rejected_epoch_counts,
        },
        "satellite_epoch_observations": {
            "raw": int(obs.n),
            "accepted": int(features.shape[0]),
            "accepted_by_constellation": accepted_system_counts,
            "raw_by_constellation": obs_system_counts,
            "exclusion_or_status_counts": exclusion_counts,
        },
        "broadcast_ephemeris_coverage": {
            "observed_constellations": sorted(obs_system_counts),
            "navigation_constellations": sorted(nav_text_counts),
            "required_constellations_present_in_navigation": all(
                system in nav_text_counts for system in obs_system_counts
            ),
            "unusable_satellite_epoch_rows_by_constellation": no_eph_system_counts,
            "historical_galileo_ephemeris_selection": int(
                prl.getseleph(prl.SYS_GAL)
            ),
            "signal_or_ephemeris_substitution_performed": False,
        },
        "features": {
            "matrix_name": "features",
            "columns": [
                {"index": index, "name": name, "unit": unit}
                for index, (name, unit) in enumerate(
                    zip(FEATURE_NAMES, FEATURE_UNITS, strict=True)
                )
            ],
            "dtype": "float64",
            "shape": list(features.shape),
            "ibiza_normalization_computed": False,
            "normalization_policy": (
                "raw features only; each frozen checkpoint applies its frozen "
                "KLT3 StandardizeLayer later"
            ),
        },
        "state_layout": {
            "indices": [
                "x_m",
                "y_m",
                "z_m",
                "GPS_clock_m",
                "BeiDou_clock_m",
                "Galileo_clock_m",
                "GLONASS_clock_m",
            ],
            "constellation_code": {
                str(code): system for code, system in CODE_TO_CONSTELLATION.items()
            },
            "exclusion_reason_code": {
                str(code): name for code, name in REASON_CODE_TO_NAME.items()
            },
        },
        "ols_diagnostics": {
            "all_accepted_epochs_converged": True,
            "all_accepted_epochs_full_column_rank": True,
            "maximum_design_condition_number": float(
                arrays["epoch_design_condition_number"].max()
            ),
            "maximum_normal_condition_number": float(
                arrays["epoch_normal_condition_number"].max()
            ),
            "condition_number_definition": (
                "2-norm condition numbers of final H and H.T@H"
            ),
        },
        "scientific_controls": {
            "ground_truth_used": False,
            "ground_truth_input_exists": False,
            "ibiza_normalization_computed": False,
            "all_architectures_share_one_raw_feature_array": True,
            "network_inference_run": False,
            "network_training_run": False,
        },
        "npz_schema": {
            name: {"dtype": str(value.dtype), "shape": list(value.shape)}
            for name, value in sorted(arrays.items())
        },
        "output": {
            "filename": DEFAULT_OUTPUT.name,
            "canonical_content_sha256": canonical_content_sha256(arrays),
        },
    }
    return arrays, manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    arrays, manifest = preprocess_dataset(
        observation_path=args.observation,
        navigation_path=args.navigation,
        runtime_dir=args.runtime_dir,
        progress_every=args.progress_every,
    )
    output_path = args.output.resolve()
    npz_sha256 = write_deterministic_npz(output_path, arrays)
    manifest["output"]["filename"] = output_path.name
    manifest["output"]["npz_sha256"] = npz_sha256
    manifest_path = args.manifest.resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": output_path.name,
                "manifest": manifest_path.name,
                "raw_epochs": manifest["epoch_counts"]["raw"],
                "accepted_epochs": manifest["epoch_counts"]["accepted"],
                "accepted_satellite_epoch_observations": manifest[
                    "satellite_epoch_observations"
                ]["accepted"],
                "npz_sha256": npz_sha256,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
