"""Command-line entry point for the controlled official-RTKLIB Ibiza baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

from .core import (
    CONFIG,
    DEFAULT_OUTPUT,
    audit_pre_run,
    execute_rtklib,
    finalize_evaluation,
    parse_rtklib_pos,
    sha256_file,
    solve_full_interval,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("audit", help="run the ground-truth-free pre-run audit")

    solve = subparsers.add_parser(
        "solve", help="run full-interval RTKLIB and serialize ECEF before ground truth"
    )
    solve.add_argument("--binary", type=Path, required=True)
    solve.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)

    evaluate = subparsers.add_parser(
        "evaluate", help="evaluate an already serialized full-interval solution"
    )
    evaluate.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)

    smoke = subparsers.add_parser(
        "smoke", help="run a replayed 60-second non-scientific smoke interval"
    )
    smoke.add_argument("--binary", type=Path, required=True)
    smoke.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT / "smoke")
    return parser.parse_args()


def run_smoke(binary: Path, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / "rtklib_smoke.pos"
    with tempfile.TemporaryDirectory(prefix="rtklib-ibiza-smoke-") as temporary:
        first_path = Path(temporary) / "first.pos"
        second_path = Path(temporary) / "second.pos"
        start = "2025/01/01 00:09:00"
        end = "2025/01/01 00:10:00"
        execute_rtklib(binary, first_path, start=start, end=end)
        execute_rtklib(binary, second_path, start=start, end=end)
        header_verified = all(
            "% elev mask : 7.0 deg" in path.read_text(encoding="ascii")
            for path in (first_path, second_path)
        )
        if not header_verified:
            raise RuntimeError("RTKLIB smoke output does not declare a 7 degree mask")
        first = parse_rtklib_pos(first_path)
        second = parse_rtklib_pos(second_path)
        identical = all(
            np.array_equal(left, right)
            for left, right in (
                (first.timestamps_gpst_s, second.timestamps_gpst_s),
                (first.ecef_m, second.ecef_m),
                (first.quality, second.quality),
                (first.satellite_count, second.satellite_count),
            )
        )
        if not identical:
            raise RuntimeError("replayed RTKLIB smoke positions are not deterministic")
        final_path.write_bytes(first_path.read_bytes())
    report: dict[str, object] = {
        "schema_version": 1,
        "scientific_result": False,
        "interval": "2025/01/01 00:09:00 through 00:10:00 GPST",
        "solution_count": int(first.timestamps_gpst_s.size),
        "deterministic_replay_identical": identical,
        "configuration": str(CONFIG.resolve()),
        "configuration_sha256": sha256_file(CONFIG),
        "elevation_mask_degrees": 7.0,
        "rtklib_output_header_elevation_mask_verified": header_verified,
        "ground_truth_opened": False,
        "raw_output": str(final_path.resolve()),
        "raw_output_sha256": sha256_file(final_path),
        "binary": str(binary.resolve()),
        "binary_sha256": sha256_file(binary),
    }
    write_json(output_dir / "smoke_report.json", report)
    return report


def main() -> None:
    arguments = parse_args()
    if arguments.command == "audit":
        result = audit_pre_run()
    elif arguments.command == "solve":
        result = solve_full_interval(arguments.binary, arguments.output_dir)
    elif arguments.command == "evaluate":
        result = finalize_evaluation(arguments.output_dir)
    else:
        result = run_smoke(arguments.binary, arguments.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
