#!/usr/bin/env python3
"""Run independent corrected-GT BiasNet experiments for seeds 0 through 9."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PREDEFINED_SEEDS = tuple(range(10))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    shared = REPOSITORY / "validation/paper_weightnet"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(PREDEFINED_SEEDS))
    parser.add_argument("--features", type=Path, default=shared / "klt3_features.npz")
    parser.add_argument(
        "--manifest", type=Path, default=shared / "klt3_feature_manifest.json"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY / "results/paper_biasnet_seed_sensitivity",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=REPOSITORY / "checkpoints/paper_biasnet_seed_sensitivity",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--observation", type=Path)
    parser.add_argument(
        "--ephemeris-glob", action="append", dest="ephemeris_patterns"
    )
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--tdl-dir", type=Path)
    parser.add_argument("--pyrtklib-site", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def validate_seeds(seeds: list[int]) -> None:
    if len(seeds) != len(set(seeds)):
        raise ValueError("--seeds must not contain duplicates")
    outside = sorted(set(seeds) - set(PREDEFINED_SEEDS))
    if outside:
        raise ValueError(f"seed sweep is fixed to 0..9; unsupported values: {outside}")


def command_for_seed(args: argparse.Namespace, seed: int) -> list[str]:
    command = [
        sys.executable,
        str(HERE / "run_seed.py"),
        "--seed",
        str(seed),
        "--features",
        str(args.features),
        "--manifest",
        str(args.manifest),
        "--output-dir",
        str(args.output_dir),
        "--checkpoint-dir",
        str(args.checkpoint_dir),
        "--device",
        args.device,
    ]
    for option, value in (
        ("--data-root", args.data_root),
        ("--observation", args.observation),
        ("--ground-truth", args.ground_truth),
        ("--tdl-dir", args.tdl_dir),
        ("--pyrtklib-site", args.pyrtklib_site),
    ):
        if value is not None:
            command.extend((option, str(value)))
    for pattern in args.ephemeris_patterns or ():
        command.extend(("--ephemeris-glob", pattern))
    if args.overwrite:
        command.append("--overwrite")
    return command


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_seeds(args.seeds)
    for index, seed in enumerate(args.seeds, start=1):
        print(
            f"starting independent seed {seed} ({index}/{len(args.seeds)})",
            flush=True,
        )
        subprocess.run(command_for_seed(args, seed), check=True, cwd=REPOSITORY)
    print("all requested seeds completed; no ranking or selection was performed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
