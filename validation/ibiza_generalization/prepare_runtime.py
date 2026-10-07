#!/usr/bin/env python3
"""Prepare or verify the persistent, pinned paper-era GNSS runtime cache."""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import Sequence
import uuid

from .runtime_cache import (
    DEFAULT_PYRTKLIB_REPOSITORY,
    DEFAULT_RUNTIME_DIR,
    DEFAULT_TDL_REPOSITORY,
    PYRTKLIB_COMMIT,
    PYRTKLIB_VERSION,
    RUNTIME_SCHEMA_VERSION,
    TDL_COMMIT,
    runtime_paths,
    runtime_signature,
    sha256_file,
    validate_runtime_cache,
)


RUNTIME_README = """# Generated paper-era GNSS runtime cache

This directory is a persistent, generated compatibility cache. It exists so
the deterministic Ibiza preprocessor and earlier paper-era validations do not
depend on files under `/tmp`, which may disappear after a reboot.

It contains a pinned `git archive` of TDL-GNSS, the pinned pyrtklib source,
and a locally compiled pyrtklib 0.2.6 target installation. These are runtime
dependencies, not raw GNSS data or scientific results. The source reference
repositories under `external_references/` are read-only and are never modified.

`runtime_manifest.json` records the source commits, Python/platform signature,
CPU-only compatibility substitution, and artifact hashes. The cache is reused
only when those checks pass. It may be deleted safely; recreate it from the
`gnss-satellite-selection-ml` repository root with:

```bash
.venv/bin/python -m validation.ibiza_generalization.prepare_runtime
```

Re-run that command after changing Python, operating system, or CPU
architecture. Do not commit this generated directory to a source repository.
"""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME_DIR)
    parser.add_argument(
        "--tdl-repository", type=Path, default=DEFAULT_TDL_REPOSITORY
    )
    parser.add_argument(
        "--pyrtklib-repository", type=Path, default=DEFAULT_PYRTKLIB_REPOSITORY
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even when the existing cache passes every integrity check.",
    )
    return parser.parse_args(argv)


def _verify_git_object(repository: Path, commit: str, label: str) -> None:
    if not (repository / ".git").exists():
        raise FileNotFoundError(f"{label} Git repository not found: {repository}")
    result = subprocess.run(
        ["git", "-C", str(repository), "cat-file", "-e", f"{commit}^{{commit}}"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{label} does not contain pinned commit {commit}: "
            + (result.stderr.strip() or result.stdout.strip())
        )


def _archive_commit(repository: Path, commit: str, destination: Path) -> None:
    archive = subprocess.run(
        ["git", "-C", str(repository), "archive", "--format=tar", commit],
        check=True,
        capture_output=True,
    ).stdout
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as stream:
        destination_root = destination.resolve()
        for member in stream.getmembers():
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination_root):
                raise RuntimeError(
                    f"git archive contains an unsafe path: {member.name!r}"
                )
            if member.issym() or member.islnk():
                raise RuntimeError(
                    f"git archive contains an unsupported link: {member.name!r}"
                )
        stream.extractall(destination)


def _cpu_patch_rtk_util(path: Path) -> tuple[int, str, str]:
    source = path.read_text(encoding="utf-8")
    upstream_hash = sha256_file(path)
    count = source.count(".to('cuda')")
    if count == 0:
        raise RuntimeError(
            "pinned rtk_util.py contains no expected `.to('cuda')` placement"
        )
    path.write_text(source.replace(".to('cuda')", ".to('cpu')"), encoding="utf-8")
    patched_hash = sha256_file(path)
    if upstream_hash == patched_hash:
        raise RuntimeError("CPU compatibility substitution did not change rtk_util.py")
    return count, upstream_hash, patched_hash


def _build_pyrtklib(source: Path, site: Path) -> None:
    site.mkdir(parents=True, exist_ok=False)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-build-isolation",
            "--no-cache-dir",
            "--target",
            str(site),
            str(source),
        ],
        check=True,
    )


def _write_runtime_metadata(stage: Path, cpu_patch_count: int, upstream_hash: str) -> None:
    paths = runtime_paths(stage)
    binaries = sorted((paths.pyrtklib_site / "pyrtklib").glob("pyrtklib*.so"))
    if len(binaries) != 1:
        raise RuntimeError(
            f"pyrtklib build produced {len(binaries)} extension modules, expected one"
        )
    manifest = {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "status": "ready",
        "purpose": "persistent generated paper-era GNSS compatibility runtime",
        "tdl_gnss_commit": TDL_COMMIT,
        "pyrtklib_commit": PYRTKLIB_COMMIT,
        "pyrtklib_version": PYRTKLIB_VERSION,
        "runtime": runtime_signature(),
        "cpu_device_substitution": {
            "expression": ".to('cuda') -> .to('cpu')",
            "replacement_count": cpu_patch_count,
            "changes_equations": False,
        },
        "upstream_rtk_util_sha256": upstream_hash,
        "rtk_util_sha256": sha256_file(paths.tdl_dir / "rtk_util.py"),
        "pyrtklib_binary_filename": binaries[0].name,
        "pyrtklib_binary_sha256": sha256_file(binaries[0]),
        "source_repositories_modified": False,
        "safe_to_delete_and_rebuild": True,
    }
    paths.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    paths.readme.write_text(RUNTIME_README, encoding="utf-8")


def _install_stage(stage: Path, destination: Path) -> None:
    backup: Path | None = None
    if destination.exists():
        backup = destination.with_name(
            destination.name + ".previous-" + uuid.uuid4().hex
        )
        os.replace(destination, backup)
    try:
        os.replace(stage, destination)
    except BaseException:
        if backup is not None and backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    if backup is not None:
        shutil.rmtree(backup)


def prepare_runtime(
    *,
    runtime_dir: Path,
    tdl_repository: Path,
    pyrtklib_repository: Path,
    force: bool = False,
) -> tuple[str, dict[str, object]]:
    runtime_dir = runtime_dir.resolve()
    tdl_repository = tdl_repository.resolve()
    pyrtklib_repository = pyrtklib_repository.resolve()
    if not force:
        try:
            return "reused", validate_runtime_cache(runtime_dir)
        except RuntimeError:
            pass

    _verify_git_object(tdl_repository, TDL_COMMIT, "TDL-GNSS")
    _verify_git_object(pyrtklib_repository, PYRTKLIB_COMMIT, "pyrtklib")
    runtime_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{runtime_dir.name.lstrip('.')}-build-", dir=runtime_dir.parent
        )
    )
    try:
        paths = runtime_paths(stage)
        _archive_commit(tdl_repository, TDL_COMMIT, paths.tdl_dir)
        patch_count, upstream_hash, _patched_hash = _cpu_patch_rtk_util(
            paths.tdl_dir / "rtk_util.py"
        )
        _archive_commit(
            pyrtklib_repository, PYRTKLIB_COMMIT, paths.pyrtklib_source
        )
        _build_pyrtklib(paths.pyrtklib_source, paths.pyrtklib_site)
        _write_runtime_metadata(stage, patch_count, upstream_hash)
        validate_runtime_cache(stage)
        _install_stage(stage, runtime_dir)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return "rebuilt", validate_runtime_cache(runtime_dir)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    action, manifest = prepare_runtime(
        runtime_dir=args.runtime_dir,
        tdl_repository=args.tdl_repository,
        pyrtklib_repository=args.pyrtklib_repository,
        force=args.force,
    )
    paths = runtime_paths(args.runtime_dir)
    print(
        json.dumps(
            {
                "action": action,
                "runtime_directory": str(paths.root),
                "tdl_gnss_commit": manifest["tdl_gnss_commit"],
                "pyrtklib_commit": manifest["pyrtklib_commit"],
                "pyrtklib_version": manifest["pyrtklib_version"],
                "status": manifest["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
