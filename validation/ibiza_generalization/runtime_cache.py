"""Paths and integrity checks for the persistent paper-era runtime cache."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from types import ModuleType
from typing import Any


TDL_COMMIT = "dd5eac669676ba0a922102047e58c2dfc9be9267"
PYRTKLIB_COMMIT = "916d3cc8eb202718a16097cea4a5729bd6b27ac5"
PYRTKLIB_VERSION = "0.2.6"
RUNTIME_SCHEMA_VERSION = 1

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PHD_ROOT = REPOSITORY_ROOT.parent
DEFAULT_RUNTIME_DIR = PHD_ROOT / "external_data/.paper_runtime"
DEFAULT_TDL_REPOSITORY = PHD_ROOT / "external_references/TDL-GNSS"
DEFAULT_PYRTKLIB_REPOSITORY = PHD_ROOT / "external_references/pyrtklib"


@dataclass(frozen=True)
class RuntimePaths:
    root: Path
    tdl_dir: Path
    pyrtklib_source: Path
    pyrtklib_site: Path
    manifest: Path
    readme: Path


def runtime_paths(root: Path = DEFAULT_RUNTIME_DIR) -> RuntimePaths:
    resolved = root.resolve()
    return RuntimePaths(
        root=resolved,
        tdl_dir=resolved / "tdl-dd5eac6",
        pyrtklib_source=resolved / "pyrtklib-0.2.6-src",
        pyrtklib_site=resolved / "pyrtklib-0.2.6-site",
        manifest=resolved / "runtime_manifest.json",
        readme=resolved / "README.md",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_signature() -> dict[str, str]:
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_cache_tag": sys.implementation.cache_tag or "",
        "operating_system": platform.system(),
        "machine": platform.machine(),
    }


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read runtime manifest {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"runtime manifest is not a JSON object: {path}")
    return value


def _pyrtklib_binary(paths: RuntimePaths) -> Path:
    candidates = sorted((paths.pyrtklib_site / "pyrtklib").glob("pyrtklib*.so"))
    if len(candidates) != 1:
        raise RuntimeError(
            "runtime cache must contain exactly one pyrtklib extension, found "
            f"{len(candidates)} under {paths.pyrtklib_site / 'pyrtklib'}"
        )
    return candidates[0]


def _probe_pyrtklib(site: Path) -> tuple[str, Path]:
    site = site.resolve()
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(site) if not existing else os.pathsep.join((str(site), existing))
    )
    command = [
        sys.executable,
        "-c",
        (
            "import importlib.metadata, json; import pyrtklib; "
            "print(json.dumps({'version': importlib.metadata.version('pyrtklib'), "
            "'module_file': pyrtklib.__file__}))"
        ),
    ]
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "pyrtklib cache import failed: "
            + (result.stderr.strip() or result.stdout.strip())
        )
    try:
        payload = json.loads(result.stdout)
        version = str(payload["version"])
        module_file = Path(payload["module_file"]).resolve()
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"pyrtklib cache probe returned invalid output: {result.stdout!r}"
        ) from error
    if not module_file.is_relative_to(site):
        raise RuntimeError(
            f"pyrtklib cache probe imported {module_file}, outside selected site {site}"
        )
    return version, module_file


def validate_runtime_cache(root: Path = DEFAULT_RUNTIME_DIR) -> dict[str, Any]:
    """Validate the cache and return its manifest, or raise with a repair hint."""

    paths = runtime_paths(root)
    try:
        manifest = _load_manifest(paths.manifest)
        expected_fields = {
            "schema_version": RUNTIME_SCHEMA_VERSION,
            "status": "ready",
            "tdl_gnss_commit": TDL_COMMIT,
            "pyrtklib_commit": PYRTKLIB_COMMIT,
            "pyrtklib_version": PYRTKLIB_VERSION,
            "runtime": runtime_signature(),
        }
        for name, expected in expected_fields.items():
            if manifest.get(name) != expected:
                raise RuntimeError(
                    f"runtime manifest field {name!r} is {manifest.get(name)!r}, "
                    f"expected {expected!r}"
                )
        if not paths.readme.is_file():
            raise RuntimeError(f"runtime README is missing: {paths.readme}")
        rtk_util = paths.tdl_dir / "rtk_util.py"
        if not rtk_util.is_file():
            raise RuntimeError(f"cached rtk_util.py is missing: {rtk_util}")
        if sha256_file(rtk_util) != manifest.get("rtk_util_sha256"):
            raise RuntimeError("cached rtk_util.py hash differs from the manifest")
        binary = _pyrtklib_binary(paths)
        if sha256_file(binary) != manifest.get("pyrtklib_binary_sha256"):
            raise RuntimeError("cached pyrtklib binary hash differs from the manifest")
        actual_version, _module_file = _probe_pyrtklib(paths.pyrtklib_site)
        if actual_version != PYRTKLIB_VERSION:
            raise RuntimeError(
                f"cached pyrtklib reports {actual_version}, expected {PYRTKLIB_VERSION}"
            )
    except RuntimeError as error:
        raise RuntimeError(
            f"paper runtime cache is absent or invalid: {error}. Rebuild it with "
            "`.venv/bin/python -m "
            "validation.ibiza_generalization.prepare_runtime`."
        ) from error
    return manifest


def resolve_runtime(
    root: Path = DEFAULT_RUNTIME_DIR,
) -> tuple[RuntimePaths, dict[str, Any]]:
    """Resolve and validate one complete persistent paper-era runtime."""

    paths = runtime_paths(root)
    return paths, validate_runtime_cache(paths.root)


def import_pyrtklib(pyrtklib_site: Path) -> ModuleType:
    """Import pyrtklib 0.2.6 and reject modules outside the selected site."""

    site = pyrtklib_site.resolve()
    if not site.is_dir():
        raise FileNotFoundError(f"pyrtklib target installation not found: {site}")
    rendered = str(site)
    if rendered not in sys.path:
        sys.path.insert(0, rendered)
    module = importlib.import_module("pyrtklib")
    module_filename = getattr(module, "__file__", None)
    if not module_filename:
        raise RuntimeError("imported pyrtklib has no __file__; cannot verify its origin")
    module_file = Path(module_filename).resolve()
    if not module_file.is_relative_to(site):
        raise RuntimeError(
            f"pyrtklib was imported from {module_file}, outside selected runtime "
            f"site {site}"
        )
    installed_version = importlib.metadata.version("pyrtklib")
    if installed_version != PYRTKLIB_VERSION:
        raise RuntimeError(
            f"expected pyrtklib {PYRTKLIB_VERSION}, found {installed_version} "
            f"under {site}"
        )
    return module


def load_rtk_util(tdl_dir: Path, *, module_name: str) -> ModuleType:
    """Load the selected cached rtk_util.py without ambient module fallback."""

    source = tdl_dir.resolve() / "rtk_util.py"
    if not source.is_file():
        raise FileNotFoundError(f"cached rtk_util.py not found: {source}")
    specification = importlib.util.spec_from_file_location(module_name, source)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"could not load cached rtk_util.py: {source}")
    module = importlib.util.module_from_spec(specification)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        specification.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


__all__ = [
    "DEFAULT_PYRTKLIB_REPOSITORY",
    "DEFAULT_RUNTIME_DIR",
    "DEFAULT_TDL_REPOSITORY",
    "PYRTKLIB_COMMIT",
    "PYRTKLIB_VERSION",
    "RUNTIME_SCHEMA_VERSION",
    "RuntimePaths",
    "TDL_COMMIT",
    "import_pyrtklib",
    "load_rtk_util",
    "resolve_runtime",
    "runtime_paths",
    "runtime_signature",
    "sha256_file",
    "validate_runtime_cache",
]
