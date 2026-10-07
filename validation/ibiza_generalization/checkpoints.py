"""Verified inventory and loading for the frozen seed checkpoints.

Checkpoint bytes are hashed before they are passed to ``torch.load``.  Models
are instantiated with inert normalization values and then populated strictly
through ``load_state_dict`` so the embedded ``seq.0.mean`` and ``seq.0.std``
are the values used by inference.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from validation.paper_biasnet.core import instantiate_released_biasnet
from validation.paper_hybrid.core import instantiate_released_hybrid
from validation.paper_weightnet.core import instantiate_released_weightnet


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = Path(__file__).with_name("frozen_checkpoint_manifest.json")
ARCHITECTURES = ("TDL-B", "TDL-W", "TDL-BW")

_CHECKPOINT_DIRECTORIES = {
    "TDL-B": Path("checkpoints/paper_biasnet_seed_sensitivity"),
    "TDL-W": Path("checkpoints/paper_weightnet_seed_sensitivity"),
    "TDL-BW": Path("checkpoints/paper_hybrid_seed_sensitivity"),
}

_MODEL_FACTORIES = {
    "TDL-B": instantiate_released_biasnet,
    "TDL-W": instantiate_released_weightnet,
    "TDL-BW": instantiate_released_hybrid,
}


@dataclass(frozen=True)
class CheckpointRecord:
    architecture: str
    seed: int
    checkpoint_path: str
    sha256: str
    expected_state_dict_keys: tuple[str, ...]
    standardize_mean: tuple[float, float, float]
    standardize_std: tuple[float, float, float]


@dataclass(frozen=True)
class FrozenModel:
    record: CheckpointRecord
    model: nn.Module


@dataclass(frozen=True)
class InventorySummary:
    records: tuple[CheckpointRecord, ...]
    architecture_counts: Mapping[str, int]
    distinct_hash_count: int
    common_mean: tuple[float, float, float]
    common_std: tuple[float, float, float]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _float_triplet(value: object, name: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise RuntimeError(f"{name} must be a three-element JSON array")
    result = tuple(float(item) for item in value)
    return result  # type: ignore[return-value]


def load_checkpoint_manifest(path: Path = DEFAULT_MANIFEST) -> tuple[CheckpointRecord, ...]:
    """Load and structurally validate the tracked checkpoint manifest."""

    text = path.read_text(encoding="utf-8")
    if "/home/" in text:
        raise RuntimeError("checkpoint manifest contains an absolute /home path")
    document = json.loads(text)
    if document.get("schema_version") != 1:
        raise RuntimeError("unsupported frozen checkpoint manifest schema")
    raw_records = document.get("checkpoints")
    if not isinstance(raw_records, list):
        raise RuntimeError("checkpoint manifest 'checkpoints' must be an array")

    records: list[CheckpointRecord] = []
    for raw in raw_records:
        if not isinstance(raw, dict):
            raise RuntimeError("each checkpoint manifest entry must be an object")
        checkpoint_path = str(raw.get("checkpoint_path", ""))
        if not checkpoint_path or Path(checkpoint_path).is_absolute():
            raise RuntimeError("checkpoint paths must be non-empty and repository-relative")
        keys = raw.get("expected_state_dict_keys")
        if not isinstance(keys, list) or not keys or not all(
            isinstance(item, str) for item in keys
        ):
            raise RuntimeError("expected_state_dict_keys must be a non-empty string array")
        record = CheckpointRecord(
            architecture=str(raw.get("architecture", "")),
            seed=int(raw.get("seed", -1)),
            checkpoint_path=checkpoint_path,
            sha256=str(raw.get("sha256", "")),
            expected_state_dict_keys=tuple(keys),
            standardize_mean=_float_triplet(
                raw.get("standardize_mean"), "standardize_mean"
            ),
            standardize_std=_float_triplet(
                raw.get("standardize_std"), "standardize_std"
            ),
        )
        if record.architecture not in ARCHITECTURES:
            raise RuntimeError(f"unknown architecture {record.architecture!r}")
        if record.seed not in range(10):
            raise RuntimeError(f"invalid seed {record.seed} for {record.architecture}")
        if len(record.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in record.sha256
        ):
            raise RuntimeError("checkpoint SHA-256 must be 64 lowercase hex characters")
        records.append(record)
    return tuple(records)


def checkpoint_record(
    architecture: str,
    seed: int,
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> CheckpointRecord:
    matches = [
        record
        for record in load_checkpoint_manifest(manifest_path)
        if record.architecture == architecture and record.seed == seed
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one manifest record for {architecture} seed {seed}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _verified_payload(record: CheckpointRecord, path: Path) -> bytes:
    if not path.is_file():
        raise FileNotFoundError(f"frozen checkpoint not found: {path}")
    payload = path.read_bytes()
    actual = sha256_bytes(payload)
    if actual != record.sha256:
        raise RuntimeError(
            "checkpoint SHA-256 mismatch for "
            f"{record.architecture} seed {record.seed}: expected {record.sha256}, "
            f"got {actual}"
        )
    return payload


def _load_verified_state_dict(
    record: CheckpointRecord,
    path: Path,
    *,
    device: torch.device,
) -> Mapping[str, torch.Tensor]:
    # Hash exactly the immutable bytes subsequently deserialized, avoiding a
    # verify/open race between separate file reads.
    payload = _verified_payload(record, path)
    loaded: Any = torch.load(
        io.BytesIO(payload), map_location=device, weights_only=True
    )
    if not isinstance(loaded, dict) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in loaded.items()
    ):
        raise RuntimeError("checkpoint is not a tensor state_dict")
    if tuple(loaded) != record.expected_state_dict_keys:
        raise RuntimeError(
            f"state_dict keys/order mismatch for {record.architecture} seed {record.seed}"
        )
    for key in ("seq.0.mean", "seq.0.std"):
        if key not in loaded:
            raise RuntimeError(f"checkpoint lacks required StandardizeLayer tensor {key}")
    expected_mean = torch.tensor(
        record.standardize_mean, dtype=loaded["seq.0.mean"].dtype, device=device
    )
    expected_std = torch.tensor(
        record.standardize_std, dtype=loaded["seq.0.std"].dtype, device=device
    )
    if not torch.equal(loaded["seq.0.mean"], expected_mean):
        raise RuntimeError("checkpoint StandardizeLayer mean differs from manifest")
    if not torch.equal(loaded["seq.0.std"], expected_std):
        raise RuntimeError("checkpoint StandardizeLayer std differs from manifest")
    return loaded


def load_frozen_model(
    architecture: str,
    seed: int,
    *,
    repository_root: Path = REPOSITORY_ROOT,
    manifest_path: Path = DEFAULT_MANIFEST,
    checkpoint_path: Path | None = None,
    device: torch.device | str = "cpu",
) -> FrozenModel:
    """Hash-check, strictly load, freeze, and switch one model to eval mode."""

    selected_device = torch.device(device)
    record = checkpoint_record(architecture, seed, manifest_path=manifest_path)
    path = (
        checkpoint_path
        if checkpoint_path is not None
        else repository_root / record.checkpoint_path
    )
    state_dict = _load_verified_state_dict(record, path, device=selected_device)

    # Dummy values are intentionally overwritten by strict state_dict loading.
    # This makes the checkpoint, rather than the evaluator or Ibiza data, the
    # sole source of inference normalization.
    factory = _MODEL_FACTORIES[architecture]
    model = factory([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], device=selected_device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    loaded = model.state_dict()
    if not torch.equal(loaded["seq.0.mean"], state_dict["seq.0.mean"]):
        raise RuntimeError("loaded model did not retain checkpoint normalization mean")
    if not torch.equal(loaded["seq.0.std"], state_dict["seq.0.std"]):
        raise RuntimeError("loaded model did not retain checkpoint normalization std")
    return FrozenModel(record=record, model=model)


def validate_checkpoint_inventory(
    *,
    repository_root: Path = REPOSITORY_ROOT,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> InventorySummary:
    """Verify counts, files, hashes, keys, and bit-identical normalization."""

    records = load_checkpoint_manifest(manifest_path)
    counts = Counter(record.architecture for record in records)
    if counts != Counter({architecture: 10 for architecture in ARCHITECTURES}):
        raise RuntimeError(f"checkpoint architecture counts are invalid: {dict(counts)}")
    pairs = {(record.architecture, record.seed) for record in records}
    expected_pairs = {
        (architecture, seed) for architecture in ARCHITECTURES for seed in range(10)
    }
    if pairs != expected_pairs or len(records) != len(pairs):
        raise RuntimeError("manifest must contain exactly seeds 0-9 once per architecture")
    hashes = {record.sha256 for record in records}
    if len(hashes) != 30:
        raise RuntimeError("all 30 checkpoint SHA-256 values must be distinct")

    manifest_paths = {record.checkpoint_path for record in records}
    disk_paths: set[str] = set()
    for architecture, relative_directory in _CHECKPOINT_DIRECTORIES.items():
        found = sorted((repository_root / relative_directory).glob("*.pth"))
        if len(found) != 10:
            raise RuntimeError(
                f"expected exactly 10 {architecture} checkpoints, found {len(found)}"
            )
        disk_paths.update(
            str(path.relative_to(repository_root).as_posix()) for path in found
        )
    if disk_paths != manifest_paths:
        raise RuntimeError("checkpoint files on disk do not exactly match the manifest")

    reference_mean: torch.Tensor | None = None
    reference_std: torch.Tensor | None = None
    for record in records:
        path = repository_root / record.checkpoint_path
        state = _load_verified_state_dict(record, path, device=torch.device("cpu"))
        mean = state["seq.0.mean"]
        std = state["seq.0.std"]
        if reference_mean is None:
            reference_mean = mean.clone()
            reference_std = std.clone()
        elif not torch.equal(mean, reference_mean):
            raise RuntimeError("the 30 embedded StandardizeLayer means are not identical")
        elif not torch.equal(std, reference_std):
            raise RuntimeError("the 30 embedded StandardizeLayer stds are not identical")

    if reference_mean is None or reference_std is None:  # Defensive; counts require 30.
        raise RuntimeError("checkpoint inventory is empty")
    return InventorySummary(
        records=records,
        architecture_counts=dict(counts),
        distinct_hash_count=len(hashes),
        common_mean=tuple(reference_mean.tolist()),
        common_std=tuple(reference_std.tolist()),
    )


__all__ = [
    "ARCHITECTURES",
    "CheckpointRecord",
    "DEFAULT_MANIFEST",
    "FrozenModel",
    "InventorySummary",
    "REPOSITORY_ROOT",
    "checkpoint_record",
    "load_checkpoint_manifest",
    "load_frozen_model",
    "sha256_file",
    "validate_checkpoint_inventory",
]
