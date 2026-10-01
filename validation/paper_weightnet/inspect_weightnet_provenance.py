#!/usr/bin/env python3
"""Verify and record the exact dd5eac6 WeightNet training provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


COMMIT = "dd5eac669676ba0a922102047e58c2dfc9be9267"
EXPECTED_HASHES = {
    "model.py": "12bab5692b899e07681c5b130aaf69b54334328d3689426db1187b8594e9a2d2",
    "weight_network_train.py": "ee7cf1f8d8ded71845a2d97c7f303bfa3f6231c8babb4f78af2447a0f9202cd5",
    "weight_network_predict.py": "d7bd7e539f69658e933ab11f2281c188ac2a2ee0cdb938b1616fd75f50e0efb5",
    "rtk_util.py": "25fa8e26ff5772960e0dd33763950868aaaf1d181ebae25b3b93051e7511692c",
    "config/weight/klt3_train.json": "66d205340529731d5ccce75375e391a582628b6f629442aa285f6bb60bca3ff3",
}


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tdl-repository",
        type=Path,
        default=here.parents[2] / "external_references/TDL-GNSS",
    )
    parser.add_argument("--output", type=Path, default=here / "weightnet_provenance.json")
    return parser.parse_args()


def git_show(repository: Path, path: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(repository), "show", f"{COMMIT}:{path}"],
        check=True,
        stdout=subprocess.PIPE,
    )
    return completed.stdout


def require_fragments(text: str, fragments: list[str], source: str) -> None:
    missing = [fragment for fragment in fragments if fragment not in text]
    if missing:
        raise RuntimeError(f"{source} no longer matches audited fragments: {missing}")


def main() -> int:
    args = parse_args()
    repository = args.tdl_repository.resolve()
    sources: dict[str, str] = {}
    source_records: dict[str, dict[str, object]] = {}
    for path, expected_hash in EXPECTED_HASHES.items():
        content = git_show(repository, path)
        actual_hash = hashlib.sha256(content).hexdigest()
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"historical source hash mismatch for {path}: "
                f"expected {expected_hash}, got {actual_hash}"
            )
        sources[path] = content.decode("utf-8")
        source_records[path] = {
            "sha256": actual_hash,
            "size_bytes": len(content),
        }

    require_fragments(
        sources["model.py"],
        [
            "class WeightNet(nn.Module):",
            "StandardizeLayer(imean, istd)",
            "nn.Linear(3, 64)",
            "nn.Linear(64,128)",
            "nn.Linear(128, 64)",
            "nn.Linear(64,1)",
            "self.seq(x) * 10",
            "torch.clamp(x,min=0,max = 10)",
        ],
        "model.py",
    )
    require_fragments(
        sources["weight_network_train.py"],
        [
            "DEVICE = 'cuda'",
            "norm_data.mean(axis=0)",
            "norm_data.std(axis=0)",
            "net.double()",
            "torch.optim.Adam(net.parameters(),lr = 0.01)",
            "epoch = conf.get('epoch',500)",
            "batch = conf.get('batch',128)",
            "lossFn = MSELoss(reduction='sum')",
            "epoch_loss = torch.norm(torch.hstack(enu[:3]))",
            "loss += epoch_loss",
            "opt.zero_grad()",
            "loss.backward()",
            "opt.step()",
            'torch.save(net.state_dict(),conf[\'model\']+"/weightnet_3d.pth")',
        ],
        "weight_network_train.py",
    )
    require_fragments(
        sources["rtk_util.py"],
        [
            "SNR.append(obsd.SNR[0]/1e3)",
            "p = np.array([0,0,0,0,0,0,0],dtype=np.float64)",
            "p[sysinfo] = p[sysinfo]+dp.squeeze()",
            "t3 = torch.matmul(torch.inverse(t2),H.T)",
        ],
        "rtk_util.py",
    )
    configuration = json.loads(sources["config/weight/klt3_train.json"])
    if configuration != {
        "obs": ["data/0610_KLT/COM38_210610_025603.obs"],
        "eph": "data/0610_KLT/sta/hksc161d.21*",
        "gt": "data/0610_KLT/20210610_100.txt",
        "start_time": 1623297151,
        "end_time": 1623297556,
        "model": "model/weight",
        "mode": "train",
        "epoch": 500,
    }:
        raise RuntimeError("historical KLT3 configuration differs from the audit")

    report = {
        "status": "passed",
        "tdl_gnss_commit": COMMIT,
        "historical_sources": source_records,
        "weightnet": {
            "source": "model.py",
            "class": "WeightNet",
            "architecture": [3, 64, 128, 64, 1],
            "activation_after_each_linear": "Sigmoid",
            "output_transformation": "sigmoid output multiplied by 10 then clamped to [0,10]",
            "normalization": "population mean/std over all retained KLT3 feature rows",
            "features": [
                "SNR[0]/1000",
                "elevation radians",
                "equal-weight OLS residual metres",
            ],
        },
        "training": {
            "executed_loss": "sum of per-epoch 3D ENU Euclidean norms",
            "imported_but_unused_loss": "MSELoss(reduction='sum')",
            "optimizer": "Adam",
            "learning_rate": 0.01,
            "epochs": 500,
            "random_seed": None,
            "model_initialization": "PyTorch Linear default, model constructed float32 then converted to float64",
            "iteration_order": "prl.sortobs then chronological split_obs traversal",
            "shuffle": False,
            "batching": "one accumulated full-dataset update per epoch; read batch=128 is unused",
            "initializer": "per-epoch equal-weight get_ls_pnt_pos result",
            "ground_truth": "nearest +18-second-aligned row, used in 3D position loss only",
            "checkpoint": "<conf.model>/weightnet_3d.pth state_dict",
            "device": "hard-coded CUDA",
            "state": "[x,y,z,b_GPS,b_BDS,b_Galileo,b_GLONASS] with active columns",
        },
        "paper_vs_code": {
            "architecture": "paper 3-64-128-1; code 3-64-128-64-1",
            "loss": "paper MSE/half squared error; code 3D Euclidean norm sum",
            "learning_rate": "paper 0.001; code 0.01",
            "state": "paper four-state clock-time form; code seven-slot metre-clock form",
            "weight_range": "paper final sigmoid; code sigmoid*10 clamped [0,10]",
        },
        "klt3_configuration": configuration,
        "primary_target": "released dd5eac6 computational implementation",
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"WeightNet provenance audit passed: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
