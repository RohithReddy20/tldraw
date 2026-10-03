import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

TRAINING = Path(__file__).resolve().parent
HERE = TRAINING / "runs/quality-next/modal"
FROZEN = TRAINING / "runs/v7-accuracy-workflow-revised"
BASE = TRAINING / "runs/base-bb327a9a-float16"
WARM = (
    TRAINING
    / "runs/kaggle/spoken/download/refinement/dual-window/adapter/adapters.safetensors"
)
SOURCE_NAMES = ("training.py", "gpu_benchmark.py", "config.yaml")


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def prepare():
    receipt = json.loads((FROZEN / "source-receipt.json").read_text())
    token_receipt = json.loads((FROZEN / "data/tokens.json").read_text())
    source = FROZEN / "data/tokens-train.npz"
    if digest(source) != token_receipt["files"][source.name]:
        raise ValueError("Frozen training cache changed.")
    payload = HERE / "payload"
    payload.mkdir(parents=True, exist_ok=True)
    for name in SOURCE_NAMES:
        if digest(FROZEN / name) != receipt["sha256"][name]:
            raise ValueError(f"Frozen source changed: {name}")
        shutil.copyfile(FROZEN / name, payload / name)
    if (
        digest(WARM)
        != "08593b33c249f6a4fc0394d6365d034c3665fb325b6a47f6fe0f4d0c069deeb3"
    ):
        raise ValueError("The retained v6 warm-start adapter changed.")
    shutil.copyfile(WARM, payload / "warm-start.safetensors")
    with np.load(source) as data:
        tokens, boundaries, offsets = [
            data[k] for k in ("tokens", "boundaries", "offsets")
        ]
    sizes = np.diff(boundaries)
    groups = np.argsort(sizes, kind="stable").reshape(-1, 8)
    quantiles = np.linspace(0.05, 1.0, 8)
    selected = [groups[min(len(groups) - 1, int(q * len(groups)))] for q in quantiles]
    arrays, batches = {}, []
    for number, indices in enumerate(selected):
        lengths = sizes[indices]
        width = 1 + 32 * ((int(max(lengths)) + 31) // 32)
        batch = np.zeros((8, width), dtype=np.int32)
        for row, index in enumerate(indices):
            begin, end = boundaries[index : index + 2]
            batch[row, : end - begin] = tokens[begin:end]
        bounds = np.column_stack((offsets[indices], lengths))
        if not np.all((bounds[:, 0] > 0) & (bounds[:, 0] < bounds[:, 1])):
            raise ValueError("Invalid completion bounds.")
        if int(np.max(bounds[:, 1] - bounds[:, 0])) > 160:
            raise ValueError("The frozen completion window truncates targets.")
        arrays[f"batch_{number}"] = batch
        arrays[f"bounds_{number}"] = bounds
        batches.append(
            {
                "quantile": float(quantiles[number]),
                "train_indices": indices.tolist(),
                "padded_width": width,
                "input_tokens": int(bounds[:, 0].sum()),
                "completion_tokens": int((bounds[:, 1] - bounds[:, 0]).sum()),
            }
        )
    np.savez_compressed(payload / "representative.npz", **arrays)
    model_files = {p.name: digest(p) for p in sorted(BASE.iterdir()) if p.is_file()}
    if "model.safetensors" not in model_files or "config.json" not in model_files:
        raise ValueError("The retained converted FP16 base is unavailable.")
    manifest = {
        "source_commit": receipt["source_commit"],
        "dataset_sha256": token_receipt["dataset_sha256"],
        "task_sha256": token_receipt["task_sha256"],
        "training_cache_sha256": token_receipt["files"][source.name],
        "model": "mlx-community/functiongemma-270m-it-bf16",
        "model_revision": "bb327a9ad61044e1496a2bee2365a6b6a6684c72",
        "local_converted_dtype": "float16",
        "source_sha256": {name: receipt["sha256"][name] for name in SOURCE_NAMES},
        "files": {
            p.name: digest(p)
            for p in sorted(payload.iterdir())
            if p.is_file() and p.name != "manifest.json"
        },
        "model_files": model_files,
        "training_examples": len(sizes),
        "unique_sampled_examples": len(set(np.concatenate(selected).tolist())),
        "sampled_batches": batches,
        "selection": (
            "Eight train-only length quantiles; last includes the longest group. "
            "This is a throughput probe, not a full-epoch estimate "
            "or accuracy evaluation."
        ),
        "holds_out_all_development_and_test_data": True,
    }
    (payload / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "payload": str(payload),
                "batches": len(batches),
                "sampled_examples": manifest["unique_sampled_examples"],
                "payload_bytes": sum(
                    p.stat().st_size for p in payload.iterdir() if p.is_file()
                ),
                "model_bytes": sum(
                    p.stat().st_size for p in BASE.iterdir() if p.is_file()
                ),
                "cloud_calls": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    argparse.ArgumentParser(
        description="Prepare local frozen train data without network calls."
    ).parse_args()
    prepare()
