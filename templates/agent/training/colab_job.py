import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


def run_command(command):
    environment = {**os.environ, "PYTHONUNBUFFERED": "1"}
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=environment,
    )
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
        if process.wait():
            raise RuntimeError(f"Command failed: {' '.join(map(str, command))}")
    except BaseException:
        process.terminate()
        process.wait()
        raise


def prepare_environment(
    environment=Path("/content/canvas-env"), *, python_version=None
):
    os.environ["MLX_CUDA_GRAPH_CACHE_SIZE"] = "4096"
    python = environment / "bin" / "python"
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("The uv installer is unavailable.")
    if not python.exists():
        print("Creating an isolated training environment.", flush=True)
        run_command(
            [uv, "venv", "--python", python_version or sys.executable, str(environment)]
        )
    run_command(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(python),
            "mlx-lm[train,cuda12]==0.31.3",
            "mlx==0.32.3",
            "pydantic>=2.12,<3",
        ]
    )
    packages = next(environment.glob("lib/python*/site-packages"))
    sys.path.insert(0, str(packages))
    return python


def main():
    os.environ["MLX_CUDA_GRAPH_CACHE_SIZE"] = "4096"
    root = Path("/content/canvas-training")
    root.mkdir(exist_ok=True)
    parts = sorted(Path("/content").glob("canvas-training.zip.part-*"))
    if parts:
        with Path("/content/canvas-training.zip").open("wb") as destination:
            for part in parts:
                with part.open("rb") as source:
                    shutil.copyfileobj(source, destination)
    with zipfile.ZipFile("/content/canvas-training.zip") as bundle:
        bundle.extractall(root)
    prepare_environment()
    # Train in the notebook kernel so Colab can observe the actual computation.
    sys.path.insert(0, str(root))
    train_job(root)


def train_job(root, *, output_dir=Path("/content")):
    import mlx.core as mx
    import yaml

    from lab import evaluate, evaluate_sessions, train_model

    if mx.default_device() != mx.gpu:
        raise RuntimeError("No MLX GPU backend is active; refusing CPU training.")
    weights = mx.ones((8, 8), dtype=mx.float16)
    value, gradient = mx.value_and_grad(lambda w: mx.mean(w @ w))(weights)
    mx.eval(value, gradient)
    print("CUDA forward and backward check passed.", flush=True)
    hardware = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    print(f"GPU: {hardware}", flush=True)
    config = yaml.safe_load((root / "config.yaml").read_text())
    # T4 Tensor Cores support FP16; BF16 matrix multiplication requires newer GPUs.
    config["model_dtype"] = "float16"
    config["cuda_graph_cache_size"] = int(os.environ["MLX_CUDA_GRAPH_CACHE_SIZE"])
    config["resume_adapter_file"] = str(root / "warm-start.safetensors")
    if (root / "resume.json").exists():
        resume = json.loads((root / "resume.json").read_text())
        config["resume_step"] = resume["step"]
        if (root / "resume" / "progress.json").exists():
            config["resume_training_state"] = str(root / "resume")
        print(
            f"Resuming retained update {resume['step']}; target {config['iters']}.",
            flush=True,
        )
    (root / "colab-config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    name = "colab-270m-v4-sessions"
    train_model(
        argparse.Namespace(run=name, config=root / "colab-config.yaml", iters=None),
        in_process=True,
    )
    run = root / "runs" / name
    (run / "hardware.json").write_text(
        json.dumps({"gpu": hardware, "backend": "cuda"}, indent=2)
    )
    mx.clear_cache()
    evaluate(
        argparse.Namespace(
            split="valid",
            adapter=run / "adapter",
            output=run / "valid.json",
            limit=None,
        )
    )
    evaluate_sessions(
        argparse.Namespace(
            split="valid",
            adapter=run / "adapter",
            output=run / "sessions-valid.json",
            limit=None,
        )
    )
    output = output_dir / "canvas-270m-v4-sessions-results.zip"
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in run.rglob("*"):
            if (
                path.is_file()
                and "checkpoints" not in path.parts
                and not re.fullmatch(r"\d+_adapters\.safetensors", path.name)
            ):
                archive.write(path, path.relative_to(root / "runs"))
        for path in root.glob("*.py"):
            archive.write(path, f"{name}/source/{path.name}")
    parts = []
    with output.open("rb") as source:
        while chunk := source.read(16 * 1024 * 1024):
            part = output.with_name(f"{output.name}.part-{len(parts):03d}")
            part.write_bytes(chunk)
            parts.append(
                {"name": part.name, "sha256": hashlib.sha256(chunk).hexdigest()}
            )
    output.with_suffix(".parts.json").write_text(json.dumps(parts))
    print(f"Training and validation complete. Download {output}", flush=True)


if __name__ == "__main__":
    main()
