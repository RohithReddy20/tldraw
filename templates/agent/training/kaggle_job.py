import argparse
import hashlib
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path


def extract_archive(bundle, root, digest):
    if hashlib.sha256(bundle.read_bytes()).hexdigest() != digest:
        raise ValueError("Kaggle input bundle checksum mismatch.")
    with zipfile.ZipFile(bundle) as archive:
        for name in archive.namelist():
            path = Path(name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Invalid path in training bundle.")
        archive.extractall(root)


def extract_bundle(bundle, root, digest):
    extract_archive(bundle, root, digest)
    if (root / "warm-start.json").exists():
        warm = json.loads((root / "warm-start.json").read_text())
        if (
            warm["initialization"] != "weights_only"
            or hashlib.sha256(
                (root / "warm-start.safetensors").read_bytes()
            ).hexdigest()
            != warm["adapter_sha256"]
        ):
            raise ValueError("Refinement starting weights failed verification.")
        if (root / "resume.json").exists():
            raise ValueError(
                "Refinement must not restore the previous dataset's optimizer."
            )
        print("Verified adapter weights for a new refinement run.", flush=True)
        return
    resume = json.loads((root / "resume.json").read_text())
    if (
        hashlib.sha256((root / "warm-start.safetensors").read_bytes()).hexdigest()
        != (resume["adapter_sha256"])
    ):
        raise ValueError("Retained adapter checksum mismatch.")
    for name in ("optimizer.safetensors", "random.safetensors", "progress.json"):
        if not (root / "resume" / name).is_file():
            raise ValueError(f"Missing retained training state: {name}")
    print(f"Verified training bundle and retained step {resume['step']}.", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument(
        "--task",
        choices=("train", "benchmark", "quality", "refine", "evaluate"),
        default=globals().get("TASK", "train"),
    )
    args = parser.parse_args()
    os.environ["MLX_CUDA_GRAPH_CACHE_SIZE"] = "4096"
    # Production resumes on one T4; experiments can measure both allocated devices.
    os.environ["CUDA_VISIBLE_DEVICES"] = (
        "0,1" if args.task in ("benchmark", "quality", "refine", "evaluate") else "0"
    )
    working = Path("/kaggle/working")
    root = working / "canvas-training"
    if args.train:
        overrides = globals().get("CONFIG_OVERRIDES", {})
        if overrides:
            import yaml

            if set(overrides) - {"refinement_micro_batch", "quality_timeout_seconds"}:
                raise ValueError("Unknown GPU profile override.")
            config = yaml.safe_load((root / "config.yaml").read_text())
            config.update(overrides)
            (root / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        if args.task == "benchmark":
            from gpu_benchmark import benchmark

            benchmark(root, working, cases=globals().get("BENCHMARK_CASES"))
        elif args.task == "quality":
            from gpu_benchmark import quality_experiment

            quality_experiment(root, working)
        elif args.task == "refine":
            from gpu_benchmark import refinement_experiment

            refinement_experiment(root, working)
        elif args.task == "evaluate":
            from gpu_benchmark import resume_refinement_evaluation

            bundles = list(Path("/kaggle/input").rglob("canvas-checkpoint.bundle"))
            if len(bundles) != 1:
                raise ValueError("Expected one mounted completed checkpoint bundle.")
            source = working / "recovered-checkpoint"
            source.mkdir(exist_ok=False)
            extract_archive(
                bundles[0], source, globals()["EVALUATION_CHECKPOINT_SHA256"]
            )
            resume_refinement_evaluation(
                root, working, globals()["EVALUATION_EXPECTED"], source=source
            )
        else:
            from colab_job import train_job

            train_job(root, output_dir=working)
        return
    bundles = list(Path("/kaggle/input").rglob("canvas-training.bundle"))
    if len(bundles) != 1:
        raise ValueError("Expected exactly one mounted canvas training bundle.")
    manifest = json.loads(bundles[0].with_name("bundle-manifest.json").read_text())
    root.mkdir(parents=True, exist_ok=False)
    extract_bundle(bundles[0], root, manifest["sha256"])
    # Freeze launcher fixes in the kernel so retries can reuse the large data upload.
    for filename, source in globals().get("TRAINING_SOURCES", {}).items():
        if Path(filename).name != filename or not filename.endswith(".py"):
            raise ValueError("Invalid frozen training source filename.")
        (root / filename).write_text(source)
    shutil.copyfile(Path(__file__), root / "kaggle_job.py")
    sys.path.insert(0, str(root))
    from colab_job import prepare_environment, run_command

    if shutil.which("uv") is None:
        run_command([sys.executable, "-m", "pip", "install", "uv"])
    python = prepare_environment(working / "canvas-env", python_version="3.12")
    system_libraries = os.environ.get("LD_LIBRARY_PATH", "")
    if globals().get("ENABLE_TORCH"):
        torch_env = working / "torch-env"
        torch_python = torch_env / "bin/python"
        run_command([shutil.which("uv"), "venv", "--python", "3.12", str(torch_env)])
        run_command(
            [
                shutil.which("uv"),
                "pip",
                "install",
                "--python",
                str(torch_python),
                "torch==2.8.0",
                "xformers==0.0.32.post2",
                "--index-url",
                "https://download.pytorch.org/whl/cu126",
            ]
        )
        run_command(
            [
                shutil.which("uv"),
                "pip",
                "install",
                "--python",
                str(torch_python),
                "numpy",
                "safetensors",
                "pyyaml",
            ]
        )
        torch_packages = next(torch_env.glob("lib/python*/site-packages"))
        os.environ["TORCH_PYTHON"] = str(torch_python)
        os.environ["TORCH_LIBRARY_PATH"] = ":".join(
            [
                str(torch_packages / "torch/lib"),
                *map(str, sorted((torch_packages / "nvidia").glob("*/lib"))),
                system_libraries,
            ]
        )
    packages = next((working / "canvas-env").glob("lib/python*/site-packages"))
    libraries = [packages / "mlx" / "lib", *sorted((packages / "nvidia").glob("*/lib"))]
    # Kaggle's system CUDA and Python hooks conflict with the isolated MLX environment.
    os.environ["LD_LIBRARY_PATH"] = ":".join(
        [*map(str, libraries), os.environ.get("LD_LIBRARY_PATH", "")]
    )
    os.environ.pop("LD_PRELOAD", None)
    os.environ.pop("PYTHONPATH", None)
    run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    )
    try:
        run_command(
            [
                str(python),
                "-u",
                str(root / "kaggle_job.py"),
                "--train",
                "--task",
                args.task,
            ]
        )
    finally:
        # Kaggle retains outputs, so remove the reproducible dependency cache.
        shutil.rmtree(working / "canvas-env", ignore_errors=True)
        shutil.rmtree(working / "torch-env", ignore_errors=True)
    print("Kaggle training and validation outputs are ready.", flush=True)


if __name__ == "__main__":
    main()
