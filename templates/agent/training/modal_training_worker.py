import argparse
import hashlib
import importlib.metadata
import json
import shutil
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path("/tmp/canvas-job")
MODEL = Path("/payload/model")


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_input():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Frozen input changed: {name}")
    for name, expected in manifest["model_files"].items():
        if digest(MODEL / name) != expected:
            raise ValueError(f"Pinned FP16 model changed: {name}")
    for package, version in (("mlx", "0.32.3"), ("mlx-lm", "0.31.3")):
        if importlib.metadata.version(package) != version:
            raise ValueError(f"Pinned library changed: {package}")
    return manifest


def adapter_folder(output, case, step, metadata):
    import yaml

    from gpu_benchmark import export_checkpoint

    trained = output / "refinement/dual-window"
    folder = output / "refinement" / case
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    baseline = "baseline" if (ROOT / "baseline.safetensors").exists() else "warm-start"
    if not (folder / "adapter/adapters.safetensors").exists():
        if case == "baseline":
            (folder / "adapter").mkdir(parents=True)
            shutil.copyfile(
                ROOT / f"{baseline}.safetensors",
                folder / "adapter/adapters.safetensors",
            )
            shutil.copyfile(
                ROOT / f"{baseline}-config.json",
                folder / "adapter/adapter_config.json",
            )
            (folder / "config.yaml").write_text(
                yaml.safe_dump({**config, "model_dtype": "float16"}, sort_keys=False)
            )
        else:
            if step not in config["checkpoint_selection_steps"]:
                raise ValueError("Only frozen selection checkpoints can be exported.")
            export_checkpoint(trained, folder, step, metadata)
    expected = (
        digest(ROOT / f"{baseline}.safetensors")
        if case == "baseline"
        else digest(trained / f"adapter/checkpoints/{step:07d}/adapters.safetensors")
    )
    if digest(folder / "adapter/adapters.safetensors") != expected:
        raise ValueError("Scored adapter differs from its retained checkpoint.")
    return folder


def score_pool(folder, pool, examples, sessions):
    import mlx.core as mx

    from gpu_benchmark import checkpoint_metrics
    from lab import evaluate, evaluate_sessions

    target = folder / pool
    target.mkdir(exist_ok=True)
    sha = digest(folder / "adapter/adapters.safetensors")
    for function, name in ((evaluate, "valid"), (evaluate_sessions, "sessions-valid")):
        path = target / f"{name}-guarded.json"
        function(
            argparse.Namespace(
                split="valid",
                adapter=folder / "adapter",
                output=path,
                limit=None,
                guarded=True,
                cache_prefix=True,
                batch_size=8 if function is evaluate else 1,
                example_ids=examples,
                session_ids=sessions,
            )
        )
        report = json.loads(path.read_text())
        actual = (
            {row["id"] for row in report["examples"]}
            if function is evaluate
            else {row["session_id"] for row in report["examples"]}
        )
        if actual != set(examples if function is evaluate else sessions):
            raise ValueError("Development reports omit requested cases.")
        report["adapter_sha256"] = sha
        path.write_text(json.dumps(report, indent=2))
        mx.clear_cache()
    metrics = checkpoint_metrics(target)
    session_report = json.loads((target / "sessions-valid-guarded.json").read_text())
    metrics.update(
        {
            key: session_report[key]
            for key in (
                "document_agreement_rate",
                "selection_agreement_rate",
                "camera_agreement_rate",
            )
        }
    )
    return metrics


def run(args):
    import mlx.core as mx
    import yaml

    shutil.copytree(Path("/job"), ROOT, dirs_exist_ok=True)
    sys.path.insert(0, str(ROOT))
    manifest = verify_input()
    if mx.default_device() != mx.gpu:
        raise RuntimeError("Refusing local or CPU training/evaluation.")
    import gpu_benchmark
    import lab

    # The verified image already contains the base; offline HF cache is unnecessary.
    lab.model_path = lambda dtype=None: str(MODEL)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((ROOT / "data/tokens.json").read_text())
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    with (output / f"{args.mode}-{args.case}.log").open("a") as log:
        with (
            redirect_stdout(lab._TrainingOutput(sys.stdout, log)),
            redirect_stderr(lab._TrainingOutput(sys.stderr, log)),
        ):
            if args.mode == "train":
                trained = output / "refinement/dual-window"
                if not (trained / "result.json").exists():
                    resume_step = 0
                    checkpoints = sorted((trained / "adapter/checkpoints").glob("*"))
                    ready = [
                        path
                        for path in checkpoints
                        if (path / "checkpoint-receipt.json").exists()
                    ]
                    if ready:
                        source = ready[-1]
                        receipt = json.loads(
                            (source / "checkpoint-receipt.json").read_text()
                        )
                        for name, sha in receipt["files"].items():
                            if digest(source / name) != sha:
                                raise ValueError(
                                    "Retained recovery checkpoint changed."
                                )
                        progress = json.loads((source / "progress.json").read_text())
                        if (
                            not isinstance(receipt["step"], int)
                            or not 0 < receipt["step"] <= config["quality_updates"]
                            or receipt["step"] != progress.get("step")
                            or source.name != f"{receipt['step']:07d}"
                        ):
                            raise ValueError(
                                "Recovery checkpoint step is inconsistent."
                            )
                        if (
                            any(
                                progress.get(key) != metadata[key]
                                for key in ("dataset_sha256", "task_sha256")
                            )
                            or progress.get("config_sha256")
                            != gpu_benchmark.quality_config_hash(config)
                            or progress.get("training_plan_sha256")
                            != metadata["files"]["train-plan.npz"]
                        ):
                            raise ValueError(
                                "Recovery differs from the frozen contract."
                            )
                        resume_step = receipt["step"]
                        if receipt["step"] == config["quality_updates"]:
                            arrays = mx.load(str(source / "adapters.safetensors"))
                            if not all(
                                bool(mx.all(mx.isfinite(value)).item())
                                for value in arrays.values()
                            ):
                                raise ValueError("Completed checkpoint is not finite.")
                            shutil.copyfile(
                                source / "adapters.safetensors",
                                trained / "adapter/adapters.safetensors",
                            )
                            (trained / "result.json").write_text(
                                json.dumps(
                                    {
                                        "end_step": receipt["step"],
                                        "training_target": receipt["step"],
                                        "training_finite": True,
                                        "recovered_completed_checkpoint": True,
                                        "dataset_sha256": metadata["dataset_sha256"],
                                        "task_sha256": metadata["task_sha256"],
                                    },
                                    indent=2,
                                )
                            )
                        else:
                            shutil.copytree(source, ROOT / "resume", dirs_exist_ok=True)
                    if not (trained / "result.json").exists():
                        previous = trained / "training-exposure.json"
                        if previous.exists():
                            shutil.copyfile(
                                previous,
                                trained / f"exposure-before-{resume_step}.json",
                            )
                        gpu_benchmark.worker(ROOT, trained, "baseline", quality=True)
                result = json.loads((trained / "result.json").read_text())
                if result["end_step"] != config["quality_updates"]:
                    raise ValueError("Training did not reach the frozen update target.")
                if not result.get("training_finite"):
                    raise ValueError("Training adapter is not finite.")
                shutil.copyfile(
                    ROOT / "planned-training-exposure.json",
                    trained / "completed-training-exposure.json",
                )
                result["mode"] = "train"
            else:
                folder = adapter_folder(output, args.case, args.step, metadata)
                if args.mode == "selection":
                    gpu_benchmark.evaluate_checkpoint_selection(folder)
                    result = gpu_benchmark.checkpoint_metrics(folder, selection=True)
                elif args.mode == "language":
                    identifiers = manifest["development_pools"].get("lexical")
                    if manifest.get("refinement_mode") != "lexical" or not identifiers:
                        raise ValueError(
                            "English scoring requires a frozen lexical pool."
                        )
                    result = score_pool(
                        folder,
                        "lexical",
                        identifiers["example_ids"],
                        identifiers["session_ids"],
                    )
                elif args.mode == "full":
                    result = {
                        pool: score_pool(
                            folder,
                            pool,
                            identifiers["example_ids"],
                            identifiers["session_ids"],
                        )
                        for pool, identifiers in manifest["development_pools"].items()
                    }
                else:
                    if args.case != "selected" or not args.native_passed:
                        raise ValueError(
                            "Fresh testing requires validation and native checks."
                        )
                    gpu_benchmark.score_selected_split(folder, "test")
                    result = {
                        "test_set_used": True,
                        "adapter_sha256": digest(
                            folder / "adapter/adapters.safetensors"
                        ),
                    }
                result = {
                    "mode": args.mode,
                    "case": args.case,
                    "step": args.step,
                    "metrics": result,
                }
    path = output / f"{args.mode}-{args.case}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {"task_complete": args.mode, "case": args.case, "report": str(path)}
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("train", "selection", "language", "full", "test"),
        required=True,
    )
    parser.add_argument("--case", default="baseline")
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--native-passed", action="store_true")
    run(parser.parse_args())
