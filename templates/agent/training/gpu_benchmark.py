import argparse
import csv
import hashlib
import json
import os
import shutil
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

CASES = {
    "baseline": {"packed": False, "checkpoint": "all", "workers": 1, "micro": 8},
    "packed": {"packed": True, "checkpoint": "all", "workers": 1, "micro": 8},
    "attention": {"packed": True, "checkpoint": "attention", "workers": 1, "micro": 8},
    "half": {"packed": True, "checkpoint": "half", "workers": 1, "micro": 8},
    "micro4": {"packed": True, "checkpoint": "none", "workers": 1, "micro": 4},
    "dual": {"packed": True, "checkpoint": "all", "workers": 2, "micro": 4},
    "dual-attention": {
        "packed": True,
        "checkpoint": "attention",
        "workers": 2,
        "micro": 4,
    },
    "dual-none": {"packed": True, "checkpoint": "none", "workers": 2, "micro": 4},
    "baseline-repeat": {
        "packed": False,
        "checkpoint": "all",
        "workers": 1,
        "micro": 8,
    },
    "dual-window": {
        "packed": False,
        "checkpoint": "all",
        "workers": 2,
        "micro": 4,
    },
    "torch": {"backend": "torch", "checkpoint": "none", "workers": 1},
    "torch-checkpoint": {"backend": "torch", "checkpoint": "all", "workers": 1},
    "torch-dual": {"backend": "torch", "checkpoint": "none", "workers": 2},
    "torch-compiled": {
        "backend": "torch",
        "checkpoint": "all",
        "workers": 1,
        "compile": True,
    },
    "reference-fp32": {
        "packed": False,
        "checkpoint": "all",
        "workers": 1,
        "micro": 8,
        "dtype": "float32",
        "check_only": True,
        "reference": "reference-fp32",
    },
    "dual-fp32": {
        "packed": False,
        "checkpoint": "all",
        "workers": 2,
        "micro": 4,
        "dtype": "float32",
        "reference": "reference-fp32",
    },
    "torch-fp32": {
        "backend": "torch",
        "checkpoint": "none",
        "workers": 1,
        "dtype": "float32",
        "reference": "reference-fp32",
    },
    "torch-fp32-checkpoint": {
        "backend": "torch",
        "checkpoint": "all",
        "workers": 1,
        "dtype": "float32",
        "reference": "reference-fp32",
    },
    "torch-fp32-dual": {
        "backend": "torch",
        "checkpoint": "none",
        "workers": 2,
        "dtype": "float32",
        "reference": "reference-fp32",
    },
    "xformers-dual": {
        "backend": "torch",
        "checkpoint": "none",
        "workers": 2,
        "attention": "xformers",
    },
    "xformers-fp32-dual": {
        "backend": "torch",
        "checkpoint": "none",
        "workers": 2,
        "attention": "xformers",
        "dtype": "float32",
        "reference": "reference-fp32",
    },
    "xformers-fp32-compiled": {
        "backend": "torch",
        "checkpoint": "none",
        "workers": 2,
        "attention": "xformers",
        "dtype": "float32",
        "reference": "reference-fp32",
        "compile": True,
    },
}


def representative_batches(path, verify=False):
    import numpy as np

    with np.load(path) as arrays:
        tokens, boundaries, offsets = [
            arrays[name] for name in ("tokens", "boundaries", "offsets")
        ]
    sizes = np.diff(boundaries)
    order = np.argsort(sizes, kind="stable")
    quantiles = np.linspace(0.05, 1, 8) if verify else (0.5, 1)
    batches = []
    for quantile in quantiles:
        end = max(8, int(len(order) * quantile))
        indices = order[end - 8 : end]
        lengths = sizes[indices]
        width = 1 + 32 * ((int(max(lengths)) + 31) // 32)
        batch = np.zeros((8, width), dtype=np.int32)
        for row, index in enumerate(indices):
            begin, finish = boundaries[index : index + 2]
            batch[row, : finish - begin] = tokens[begin:finish]
        batches.append((batch, np.column_stack((offsets[indices], lengths))))
    return batches


def quality_batches(path, seed, resume_step, updates=500):
    import numpy as np

    with np.load(path) as arrays:
        tokens, boundaries, offsets = [
            arrays[name] for name in ("tokens", "boundaries", "offsets")
        ]
    sizes = np.diff(boundaries)
    order = np.argsort(sizes, kind="stable")
    groups = order[: len(order) // 8 * 8].reshape(-1, 8)
    random = np.random.RandomState(seed)
    batches, selected = [], []
    step = 0
    while len(batches) < updates:
        for group in random.permutation(len(groups)):
            if step >= resume_step:
                indices = groups[group]
                lengths = sizes[indices]
                width = 1 + 32 * ((int(max(lengths)) + 31) // 32)
                batch = np.zeros((8, width), dtype=np.int32)
                for row, index in enumerate(indices):
                    begin, end = boundaries[index : index + 2]
                    batch[row, : end - begin] = tokens[begin:end]
                batches.append((batch, np.column_stack((offsets[indices], lengths))))
                selected.extend(map(int, indices))
                if len(batches) == updates:
                    break
            step += 1
    digest = hashlib.sha256(np.asarray(selected, dtype="<i8").tobytes()).hexdigest()
    return batches, digest


def checkpoint_layers(model, mode):
    from mlx_lm.tuner.trainer import grad_checkpoint

    if mode == "all":
        grad_checkpoint(model.layers[0])
    elif mode == "attention":
        grad_checkpoint(model.layers[0].self_attn)
    elif mode == "half":
        layer_type = type(model.layers[0])
        original = layer_type.__call__
        grad_checkpoint(model.layers[0])
        checkpointed = layer_type.__call__
        selected = {id(layer) for layer in model.layers[::2]}

        def forward(layer, *args, **kwargs):
            call = checkpointed if id(layer) in selected else original
            return call(layer, *args, **kwargs)

        layer_type.__call__ = forward


def worker(root, output, case, verify=False, quality=False):
    if CASES[case].get("backend") == "torch":
        from torch_benchmark import worker as torch_worker

        return torch_worker(root, output, case, verify)

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import numpy as np
    import yaml
    from mlx.nn.utils import average_gradients
    from mlx.utils import tree_flatten, tree_map, tree_unflatten
    from mlx_lm import load
    from mlx_lm.tuner.callbacks import TrainingCallback
    from mlx_lm.tuner.trainer import TrainingArgs, train
    from mlx_lm.tuner.utils import build_schedule, linear_to_lora_layers
    from mlx_lm.utils import save_config

    from lab import model_path
    from training import (
        completion_indices,
        completion_loss,
        packed_completion_loss,
        restore_random_state,
        save_training_checkpoint,
    )

    settings = CASES[case]
    world = mx.distributed.init(
        strict=settings["workers"] > 1,
        backend="nccl" if settings["workers"] > 1 else "any",
    )
    if mx.default_device() != mx.gpu or world.size() != settings["workers"]:
        raise RuntimeError("The requested CUDA worker count is unavailable.")
    rank, workers = world.rank(), world.size()
    if workers > 1:
        original_all_sum = mx.distributed.all_sum

        def gpu_all_sum(*args, **kwargs):
            # MLX-LM's CPU reporting stream cannot be used by the NCCL backend.
            if kwargs.get("stream") is mx.cpu:
                kwargs["stream"] = mx.gpu
            return original_all_sum(*args, **kwargs)

        mx.distributed.all_sum = gpu_all_sum
    accumulation = 8 // (settings["micro"] * workers)
    config = yaml.safe_load((root / "config.yaml").read_text())
    model, _ = load(model_path("float16"))
    model.freeze()
    linear_to_lora_layers(model, config["num_layers"], config["lora_parameters"])
    model.load_weights(str(root / "warm-start.safetensors"), strict=False)
    if settings.get("dtype") == "float32":
        model.set_dtype(mx.float32)
    warm_start = (root / "warm-start.json").exists()
    if warm_start:
        np.random.seed(config["seed"])
        mx.random.seed(config["seed"])
    else:
        restore_random_state(mx.load(str(root / "resume/random.safetensors"))["key"])
    checkpoint_layers(model, settings["checkpoint"])
    if quality:
        if warm_start:
            resume = {
                **json.loads((root / "data/tokens.json").read_text()),
                "step": 0,
                "schedule_offset": 0,
            }
        else:
            resume = json.loads((root / "resume/progress.json").read_text())
        raw_batches, batch_digest = quality_batches(
            root / "data/tokens-train.npz",
            config["seed"],
            resume["step"],
            updates=config.get("quality_updates", 500),
        )
    else:
        raw_batches = representative_batches(root / "data/tokens-train.npz", verify)
    output.mkdir(parents=True, exist_ok=True)
    batches = []
    for raw, bounds in raw_batches:
        total = int((bounds[:, 1] - bounds[:, 0]).sum())
        group = []
        for start in range(0, 8, settings["micro"] * workers):
            rows = np.arange(start + rank, start + settings["micro"] * workers, workers)
            batch, lengths = raw[rows], bounds[rows]
            count = int((lengths[:, 1] - lengths[:, 0]).sum())
            # Averaging local means would give short answers too much weight.
            weight = mx.array(workers * accumulation * count / total, mx.float32)
            if settings["packed"]:
                indices, mask = completion_indices(lengths, batch.shape[1])
                group.append(
                    (mx.array(batch), mx.array(indices), mx.array(mask), weight)
                )
            else:
                group.append((mx.array(batch), mx.array(lengths, mx.int32), weight))
        batches.append(group)

    def loss(model, *batch):
        data, weight = batch[:-1], batch[-1]
        value, count = (
            packed_completion_loss(model, *data)
            if settings["packed"]
            else completion_loss(model, *data, 96)
        )
        return value * weight, count

    check_started = time.time()
    value_and_grad = nn.value_and_grad(model, loss)
    gradients = None
    initial_loss = 0
    for batch in batches[-1]:
        (value, _), grad = value_and_grad(model, *batch)
        gradients = (
            grad if gradients is None else tree_map(lambda x, y: x + y, gradients, grad)
        )
        initial_loss += value
        mx.eval(initial_loss, gradients)
    gradients = average_gradients(gradients)
    gradients = tree_map(lambda value: value / accumulation, gradients)
    initial_loss = mx.distributed.all_sum(initial_loss) / (workers * accumulation)
    mx.eval(initial_loss, gradients)
    initial_loss = initial_loss.item()
    if rank == 0:
        mx.save_safetensors(
            str(output / "gradients.safetensors"), dict(tree_flatten(gradients))
        )
    if settings.get("check_only"):
        result = {
            "case": case,
            "settings": settings,
            "verification": verify,
            "initial_loss": initial_loss,
            "updates_per_second": 0,
            "steady_started_at": check_started,
            "finished_at": time.time(),
            "note": "Gradient reference only; no optimizer updates measured.",
        }
        (output / "result.json").write_text(json.dumps(result, indent=2))
        return
    del gradients, grad, value
    mx.clear_cache()
    mx.reset_peak_memory()
    if quality:
        schedule = build_schedule(config["lr_schedule"])
        optimizer = optim.Adam(
            learning_rate=lambda step: schedule(step + resume["schedule_offset"])
        )
        if not warm_start:
            optimizer.state = tree_unflatten(
                list(mx.load(str(root / "resume/optimizer.safetensors")).items())
            )
            restored_step = int(optimizer.state["step"].item())
            if restored_step + resume["schedule_offset"] != resume["step"]:
                raise ValueError("Optimizer state does not match the retained update.")
        adapter_file = output / "adapter/adapters.safetensors"
        adapter_file.parent.mkdir(exist_ok=True)
        if rank == 0:
            save_config(config, adapter_file.parent / "adapter_config.json")
            (output / "config.yaml").write_text(
                yaml.safe_dump({**config, "model_dtype": "float16"}, sort_keys=False)
            )
            if warm_start:
                (output / "data").mkdir(exist_ok=True)
                shutil.copyfile(root / "data/tokens.json", output / "data/tokens.json")
    else:
        optimizer = optim.Adam(learning_rate=3e-5)
        adapter_file = output / "benchmark-adapters.safetensors"
    reports = []
    updates, interval, warmup = (
        (config.get("quality_updates", 500), 50, 50)
        if quality
        else ((80, 8, 16) if verify else (24, 4, 12))
    )
    steady_started = None

    class Reports(TrainingCallback):
        def on_train_loss_report(self, info):
            nonlocal steady_started
            reports.append(info)
            if info["iteration"] == warmup * accumulation:
                steady_started = time.time()
            if warm_start and rank == 0 and info["iteration"] % 500 == 0:
                save_training_checkpoint(
                    model,
                    optimizer,
                    adapter_file.parent,
                    info["iteration"] // accumulation,
                    0,
                    archive=False,
                )

    def iterator(**kwargs):
        while True:
            for group in batches:
                yield from group

    started = time.monotonic()
    train(
        model,
        optimizer,
        batches,
        args=TrainingArgs(
            batch_size=settings["micro"] * workers,
            grad_accumulation_steps=accumulation,
            iters=updates * accumulation,
            steps_per_report=interval * accumulation,
            steps_per_save=10000,
            adapter_file=str(adapter_file),
            max_seq_length=2048,
        ),
        loss=loss,
        iterate_batches=iterator,
        training_callback=Reports(),
    )
    if rank != 0:
        return
    steady = [
        report["iterations_per_second"] / accumulation
        for report in reports
        if report["iteration"] > warmup * accumulation
    ]
    rate = statistics.median(steady)
    result = {
        "case": case,
        "settings": settings,
        "verification": verify,
        "global_batch": 8,
        "updates": updates,
        "batch_shapes": [list(batch.shape) for batch, _ in raw_batches],
        "initial_loss": initial_loss,
        "updates_per_second": rate,
        "examples_per_second": 8 * rate,
        "min_updates_per_second": min(steady),
        "max_updates_per_second": max(steady),
        "peak_memory_gb_per_worker": mx.get_peak_memory() / 1e9,
        "training_seconds_including_warmup": time.monotonic() - started,
        "steady_started_at": steady_started,
        "finished_at": time.time(),
        "reports": reports,
    }
    if quality:
        result.update(
            start_step=resume["step"],
            end_step=resume["step"] + updates,
            training_batch_sha256=batch_digest,
            dataset_sha256=resume["dataset_sha256"],
            task_sha256=resume["task_sha256"],
            warm_start_sha256=hashlib.sha256(
                (root / "warm-start.safetensors").read_bytes()
            ).hexdigest(),
            optimizer_sha256=None
            if warm_start
            else hashlib.sha256(
                (root / "resume/optimizer.safetensors").read_bytes()
            ).hexdigest(),
            model_revision="bb327a9ad61044e1496a2bee2365a6b6a6684c72",
            initialization="weights_only" if warm_start else "exact_resume",
            warm_start_source_step=config.get("warm_start_source_step"),
        )
    (output / "result.json").write_text(json.dumps(result, indent=2))
    print("RESULT " + json.dumps(result), flush=True)


def compare(reference, candidate):
    import numpy as np
    from safetensors.numpy import load_file

    original = load_file(str(reference / "gradients.safetensors"))
    actual = load_file(str(candidate / "gradients.safetensors"))
    if original.keys() != actual.keys():
        raise ValueError("Benchmark gradient parameters do not match.")
    squared_error = squared_norm = candidate_norm = dot = max_error = 0.0
    for name, values in original.items():
        delta = actual[name].astype(np.float64) - values.astype(np.float64)
        squared_error += float(np.square(delta).sum())
        squared_norm += float(np.square(values.astype(np.float64)).sum())
        candidate_norm += float(np.square(actual[name].astype(np.float64)).sum())
        dot += float(
            (actual[name].astype(np.float64) * values.astype(np.float64)).sum()
        )
        max_error = max(max_error, float(np.abs(delta).max()))
    result = json.loads((candidate / "result.json").read_text())
    original_loss = json.loads((reference / "result.json").read_text())["initial_loss"]
    result["gradient_relative_l2_error"] = (
        squared_error / max(squared_norm, 1e-30)
    ) ** 0.5
    result["gradient_max_absolute_error"] = max_error
    result["gradient_norm_ratio"] = (candidate_norm / max(squared_norm, 1e-30)) ** 0.5
    result["gradient_cosine"] = dot / max((candidate_norm * squared_norm) ** 0.5, 1e-30)
    result["loss_absolute_error"] = abs(result["initial_loss"] - original_loss)
    result["training_finite"] = all(
        np.isfinite(report["train_loss"]) for report in result.get("reports", [])
    )
    adapter = candidate / "benchmark-adapters.safetensors"
    if not result["settings"].get("check_only"):
        result["training_finite"] = result["training_finite"] and all(
            np.isfinite(value).all() for value in load_file(str(adapter)).values()
        )
    result["training_finite"] = bool(result["training_finite"])
    result["gradient_check_passed"] = (
        result["gradient_relative_l2_error"] < 0.01
        and result["loss_absolute_error"] < 1e-5
        and result["training_finite"]
    )
    result["gpu_samples"] = summarize_gpu_samples(candidate, result)
    return result


def summarize_gpu_samples(folder, result):
    utilization = {}
    for row in csv.reader((folder / "gpu.csv").open()):
        try:
            timestamp = datetime.strptime(
                row[0].strip(), "%Y/%m/%d %H:%M:%S.%f"
            ).timestamp()
            if not result["steady_started_at"] <= timestamp <= result["finished_at"]:
                continue
            device = row[1].strip()
            utilization.setdefault(device, []).append(
                [float(value.strip()) for value in row[2:]]
            )
        except (ValueError, IndexError):
            continue
    return {
        device: {
            "samples": len(samples),
            "mean_utilization_percent": statistics.mean(
                sample[0] for sample in samples
            ),
            "mean_memory_utilization_percent": statistics.mean(
                sample[1] for sample in samples
            ),
            "max_memory_mib": max(sample[2] for sample in samples),
            "mean_power_watts": statistics.mean(sample[3] for sample in samples),
            "power_limit_watts": samples[0][4],
            "mean_sm_clock_mhz": statistics.mean(sample[5] for sample in samples),
        }
        for device, samples in utilization.items()
    }


def run_case(root, output, case, verify=False, quality=False):
    folder = output / (("verify-" if verify else "") + case)
    reference = output / CASES[case].get(
        "reference", "verify-baseline" if verify else "baseline"
    )
    if (
        not quality
        and reference != folder
        and not (reference / "result.json").is_file()
    ):
        return {
            "case": case,
            "verification": verify,
            "error": f"Required gradient reference is unavailable: {reference.name}",
        }
    folder.mkdir(parents=True)
    torch_case = CASES[case].get("backend") == "torch"
    command = [
        os.environ["TORCH_PYTHON"] if torch_case else sys.executable,
        "-u",
        str(root / "gpu_benchmark.py"),
        "--root",
        str(root),
        "--output",
        str(folder),
        "--case",
        case,
    ]
    if verify:
        command.append("--verify")
    if quality:
        command.append("--quality")
    timeout = (
        1800 if quality else (600 if verify or CASES[case].get("compile") else 360)
    )
    print(f"Benchmarking {folder.name}; maximum {timeout} seconds.", flush=True)
    workers = CASES[case]["workers"]
    handles, processes = [], []
    query = (
        "timestamp,index,utilization.gpu,utilization.memory,memory.used,"
        "power.draw,power.limit,clocks.sm"
    )
    gpu_log = (folder / "gpu.csv").open("w")
    sampler = subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=" + query,
            "--format=csv,noheader,nounits",
            "-l",
            "1",
        ],
        stdout=gpu_log,
        stderr=subprocess.STDOUT,
    )
    deadline = time.monotonic() + timeout
    error = None
    next_report = time.monotonic() + 30
    try:
        for rank in range(workers):
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = str(rank)
            if torch_case:
                environment["LD_LIBRARY_PATH"] = environment["TORCH_LIBRARY_PATH"]
                environment["TORCHINDUCTOR_COMPILE_THREADS"] = "2"
            if workers > 1:
                environment.update(
                    MLX_RANK=str(rank),
                    MLX_WORLD_SIZE=str(workers),
                    NCCL_HOST_IP="127.0.0.1",
                    NCCL_PORT="29452",
                    MLX_NCCL_TIMEOUT="30000",
                    RANK=str(rank),
                    WORLD_SIZE=str(workers),
                    LOCAL_RANK="0",
                    MASTER_ADDR="127.0.0.1",
                    MASTER_PORT="29453",
                )
            log = (folder / f"worker-{rank}.log").open("w")
            handles.append(log)
            processes.append(
                subprocess.Popen(
                    command,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
        while any(process.poll() is None for process in processes):
            failed = [
                process.returncode
                for process in processes
                if process.poll() not in (0, None)
            ]
            if failed or time.monotonic() > deadline:
                error = f"Worker failed: {failed}" if failed else "Case timed out"
                break
            if quality and time.monotonic() >= next_report:
                lines = (folder / "worker-0.log").read_text().splitlines()
                if lines:
                    print(f"QUALITY PROGRESS {case}: {lines[-1]}", flush=True)
                next_report = time.monotonic() + 30
            time.sleep(0.5)
        if any(process.returncode not in (0, None) for process in processes):
            error = error or "Worker failed"
    finally:
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        for log in handles:
            log.close()
        sampler.terminate()
        sampler.wait()
        gpu_log.close()
    logs = "\n".join(
        (folder / f"worker-{rank}.log").read_text()[-4500:] for rank in range(workers)
    )
    print(logs, flush=True)
    if error:
        return {"case": case, "verification": verify, "error": error, "log_tail": logs}
    if quality:
        import numpy as np
        from safetensors.numpy import load_file

        result = json.loads((folder / "result.json").read_text())
        result["training_finite"] = bool(
            all(np.isfinite(report["train_loss"]) for report in result["reports"])
            and all(
                np.isfinite(value).all()
                for value in load_file(
                    str(folder / "adapter/adapters.safetensors")
                ).values()
            )
        )
        if not result["training_finite"]:
            raise ValueError(f"Non-finite training result: {case}")
        result["gpu_samples"] = summarize_gpu_samples(folder, result)
        return result
    return compare(reference, folder)


def quality_comparison(baseline, candidate, *, guarded=False):
    from sessions import summarize_sessions

    result = {}
    for name in ("valid", "sessions-valid"):
        suffix = "-guarded" if guarded else ""
        before = json.loads((baseline / f"{name}{suffix}.json").read_text())
        after = json.loads((candidate / f"{name}{suffix}.json").read_text())
        if (
            before.get("execution_guards", False) != guarded
            or after.get("execution_guards", False) != guarded
        ):
            raise ValueError("Raw and guarded accuracy reports must not be mixed.")
        old, new = before["examples"], after["examples"]
        if [(r["id"], r["command"], r["expected"]) for r in old] != [
            (r["id"], r["command"], r["expected"]) for r in new
        ]:
            raise ValueError("Accuracy reports contain different validation cases.")
        for key in ("model", "revision", "dataset_sha256", "task_sha256", "split"):
            if before[key] != after[key]:
                raise ValueError(f"Accuracy reports differ in {key}.")
        if before["split"] != "valid":
            raise ValueError(
                "Experiments must use validation, preserving the final test set."
            )
        metrics = {
            "total": len(old),
            "baseline_correct": sum(r["correct"] for r in old),
            "candidate_correct": sum(r["correct"] for r in new),
            "regressions": [
                a["id"]
                for a, b in zip(old, new, strict=True)
                if a["correct"] and not b["correct"]
            ],
            "improvements": [
                a["id"]
                for a, b in zip(old, new, strict=True)
                if not a["correct"] and b["correct"]
            ],
        }
        metrics["accuracy_difference_percentage_points"] = (
            100
            * (metrics["candidate_correct"] - metrics["baseline_correct"])
            / len(old)
        )
        if name == "valid":
            metrics["baseline_per_action"] = before["per_action"]
            metrics["candidate_per_action"] = after["per_action"]
        else:
            metrics["baseline_sessions"] = summarize_sessions(old)
            metrics["candidate_sessions"] = summarize_sessions(new)
        result[name] = metrics
    commands, sessions = result["valid"], result["sessions-valid"]
    old_sessions, new_sessions = (
        sessions["baseline_sessions"],
        sessions["candidate_sessions"],
    )
    result["no_observed_accuracy_drop"] = bool(
        commands["candidate_correct"] >= commands["baseline_correct"]
        and all(
            commands["candidate_per_action"][name]["correct"] >= value["correct"]
            for name, value in commands["baseline_per_action"].items()
        )
        and sessions["candidate_correct"] >= sessions["baseline_correct"]
        and new_sessions["state_agreement_rate"] >= old_sessions["state_agreement_rate"]
        and new_sessions["perfect_sessions"] >= old_sessions["perfect_sessions"]
        and new_sessions["final_state_matches"] >= old_sessions["final_state_matches"]
        and new_sessions["unwanted_mutations"] <= old_sessions["unwanted_mutations"]
    )
    result["scope"] = (
        "One paired 500-update trial on validation commands and sessions; "
        "long-run convergence remains unmeasured."
    )
    return result


def refinement_experiment(root, working):
    import yaml

    if not (root / "warm-start.json").exists():
        raise ValueError("Refinement requires verified weights and a fresh optimizer.")
    config = yaml.safe_load((root / "config.yaml").read_text())
    output = working / "refinement"
    baseline = output / "baseline"
    (baseline / "adapter").mkdir(parents=True, exist_ok=False)
    shutil.copyfile(
        root / "warm-start.safetensors", baseline / "adapter/adapters.safetensors"
    )
    shutil.copyfile(
        root / "warm-adapter-config.json", baseline / "adapter/adapter_config.json"
    )
    (baseline / "config.yaml").write_text(
        yaml.safe_dump({**config, "model_dtype": "float16"})
    )
    trained = run_case(root, output, "dual-window", quality=True)
    (working / "refinement-training.json").write_text(json.dumps(trained, indent=2))
    if "error" in trained:
        raise RuntimeError(
            "Refinement training failed; the original adapter is retained."
        )
    evaluate_pair(root, output)
    result = {
        "raw": quality_comparison(baseline, output / "dual-window"),
        "guarded": quality_comparison(baseline, output / "dual-window", guarded=True),
        "training": trained,
        "scope": (
            "One targeted refinement from the retained 15,000-update model; "
            "validation only."
        ),
    }
    for mode in ("raw", "guarded"):
        result[mode]["scope"] = result["scope"]
    (working / "refinement-comparison.json").write_text(json.dumps(result, indent=2))
    print(
        "REFINEMENT COMPARISON "
        + json.dumps({k: v for k, v in result.items() if k != "training"}),
        flush=True,
    )


def quality_experiment(root, working):
    output = working / "quality"
    output.mkdir(exist_ok=True)
    training = []
    for case in ("baseline", "dual-window"):
        result = run_case(root, output, case, quality=True)
        training.append(result)
        (working / "quality-training.json").write_text(json.dumps(training, indent=2))
        if "error" in result:
            raise RuntimeError(f"Accuracy trial training failed: {case}")
        print(
            "QUALITY TRAINED "
            + json.dumps(
                {
                    key: value
                    for key, value in result.items()
                    if key not in ("reports", "batch_shapes")
                }
            ),
            flush=True,
        )
    for key in (
        "training_batch_sha256",
        "warm_start_sha256",
        "optimizer_sha256",
        "dataset_sha256",
        "task_sha256",
        "model_revision",
        "start_step",
        "end_step",
        "global_batch",
        "updates",
    ):
        if training[0][key] != training[1][key]:
            raise ValueError(f"Paired training conditions differ in {key}.")
    evaluate_pair(root, output)
    result = quality_comparison(output / "baseline", output / "dual-window")
    result["training"] = training
    result["measured_speed_ratio"] = (
        training[1]["updates_per_second"] / training[0]["updates_per_second"]
    )
    (working / "quality-comparison.json").write_text(json.dumps(result, indent=2))
    print(
        "QUALITY COMPARISON "
        + json.dumps(
            {key: value for key, value in result.items() if key != "training"}
        ),
        flush=True,
    )


def evaluate_pair(root, output):
    evaluations = []
    handles = []
    try:
        for device, case in enumerate(("baseline", "dual-window")):
            environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(device)}
            environment.pop("MLX_RANK", None)
            environment.pop("MLX_WORLD_SIZE", None)
            log = (output / case / "evaluation.log").open("w")
            handles.append(log)
            evaluations.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        str(root / "gpu_benchmark.py"),
                        "--root",
                        str(root),
                        "--output",
                        str(output / case),
                        "--case",
                        case,
                        "--evaluate-quality",
                    ],
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
        deadline = time.monotonic() + 5400
        next_report = time.monotonic() + 30
        while any(process.poll() is None for process in evaluations):
            if any(process.poll() not in (None, 0) for process in evaluations):
                raise RuntimeError(
                    "GPU accuracy evaluation failed; inspect evaluation logs."
                )
            if time.monotonic() >= deadline:
                raise TimeoutError("GPU accuracy evaluation exceeded its fixed limit.")
            if time.monotonic() >= next_report:
                for case in ("baseline", "dual-window"):
                    lines = (output / case / "evaluation.log").read_text().splitlines()
                    if lines:
                        print(f"QUALITY EVALUATING {case}: {lines[-1]}", flush=True)
                next_report = time.monotonic() + 30
            time.sleep(0.5)
        if any(process.returncode for process in evaluations):
            raise RuntimeError("GPU accuracy evaluation failed.")
    finally:
        for process in evaluations:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        for log in handles:
            log.close()


def evaluate_quality(output):
    import mlx.core as mx
    import yaml

    from lab import evaluate, evaluate_sessions

    if mx.default_device() != mx.gpu:
        raise RuntimeError("Accuracy evaluation requires the CUDA GPU.")
    for function, name in ((evaluate, "valid"), (evaluate_sessions, "sessions-valid")):
        function(
            argparse.Namespace(
                split="valid",
                adapter=output / "adapter",
                output=output / f"{name}.json",
                limit=None,
            )
        )
        mx.clear_cache()
    config = yaml.safe_load((output / "config.yaml").read_text())
    if config.get("evaluate_guards"):
        for function, name in (
            (evaluate, "valid"),
            (evaluate_sessions, "sessions-valid"),
        ):
            function(
                argparse.Namespace(
                    split="valid",
                    adapter=output / "adapter",
                    output=output / f"{name}-guarded.json",
                    limit=None,
                    guarded=True,
                )
            )
            mx.clear_cache()


def benchmark(root, working, cases=None):
    output = working / "throughput"
    output.mkdir(exist_ok=True)
    results = []
    started = time.monotonic()

    def record(result):
        results.append(result)
        (working / "throughput-results.json").write_text(json.dumps(results, indent=2))
        print(
            "CASE SUMMARY "
            + json.dumps(
                {
                    key: value
                    for key, value in result.items()
                    if key not in ("reports", "log_tail")
                }
            ),
            flush=True,
        )

    cases = cases or list(CASES)[:8]
    for case in cases:
        if time.monotonic() - started > 2400:
            break
        result = run_case(root, output, case)
        record(result)
        if case == "baseline" and "error" in result:
            raise RuntimeError("Baseline failed; refusing to rank candidates.")
    eligible = [result for result in results if result.get("gradient_check_passed")]
    winner = max(eligible, key=lambda result: result["updates_per_second"])
    record(run_case(root, output, "baseline", verify=True))
    if winner["case"] != "baseline":
        record(run_case(root, output, winner["case"], verify=True))
    verified = [
        result
        for result in results
        if result.get("verification") and result.get("gradient_check_passed")
    ]
    if not verified:
        raise RuntimeError("No candidate completed the verification workload.")
    best = max(verified, key=lambda result: result["updates_per_second"])
    summary = {
        "best_verified_case": best["case"],
        "updates_per_second": best["updates_per_second"],
        "global_batch": 8,
        "elapsed_seconds": time.monotonic() - started,
        "scope": (
            "Bounded configurations; same model, data, LoRA, loss targets "
            "and effective batch."
        ),
    }
    (working / "throughput-best.json").write_text(json.dumps(summary, indent=2))
    print("BEST VERIFIED " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--quality", action="store_true")
    parser.add_argument("--evaluate-quality", action="store_true")
    args = parser.parse_args()
    if args.evaluate_quality:
        evaluate_quality(args.output)
    else:
        worker(args.root, args.output, args.case, args.verify, args.quality)
