import argparse
import csv
import hashlib
import json
import math
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


def quality_batches(path, seed, resume_step, updates=500, *, exposure_path=None):
    import numpy as np

    with np.load(path) as arrays:
        tokens, boundaries, offsets = [
            arrays[name] for name in ("tokens", "boundaries", "offsets")
        ]
    sizes = np.diff(boundaries)
    if len(sizes) < 8 or updates < 1 or resume_step < 0:
        raise ValueError("Quality batches require eight rows and positive updates.")
    order = np.argsort(sizes, kind="stable")
    groups = order[: len(order) // 8 * 8].reshape(-1, 8)
    random = np.random.RandomState(seed)
    batches, selected = [], []
    plan_path = path.with_name("train-plan.npz")
    if plan_path.exists():
        metadata = json.loads(path.with_name("tokens.json").read_text())
        expected = metadata.get("files", {}).get(plan_path.name)
        if (
            not expected
            or hashlib.sha256(plan_path.read_bytes()).hexdigest() != expected
        ):
            raise ValueError("Frozen training plan checksum mismatch.")
        with np.load(plan_path, allow_pickle=False) as arrays:
            if "indices" not in arrays:
                raise ValueError("Frozen training plan requires an indices matrix.")
            plan = arrays["indices"]
        if (
            plan.ndim != 2
            or plan.shape[1] != 8
            or len(plan) == 0
            or plan.dtype.kind not in "iu"
            or np.any(plan < 0)
            or np.any(plan >= len(sizes))
        ):
            raise ValueError(
                "Frozen training plan needs valid integer batches of eight."
            )
        if np.unique(plan).size != len(sizes):
            raise ValueError("Frozen training plan must cover every training example.")
        ordered = np.sort(plan, axis=1)
        if np.any(ordered[:, 1:] == ordered[:, :-1]):
            raise ValueError("Frozen training plan repeats an example within a batch.")
        if resume_step + updates > len(plan):
            raise ValueError("Requested updates exceed the frozen training plan.")
        indices_by_step = plan[resume_step : resume_step + updates]
    else:
        indices_by_step = []
        step = 0
        while len(indices_by_step) < updates:
            for group in random.permutation(len(groups)):
                if step >= resume_step:
                    indices_by_step.append(groups[group])
                    if len(indices_by_step) == updates:
                        break
                step += 1
    for indices in indices_by_step:
        lengths = sizes[indices]
        width = 1 + 32 * ((int(max(lengths)) + 31) // 32)
        batch = np.zeros((8, width), dtype=np.int32)
        for row, index in enumerate(indices):
            begin, end = boundaries[index : index + 2]
            batch[row, : end - begin] = tokens[begin:end]
        batches.append((batch, np.column_stack((offsets[indices], lengths))))
        selected.extend(map(int, indices))
    digest = hashlib.sha256(np.asarray(selected, dtype="<i8").tobytes()).hexdigest()
    if exposure_path is not None:
        from training import write_training_exposure

        exposure = write_training_exposure(
            path, selected, len(sizes), seed, resume_step, updates
        )
        exposure["training_batch_sha256"] = digest
        metadata_path = path.with_name("tokens.json")
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            exposure.update(
                {key: metadata[key] for key in ("dataset_sha256", "task_sha256")}
            )
        exposure_path.parent.mkdir(parents=True, exist_ok=True)
        exposure_path.write_text(json.dumps(exposure, indent=2))
    return batches, digest


def quality_config_hash(config):
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def quality_settings(settings, config):
    checkpoint = config.get("refinement_checkpoint", "all")
    if checkpoint not in {"all", "none"}:
        raise ValueError("Refinement checkpointing must be all or none.")
    return {
        **settings,
        "micro": config.get("refinement_micro_batch", settings["micro"]),
        "checkpoint": checkpoint,
    }


def quality_training_state(root, config):
    metadata = json.loads((root / "data/tokens.json").read_text())
    refinement = (root / "warm-start.json").exists()
    state_path = root / "resume"
    files = (
        "adapters.safetensors",
        "optimizer.safetensors",
        "random.safetensors",
        "progress.json",
    )
    exact_resume = refinement and any((state_path / name).exists() for name in files)
    if refinement:
        warm = json.loads((root / "warm-start.json").read_text())
        if warm.get("initialization") != "weights_only" or hashlib.sha256(
            (root / "warm-start.safetensors").read_bytes()
        ).hexdigest() != warm["adapter_sha256"]:
            raise ValueError("Refinement starting weights failed verification.")
    if refinement and not exact_resume:
        return {**metadata, "step": 0, "schedule_offset": 0}, True
    required = files if exact_resume else files[1:]
    if any(not (state_path / name).is_file() for name in required):
        raise ValueError("Retained training state is incomplete.")
    resume = json.loads((state_path / "progress.json").read_text())
    if any(
        resume.get(key) != metadata[key] for key in ("dataset_sha256", "task_sha256")
    ):
        raise ValueError("Retained training state differs from the dataset or task.")
    step, offset = resume.get("step"), resume.get("schedule_offset")
    if type(step) is not int or type(offset) is not int or not 0 <= offset <= step:
        raise ValueError("Retained training state has an invalid update step.")
    if exact_resume:
        if not 0 < step < config.get("quality_updates", 500):
            raise ValueError("Retained update must precede the training target.")
        if resume.get("config_sha256") != quality_config_hash(config):
            raise ValueError("Retained training configuration checksum mismatch.")
        if resume.get("training_plan_sha256") != metadata.get("files", {}).get(
            "train-plan.npz"
        ):
            raise ValueError("Retained training plan checksum mismatch.")
        for name in files[:-1]:
            if resume.get("state_files", {}).get(name) != hashlib.sha256(
                (state_path / name).read_bytes()
            ).hexdigest():
                raise ValueError(f"Retained training state checksum mismatch: {name}")
    return resume, False


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
    config = yaml.safe_load((root / "config.yaml").read_text())
    refinement = (root / "warm-start.json").exists()
    warm_start = refinement
    if quality:
        resume, warm_start = quality_training_state(root, config)
    if quality and refinement:
        settings = quality_settings(settings, config)
    accumulation = 8 // (settings["micro"] * workers)
    if accumulation < 1 or settings["micro"] * workers * accumulation != 8:
        raise ValueError("Micro batches must preserve global batch eight.")
    if quality:
        from lab import verify_token_cache

        if not verify_token_cache(config):
            raise ValueError("A verified token cache is required for GPU experiments.")
    model, _ = load(model_path("float16"))
    model.freeze()
    linear_to_lora_layers(model, config["num_layers"], config["lora_parameters"])
    resumed_adapter = root / "resume/adapters.safetensors"
    adapter_source = (
        resumed_adapter
        if quality and refinement and not warm_start
        else root / "warm-start.safetensors"
    )
    model.load_weights(str(adapter_source), strict=False)
    if settings.get("dtype") == "float32":
        model.set_dtype(mx.float32)
    if warm_start:
        np.random.seed(config["seed"])
        mx.random.seed(config["seed"])
    else:
        restore_random_state(mx.load(str(root / "resume/random.safetensors"))["key"])
    checkpoint_layers(model, settings["checkpoint"])
    if quality:
        remaining_updates = config.get("quality_updates", 500) - (
            resume["step"] if refinement else 0
        )
        raw_batches, batch_digest = quality_batches(
            root / "data/tokens-train.npz",
            config["seed"],
            resume["step"],
            updates=remaining_updates,
            exposure_path=output / "training-exposure.json"
            if rank == 0 and config.get("checkpoint_selection_steps")
            else None,
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
            else completion_loss(model, *data, config.get("completion_window", 96))
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
    if quality and (
        not math.isfinite(initial_loss)
        or not mx.stack(
            [mx.all(mx.isfinite(value)) for _, value in tree_flatten(gradients)]
        ).all().item()
    ):
        raise ValueError("Initial training loss or gradients are not finite.")
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
            if refinement:
                (output / "data").mkdir(exist_ok=True)
                shutil.copyfile(root / "data/tokens.json", output / "data/tokens.json")
    else:
        optimizer = optim.Adam(learning_rate=3e-5)
        adapter_file = output / "benchmark-adapters.safetensors"
    reports = []
    updates, interval, warmup = (
        (remaining_updates, 50, 50)
        if quality
        else ((80, 8, 16) if verify else (24, 4, 12))
    )
    steady_started = None

    class Reports(TrainingCallback):
        def on_train_loss_report(self, info):
            nonlocal steady_started
            if quality and not math.isfinite(info["train_loss"]):
                raise ValueError("Training loss is not finite.")
            reports.append(info)
            if info["iteration"] == warmup * accumulation:
                steady_started = time.time()
            completed = info["iteration"] // accumulation + (
                resume["step"] if quality else 0
            )
            if (
                refinement
                and quality
                and rank == 0
                and (
                    completed == 50
                    or completed == config.get("quality_updates", 500)
                    or completed % config.get("save_every", 500) == 0
                    or completed in config.get("checkpoint_selection_steps", [])
                )
            ):
                save_training_checkpoint(
                    model,
                    optimizer,
                    adapter_file.parent,
                    completed,
                    resume["schedule_offset"],
                    archive=False,
                )
                checkpoint = adapter_file.parent / "checkpoints" / f"{completed:07d}"
                progress = json.loads((checkpoint / "progress.json").read_text())
                progress.update(
                    config_sha256=quality_config_hash(config),
                    training_plan_sha256=json.loads(
                        (root / "data/tokens.json").read_text()
                    )
                    .get("files", {})
                    .get("train-plan.npz"),
                    state_files={
                        name: hashlib.sha256(
                            (checkpoint / name).read_bytes()
                        ).hexdigest()
                        for name in (
                            "adapters.safetensors",
                            "optimizer.safetensors",
                            "random.safetensors",
                        )
                    },
                )
                pending = checkpoint / "progress.json.tmp"
                pending.write_text(json.dumps(progress))
                pending.replace(checkpoint / "progress.json")

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
            max_seq_length=config["max_seq_length"],
        ),
        loss=loss,
        iterate_batches=iterator,
        training_callback=Reports(),
    )
    if rank != 0:
        return
    if quality and not mx.stack(
        [
            mx.all(mx.isfinite(value))
            for _, value in tree_flatten(model.trainable_parameters())
        ]
    ).all().item():
        raise ValueError("Final trained adapter is not finite.")
    steady = [
        report["iterations_per_second"] / accumulation
        for report in reports
        if report["iteration"] > warmup * accumulation
    ]
    measured = steady or [
        report["iterations_per_second"] / accumulation for report in reports
    ]
    rate = statistics.median(measured)
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
        "min_updates_per_second": min(measured),
        "max_updates_per_second": max(measured),
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
            training_target=config.get("quality_updates", 500)
            if refinement
            else resume["step"] + updates,
            training_finite=True,
            resumed_adapter_sha256=hashlib.sha256(adapter_source.read_bytes()).hexdigest()
            if not warm_start
            else None,
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
    if quality:
        import yaml

        timeout = yaml.safe_load((root / "config.yaml").read_text()).get(
            "quality_timeout_seconds", 1800
        )
    else:
        timeout = 600 if verify or CASES[case].get("compile") else 360
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
    last_progress = time.monotonic()
    last_log_size = 0
    progress_started = False
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
            if quality:
                # Kaggle's streamed logs can lag behind the worker's local output.
                worker_log = folder / "worker-0.log"
                size = worker_log.stat().st_size
                if size != last_log_size:
                    last_progress = time.monotonic()
                    last_log_size = size
                    progress_started = (
                        progress_started or "Iter " in worker_log.read_text()
                    )
                if progress_started and time.monotonic() - last_progress > 900:
                    error = (
                        "No new training report for 900 seconds; "
                        "retained checkpoints are preserved."
                    )
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


def quality_comparison(baseline, candidate, *, guarded=False, report_prefix=""):
    from sessions import summarize_sessions

    result = {}
    for name in ("valid", "sessions-valid"):
        suffix = "-guarded" if guarded else ""
        before = json.loads(
            (baseline / f"{report_prefix}{name}{suffix}.json").read_text()
        )
        after = json.loads(
            (candidate / f"{report_prefix}{name}{suffix}.json").read_text()
        )
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


def checkpoint_selection_settings(config):
    steps = config["checkpoint_selection_steps"]
    updates = config.get("quality_updates", 500)
    if (
        not isinstance(steps, list)
        or not 1 <= len(steps) <= 3
        or any(
            type(step) is not int or not 1 <= step <= updates or step % 50
            for step in steps
        )
        or len(set(steps)) != len(steps)
    ):
        raise ValueError(
            "Checkpoint selection requires up to three reported update steps."
        )
    for key, size in (("selection_example_ids", 256), ("selection_session_ids", 6)):
        identifiers = config.get(key)
        if (
            not isinstance(identifiers, list)
            or len(identifiers) != size
            or any(
                not isinstance(identifier, str) or not identifier
                for identifier in identifiers
            )
            or len(set(identifiers)) != size
        ):
            raise ValueError(f"{key} must contain {size} unique validation IDs.")
    return sorted(steps)


def checkpoint_metrics(folder, *, selection=False):
    from sessions import summarize_sessions
    from training import action_stratum

    prefix = "selection-" if selection else ""
    command = json.loads((folder / f"{prefix}valid-guarded.json").read_text())
    session = json.loads((folder / f"{prefix}sessions-valid-guarded.json").read_text())
    if any(
        report["split"] != "valid" or not report.get("execution_guards")
        for report in (command, session)
    ):
        raise ValueError("Checkpoint selection must use guarded validation only.")
    if selection:
        import yaml

        config = yaml.safe_load((folder / "config.yaml").read_text())
        checkpoint_selection_settings(config)
        if (
            len(command["examples"]) != len(config["selection_example_ids"])
            or {row["id"] for row in command["examples"]}
            != set(config["selection_example_ids"])
            or {row["session_id"] for row in session["examples"]}
            != set(config["selection_session_ids"])
        ):
            raise ValueError(
                "Checkpoint reports do not cover the selected validation IDs."
            )
    strata = {}
    for row in command["examples"]:
        count = strata.setdefault(
            action_stratum(row["expected"]), {"correct": 0, "total": 0}
        )
        count["total"] += 1
        count["correct"] += row["correct"]
    sessions = summarize_sessions(session["examples"])
    result = {
        "command_accuracy": sum(row["correct"] for row in command["examples"])
        / len(command["examples"]),
        "session_action_accuracy": sessions["exact_action_accuracy"],
        **{
            key: sessions[key]
            for key in (
                "state_agreement_rate",
                "perfect_sessions",
                "final_state_matches",
                "unwanted_mutations",
            )
        },
        "per_stratum": dict(sorted(strata.items())),
        "adapter_sha256": command["adapter_sha256"],
    }
    if result["adapter_sha256"] != session["adapter_sha256"]:
        raise ValueError(
            "Command and session selection reports score different adapters."
        )
    result["joint_score"] = (
        sum(
            result[key]
            for key in (
                "command_accuracy",
                "session_action_accuracy",
                "state_agreement_rate",
            )
        )
        / 3
    )
    return result


def checkpoint_regressions(baseline, candidate):
    if set(baseline["per_stratum"]) != set(candidate["per_stratum"]):
        raise ValueError("Checkpoint metrics use different validation strata.")
    regressions = []
    for key in (
        "command_accuracy",
        "session_action_accuracy",
        "state_agreement_rate",
        "perfect_sessions",
        "final_state_matches",
    ):
        if candidate[key] < baseline[key]:
            regressions.append(key)
    if candidate["unwanted_mutations"] > baseline["unwanted_mutations"]:
        regressions.append("unwanted_mutations")
    for key, old in baseline["per_stratum"].items():
        new = candidate["per_stratum"][key]
        if new["total"] != old["total"]:
            raise ValueError("Checkpoint metrics use different validation counts.")
        if new["correct"] < old["correct"]:
            regressions.append(f"stratum:{key}")
    return regressions


def choose_checkpoint(baseline, candidates):
    eligible, rejected = [], {}
    for step, candidate in candidates.items():
        regressions = checkpoint_regressions(baseline, candidate)
        if not regressions and candidate["joint_score"] > baseline["joint_score"]:
            eligible.append(step)
        else:
            rejected[str(step)] = regressions or ["no_joint_improvement"]
    chosen = (
        max(eligible, key=lambda step: (candidates[step]["joint_score"], -step))
        if eligible
        else 0
    )
    return {
        "chosen_step": chosen,
        "eligible_steps": sorted(eligible),
        "rejected_steps": rejected,
    }


def export_checkpoint(source, destination, step, metadata):
    checkpoint = source / "adapter/checkpoints" / f"{step:07d}"
    progress = json.loads((checkpoint / "progress.json").read_text())
    if progress["step"] != step or any(
        progress[key] != metadata[key] for key in ("dataset_sha256", "task_sha256")
    ):
        raise ValueError(
            "Selection checkpoint differs from the requested step or task."
        )
    adapter = destination / "adapter"
    adapter.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source / "config.yaml", destination / "config.yaml")
    shutil.copyfile(
        source / "adapter/adapter_config.json", adapter / "adapter_config.json"
    )
    shutil.copyfile(
        checkpoint / "adapters.safetensors", adapter / "adapters.safetensors"
    )
    shutil.copytree(checkpoint, adapter / "checkpoints" / f"{step:07d}")


def select_refinement_checkpoint(root, working, config):
    steps = checkpoint_selection_settings(config)
    output = working / "refinement"
    baseline = output / "baseline"
    final = output / "dual-window"
    metadata = json.loads((root / "data/tokens.json").read_text())
    candidates = {}
    for step in steps:
        folder = output / "checkpoint-selection" / f"step-{step}"
        export_checkpoint(final, folder, step, metadata)
        candidates[step] = folder
    folders = [
        ("baseline", baseline),
        *((f"step-{step}", candidates[step]) for step in steps),
    ]
    for offset in range(0, len(folders), 2):
        evaluate_pair(
            root, output, folders=dict(folders[offset : offset + 2]), selection=True
        )
    before = checkpoint_metrics(baseline, selection=True)
    metrics = {}
    for step, folder in candidates.items():
        quality_comparison(baseline, folder, guarded=True, report_prefix="selection-")
        metrics[step] = checkpoint_metrics(folder, selection=True)
    result = {
        **choose_checkpoint(before, metrics),
        "baseline": before,
        "candidates": {str(step): value for step, value in metrics.items()},
        "selection_example_ids": config["selection_example_ids"],
        "selection_session_ids": config["selection_session_ids"],
        "dataset_sha256": metadata["dataset_sha256"],
        "task_sha256": metadata["task_sha256"],
        "selection_criteria": {
            "score": (
                "Mean of command accuracy, session action accuracy, "
                "and canvas state agreement"
            ),
            "regression_gate": (
                "No decreases in tool/operation/reason correctness, action accuracy, "
                "state agreement, perfect/final-state sessions; "
                "no additional unwanted mutations"
            ),
            "ties": "Earlier checkpoint",
            "full_validation_required": True,
        },
        "promotion_passed": False,
        "chosen_adapter_sha256": before["adapter_sha256"],
        "full_validation": None,
        "test_set_used": False,
    }
    step = result["chosen_step"]
    if step:
        selected = output / "selected"
        export_checkpoint(final, selected, step, metadata)
        result["chosen_adapter_sha256"] = hashlib.sha256(
            (selected / "adapter/adapters.safetensors").read_bytes()
        ).hexdigest()
        if result["chosen_adapter_sha256"] != metrics[step]["adapter_sha256"]:
            raise ValueError("Selected export differs from the scored checkpoint.")
        evaluate_pair(
            root, output, folders={"baseline": baseline, "selected": selected}
        )
        comparison = quality_comparison(baseline, selected, guarded=True)
        full_before, full_after = (
            checkpoint_metrics(baseline),
            checkpoint_metrics(selected),
        )
        regressions = checkpoint_regressions(full_before, full_after)
        result["full_validation"] = {
            "baseline": full_before,
            "selected": full_after,
            "comparison": comparison,
            "regressions": regressions,
        }
        result["promotion_passed"] = bool(
            comparison["no_observed_accuracy_drop"] and not regressions
        )
    (working / "checkpoint-selection.json").write_text(json.dumps(result, indent=2))
    print(
        "CHECKPOINT SELECTION "
        + json.dumps(
            {
                key: result[key]
                for key in ("chosen_step", "chosen_adapter_sha256", "promotion_passed")
            }
        ),
        flush=True,
    )
    return result


def refinement_experiment(root, working, *, training_only=False):
    import yaml

    if not (root / "warm-start.json").exists():
        raise ValueError("Refinement requires verified weights and a fresh optimizer.")
    config = yaml.safe_load((root / "config.yaml").read_text())
    if config.get("checkpoint_selection_steps"):
        checkpoint_selection_settings(config)
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
    if config.get("checkpoint_selection_steps"):
        select_refinement_checkpoint(root, working, config)
        return
    if training_only:
        print(
            "Refinement training complete; checkpoint ready for native checks.",
            flush=True,
        )
        return
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


def resume_refinement_evaluation(
    root, working, expected, *, source=None, selected=False
):
    from kaggle_follow import verify_workflow_checkpoint
    from lab import _metadata

    if selected and expected.get("native_verification") != "passed":
        raise ValueError(
            "Final scoring requires passed native checks for this adapter."
        )
    if source is None:
        sources = list(Path("/kaggle/input").rglob("refinement-training.json"))
        if len(sources) != 1:
            raise ValueError("Expected one mounted completed refinement run.")
        source = sources[0].parent
    trained, weights = verify_workflow_checkpoint(source, expected)
    metadata = _metadata()
    for key in ("dataset_sha256", "task_sha256"):
        if metadata[key] != expected[key]:
            raise ValueError(f"Evaluation input differs from training in {key}.")
    if hashlib.sha256(weights.read_bytes()).hexdigest() != expected["adapter_sha256"]:
        raise ValueError("Evaluation candidate differs from the recovered adapter.")
    baseline = source / "refinement/baseline/adapter/adapters.safetensors"
    if (
        hashlib.sha256(baseline.read_bytes()).hexdigest()
        != expected["warm_start_sha256"]
    ):
        raise ValueError("Evaluation baseline differs from the retained model.")
    output = working / "refinement"
    retained_step = expected.get("selected_step", expected["updates"])
    selected_source = source / "refinement/selected"
    if "selected_step" in expected:
        if not expected.get("promotion_passed"):
            raise ValueError(
                "Selected checkpoint scoring requires a passed promotion gate."
            )
        shutil.copytree(selected_source, output / "selected")
        shutil.copyfile(
            source / "checkpoint-selection.json", working / "checkpoint-selection.json"
        )
        shutil.copytree(source / "refinement/dual-window", output / "training-final")
    for case in ("baseline", "dual-window"):
        origin = (
            selected_source
            if case == "dual-window" and "selected_step" in expected
            else source / "refinement" / case
        )
        destination = output / case
        (destination / "adapter").mkdir(parents=True)
        for filename in (
            "config.yaml",
            "adapter/adapters.safetensors",
            "adapter/adapter_config.json",
        ):
            shutil.copyfile(origin / filename, destination / filename)
        # Completed command reports survive a later session-evaluator failure.
        if (origin / "valid.json").exists():
            report = json.loads((origin / "valid.json").read_text())
            if any(
                report[key] != metadata[key]
                for key in ("dataset_sha256", "task_sha256")
            ):
                raise ValueError(
                    "Retained command report uses different validation data."
                )
            if report["split"] != "valid" or report.get("execution_guards", False):
                raise ValueError(
                    "Retained command report has the wrong evaluation mode."
                )
            report["adapter_sha256"] = hashlib.sha256(
                (destination / "adapter/adapters.safetensors").read_bytes()
            ).hexdigest()
            (destination / "valid.json").write_text(json.dumps(report, indent=2))
    checkpoint = f"adapter/checkpoints/{retained_step:07d}"
    shutil.copytree(
        (
            selected_source
            if "selected_step" in expected
            else source / "refinement/dual-window"
        )
        / checkpoint,
        output / "dual-window" / checkpoint,
    )
    shutil.copyfile(
        source / "refinement-training.json", working / "refinement-training.json"
    )
    print(
        f"Recovered update {trained['end_step']}; evaluating without training.",
        flush=True,
    )
    evaluate_pair(root, output, selected=selected)
    if selected:
        reports = {}
        candidate = output / "dual-window"
        for name in (
            "valid",
            "sessions-valid",
            "valid-guarded",
            "sessions-valid-guarded",
            "test-guarded",
            "sessions-test-guarded",
        ):
            report = json.loads((candidate / f"{name}.json").read_text())
            if report["adapter_sha256"] != expected["adapter_sha256"]:
                raise ValueError("Final reports score a different adapter.")
            reports[name] = {
                key: value for key, value in report.items() if key != "examples"
            }
        (working / "refinement-selected-evaluation.json").write_text(
            json.dumps(
                {
                    "adapter_sha256": expected["adapter_sha256"],
                    "test_set_used": True,
                    "reports": reports,
                },
                indent=2,
            )
        )
        return
    result = {
        mode: quality_comparison(
            output / "baseline", output / "dual-window", guarded=mode == "guarded"
        )
        for mode in ("raw", "guarded")
    }
    (working / "refinement-comparison.json").write_text(json.dumps(result, indent=2))


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


def evaluate_pair(root, output, *, selected=False, folders=None, selection=False):
    evaluations = []
    handles = []
    log_paths = {}
    if selected and (folders is not None or selection):
        raise ValueError(
            "Final test scoring must not be mixed with checkpoint selection."
        )
    if folders is not None and not 1 <= len(folders) <= 2:
        raise ValueError("Validation evaluation requires one or two GPU workers.")
    cases = (
        tuple(folders)
        if folders is not None
        else ("valid", "test")
        if selected
        else ("baseline", "dual-window")
    )
    try:
        # Concurrent conversion can expose an unfinished checkpoint to the other worker.
        from lab import model_path

        model_path("float16")
        for device, case in enumerate(cases):
            environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(device)}
            environment.pop("MLX_RANK", None)
            environment.pop("MLX_WORLD_SIZE", None)
            folder = (
                folders[case]
                if folders is not None
                else output / "dual-window"
                if selected
                else output / case
            )
            log_paths[case] = folder / (
                f"evaluation-{case}.log"
                if selected
                else "selection-evaluation.log"
                if selection
                else "evaluation.log"
            )
            log = log_paths[case].open("w")
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
                        str(folder),
                        "--case",
                        "baseline" if case == "baseline" else "dual-window",
                        *(
                            ["--score-split", case]
                            if selected
                            else ["--evaluate-selection"]
                            if selection
                            else ["--evaluate-quality"]
                        ),
                    ],
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
        deadline = time.monotonic() + (10800 if selected else 7200)
        next_report = time.monotonic() + 30
        while any(process.poll() is None for process in evaluations):
            if any(process.poll() not in (None, 0) for process in evaluations):
                raise RuntimeError(
                    "GPU accuracy evaluation failed; inspect evaluation logs."
                )
            if time.monotonic() >= deadline:
                raise TimeoutError("GPU accuracy evaluation exceeded its fixed limit.")
            if time.monotonic() >= next_report:
                for case in cases:
                    lines = log_paths[case].read_text().splitlines()
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


def evaluate_checkpoint_selection(output):
    import mlx.core as mx
    import yaml

    from lab import evaluate, evaluate_sessions

    if mx.default_device() != mx.gpu:
        raise RuntimeError("Checkpoint selection requires the CUDA GPU.")
    config = yaml.safe_load((output / "config.yaml").read_text())
    checkpoint_selection_settings(config)
    digest = hashlib.sha256(
        (output / "adapter/adapters.safetensors").read_bytes()
    ).hexdigest()
    for function, name in ((evaluate, "valid"), (evaluate_sessions, "sessions-valid")):
        path = output / f"selection-{name}-guarded.json"
        function(
            argparse.Namespace(
                split="valid",
                adapter=output / "adapter",
                output=path,
                limit=None,
                guarded=True,
                cache_prefix=True,
                batch_size=8 if function is evaluate else 1,
                example_ids=config["selection_example_ids"],
                session_ids=config["selection_session_ids"],
            )
        )
        report = json.loads(path.read_text())
        report["adapter_sha256"] = digest
        path.write_text(json.dumps(report, indent=2))
        mx.clear_cache()


def evaluate_quality(output):
    import mlx.core as mx
    import yaml

    from lab import _metadata, evaluate, evaluate_sessions

    if mx.default_device() != mx.gpu:
        raise RuntimeError("Accuracy evaluation requires the CUDA GPU.")
    metadata = _metadata()
    adapter_sha256 = hashlib.sha256(
        (output / "adapter/adapters.safetensors").read_bytes()
    ).hexdigest()
    config = yaml.safe_load((output / "config.yaml").read_text())
    for guarded in (False, True) if config.get("evaluate_guards") else (False,):
        for function, name in (
            (evaluate, "valid"),
            (evaluate_sessions, "sessions-valid"),
        ):
            suffix = "-guarded" if guarded else ""
            report_path = output / f"{name}{suffix}.json"
            if name == "valid" and report_path.exists():
                report = json.loads(report_path.read_text())
                if (
                    all(
                        report[key] == metadata[key]
                        for key in ("dataset_sha256", "task_sha256")
                    )
                    and report.get("adapter_sha256") == adapter_sha256
                    and report["split"] == "valid"
                    and report.get("execution_guards", False) == guarded
                ):
                    print(f"Reusing verified {report_path.name}.", flush=True)
                    continue
            function(
                argparse.Namespace(
                    split="valid",
                    adapter=output / "adapter",
                    output=report_path,
                    limit=None,
                    guarded=guarded,
                    cache_prefix=True,
                    batch_size=8 if function is evaluate else 1,
                )
            )
            report = json.loads(report_path.read_text())
            report["adapter_sha256"] = adapter_sha256
            report_path.write_text(json.dumps(report, indent=2))
            mx.clear_cache()


def score_selected_split(output, split):
    import mlx.core as mx

    from lab import evaluate, evaluate_sessions

    if mx.default_device() != mx.gpu:
        raise RuntimeError("Final scoring requires the CUDA GPU.")
    digest = hashlib.sha256(
        (output / "adapter/adapters.safetensors").read_bytes()
    ).hexdigest()
    for guarded in (False, True) if split == "valid" else (True,):
        for function, prefix in ((evaluate, ""), (evaluate_sessions, "sessions-")):
            suffix = "-guarded" if guarded else ""
            path = output / f"{prefix}{split}{suffix}.json"
            function(
                argparse.Namespace(
                    split=split,
                    adapter=output / "adapter",
                    output=path,
                    limit=None,
                    guarded=guarded,
                    cache_prefix=True,
                    batch_size=8 if function is evaluate else 1,
                )
            )
            report = json.loads(path.read_text())
            report["adapter_sha256"] = digest
            path.write_text(json.dumps(report, indent=2))
            print(
                "FINAL SCORE "
                + json.dumps(
                    {
                        "report": path.name,
                        **{
                            key: report[key]
                            for key in (
                                "total",
                                "turns",
                                "correct",
                                "exact_action_accuracy",
                                "state_agreement_rate",
                            )
                            if key in report
                        },
                    }
                ),
                flush=True,
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
    parser.add_argument("--evaluate-selection", action="store_true")
    parser.add_argument("--score-split", choices=("valid", "test"))
    args = parser.parse_args()
    if args.evaluate_selection:
        if args.score_split or args.evaluate_quality:
            parser.error("Checkpoint selection accepts validation subsets only.")
        evaluate_checkpoint_selection(args.output)
    elif args.score_split:
        score_selected_split(args.output, args.score_split)
    elif args.evaluate_quality:
        evaluate_quality(args.output)
    else:
        worker(args.root, args.output, args.case, args.verify, args.quality)
