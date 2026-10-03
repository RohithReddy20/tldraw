import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/payload")
UPDATES = 48
WARMUP = 8
BASELINE_RATE = 0.3254031406462446


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(stage):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import numpy as np
    import yaml
    from mlx.utils import tree_flatten, tree_map
    from mlx_lm import load
    from mlx_lm.tuner.callbacks import TrainingCallback
    from mlx_lm.tuner.trainer import TrainingArgs, train
    from mlx_lm.tuner.utils import build_schedule, linear_to_lora_layers

    sys.path.insert(0, str(ROOT))
    from gpu_benchmark import checkpoint_layers
    from training import completion_indices, completion_loss, packed_completion_loss

    for package, version in (("mlx", "0.32.3"), ("mlx-lm", "0.31.3")):
        if importlib.metadata.version(package) != version:
            raise RuntimeError(f"Pinned library mismatch: {package}")
    if mx.default_device() != mx.gpu:
        raise RuntimeError("Refusing a CPU benchmark.")
    devices = (
        subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        .strip()
        .splitlines()
    )
    if len(devices) != 1 or "H100" not in devices[0]:
        raise RuntimeError("Benchmark requires exactly one H100.")
    manifest = json.loads((ROOT / "manifest.json").read_text())
    for name, sha in manifest["files"].items():
        if digest(ROOT / name) != sha:
            raise ValueError(f"Frozen payload changed: {name}")
    for name, sha in manifest["model_files"].items():
        if digest(ROOT / "model" / name) != sha:
            raise ValueError(f"Converted model changed: {name}")
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    if (
        config["batch_size"] != 8
        or config["lora_parameters"]["rank"] != 32
        or config["loss_mode"] != "completion_tokens_v1"
    ):
        raise ValueError("Frozen training contract changed.")
    micro = 2 if stage == "safe" else 8
    accumulation = 8 // micro
    packed = stage == "candidate"
    output = Path("/results") / stage
    output.mkdir(parents=True, exist_ok=True)
    model, _ = load(str(ROOT / "model"))
    model.freeze()
    linear_to_lora_layers(model, config["num_layers"], config["lora_parameters"])
    model.load_weights(str(ROOT / "warm-start.safetensors"), strict=False)
    np.random.seed(config["seed"])
    mx.random.seed(config["seed"])
    if not packed:
        checkpoint_layers(model, "all")
    with np.load(ROOT / "representative.npz") as archive:
        raw = [(archive[f"batch_{i}"], archive[f"bounds_{i}"]) for i in range(8)]
    batches = []
    for array, bounds in raw:
        total = int((bounds[:, 1] - bounds[:, 0]).sum())
        group = []
        for start in range(0, 8, micro):
            values, lengths = (
                array[start : start + micro],
                bounds[start : start + micro],
            )
            count = int((lengths[:, 1] - lengths[:, 0]).sum())
            weight = mx.array(accumulation * count / total, mx.float32)
            if packed:
                indices, mask = completion_indices(lengths, values.shape[1])
                group.append(
                    (mx.array(values), mx.array(indices), mx.array(mask), weight)
                )
            else:
                group.append((mx.array(values), mx.array(lengths, mx.int32), weight))
        batches.append(group)

    def loss(model, *data):
        args, weight = data[:-1], data[-1]
        value, count = (
            packed_completion_loss(model, *args)
            if packed
            else completion_loss(model, *args, config["completion_window"])
        )
        return value * weight, count

    value_and_grad = nn.value_and_grad(model, loss)
    gates = []
    for index in (3, 7):
        grads, value = None, mx.array(0, mx.float32)
        for data in batches[index]:
            (part, _), current = value_and_grad(model, *data)
            value += part
            grads = (
                current
                if grads is None
                else tree_map(lambda a, b: a + b, grads, current)
            )
            mx.eval(value, grads)
        value, grads = value / accumulation, tree_map(lambda g: g / accumulation, grads)
        mx.eval(value, grads)
        flat = dict(tree_flatten(grads))
        finite = all(
            bool(mx.all(mx.isfinite(g)).item()) for g in flat.values()
        ) and math.isfinite(float(value.item()))
        gate = {
            "batch_index": index,
            "initial_loss": float(value.item()),
            "finite": finite,
        }
        if stage == "safe":
            if not finite:
                raise ValueError("Non-finite baseline loss or gradients.")
            mx.save_safetensors(str(output / f"gradient-{index}.safetensors"), flat)
            (output / f"gate-{index}.json").write_text(json.dumps(gate))
        else:
            reference = mx.load(
                str(Path("/results/safe") / f"gradient-{index}.safetensors")
            )
            original = json.loads(
                (Path("/results/safe") / f"gate-{index}.json").read_text()
            )
            if set(reference) != set(flat):
                raise ValueError("Candidate trainable parameters changed.")
            error = norm = dot = candidate_norm = 0.0
            for name, ref in reference.items():
                current = flat[name].astype(mx.float32)
                ref = ref.astype(mx.float32)
                values = [
                    mx.sum((current - ref) ** 2),
                    mx.sum(ref**2),
                    mx.sum(current * ref),
                    mx.sum(current**2),
                ]
                mx.eval(values)
                err, base, product, cand = [float(v.item()) for v in values]
                error += err
                norm += base
                dot += product
                candidate_norm += cand
            gate.update(
                loss_absolute_error=abs(
                    gate["initial_loss"] - original["initial_loss"]
                ),
                gradient_relative_l2_error=math.sqrt(error / max(norm, 1e-30)),
                gradient_cosine=dot / max(math.sqrt(norm * candidate_norm), 1e-30),
            )
            gate["passed"] = (
                finite
                and gate["loss_absolute_error"] < 1e-5
                and gate["gradient_relative_l2_error"] < 0.01
            )
            if not gate["passed"]:
                result = {
                    "stage": stage,
                    "status": "gradient_gate_failed",
                    "gates": gates + [gate],
                    "trained_updates": 0,
                }
                (output / "result.json").write_text(json.dumps(result, indent=2))
                print(json.dumps(result), flush=True)
                return
        gates.append(gate)
    del grads, flat, current, value_and_grad
    mx.clear_cache()
    mx.reset_peak_memory()
    schedule = build_schedule(config["lr_schedule"])
    optimizer = optim.Adam(learning_rate=lambda step: schedule(step))
    reports, timing = [], {"start": None, "finish": None}

    class Reports(TrainingCallback):
        def on_train_loss_report(self, info):
            reports.append(info)
            update = info["iteration"] // accumulation
            if not math.isfinite(info["train_loss"]):
                raise ValueError("Non-finite training loss.")
            if update == WARMUP:
                timing["start"] = time.time()
            if update == UPDATES:
                timing["finish"] = time.time()

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
            batch_size=micro,
            grad_accumulation_steps=accumulation,
            iters=UPDATES * accumulation,
            steps_per_report=8 * accumulation,
            steps_per_save=10000,
            adapter_file=str(output / "benchmark-adapters.safetensors"),
            max_seq_length=config["max_seq_length"],
        ),
        loss=loss,
        iterate_batches=iterator,
        training_callback=Reports(),
    )
    finite = all(
        bool(mx.all(mx.isfinite(p)).item())
        for _, p in tree_flatten(model.trainable_parameters())
    )
    rates = [
        r["iterations_per_second"] / accumulation
        for r in reports
        if r["iteration"] > WARMUP * accumulation
    ]
    elapsed = timing["finish"] - timing["start"]
    steady_indices = [i % len(raw) for i in range(WARMUP, UPDATES)]
    inputs = sum(int(raw[i][1][:, 0].sum()) for i in steady_indices)
    completions = sum(
        int((raw[i][1][:, 1] - raw[i][1][:, 0]).sum()) for i in steady_indices
    )
    samples = []
    with Path("/results/gpu.csv").open() as stream:
        for row in csv.DictReader(stream):
            if timing["start"] <= float(row["time"]) <= timing["finish"]:
                samples.append({k: float(v) for k, v in row.items()})
    rate = statistics.median(rates)
    result = {
        "stage": stage,
        "status": "completed" if finite else "non_finite_weights",
        "global_batch": 8,
        "micro_batch": micro,
        "accumulation": accumulation,
        "workers": 1,
        "checkpoint": "none" if packed else "all",
        "head": "packed_exact_completion" if packed else "window_160",
        "model_dtype": "float16",
        "gates": gates,
        "updates": UPDATES,
        "warmup_updates": WARMUP,
        "steady_updates": UPDATES - WARMUP,
        "updates_per_second_median": rate,
        "updates_per_second_elapsed": (UPDATES - WARMUP) / elapsed,
        "examples_per_second": 8 * rate,
        "input_tokens_per_second": inputs / elapsed,
        "completion_tokens_per_second": completions / elapsed,
        "total_unpadded_tokens_per_second": (inputs + completions) / elapsed,
        "mlx_peak_allocations_gb": mx.get_peak_memory() / 1e9,
        "gpu_sample_count": len(samples),
        "gpu_mean_utilization_percent": statistics.mean(
            s["utilization"] for s in samples
        )
        if samples
        else None,
        "gpu_max_vram_mib": max(s["vram_mib"] for s in samples) if samples else None,
        "gpu_mean_power_watts": statistics.mean(s["power_watts"] for s in samples)
        if samples
        else None,
        "seconds_including_warmup": time.monotonic() - started,
        "baseline_dual_t4_updates_per_second": BASELINE_RATE,
        "speedup_vs_dual_t4": rate / BASELINE_RATE,
        "hardware": devices,
        "training_finite": finite,
        "steady_started_at": timing["start"],
        "finished_at": timing["finish"],
        "dataset_sha256": manifest["dataset_sha256"],
        "task_sha256": manifest["task_sha256"],
        "reports": reports,
        "comparison_limit": (
            "Representative quantile batches and short warmup differ from full "
            "v7 epoch; speedup is a probe, not a guaranteed full-run speedup "
            "or accuracy result."
        ),
    }
    (output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("safe", "candidate"), required=True)
    run(parser.parse_args().stage)
