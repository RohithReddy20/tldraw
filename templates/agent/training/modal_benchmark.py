import csv
import json
import logging
import os
import signal
import subprocess
import sys
import sysconfig
import threading
import time
from pathlib import Path

import modal

TRAINING = Path(__file__).resolve().parent
BENCHMARK_DIR = TRAINING / "runs/quality-next/modal"
PROFILE_DIR = TRAINING / "runs/v8-quality-modal/profile"
TRAINING_PROFILE = "--training-profile" in sys.argv
HERE = PROFILE_DIR if TRAINING_PROFILE else BENCHMARK_DIR
PAYLOAD = BENCHMARK_DIR / "payload"
BASE = TRAINING / "runs/base-bb327a9a-float16"
BASE_REPRO = BENCHMARK_DIR / "base-repro"
BUILD_SECONDS = 300
ACTIVE_SECONDS = 690
STOP_SECONDS = 15
FORCE_EXIT_SECONDS = 20
RESOURCE_RATE_USD = 0.001097 + 4 * 0.0000131 + 16 * 0.00000222
SETUP_ONLY = "--setup-only" in sys.argv
app = modal.App("canvas-270m-h100-bounded-probe")


def prepare_base_reproduction():
    import shutil
    import struct

    manifest = json.loads((PAYLOAD / "manifest.json").read_text())
    BASE_REPRO.mkdir(exist_ok=True)
    with (BASE / "model.safetensors").open("rb") as stream:
        prefix = stream.read(8)
        header = stream.read(struct.unpack("<Q", prefix)[0])
    (BASE_REPRO / "fp16-header.bin").write_bytes(prefix + header)
    overrides = ["README.md", "config.json", "tokenizer_config.json"]
    for name in overrides:
        shutil.copyfile(BASE / name, BASE_REPRO / name)
    spec = {
        "repo": manifest["model"],
        "revision": manifest["model_revision"],
        "source_bf16_sha256": (
            "fb64bf18b2911fcaa59d44c1b7d5842a011a874530be9dc5bc9d307e82b4edee"
        ),
        "model_files": manifest["model_files"],
        "overrides": overrides,
    }
    (BASE_REPRO / "reproduction.json").write_text(json.dumps(spec, indent=2) + "\n")


def build_cpu_base():
    import hashlib
    import mmap
    import shutil
    import struct

    import numpy as np
    from huggingface_hub import snapshot_download

    def digest(path):
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    repro = Path("/base-repro")
    spec = json.loads((repro / "reproduction.json").read_text())
    source = Path(
        snapshot_download(
            spec["repo"],
            revision=spec["revision"],
            token=False,
            allow_patterns=["*.json", "*.safetensors", "*.jinja"],
            local_dir="/tmp/base-bf16",
            cache_dir="/tmp/base-hf-cache",
        )
    )
    original = source / "model.safetensors"
    if digest(original) != spec["source_bf16_sha256"]:
        raise ValueError("Pinned BF16 model checksum does not match the local source.")
    target = Path("/payload/model")
    target.mkdir(parents=True, exist_ok=True)
    prefix = (repro / "fp16-header.bin").read_bytes()
    converted_header = json.loads(prefix[8:])
    with original.open("rb") as stream:
        original_size = struct.unpack("<Q", stream.read(8))[0]
        original_header = json.loads(stream.read(original_size))
        source_keys = set(original_header) - {"__metadata__"}
        target_keys = set(converted_header) - {"__metadata__"}
        if source_keys != target_keys:
            raise ValueError("The pinned source tensor names changed.")
        position = 0
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            with (target / "model.safetensors").open("wb") as output:
                output.write(prefix)
                for name in sorted(
                    target_keys,
                    key=lambda key: converted_header[key]["data_offsets"][0],
                ):
                    left, right = original_header[name], converted_header[name]
                    begin, end = right["data_offsets"]
                    source_begin, source_end = left["data_offsets"]
                    if (
                        left["dtype"] != "BF16"
                        or right["dtype"] != "F16"
                        or left["shape"] != right["shape"]
                        or begin != position
                        or end - begin != source_end - source_begin
                    ):
                        raise ValueError(f"Unexpected tensor metadata: {name}")
                    for offset in range(0, end - begin, 8 * 1024 * 1024):
                        size = min(8 * 1024 * 1024, end - begin - offset)
                        bf16 = np.frombuffer(
                            mapped,
                            dtype="<u2",
                            count=size // 2,
                            offset=8 + original_size + source_begin + offset,
                        )
                        fp16 = (bf16.astype("<u4") << 16).view("<f4").astype("<f2")
                        output.write(fp16.tobytes())
                        del bf16, fp16
                    position = end
    for name in spec["model_files"]:
        if name != "model.safetensors":
            location = repro if name in spec["overrides"] else source
            shutil.copyfile(location / name, target / name)
    actual = {name: digest(target / name) for name in spec["model_files"]}
    if actual != spec["model_files"]:
        raise ValueError("Reconstructed FP16 base differs from the frozen model files.")
    shutil.rmtree("/tmp/base-bf16")
    shutil.rmtree("/tmp/base-hf-cache", ignore_errors=True)
    print(json.dumps({"cpu_base_verified": True, "model_files": actual}), flush=True)


image = (
    # MLX gradient kernels need CUDA headers omitted by the pip runtime wheels.
    modal.Image.from_registry(
        "nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python="3.12"
    )
    .pip_install("mlx-lm[train,cuda12]==0.31.3", "mlx==0.32.3", "pydantic>=2.12,<3")
    .add_local_dir(str(BASE_REPRO), remote_path="/base-repro", copy=True)
    .run_function(build_cpu_base, gpu=None, cpu=2.0, memory=4096, timeout=180)
    .env(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "CUDA_PATH": "/usr/local/cuda",
            "MLX_CUDA_GRAPH_CACHE_SIZE": "4096",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    .add_local_dir(str(PAYLOAD), remote_path="/payload")
    .add_local_file(
        str(TRAINING / "modal_worker.py"), remote_path="/benchmark-worker.py"
    )
)


@app.function(
    image=image,
    gpu="H100!",
    cpu=(4.0, 4.0),
    memory=(16384, 16384),
    timeout=720,
    startup_timeout=120,
    retries=0,
    max_containers=1,
    min_containers=0,
    buffer_containers=0,
    scaledown_window=2,
    serialized=True,
)
def benchmark(deadline_unix: float, training_profile: bool = False):
    if time.time() >= deadline_unix - 15:
        return {"status": "absolute_deadline_expired", "stages": {}}
    results = Path("/results")
    results.mkdir(exist_ok=True)
    stop = threading.Event()
    packages = Path(sysconfig.get_paths()["purelib"])
    library_paths = [packages / "mlx/lib", *sorted((packages / "nvidia").glob("*/lib"))]
    environment = {
        **os.environ,
        "LD_LIBRARY_PATH": ":".join(
            [*map(str, library_paths), os.environ.get("LD_LIBRARY_PATH", "")]
        ),
    }
    for key in (
        "MLX_WORLD_SIZE",
        "MLX_RANK",
        "MLX_HOSTFILE",
        "WORLD_SIZE",
        "RANK",
        "LOCAL_RANK",
        "LD_PRELOAD",
    ):
        environment.pop(key, None)

    def telemetry():
        with (results / "gpu.csv").open("w", buffering=1) as stream:
            writer = csv.writer(stream)
            writer.writerow(["time", "utilization", "vram_mib", "power_watts"])
            while not stop.is_set():
                try:
                    row = (
                        subprocess.check_output(
                            [
                                "nvidia-smi",
                                "--query-gpu=utilization.gpu,memory.used,power.draw",
                                "--format=csv,noheader,nounits",
                            ],
                            text=True,
                            timeout=3,
                        )
                        .strip()
                        .split(",")
                    )
                    writer.writerow([time.time(), *[float(v) for v in row]])
                except (ValueError, subprocess.SubprocessError):
                    pass
                stop.wait(1)

    thread = threading.Thread(target=telemetry, daemon=True)
    thread.start()
    started = time.monotonic()
    report = {
        "gpu": "H100!",
        "training_profile": training_profile,
        "measurement_scope": (
            "throughput_only" if training_profile else "equivalence_benchmark"
        ),
        "modal_timeout_seconds": 720,
        "internal_timeout_seconds": 600,
        "retries": 0,
        "max_containers": 1,
        "absolute_deadline_unix": deadline_unix,
        "cpu_request_and_limit": [4.0, 4.0],
        "memory_request_and_limit_mib": [16384, 16384],
        "stages": {},
        "no_validation_or_test_evaluation": True,
    }

    def remaining_seconds():
        return min(600 - (time.monotonic() - started), deadline_unix - time.time() - 15)

    def stage(name, timeout):
        stage_started = time.monotonic()
        timeout = min(timeout, remaining_seconds())
        if timeout <= 0:
            return {"stage": name, "status": "absolute_deadline_expired"}
        with (results / f"{name}.log").open("w") as log:
            try:
                arguments = [
                    sys.executable, "-u", "/benchmark-worker.py", "--stage", name
                ]
                if training_profile:
                    arguments.append("--training-profile")
                completed = subprocess.run(
                    arguments,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=environment,
                    timeout=timeout,
                    check=False,
                )
                path = results / name / "result.json"
                if completed.returncode == 0 and path.exists():
                    result = json.loads(path.read_text())
                    stage_seconds = time.monotonic() - stage_started
                    result["stage_seconds"] = stage_seconds
                    result["requested_stage_resource_cost_estimate_usd"] = (
                        stage_seconds * RESOURCE_RATE_USD
                    )
                    if result.get("trained_updates"):
                        result["requested_stage_resource_cost_per_update_usd"] = (
                            stage_seconds
                            * RESOURCE_RATE_USD
                            / result["trained_updates"]
                        )
                        result["requested_steady_resource_cost_per_update_usd"] = (
                            RESOURCE_RATE_USD / result["updates_per_second_elapsed"]
                        )
                    return result
                return {
                    "stage": name,
                    "status": "worker_failed",
                    "returncode": completed.returncode,
                    "log_tail": (results / f"{name}.log").read_text()[-4000:],
                }
            except subprocess.TimeoutExpired:
                return {
                    "stage": name,
                    "status": "internal_timeout",
                    "timeout_seconds": timeout,
                }

    try:
        report["stages"]["safe"] = stage("safe", 280)
        safe = report["stages"]["safe"]
        remaining = remaining_seconds()
        memory_total = float(safe.get("hardware", [",0,"])[0].split(",")[1])
        head_ratio = 160 / max(
            1,
            max(
                b["completion_tokens"] / 8
                for b in json.loads(Path("/payload/manifest.json").read_text())[
                    "sampled_batches"
                ]
            ),
        )
        sampled_vram = safe.get("gpu_max_vram_mib")
        memory_margin = (
            sampled_vram is not None
            and memory_total > 0
            and sampled_vram < memory_total / 4
        )
        justified = (
            safe.get("status") == "completed"
            and safe.get("training_finite")
            and memory_margin
            and (training_profile or head_ratio >= 2)
            and remaining >= 150
        )
        report["candidate_justification"] = {
            "baseline_completed_and_finite": safe.get("status") == "completed"
            and safe.get("training_finite"),
            "safe_vram_below_quarter_capacity": memory_margin,
            "head_window_to_mean_completion_ratio": head_ratio,
            "candidate_configuration": (
                "micro8/global8, no activation checkpointing, completion window160"
                if training_profile
                else (
                    "micro8/global8, no activation checkpointing, "
                    "packed completion head"
                )
            ),
            "remaining_internal_seconds": remaining,
            "attempt_candidate": bool(justified),
        }
        if justified:
            report["stages"]["candidate"] = stage(
                "candidate", min(280, max(1, remaining - 15))
            )
        else:
            report["stages"]["candidate"] = {
                "status": "skipped",
                "reason": (
                    "Insufficient successful baseline evidence, VRAM margin, "
                    "head padding, or remaining time."
                ),
            }
    except Exception as error:
        report["status"] = "orchestrator_failed"
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        stop.set()
        thread.join(timeout=5)
    report["function_seconds"] = time.monotonic() - started
    report["requested_resource_cost_estimate_usd"] = (
        report["function_seconds"] * RESOURCE_RATE_USD
    )
    report["budget_limit"] = (
        "Estimate excludes image build/startup/idle. A fixed caller deadline "
        "survives platform rescheduling and the caller stops this specific "
        "ephemeral app. These controls are not an account spending cap."
    )
    report["gpu_csv"] = (results / "gpu.csv").read_text()
    return report


def stop_specific_app(reason):
    app_id = app.app_id
    receipt = {"app_id": app_id, "reason": reason, "time_unix": time.time()}
    if app_id is None:
        receipt["status"] = "no_app_created"
    else:
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "modal", "app", "stop", "--yes", app_id],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=STOP_SECONDS,
                check=False,
            )
            receipt["returncode"] = completed.returncode
            receipt["status"] = (
                "stopped" if completed.returncode == 0 else "stop_failed"
            )
            receipt["output_tail"] = completed.stdout[-2000:]
        except subprocess.TimeoutExpired:
            receipt["status"] = "stop_rpc_timeout"
    (HERE / f"stop-{reason}.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def deadline_watchdog(seconds, reason):
    def expired():
        # App termination also stops containers recreated after a platform crash.
        force_exit = threading.Timer(FORCE_EXIT_SECONDS, lambda: os._exit(124))
        force_exit.daemon = True
        force_exit.start()
        try:
            stop_specific_app(reason)
        finally:
            os.kill(os.getpid(), signal.SIGINT)

    timer = threading.Timer(seconds, expired)
    timer.daemon = True
    timer.start()
    return timer


def run_benchmark(*, training_profile=False):
    global HERE
    HERE = PROFILE_DIR if training_profile else BENCHMARK_DIR
    HERE.mkdir(parents=True, exist_ok=True)
    if not (PAYLOAD / "manifest.json").exists():
        raise RuntimeError(
            "Run prepare_modal_benchmark.py first and review the resource limits."
        )
    if app.app_id is None:
        raise RuntimeError("An ephemeral app must be running before the benchmark.")
    deadline_unix = time.time() + ACTIVE_SECONDS
    launch = {
        "app_id": app.app_id,
        "absolute_deadline_unix": deadline_unix,
        "active_seconds": ACTIVE_SECONDS,
        "training_profile": training_profile,
        "authorized_allowance_usd": 1.0 if training_profile else 2.0,
        "one_function_call_only": True,
        "caller_retries": 0,
    }
    if training_profile:
        launch["overall_accuracy_run_allowance_usd"] = 8.0
    (HERE / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    watchdog = deadline_watchdog(ACTIVE_SECONDS, "active-deadline")
    succeeded = False
    try:
        call = benchmark.spawn(deadline_unix, training_profile=training_profile)
        report = call.get(timeout=max(1, deadline_unix - time.time() - 5))
        telemetry = report.pop("gpu_csv", "")
        (HERE / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        (HERE / "gpu.csv").write_text(telemetry)
        print(json.dumps(report, indent=2))
        succeeded = True
    finally:
        stop_specific_app("completed" if succeeded else "caller-failed")
        watchdog.cancel()


def setup_progress_logging():
    allowed = (
        "Computing checksums for ",
        "Creating blob file for ",
        "Uploading blob file ",
        "Uploading file ",
        "Uploaded ",
        "Created new app with id ",
        "Serializing ",
    )

    class ProgressFilter(logging.Filter):
        def filter(self, record):
            return record.levelno >= logging.INFO or record.getMessage().startswith(
                allowed
            )

    sdk_logger = logging.getLogger("modal-client")
    sdk_logger.setLevel(logging.DEBUG)
    for handler in sdk_logger.handlers:
        handler.setLevel(logging.DEBUG)
        handler.addFilter(ProgressFilter())


@app.local_entrypoint()
def main(training_profile: bool = False):
    run_benchmark(training_profile=training_profile)


if __name__ == "__main__":
    HERE.mkdir(parents=True, exist_ok=True)
    if TRAINING_PROFILE:
        if not (BASE_REPRO / "reproduction.json").exists():
            raise RuntimeError(
                "Training profile requires the frozen base reproduction."
            )
    else:
        prepare_base_reproduction()
    setup_progress_logging()
    setup_seconds = 600 if SETUP_ONLY else BUILD_SECONDS
    setup_started = time.monotonic()
    print(json.dumps({"phase": "setup_started", "setup_only": SETUP_ONLY}), flush=True)
    build_watchdog = deadline_watchdog(setup_seconds, "image-build-deadline")
    try:
        with modal.enable_output(), app.run(detach=False):
            build_watchdog.cancel()
            setup_receipt = {
                "phase": "setup_complete",
                "app_id": app.app_id,
                "setup_seconds": time.monotonic() - setup_started,
                "setup_only": SETUP_ONLY,
                "gpu_inputs_spawned": 0,
                "serialized_function": True,
            }
            (HERE / "setup.json").write_text(
                json.dumps(setup_receipt, indent=2) + "\n"
            )
            print(json.dumps(setup_receipt), flush=True)
            if SETUP_ONLY:
                stop_specific_app("setup-only-completed")
            else:
                run_benchmark(training_profile=TRAINING_PROFILE)
    finally:
        build_watchdog.cancel()
