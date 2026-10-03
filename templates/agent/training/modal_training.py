import csv
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import sysconfig
import threading
import time
import zipfile
from pathlib import Path

import modal

from modal_benchmark import image as benchmark_image
from modal_benchmark import setup_progress_logging

TRAINING = Path(__file__).resolve().parent
HERE = Path(
    os.environ.get("CANVAS_MODAL_RUN_DIRECTORY", TRAINING / "runs/v8-quality-modal")
)
INPUT = HERE / "input"
INPUT_ZIP = HERE / "input.zip"
RUN_PREFIX = os.environ.get("CANVAS_MODAL_RUN_PREFIX", "v8")
if not re.fullmatch(r"v[89]", RUN_PREFIX):
    raise ValueError("Use a separate v8 or v9 frozen run directory.")
VOLUME_NAME = "canvas-270m-quality-checkpoints"
BUDGET_USD = 8.0
BILLING_BASELINE = Path(
    os.environ.get(
        "CANVAS_MODAL_BILLING_BASELINE", HERE / "profile/billing-before.json"
    )
)
ACTIVE_SECONDS = 6300
RESOURCE_RATE_USD = 0.00118492
app = modal.App(f"canvas-270m-{RUN_PREFIX}-quality")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = benchmark_image.add_local_file(
    str(INPUT_ZIP), remote_path="/job-input.zip"
).add_local_file(
    str(TRAINING / "modal_training_worker.py"), remote_path="/training-worker.py"
)


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


@app.function(
    image=image,
    gpu="H100!",
    cpu=(4.0, 4.0),
    memory=(16384, 16384),
    volumes={"/results": volume},
    timeout=6400,
    startup_timeout=120,
    retries=0,
    max_containers=1,
    min_containers=0,
    buffer_containers=0,
    scaledown_window=2,
    serialized=True,
)
def run_task(run_name, mode, case, step, deadline_unix, native_passed=False):
    if not re.fullmatch(r"v[89]-[a-f0-9]{12}", run_name):
        raise ValueError("Invalid frozen run name.")
    if mode not in ("train", "selection", "language", "full", "test"):
        raise ValueError("Invalid training task.")
    job = Path("/job")
    with zipfile.ZipFile("/job-input.zip") as archive:
        for member in archive.infolist():
            path = Path(member.filename)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Invalid frozen input archive path.")
        archive.extractall(job)
    volume.reload()
    output = Path("/results") / run_name
    output.mkdir(exist_ok=True)
    incoming_manifest = Path("/job/manifest.json").read_bytes()
    retained_manifest = output / "input-manifest.json"
    if (
        retained_manifest.exists()
        and retained_manifest.read_bytes() != incoming_manifest
    ):
        raise ValueError("Stored recovery output belongs to another frozen input.")
    retained_manifest.write_bytes(incoming_manifest)
    stop = threading.Event()
    packages = Path(sysconfig.get_paths()["purelib"])
    paths = [packages / "mlx/lib", *sorted((packages / "nvidia").glob("*/lib"))]
    environment = {**os.environ, "LD_LIBRARY_PATH": ":".join(map(str, paths))}
    for name in (
        "MLX_WORLD_SIZE",
        "MLX_RANK",
        "MLX_HOSTFILE",
        "WORLD_SIZE",
        "RANK",
        "LOCAL_RANK",
        "LD_PRELOAD",
    ):
        environment.pop(name, None)
    hardware = (
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
    if len(hardware) != 1 or "H100" not in hardware[0]:
        raise RuntimeError("The accuracy run requires exactly one H100.")

    def telemetry():
        with (output / "gpu.csv").open("a", buffering=1) as stream:
            writer = csv.writer(stream)
            if stream.tell() == 0:
                writer.writerow(
                    ["time", "mode", "case", "utilization", "vram_mib", "power_watts"]
                )
            while not stop.is_set():
                try:
                    values = (
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
                    writer.writerow(
                        [time.time(), mode, case, *[float(v) for v in values]]
                    )
                except (ValueError, subprocess.SubprocessError):
                    pass
                stop.wait(1)

    def commit_checkpoints():
        while True:
            for path in sorted(
                (output / "refinement/dual-window/adapter/checkpoints").glob(
                    "*/progress.json"
                )
            ):
                receipt = path.with_name("checkpoint-receipt.json")
                if receipt.exists():
                    continue
                try:
                    progress = json.loads(path.read_text())
                    if not progress.get("config_sha256") or not progress.get(
                        "state_files"
                    ):
                        continue
                    files = {
                        name: digest(path.parent / name)
                        for name in (
                            "adapters.safetensors",
                            "optimizer.safetensors",
                            "random.safetensors",
                            "progress.json",
                        )
                    }
                    if any(
                        files[name] != sha
                        for name, sha in progress["state_files"].items()
                    ):
                        continue
                except (FileNotFoundError, json.JSONDecodeError):
                    continue
                receipt.write_text(
                    json.dumps({"step": progress["step"], "files": files}, indent=2)
                    + "\n"
                )
                volume.commit()
                print(
                    json.dumps({"checkpoint_committed": progress["step"]}), flush=True
                )
            if stop.wait(5):
                return

    threads = [
        threading.Thread(target=fn, daemon=True)
        for fn in (telemetry, commit_checkpoints)
    ]
    for thread in threads:
        thread.start()
    started = time.monotonic()
    try:
        remaining = deadline_unix - time.time() - 20
        if remaining <= 0:
            raise TimeoutError("The run's absolute deadline expired.")
        subprocess.run(
            [
                sys.executable,
                "-u",
                "/training-worker.py",
                "--mode",
                mode,
                "--case",
                case,
                "--step",
                str(step),
                "--output",
                str(output),
                *(["--native-passed"] if native_passed else []),
            ],
            env=environment,
            timeout=min(4800 if mode == "train" else 2400, remaining),
            check=True,
        )
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=10)
        commit_checkpoints()
        volume.commit()
    report_path = output / f"{mode}-{case}.json"
    report = json.loads(report_path.read_text())
    artifacts = {
        str(path.relative_to(output)): digest(path)
        for path in output.rglob("*")
        if path.is_file()
        and "/checkpoints/" not in str(path)
        and path.suffix in (".json", ".yaml", ".log", ".csv", ".safetensors")
    }
    return {
        "report": report,
        "artifacts": artifacts,
        "hardware": hardware,
        "function_seconds": time.monotonic() - started,
    }


def billing():
    return json.loads(
        subprocess.check_output(
            [sys.executable, "-m", "modal", "billing", "summary", "--json"],
            text=True,
            timeout=20,
        )
    )


def save_run(run):
    (HERE / "run.json").write_text(json.dumps(run, indent=2) + "\n")


def stop_app(reason):
    receipt = {"app_id": app.app_id, "reason": reason, "time_unix": time.time()}
    if app.app_id:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "modal", "app", "stop", "--yes", app.app_id],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            receipt.update(
                returncode=result.returncode,
                status="stopped" if result.returncode == 0 else "stop_failed",
            )
        except subprocess.TimeoutExpired:
            receipt["status"] = "stop_rpc_timeout"
    (HERE / f"stop-{reason}.json").write_text(json.dumps(receipt, indent=2) + "\n")


def watchdog(seconds, reason):
    def expire():
        force = threading.Timer(20, lambda: os._exit(124))
        force.daemon = True
        force.start()
        try:
            stop_app(reason)
        finally:
            os.kill(os.getpid(), signal.SIGINT)

    timer = threading.Timer(seconds, expire)
    timer.daemon = True
    timer.start()
    return timer


def download_artifacts(run_name, files):
    destination = HERE / "download"
    destination.mkdir(exist_ok=True)
    for name, expected in files.items():
        path = destination / name
        if path.exists() and digest(path) == expected:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".partial")
        with temporary.open("wb") as stream:
            for chunk in volume.read_file(f"/{run_name}/{name}"):
                stream.write(chunk)
        if digest(temporary) != expected:
            raise ValueError(f"Downloaded artifact checksum mismatch: {name}")
        temporary.replace(path)


def pipeline(run, deadline):
    from gpu_benchmark import checkpoint_regressions, choose_checkpoint

    manifest = json.loads((INPUT / "manifest.json").read_text())
    config = manifest["configuration"]
    stop_budget = threading.Event()

    def monitor_budget():
        while not stop_budget.wait(30):
            try:
                used = float(billing()["metered_cost"]) - run["billing_baseline_usd"]
            except (subprocess.SubprocessError, ValueError, KeyError):
                continue
            if used >= BUDGET_USD - 0.30:
                stop_app("credit-budget")
                os.kill(os.getpid(), signal.SIGINT)
                return

    monitor = threading.Thread(target=monitor_budget, daemon=True)
    monitor.start()

    def task(mode, case="baseline", step=0, native=False):
        saved = HERE / "task-receipts" / f"{mode}-{case}.json"
        report_path = HERE / "download" / f"{mode}-{case}.json"
        if saved.exists() and report_path.exists():
            receipt = json.loads(saved.read_text())
            if (
                receipt["input_manifest_sha256"] == run["input_manifest_sha256"]
                and digest(report_path) == receipt["report_sha256"]
                and all(
                    (HERE / "download" / name).exists()
                    and digest(HERE / "download" / name) == sha
                    for name, sha in receipt["case_artifacts"].items()
                )
            ):
                print(
                    json.dumps({"reused_verified_task": mode, "case": case}), flush=True
                )
                return json.loads(report_path.read_text())
        run.update(status=mode, app_id=app.app_id, active_case=case)
        save_run(run)
        print(json.dumps({"phase": mode, "case": case, "step": step}), flush=True)
        response = run_task.spawn(
            run["run_name"], mode, case, step, deadline, native
        ).get(timeout=max(1, deadline - time.time() - 10))
        download_artifacts(run["run_name"], response["artifacts"])
        if report_path.exists():
            saved.parent.mkdir(exist_ok=True)
            saved.write_text(
                json.dumps(
                    {
                        "input_manifest_sha256": run["input_manifest_sha256"],
                        "report_sha256": digest(report_path),
                        "case_artifacts": {
                            name: sha
                            for name, sha in response["artifacts"].items()
                            if name.startswith(f"refinement/{case}/")
                        },
                    },
                    indent=2,
                )
                + "\n"
            )
        run.setdefault("tasks", []).append(
            {
                "mode": mode,
                "case": case,
                "function_seconds": response["function_seconds"],
            }
        )
        save_run(run)
        return response["report"]

    try:
        task("train")
        baseline = task("selection")["metrics"]
        candidates = {
            step: task("selection", f"step-{step}", step)["metrics"]
            for step in config["checkpoint_selection_steps"]
        }
        selection = choose_checkpoint(baseline, candidates)
        run.update(selection=selection, promotion_passed=False, test_set_used=False)
        save_run(run)
        if manifest.get("refinement_mode") == "lexical":
            language_baseline = task("language")["metrics"]
            language_candidates = {
                step: task("language", f"step-{step}", step)["metrics"]
                for step in candidates
            }
            run["language_validation"] = {
                "baseline": language_baseline,
                "candidates": language_candidates,
                "regressions": {
                    step: checkpoint_regressions(language_baseline, metrics)
                    + [
                        key
                        for key in (
                            "document_agreement_rate",
                            "selection_agreement_rate",
                            "camera_agreement_rate",
                        )
                        if metrics[key] < language_baseline[key]
                    ]
                    for step, metrics in language_candidates.items()
                },
                "development_set_used": True,
                "test_set_used": False,
            }
            save_run(run)
        chosen = selection["chosen_step"]
        if not chosen:
            run["status"] = "complete_retained_baseline"
            return
        if run.get("language_validation", {}).get("regressions", {}).get(chosen):
            run["status"] = "complete_retained_baseline"
            return
        before = task("full", "baseline")["metrics"]
        after = task("full", "selected", chosen)["metrics"]
        regressions = {}
        for pool, old in before.items():
            changes = checkpoint_regressions(old, after[pool])
            changes.extend(
                key
                for key in (
                    "document_agreement_rate",
                    "selection_agreement_rate",
                    "camera_agreement_rate",
                )
                if after[pool][key] < old[key]
            )
            regressions[pool] = changes
        run.update(
            full_validation={
                "baseline": before,
                "selected": after,
                "regressions": regressions,
            }
        )
        if any(regressions.values()):
            run["status"] = "complete_retained_baseline"
            return
        selected = HERE / "download/refinement/selected/adapter"
        run.update(
            status="trained",
            promotion_passed=True,
            adapter_sha256=digest(selected / "adapters.safetensors"),
            downloaded_adapter=str(selected.resolve()),
        )
        save_run(run)
        native = subprocess.run(
            [
                str(TRAINING / ".venv/bin/python"),
                "-c",
                (
                    "from pathlib import Path; "
                    "from kaggle_follow import verify_native; "
                    "verify_native(Path(__import__('sys').argv[1]))"
                ),
                str(HERE),
            ],
            cwd=TRAINING,
            timeout=600,
            check=False,
        )
        native_result = HERE / "native-check/result.json"
        report = json.loads(native_result.read_text()) if native_result.exists() else {}
        run["native_verification"] = report
        if (
            native.returncode
            or report.get("status") != "passed"
            or report.get("adapter_sha256") != run["adapter_sha256"]
        ):
            run.update(status="native_check_failed", promotion_passed=False)
            return
        run["native_check_passed"] = True
        task("test", "selected", chosen, native=True)
        run.update(status="complete", test_set_used=True)
    finally:
        stop_budget.set()
        monitor.join(timeout=25)
        save_run(run)


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((INPUT / "manifest.json").read_text())
    if (manifest.get("refinement_mode") == "lexical") != (RUN_PREFIX == "v9"):
        raise ValueError("The run prefix must match its frozen refinement mode.")
    for name in ("modal_training.py", "modal_training_worker.py"):
        expected = manifest.get("source_sha256", {}).get(name)
        if expected and digest(TRAINING / name) != expected:
            raise ValueError(f"Frozen Modal source changed: {name}")
    for name, sha in manifest["files"].items():
        if digest(INPUT / name) != sha:
            raise ValueError(f"Prepared input changed: {name}")
    if not INPUT_ZIP.exists():
        pending = INPUT_ZIP.with_suffix(".zip.partial")
        with zipfile.ZipFile(pending, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(INPUT.rglob("*")):
                if path.is_file():
                    archive.write(path, str(path.relative_to(INPUT)))
        pending.replace(INPUT_ZIP)
    with zipfile.ZipFile(INPUT_ZIP) as archive:
        expected = {
            **manifest["files"],
            "manifest.json": digest(INPUT / "manifest.json"),
        }
        if set(archive.namelist()) != set(expected):
            raise ValueError("Frozen input archive has different files.")
        for name, sha in expected.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != sha:
                raise ValueError(f"Frozen input archive changed: {name}")
    if (HERE / "run.json").exists():
        run = json.loads((HERE / "run.json").read_text())
        if run["input_manifest_sha256"] != digest(INPUT / "manifest.json"):
            raise ValueError("Recovery requires the original frozen input.")
        if run["status"].startswith("complete"):
            print(json.dumps({"status": run["status"], "already_completed": True}))
            return
        if run.get("app_id"):
            apps = json.loads(
                subprocess.check_output(
                    [sys.executable, "-m", "modal", "app", "list", "--json"],
                    text=True,
                    timeout=20,
                )
            )
            if any(
                row["app_id"] == run["app_id"] and row["state"] != "stopped"
                for row in apps
            ):
                raise RuntimeError(
                    "The original app is still active; do not duplicate it."
                )
        if run.get("absolute_deadline_unix", float("inf")) <= time.time() + 30:
            raise TimeoutError("The original run deadline has expired.")
    else:
        initial = billing()
        baseline_file = BILLING_BASELINE
        baseline = (
            json.loads(baseline_file.read_text()) if baseline_file.exists() else initial
        )
        run = {
            "status": "prepared",
            "budget_usd": BUDGET_USD,
            "billing_baseline_usd": float(baseline["metered_cost"]),
            "billing_before": initial,
            "billing_scope_includes_profile": baseline_file.exists(),
            "input_manifest_sha256": digest(INPUT / "manifest.json"),
            "run_name": RUN_PREFIX + "-" + digest(INPUT / "manifest.json")[:12],
            "test_set_used": False,
            "source_sha256": {
                name: digest(TRAINING / name)
                for name in ("modal_training.py", "modal_training_worker.py")
            },
        }
        save_run(run)
    setup_progress_logging()
    setup = watchdog(300, "image-build-deadline")
    active = None
    try:
        with modal.enable_output(), app.run(detach=False):
            setup.cancel()
            remaining_usd = (
                BUDGET_USD
                - 0.30
                - (float(billing()["metered_cost"]) - run["billing_baseline_usd"])
            )
            allowed_seconds = min(ACTIVE_SECONDS, remaining_usd / RESOURCE_RATE_USD)
            deadline = run.get("absolute_deadline_unix", time.time() + allowed_seconds)
            if deadline <= time.time() + 30:
                stop_app("credit-budget")
                raise TimeoutError("Insufficient remaining time or credit allowance.")
            active = watchdog(deadline - time.time(), "active-deadline")
            run.update(app_id=app.app_id, absolute_deadline_unix=deadline)
            save_run(run)
            try:
                pipeline(run, deadline)
            finally:
                stop_app(
                    "completed"
                    if run["status"].startswith("complete")
                    else "caller-failed"
                )
    except BaseException as error:
        run.update(status="interrupted", error=f"{type(error).__name__}: {error}")
        save_run(run)
        raise
    finally:
        setup.cancel()
        if active:
            active.cancel()
        run["billing_after"] = billing()
        save_run(run)
        print(
            json.dumps(
                {
                    "status": run["status"],
                    "app_id": run.get("app_id"),
                    "billing": run["billing_after"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
