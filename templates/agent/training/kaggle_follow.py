import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
NAME = "colab-270m-v4-sessions"
OUTPUT = ROOT / "runs" / "kaggle" / "download"


def cli(*arguments, timeout=60):
    result = subprocess.run(
        ["kaggle", *arguments], capture_output=True, text=True, timeout=timeout
    )
    result.check_returncode()
    return result.stdout


def wait_for_dataset(dataset):
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        try:
            if cli("datasets", "status", dataset).strip().lower() == "ready":
                return
        except subprocess.SubprocessError:
            print("Kaggle upload readiness check unavailable; retrying.", flush=True)
        time.sleep(30)
    raise TimeoutError("Final checkpoint upload did not become ready.")


def finish(kernel):
    cli(
        "kernels",
        "output",
        kernel,
        "--path",
        str(OUTPUT),
        "--file-pattern",
        r"^canvas-270m-v4-sessions-results\.(zip|parts\.json)$",
        "--page-size",
        "200",
        timeout=1200,
    )
    output = OUTPUT / "canvas-270m-v4-sessions-results.zip"
    parts = json.loads(output.with_suffix(".parts.json").read_text())
    with output.open("rb") as source:
        for part in parts:
            if (
                hashlib.sha256(source.read(16 * 1024 * 1024)).hexdigest()
                != (part["sha256"])
            ):
                raise ValueError("Downloaded result checksum mismatch.")
        if source.read(1):
            raise ValueError("Result archive contains unexpected trailing bytes.")
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None or any(
            Path(name).is_absolute() or ".." in Path(name).parts
            for name in archive.namelist()
        ):
            raise ValueError("Downloaded result archive failed validation.")
        archive.extractall(OUTPUT)
    run = OUTPUT / NAME
    metadata = json.loads((run / "training.json").read_text())
    if metadata["configuration"]["iters"] != 15000:
        raise ValueError("The downloaded run did not use the agreed update target.")
    for filename in ("examples.jsonl", "sessions.jsonl"):
        if (run / filename).read_bytes() != (ROOT / filename).read_bytes():
            raise ValueError("Local data changed; evaluate the frozen data first.")
    if (
        hashlib.sha256((run / "examples.jsonl").read_bytes()).hexdigest()
        != (metadata["dataset_sha256"])
    ):
        raise ValueError("Frozen training data checksum mismatch.")
    for filename in ("actions.py", "dataset.py", "lab.py", "sessions.py"):
        if (run / "source" / filename).read_bytes() != (ROOT / filename).read_bytes():
            raise ValueError("Local evaluation code changed; use the frozen source.")
    if (
        "Global step 15000: Saved resumable checkpoint."
        not in (run / "training.log").read_text()
    ):
        raise ValueError("The final checkpoint did not reach 15,000 updates.")
    print(
        "Verified the completed 15,000-update run and downloaded adapter.", flush=True
    )
    for mode, filename in (
        ("evaluate", "test.json"),
        ("evaluate-sessions", "sessions-test.json"),
    ):
        if (run / filename).exists():
            continue
        with (run / f"{mode}-test.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "lab.py"),
                    mode,
                    "--split",
                    "test",
                    "--adapter",
                    str(run / "adapter"),
                    "--output",
                    str(run / filename),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    print(f"Fresh test reports saved to {run}.", flush=True)


def finish_workflow(kernel, directory, *, training_only=False):
    from gpu_benchmark import quality_comparison

    download = directory / "download"
    launch = json.loads((directory / "run.json").read_text())
    final_step = f"{launch['updates']:07d}"
    pattern = (
        r"^refinement-[^/]+\.json$|^refinement/(?:baseline|dual-window)/"
        r"(?:[^/]+\.(?:json|yaml|log|csv)|data/tokens\.json|adapter/"
        r"(?:adapters\.safetensors|adapter_config\.json|checkpoints/"
        + final_step
        + r"/(?:adapters\.safetensors|optimizer\.safetensors|"
        r"random\.safetensors|progress\.json)))$"
    )
    cli(
        "kernels",
        "output",
        kernel,
        "--path",
        str(download),
        "--file-pattern",
        pattern,
        "--page-size",
        "200",
        timeout=1200,
    )
    result, weights = verify_workflow_checkpoint(download, launch)
    launch.update(
        status="trained",
        downloaded_adapter=str(weights.parent),
        verification="checkpoint weights, task, dataset and warm-start hashes match",
        adapter_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
        training=result,
        test_set_used=launch.get("test_set_used", False),
    )
    (directory / "run.json").write_text(json.dumps(launch, indent=2))
    if training_only:
        print("Recovered and verified the finished training checkpoint.", flush=True)
        return
    candidate = weights.parent.parent
    reports = {
        mode: quality_comparison(
            download / "refinement/baseline", candidate, guarded=mode == "guarded"
        )
        for mode in ("raw", "guarded")
    }
    for report in reports.values():
        report["scope"] = (
            f"General workflow validation after {launch['updates']} updates; "
            "native layout is verified separately in browser tests."
        )
    (directory / "verified-comparison.json").write_text(json.dumps(reports, indent=2))
    launch.update(status="complete", validation=reports)
    (directory / "run.json").write_text(json.dumps(launch, indent=2))
    print(
        f"Verified the completed general workflow model. Reports: {directory}",
        flush=True,
    )


def verify_workflow_checkpoint(download, launch):
    candidate = download / "refinement/dual-window"
    result = json.loads((download / "refinement-training.json").read_text())
    for key, expected in (
        ("end_step", launch["updates"]),
        ("dataset_sha256", launch["dataset_sha256"]),
        ("task_sha256", launch["task_sha256"]),
        ("warm_start_sha256", launch["warm_start_sha256"]),
        ("initialization", "weights_only"),
    ):
        if result[key] != expected:
            raise ValueError(f"Downloaded workflow run differs in {key}.")
    weights = candidate / "adapter/adapters.safetensors"
    checkpoint = candidate / "adapter/checkpoints" / f"{launch['updates']:07d}"
    progress = json.loads((checkpoint / "progress.json").read_text())
    if progress["step"] != launch["updates"] or any(
        progress[key] != launch[key] for key in ("dataset_sha256", "task_sha256")
    ):
        raise ValueError("Workflow adapter does not match its final checkpoint.")
    import numpy as np
    from safetensors.numpy import load_file

    final = load_file(str(weights))
    saved = load_file(str(checkpoint / "adapters.safetensors"))
    if (
        not result["training_finite"]
        or final.keys() != saved.keys()
        or not all(np.array_equal(values, saved[key]) for key, values in final.items())
    ):
        raise ValueError("Workflow weights differ from the completed checkpoint.")
    return result, weights


def finish_selected_workflow(kernel, directory):
    finish_workflow(kernel, directory, training_only=True)
    launch = json.loads((directory / "run.json").read_text())
    result = json.loads(
        (directory / "download/refinement-selected-evaluation.json").read_text()
    )
    if (
        result["adapter_sha256"] != launch["adapter_sha256"]
        or not result["test_set_used"]
    ):
        raise ValueError("Final scores do not match the verified adapter.")
    totals = {
        "valid": 1200,
        "valid-guarded": 1200,
        "sessions-valid": 480,
        "sessions-valid-guarded": 480,
        "test-guarded": 2880,
        "sessions-test-guarded": 1440,
    }
    for name, total in totals.items():
        report = result["reports"][name]
        count = report["turns"] if name.startswith("sessions-") else report["total"]
        if count != total or any(
            report[key] != launch[key]
            for key in ("adapter_sha256", "dataset_sha256", "task_sha256")
        ):
            raise ValueError(f"Final report differs in {name}.")
    launch.update(
        status="evaluated", test_set_used=True, final_scores=result["reports"]
    )
    (directory / "run.json").write_text(json.dumps(launch, indent=2))
    print("Final validation and fresh holdout scores are saved.", flush=True)


def launch_final_scoring(directory):
    directory = directory.resolve()
    launch = json.loads((directory / "run.json").read_text())
    native = launch.get("native_verification", {})
    if native.get("status") != "passed" or native.get("adapter_sha256") != launch.get(
        "adapter_sha256"
    ):
        raise ValueError(
            "Final scoring requires passed native checks for this adapter."
        )
    if launch.get("test_set_used") or launch.get("test_set_started"):
        raise ValueError("The final holdout has already been used for this adapter.")
    verify_workflow_checkpoint(directory / "download", launch)
    output = directory / "final-score"
    inputs = output / "input"
    inputs.mkdir(parents=True, exist_ok=True)
    bundle = inputs / "canvas-checkpoint.bundle"
    source = directory / "download"
    with zipfile.ZipFile(
        bundle, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1
    ) as archive:
        archive.write(source / "refinement-training.json", "refinement-training.json")
        for case in ("baseline", "dual-window"):
            for filename in (
                "config.yaml",
                "adapter/adapters.safetensors",
                "adapter/adapter_config.json",
            ):
                path = source / "refinement" / case / filename
                archive.write(path, path.relative_to(source))
        for filename in ("adapters.safetensors", "progress.json"):
            path = (
                source
                / "refinement/dual-window/adapter/checkpoints"
                / f"{launch['updates']:07d}"
                / filename
            )
            archive.write(path, path.relative_to(source))
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    dataset = "rohithresearch/canvas-270m-v6-final-checkpoint"
    kernel_id = "rohithresearch/canvas-270m-v6-final-score"
    (inputs / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "Canvas 270m v6 final checkpoint",
                "id": dataset,
                "licenses": [{"name": "CC0-1.0"}],
            },
            indent=2,
        )
    )
    receipt = output / "checkpoint-upload.json"
    upload = {"dataset": dataset, "sha256": digest}
    if receipt.exists():
        if json.loads(receipt.read_text()) != upload:
            raise ValueError("Uploaded checkpoint receipt differs from this bundle.")
        print("Resuming the accepted checkpoint upload.", flush=True)
    else:
        created = cli(
            "datasets",
            "create",
            "--path",
            str(inputs),
            "--quiet",
            "--keep-tabular",
            timeout=1200,
        )
        if "Your private Dataset is being created." not in created:
            raise RuntimeError("Kaggle did not accept the checkpoint dataset.")
        receipt.write_text(json.dumps(upload, indent=2))
    wait_for_dataset(dataset)
    print("Checkpoint upload is ready; preparing the GPU accuracy job.", flush=True)
    kernel = output / "kernel"
    kernel.mkdir(exist_ok=True)
    metadata = json.loads((directory / "kernel/kernel-metadata.json").read_text())
    metadata.update(
        id=kernel_id,
        title="Canvas 270m v6 final score",
        dataset_sources=[launch["dataset"], dataset],
    )
    (kernel / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
    expected = {
        key: launch[key]
        for key in (
            "updates",
            "dataset_sha256",
            "task_sha256",
            "warm_start_sha256",
            "adapter_sha256",
        )
    }
    expected["native_verification"] = "passed"
    sources = {path.name: path.read_text() for path in ROOT.glob("*.py")}
    for name in ("actions.py", "dataset.py", "lab.py", "sessions.py"):
        sources[name] = (Path(launch["frozen_data"]) / name).read_text()
    code = (
        "TASK = 'score'\nEVALUATION_EXPECTED = "
        + repr(expected)
        + "\nEVALUATION_CHECKPOINT_SHA256 = "
        + repr(digest)
        + "\nTRAINING_SOURCES = "
        + repr(sources)
        + "\n"
        + (ROOT / "kaggle_job.py").read_text()
    )
    compile(code, "kaggle_job.py", "exec")
    (kernel / "kaggle_job.py").write_text(code)
    final = {
        **launch,
        "kernel": kernel_id,
        "status": "submitting",
        "test_set_started": True,
        "source_training_kernel": launch["kernel"],
        "checkpoint_bundle_sha256": digest,
    }
    (output / "run.json").write_text(json.dumps(final, indent=2))
    # A timed-out push may still start scoring; preserve that exposure on retry.
    launch.update(
        test_set_started=True,
        final_scoring={"status": "submitting", "kernel": kernel_id},
    )
    (directory / "run.json").write_text(json.dumps(launch, indent=2))
    pushed = cli(
        "kernels",
        "push",
        "--path",
        str(kernel),
        "--timeout",
        "14400",
        "--accelerator",
        "NvidiaTeslaT4",
        timeout=300,
    )
    if "successfully pushed" not in pushed or "not valid" in pushed:
        raise RuntimeError("Kaggle did not accept the final scoring job.")
    final["status"] = "running"
    (output / "run.json").write_text(json.dumps(final, indent=2))
    launch["final_scoring"]["status"] = "running"
    (directory / "run.json").write_text(json.dumps(launch, indent=2))
    print(
        "Selected-model validation and fresh holdout scoring launched on Kaggle.",
        flush=True,
    )
    with (output / "follow.log").open("w") as log:
        subprocess.run(
            [
                sys.executable,
                "-u",
                str(ROOT / "kaggle_follow.py"),
                "--kernel",
                kernel_id,
                "--workflow",
                str(output),
                "--selected-score",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=5 * 60 * 60,
        )
    scored = json.loads((output / "run.json").read_text())
    launch.update(
        status="evaluated",
        test_set_used=True,
        final_scores=scored["final_scores"],
        final_score_kernel=kernel_id,
        final_scoring={"status": "complete", "kernel": kernel_id},
    )
    (directory / "run.json").write_text(json.dumps(launch, indent=2))


def await_final_scoring(directory):
    deadline = time.monotonic() + 5 * 60 * 60
    print("Waiting for completed training and native integration checks.", flush=True)
    while time.monotonic() < deadline:
        try:
            launch = json.loads((directory / "run.json").read_text())
        except json.JSONDecodeError:
            time.sleep(5)
            continue
        if launch.get("test_set_used") or launch.get("test_set_started"):
            print("Final scoring was already scheduled; see final-score/.", flush=True)
            return
        native = launch.get("native_verification", {})
        if native.get("status") == "failed":
            launch["final_scoring"] = {"status": "needs_native_fix"}
            (directory / "run.json").write_text(json.dumps(launch, indent=2))
            print(
                "Native checks failed; the final holdout remains untouched.", flush=True
            )
            return
        if native.get("status") == "passed":
            try:
                launch_final_scoring(directory)
            except (
                OSError,
                ValueError,
                RuntimeError,
                subprocess.SubprocessError,
            ) as error:
                launch = json.loads((directory / "run.json").read_text())
                launch.setdefault("final_scoring", {}).update(
                    status="failed", error=str(error)
                )
                (directory / "run.json").write_text(json.dumps(launch, indent=2))
                raise
            return
        time.sleep(30)
    raise TimeoutError("Training and native checks did not finish within five hours.")


def verify_native(directory):
    directory = directory.resolve()
    launch = json.loads((directory / "run.json").read_text())
    if launch["status"] not in ("trained", "complete") or not launch.get(
        "adapter_sha256"
    ):
        raise ValueError(
            "Native checks require a verified, completed workflow checkpoint."
        )
    output = directory / "native-check"
    output.mkdir(exist_ok=True)
    report = {
        "adapter_sha256": launch["adapter_sha256"],
        "scope": "Native model integration with synthetic speech",
        "browser_report": str(output / "browser.json"),
        "speech_fixture_used": (ROOT / "runs/voice-smoke.wav").exists(),
    }
    process = None
    started = time.monotonic()
    with (output / "service.log").open("w") as service_log:
        try:
            with socket.socket() as address:
                address.bind(("127.0.0.1", 0))
                port = address.getsockname()[1]
            url = f"http://127.0.0.1:{port}"
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    str(ROOT / "voice_server.py"),
                    "--adapter",
                    launch["downloaded_adapter"],
                    "--port",
                    str(port),
                ],
                stdout=service_log,
                stderr=subprocess.STDOUT,
            )
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("The local model service failed to start.")
                try:
                    with urlopen(url + "/health", timeout=2) as response:
                        if response.status == 200:
                            break
                except OSError:
                    time.sleep(1)
            else:
                raise TimeoutError("The local model service did not become ready.")
            environment = {
                **os.environ,
                "VOICE_MODEL_URL": url,
                "PLAYWRIGHT_JSON_OUTPUT_NAME": str(output / "browser.json"),
            }
            if report["speech_fixture_used"]:
                environment["VOICE_SMOKE_AUDIO"] = str(ROOT / "runs/voice-smoke.wav")
            command = [
                "pnpm",
                "--filter",
                "tldraw-agent",
                "test:voice",
                "--grep",
                "model integration",
                "--reporter=line,json",
            ]
            keep_awake = (
                shutil.which("caffeinate") if sys.platform == "darwin" else None
            )
            report["idle_sleep_prevented"] = keep_awake is not None
            if keep_awake:
                # Idle sleep can expire browser timeouts while inference is suspended.
                command = [keep_awake, "-i", *command]
            with (output / "browser.log").open("w") as browser_log:
                result = subprocess.run(
                    command,
                    cwd=ROOT.parents[2],
                    env=environment,
                    stdout=browser_log,
                    stderr=subprocess.STDOUT,
                    timeout=420,
                )
            report.update(
                status="passed" if result.returncode == 0 else "failed",
                returncode=result.returncode,
            )
            if result.returncode == 0:
                browser = json.loads((output / "browser.json").read_text())
                report["tests"] = browser["stats"]
                expected = 2 if report["speech_fixture_used"] else 1
                if (
                    browser["stats"]["unexpected"]
                    or browser["stats"]["expected"] < expected
                ):
                    raise RuntimeError("The native model checks did not all pass.")
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
            report.update(status="failed", error=str(error))
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    report["seconds"] = time.monotonic() - started
    (output / "result.json").write_text(json.dumps(report, indent=2))
    launch["native_verification"] = report
    (directory / "run.json").write_text(json.dumps(launch, indent=2))
    print(f"Native integration check {report['status']}; reports: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Retrieve and evaluate Kaggle training."
    )
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--workflow", type=Path)
    parser.add_argument("--native-check", action="store_true")
    parser.add_argument("--training-only", action="store_true")
    parser.add_argument("--selected-score", action="store_true")
    parser.add_argument("--await-final-score", action="store_true")
    args = parser.parse_args()
    if args.native_check and not args.workflow:
        parser.error("--native-check requires --workflow.")
    if args.training_only and not args.workflow:
        parser.error("--training-only requires --workflow.")
    if args.selected_score and (not args.workflow or args.training_only):
        parser.error("--selected-score requires --workflow without --training-only.")
    if args.await_final_score:
        if not args.workflow or any(
            (args.selected_score, args.native_check, args.training_only)
        ):
            parser.error("--await-final-score requires only --kernel and --workflow.")
        launch = json.loads((args.workflow / "run.json").read_text())
        if launch["kernel"] != args.kernel:
            parser.error("--kernel differs from the workflow being monitored.")
        await_final_scoring(args.workflow)
        return
    output = args.workflow or OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + 13 * 60 * 60
    failures = 0
    while time.monotonic() < deadline:
        try:
            status = cli("kernels", "status", args.kernel)
            print(status, end="", flush=True)
            failures = 0
        except subprocess.SubprocessError as error:
            failures += 1
            print(
                f"Kaggle status check failed ({failures}): {error}; "
                "retrying within the supervised runtime.",
                flush=True,
            )
            time.sleep(60)
            continue
        match = re.search(r'has status "([^"]+)"', status)
        if match is None:
            raise ValueError("Unrecognized Kaggle runtime status.")
        state = match[1].split(".")[-1].lower()
        if state in ("complete", "error", "cancel_acknowledged", "cancelled"):
            try:
                log = cli("kernels", "logs", args.kernel)
                (output / "kernel.log").write_text(log)
                if state != "complete":
                    if args.workflow:
                        finish_workflow(args.kernel, args.workflow, training_only=True)
                    raise RuntimeError(
                        f"Kaggle job failed; any completed workflow checkpoint was "
                        f"recovered. See {output / 'kernel.log'}"
                    )
                if args.workflow:
                    if args.selected_score:
                        finish_selected_workflow(args.kernel, args.workflow)
                    elif args.training_only:
                        finish_workflow(args.kernel, args.workflow, training_only=True)
                    else:
                        finish_workflow(args.kernel, args.workflow)
                else:
                    finish(args.kernel)
            except subprocess.SubprocessError as error:
                print(f"Result download interrupted: {error}; retrying.", flush=True)
                time.sleep(60)
                continue
            if args.native_check:
                verify_native(args.workflow)
            return
        time.sleep(60)
    raise TimeoutError("Kaggle training exceeded its supervised runtime.")


if __name__ == "__main__":
    main()
