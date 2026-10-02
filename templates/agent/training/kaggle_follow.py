import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAME = "colab-270m-v4-sessions"
OUTPUT = ROOT / "runs" / "kaggle" / "download"


def cli(*arguments, timeout=60):
    result = subprocess.run(
        ["kaggle", *arguments], capture_output=True, text=True, timeout=timeout
    )
    result.check_returncode()
    return result.stdout


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


def finish_workflow(kernel, directory):
    from gpu_benchmark import quality_comparison

    download = directory / "download"
    cli(
        "kernels",
        "output",
        kernel,
        "--path",
        str(download),
        "--file-pattern",
        r"^refinement(?:/|-)",
        "--page-size",
        "200",
        timeout=1200,
    )
    launch = json.loads((directory / "run.json").read_text())
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
    launch.update(
        status="complete",
        downloaded_adapter=str(weights.parent),
        verification="checkpoint weights, task, dataset and warm-start hashes match",
        adapter_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
        training=result,
        validation=reports,
        test_set_used=False,
    )
    (directory / "run.json").write_text(json.dumps(launch, indent=2))
    print(
        f"Verified the completed general workflow model. Reports: {directory}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Retrieve and evaluate Kaggle training."
    )
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--workflow", type=Path)
    args = parser.parse_args()
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
            print(f"Kaggle status check failed ({failures}/3): {error}", flush=True)
            if failures >= 3:
                raise
            time.sleep(60)
            continue
        match = re.search(r'has status "([^"]+)"', status)
        if match is None:
            raise ValueError("Unrecognized Kaggle runtime status.")
        state = match[1].split(".")[-1].lower()
        if state in ("complete", "error", "cancel_acknowledged", "cancelled"):
            log = cli("kernels", "logs", args.kernel)
            (output / "kernel.log").write_text(log)
            if state != "complete":
                raise RuntimeError(
                    f"Kaggle training failed; see {output / 'kernel.log'}"
                )
            if args.workflow:
                finish_workflow(args.kernel, args.workflow)
            else:
                finish(args.kernel)
            return
        time.sleep(60)
    raise TimeoutError("Kaggle training exceeded its supervised runtime.")


if __name__ == "__main__":
    main()
