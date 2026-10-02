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
REMOTE = f"/content/canvas-training/runs/{NAME}"


class RuntimeDisappeared(RuntimeError):
    pass


def cli(session, *arguments):
    result = subprocess.run(
        ["colab", "--auth=oauth2", arguments[0], "-s", session, *arguments[1:]],
        timeout=60 if arguments[0] == "status" else 300,
        capture_output=True,
        text=True,
    )
    output = re.sub(
        r"colab-runtime-proxy-token=[^&\s')]+",
        "colab-runtime-proxy-token=[redacted]",
        result.stdout + result.stderr,
    )
    print(output, flush=True)
    result.check_returncode()
    return output


def require_live_session(output):
    if re.search(r"Session .* not found|No active sessions|No sessions", output, re.I):
        raise RuntimeDisappeared(
            "Colab runtime disappeared; training is interrupted. "
            "Resume the newest verified local checkpoint."
        )


def newest_checkpoint():
    from lab import _task_hash, dataset_hash

    verified = []
    expected_dataset = dataset_hash()
    expected_task = _task_hash()
    for path in (ROOT / "runs").glob(f"{NAME}-checkpoint-*"):
        try:
            saved = json.loads((path / "checkpoint.json").read_text())
            weights = path / "adapter" / "adapters.safetensors"
            if (
                hashlib.sha256(weights.read_bytes()).hexdigest()
                != saved["adapter_sha256"]
            ):
                continue
            progress = path / "adapter" / "progress.json"
            if progress.exists():
                state = json.loads(progress.read_text())
                if (
                    state["step"] != saved["step"]
                    or state["dataset_sha256"] != expected_dataset
                    or state["task_sha256"] != expected_task
                ):
                    continue
                if not all(
                    (path / "adapter" / name).exists()
                    for name in ("optimizer.safetensors", "random.safetensors")
                ):
                    continue
            verified.append((saved["step"], path))
        except (OSError, ValueError, KeyError):
            continue
    if not verified:
        raise RuntimeError("No verified checkpoint is available for recovery.")
    return max(verified)[1]


def recover(session, attempt, console):
    from lab import bundle_colab

    checkpoint = newest_checkpoint()
    saved = json.loads((checkpoint / "checkpoint.json").read_text())
    if saved["step"] >= 15000:
        raise RuntimeError(
            "All training updates are retained; validation still needs to run "
            "using the saved final adapter."
        )
    bundle_colab(checkpoint)
    if console.exists():
        console.rename(
            console.with_name(
                f"{NAME}-interrupted-recovery-{time.time_ns()}-console.log"
            )
        )
    session = f"{session}-r{attempt}"
    print(
        f"Recovering on Colab from retained update {saved['step']} in {session}.",
        flush=True,
    )
    for allocation in range(4):
        try:
            cli(session, "new", "--gpu", "T4")
            break
        except subprocess.CalledProcessError as error:
            if allocation == 3 or "Service Unavailable" not in (
                (error.stdout or "") + (error.stderr or "")
            ):
                raise
            delay = 180 * (allocation + 1)
            print(
                f"Colab allocation unavailable; retrying in {delay}s "
                f"({allocation + 1}/3). Saved checkpoint is unchanged.",
                flush=True,
            )
            time.sleep(delay)
    bootstrap = ROOT / "runs" / "colab-recovery-bootstrap.py"
    source = (ROOT / "colab_job.py").read_text()
    bootstrap.write_text(
        "scope = {'__name__': 'colab_preparation'}\n"
        f"exec({source!r}, scope)\n"
        "scope['prepare_environment']()\n"
    )
    bootstrap_log = ROOT / "runs" / f"{session}-bootstrap.log"
    try:
        with bootstrap_log.open("w") as log:
            preparation = subprocess.Popen(
                [
                    "colab",
                    "--auth=oauth2",
                    "exec",
                    "-s",
                    session,
                    "--timeout",
                    "900",
                    "-f",
                    str(bootstrap),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        for part in sorted((ROOT / "runs").glob("canvas-training.zip.part-*")):
            cli(session, "upload", str(part), f"/content/{part.name}")
        if (
            preparation.wait(timeout=900)
            or "Traceback (most recent call last)" in bootstrap_log.read_text()
        ):
            raise RuntimeError("Colab recovery environment preparation failed.")
        with console.open("w") as log:
            subprocess.Popen(
                [
                    "colab",
                    "--auth=oauth2",
                    "exec",
                    "-s",
                    session,
                    "--timeout",
                    "43200",
                    "-f",
                    str(ROOT / "colab_job.py"),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
    except BaseException:
        cli(session, "stop")
        raise
    ledger = ROOT / "runs" / f"{NAME}-recovery.json"
    entries = json.loads(ledger.read_text()) if ledger.exists() else []
    entries.append(
        {
            "session": session,
            "resume_step": saved["step"],
            "adapter_sha256": saved["adapter_sha256"],
            "started_at": time.time(),
        }
    )
    ledger.write_text(json.dumps(entries, indent=2))
    return session


def backup(session, step):
    destination = ROOT / "runs" / f"{NAME}-checkpoint-{step}"
    (destination / "adapter").mkdir(parents=True, exist_ok=True)
    remote = f"{REMOTE}/adapter/checkpoints/{step:07d}"
    manifest = destination / "checkpoint.parts.json"
    cli(session, "download", f"{remote}/{manifest.name}", str(manifest))
    archive_path = destination / "checkpoint.zip"
    with archive_path.open("wb") as archive:
        for part in json.loads(manifest.read_text()):
            if Path(part["name"]).name != part["name"]:
                raise RuntimeError("Checkpoint manifest contains an invalid file name.")
            local = destination / part["name"]
            cli(session, "download", f"{remote}/{part['name']}", str(local))
            content = local.read_bytes()
            if hashlib.sha256(content).hexdigest() != part["sha256"]:
                raise RuntimeError("Checkpoint part checksum mismatch.")
            archive.write(content)
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None or any(
            Path(name).is_absolute() or ".." in Path(name).parts
            for name in archive.namelist()
        ):
            raise RuntimeError("Checkpoint archive failed verification.")
        archive.extractall(destination)
    if (
        json.loads((destination / "adapter" / "progress.json").read_text())["step"]
        != step
    ):
        raise RuntimeError("Checkpoint progress does not match the requested step.")
    digest = hashlib.sha256(
        (destination / "adapter/adapters.safetensors").read_bytes()
    ).hexdigest()
    (destination / "checkpoint.json").write_text(
        json.dumps({"step": step, "adapter_sha256": digest}, indent=2)
    )
    print(
        f"Backed up checkpoint {step}, including optimizer and random state: {digest}",
        flush=True,
    )
    for path in [archive_path, *destination.glob("checkpoint.zip.part-*")]:
        path.unlink()


def finish(session):
    output = ROOT / "runs" / f"{NAME}-results.zip"
    manifest = output.with_suffix(".parts.json")
    cli(
        session,
        "download",
        "/content/canvas-270m-v4-sessions-results.parts.json",
        str(manifest),
    )
    with output.open("wb") as destination:
        for part in json.loads(manifest.read_text()):
            if Path(part["name"]).name != part["name"]:
                raise RuntimeError("Result manifest contains an invalid file name.")
            local = output.parent / part["name"]
            cli(session, "download", f"/content/{part['name']}", str(local))
            content = local.read_bytes()
            if hashlib.sha256(content).hexdigest() != part["sha256"]:
                raise RuntimeError("Result part checksum mismatch.")
            destination.write(content)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Downloaded archive failed its CRC check.")
        if any(
            Path(name).is_absolute() or ".." in Path(name).parts
            for name in archive.namelist()
        ):
            raise RuntimeError("Archive contains a path outside the run directory.")
        archive.extractall(ROOT / "runs")
    run = ROOT / "runs" / NAME
    metadata = json.loads((run / "training.json").read_text())
    if (
        hashlib.sha256((run / "examples.jsonl").read_bytes()).hexdigest()
        != metadata["dataset_sha256"]
    ):
        raise RuntimeError(
            "Frozen training dataset does not match its recorded checksum."
        )
    cli(session, "stop")
    print("Results saved and Colab GPU stopped.", flush=True)
    for name in ("actions.py", "dataset.py", "lab.py", "sessions.py"):
        source = run / "source" / name
        if source.read_bytes() != (ROOT / source.name).read_bytes():
            raise RuntimeError(
                "Local evaluation code changed; use the frozen source before testing."
            )
    for file in ("examples.jsonl", "sessions.jsonl"):
        if (
            hashlib.sha256((ROOT / file).read_bytes()).digest()
            != hashlib.sha256((run / file).read_bytes()).digest()
        ):
            raise RuntimeError(
                "Local data changed; evaluate the frozen dataset before testing."
            )
    for mode, file in [
        ("evaluate", "test.json"),
        ("evaluate-sessions", "sessions-test.json"),
    ]:
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
                    str(run / file),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    with zipfile.ZipFile(output, "a", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in (
            "test.json",
            "sessions-test.json",
            "evaluate-test.log",
            "evaluate-sessions-test.log",
        ):
            archive.write(run / name, f"{NAME}/{name}")
        ledger = ROOT / "runs" / f"{NAME}-recovery.json"
        if ledger.exists():
            archive.write(ledger, f"{NAME}/recovery.json")
    print("Fresh single-command and closed-loop session tests complete.", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Back up Colab, retrieve results, stop its GPU, and test locally."
    )
    parser.add_argument("--session", default="canvas-270m-v4-sessions")
    parser.add_argument("--max-recoveries", type=int, default=2)
    args = parser.parse_args()
    console = ROOT / "runs" / f"{NAME}-console.log"
    backed_up = set()
    deadline = time.monotonic() + 13 * 60 * 60
    next_status = 0
    status_failures = 0
    recoveries = 0
    while time.monotonic() < deadline:
        text = console.read_text() if console.exists() else ""
        if time.monotonic() >= next_status:
            try:
                require_live_session(cli(args.session, "status"))
                status_failures = 0
            except RuntimeDisappeared:
                if recoveries >= args.max_recoveries:
                    raise
                recoveries += 1
                args.session = recover(args.session, recoveries, console)
                status_failures = 0
                next_status = 0
                continue
            except subprocess.SubprocessError as error:
                status_failures += 1
                print(
                    f"Runtime status check failed ({status_failures}/3): {error}",
                    flush=True,
                )
                if status_failures >= 3:
                    raise RuntimeError(
                        "Cannot verify Colab liveness after three checks."
                    ) from error
            next_status = time.monotonic() + 120
        for step in map(
            int, re.findall(r"Global step (\d+): Saved resumable checkpoint", text)
        ):
            if step not in backed_up:
                try:
                    backup(args.session, step)
                    backed_up.add(step)
                except subprocess.SubprocessError as error:
                    print(f"Checkpoint {step} download will retry: {error}", flush=True)
        if "Training and validation complete. Download" in text:
            try:
                finish(args.session)
                return
            except subprocess.SubprocessError as error:
                print(f"Result retrieval or evaluation failed: {error}", flush=True)
                if (ROOT / "runs" / NAME / "training.json").exists():
                    raise
        if "Traceback (most recent call last)" in text or "[colab] Error" in text:
            cli(args.session, "stop")
            raise RuntimeError(f"Colab failed; inspect {console}")
        time.sleep(30)
    cli(args.session, "stop")
    raise TimeoutError("The Colab job exceeded its supervised runtime.")


if __name__ == "__main__":
    main()
