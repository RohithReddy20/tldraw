import argparse
import hashlib
import json
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from datetime import UTC, datetime
from pathlib import Path

from actions import INPUT_FORMAT_VERSION, SYSTEM_PROMPT, TOOLS, messages_for, parse_call
from dataset import DATA, ROOT, dataset_hash, prepare_data, read_examples, training_row

MODEL = "mlx-community/functiongemma-270m-it-bf16"
REVISION = "bb327a9ad61044e1496a2bee2365a6b6a6684c72"


def model_path(dtype=None):
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        MODEL,
        revision=REVISION,
        allow_patterns=["*.json", "*.safetensors", "*.jinja"],
        token=False,
    )
    if dtype is None or dtype == "bfloat16":
        return path
    if dtype != "float16":
        raise ValueError("Supported checkpoint dtypes are bfloat16 and float16.")
    converted = ROOT / "runs" / f"base-{REVISION[:8]}-float16"
    if not (converted / "config.json").exists():
        from mlx_lm.convert import convert

        converted.parent.mkdir(parents=True, exist_ok=True)
        convert(hf_path=path, mlx_path=converted, dtype="float16")
    return str(converted)


def load_model(adapter=None):
    from mlx_lm import load

    dtype = None
    if adapter and (Path(adapter).parent / "config.yaml").exists():
        import yaml

        dtype = yaml.safe_load((Path(adapter).parent / "config.yaml").read_text()).get(
            "model_dtype"
        )
    return load(model_path(dtype), adapter_path=str(adapter) if adapter else None)


def inspect_data():
    from transformers import AutoTokenizer

    counts = prepare_data()
    tokenizer = AutoTokenizer.from_pretrained(model_path())
    lengths = check_tokenization(tokenizer, cache=True)
    print(
        json.dumps(
            {
                "splits": counts,
                "max_tokens": max(lengths),
                "mean_tokens": statistics.mean(lengths),
            },
            indent=2,
        )
    )
    row = training_row(read_examples()[0])
    print("\nFirst formatted training example:\n")
    print(tokenizer.apply_chat_template(row["messages"], tools=TOOLS, tokenize=False))


def check_tokenization(tokenizer, max_length=None, *, cache=False):
    import numpy as np

    lengths = []
    encoded = {split: [] for split in ("train", "valid", "test")}
    offsets = {split: [] for split in encoded}
    max_completion = 0
    for index, example in enumerate(read_examples()):
        row = training_row(example)
        full = tokenizer.apply_chat_template(
            row["messages"], tools=TOOLS, return_dict=False
        )
        prompt = tokenizer.apply_chat_template(
            row["messages"][:-1],
            tools=TOOLS,
            add_generation_prompt=True,
            return_dict=False,
        )
        if full[: len(prompt)] != prompt:
            raise ValueError(f"Prompt masking mismatch: {example['id']}")
        completion = tokenizer.decode(full[len(prompt) :])
        if parse_call(completion, example["canvas"]) != example["expected"]:
            raise ValueError(f"Function-call round trip failed: {example['id']}")
        if max_length is not None and len(full) > max_length:
            raise ValueError(
                f"{example['id']} exceeds the sequence limit; increase max_seq_length."
            )
        lengths.append(len(full))
        max_completion = max(max_completion, len(full) - len(prompt))
        if cache:
            encoded[example["split"]].append(np.asarray(full, dtype=np.int32))
            offsets[example["split"]].append(len(prompt))
        if cache and (index + 1) % 20000 == 0:
            print(f"Validated and tokenized {index + 1} examples.", flush=True)
    if cache:
        files = {}
        for split, arrays in encoded.items():
            output = DATA / f"tokens-{split}.npz"
            boundaries = np.concatenate(([0], np.cumsum([len(a) for a in arrays])))
            np.savez_compressed(
                output,
                tokens=np.concatenate(arrays),
                boundaries=boundaries,
                offsets=np.asarray(offsets[split], dtype=np.int32),
            )
            files[output.name] = hashlib.sha256(output.read_bytes()).hexdigest()
        (DATA / "tokens.json").write_text(
            json.dumps(
                {
                    "dataset_sha256": dataset_hash(),
                    "task_sha256": _task_hash(),
                    "max_tokens": max(lengths),
                    "max_completion_tokens": max_completion,
                    "files": files,
                },
                indent=2,
            )
        )
    return lengths


def _task_hash():
    return hashlib.sha256(
        json.dumps(
            [INPUT_FORMAT_VERSION, SYSTEM_PROMPT, TOOLS], sort_keys=True
        ).encode()
    ).hexdigest()


def verify_token_cache(config):
    manifest = DATA / "tokens.json"
    if not manifest.exists():
        return False
    saved = json.loads(manifest.read_text())
    if (
        saved["dataset_sha256"] != dataset_hash()
        or saved["task_sha256"] != _task_hash()
    ):
        return False
    if (
        saved["max_tokens"] > config["max_seq_length"]
        or saved["max_completion_tokens"] > config["completion_window"]
    ):
        raise ValueError(
            "Validated token cache exceeds the configured sequence or completion limit."
        )
    for name, digest in saved["files"].items():
        if hashlib.sha256((DATA / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Token cache checksum mismatch: {name}")
    return True


def train_model(args, *, in_process=False):
    import yaml
    from transformers import AutoTokenizer

    prepare_data()
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", args.run):
        raise ValueError(
            "Run names can contain letters, digits, underscores and hyphens."
        )
    run = ROOT / "runs" / args.run
    adapter = run / "adapter"
    if run.exists():
        raise ValueError("This run already exists. Choose a new --run name.")
    config = yaml.safe_load(args.config.read_text())
    config.update(
        {
            "model": model_path(config.get("model_dtype")),
            "data": str(DATA),
            "adapter_path": str(adapter),
        }
    )
    if args.iters is not None:
        config["iters"] = args.iters
    tokenizer = AutoTokenizer.from_pretrained(config["model"])
    if config.get("loss_mode") == "completion_tokens_v1" and verify_token_cache(config):
        print(
            "Using the validated token cache; dataset and task checksums match.",
            flush=True,
        )
    else:
        check_tokenization(tokenizer, config["max_seq_length"], cache=True)
    run.mkdir(parents=True, exist_ok=True)
    from mlx_lm.lora import CONFIG_DEFAULTS

    metadata = _metadata()
    if config.get("resume_adapter_file"):
        metadata["warm_start_sha256"] = hashlib.sha256(
            Path(config["resume_adapter_file"]).read_bytes()
        ).hexdigest()
    (run / "examples.jsonl").write_bytes((ROOT / "examples.jsonl").read_bytes())
    (run / "task.json").write_text(
        json.dumps([INPUT_FORMAT_VERSION, SYSTEM_PROMPT, TOOLS], indent=2)
    )
    shutil.copytree(DATA, run / "data")
    if (ROOT / "sessions.jsonl").exists():
        shutil.copyfile(ROOT / "sessions.jsonl", run / "sessions.jsonl")
    config["data"] = str(run / "data")
    (run / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    settings = {**CONFIG_DEFAULTS, **config}
    start = time.perf_counter()
    with (run / "training.log").open("w") as log:
        if in_process or config.get("loss_mode") == "completion_tokens_v1":
            if config.get("loss_mode") == "completion_tokens_v1":
                from training import run_training

                def tune(args):
                    run_training(vars(args))
            else:
                from mlx_lm.lora import run as tune

            with (
                redirect_stdout(_TrainingOutput(sys.stdout, log)),
                redirect_stderr(_TrainingOutput(sys.stderr, log)),
            ):
                tune(argparse.Namespace(**settings))
        else:
            _run_training_process(run / "config.yaml", log)
    metadata.update(
        {"training_seconds": time.perf_counter() - start, "configuration": settings}
    )
    (run / "training.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved adapter and training log to {run}")


class _TrainingOutput:
    def __init__(self, console, log):
        self.console = console
        self.log = log

    def write(self, value):
        self.console.write(value)
        self.log.write(value)
        self.flush()
        return len(value)

    def flush(self):
        self.console.flush()
        self.log.flush()


def _run_training_process(config_path, log):
    process = subprocess.Popen(
        [sys.executable, "-m", "mlx_lm", "lora", "--config", str(config_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        if process.wait():
            raise RuntimeError(f"Training failed; see {log.name}")
    except BaseException:
        process.terminate()
        process.wait()
        raise


def _metadata():
    from importlib.metadata import version

    return {
        "model": MODEL,
        "revision": REVISION,
        "mlx_lm_version": version("mlx-lm"),
        "mlx_version": version("mlx"),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_sha256": dataset_hash(),
        "task_sha256": _task_hash(),
    }


def bundle_colab(resume=None):
    prepare_data()
    output = ROOT / "runs" / "canvas-training.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    files = [
        *ROOT.glob("*.py"),
        ROOT / "examples.jsonl",
        ROOT / "config.yaml",
        ROOT / "sessions.jsonl",
        *DATA.glob("tokens-*.npz"),
        DATA / "tokens.json",
    ]
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT))
        warm_start = (
            resume / "adapter" / "adapters.safetensors"
            if resume
            else ROOT
            / "runs"
            / "colab-270m-v3-stable"
            / "adapter"
            / "adapters.safetensors"
        )
        archive.write(warm_start, "warm-start.safetensors")
        if resume:
            checkpoint = json.loads((resume / "checkpoint.json").read_text())
            if (
                hashlib.sha256(warm_start.read_bytes()).hexdigest()
                != checkpoint["adapter_sha256"]
            ):
                raise ValueError(
                    "Resume adapter does not match its checkpoint checksum."
                )
            archive.writestr("resume.json", json.dumps(checkpoint))
            for name in (
                "optimizer.safetensors",
                "random.safetensors",
                "progress.json",
            ):
                path = resume / "adapter" / name
                if path.exists():
                    archive.write(path, f"resume/{name}")
    for path in output.parent.glob(f"{output.name}.part-*"):
        path.unlink()
    with output.open("rb") as source:
        index = 0
        while chunk := source.read(16 * 1024 * 1024):
            output.with_name(f"{output.name}.part-{index:03d}").write_bytes(chunk)
            index += 1
    print(f"Saved frozen training bundle to {output}")


def predict(model, tokenizer, command, canvas, max_tokens=256, history=None):
    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    prompt = tokenizer.apply_chat_template(
        messages_for(command, canvas, history),
        tools=TOOLS,
        add_generation_prompt=True,
        tokenize=False,
    )
    mx.reset_peak_memory()
    start = time.perf_counter()
    chunks = []
    first_token = None
    for response in stream_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=make_sampler(temp=0),
    ):
        if first_token is None:
            first_token = time.perf_counter() - start
        chunks.append(response.text)
        if "<end_function_call>" in response.text or "<end_function_call>" in "".join(
            chunks
        ):
            break
    text = "".join(chunks)
    result = {
        "raw_output": text,
        "seconds": time.perf_counter() - start,
        "first_token_seconds": first_token,
        "peak_model_memory_gb": mx.get_peak_memory() / 1e9,
    }
    try:
        result["prediction"] = parse_call(text, canvas)
        result["error"] = None
    except ValueError as error:
        result["prediction"] = None
        result["error"] = str(error)
    return result


def evaluate(args):
    prepare_data()
    model, tokenizer = load_model(args.adapter)
    examples = [e for e in read_examples() if e["split"] == args.split]
    if args.limit:
        examples = examples[: args.limit]
    rows = []
    for i, example in enumerate(examples):
        result = predict(
            model,
            tokenizer,
            example["command"],
            example["canvas"],
            history=example.get("history"),
        )
        result.update(
            {
                "id": example["id"],
                "command": example["command"],
                "expected": example["expected"],
            }
        )
        result["correct"] = result["prediction"] == result["expected"]
        rows.append(result)
        status = "correct" if result["correct"] else "incorrect"
        print(f"{i + 1}/{len(examples)} {example['id']}: {status}", flush=True)
    report = _summarize(rows)
    report.update(
        {
            **_metadata(),
            "split": args.split,
            "adapter": str(args.adapter) if args.adapter else None,
            "examples": rows,
            "evaluation_max_tokens": 256,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "examples"}, indent=2))


def evaluate_sessions(args):
    from sessions import CanvasSession, summarize_sessions

    model, tokenizer = load_model(args.adapter)
    cases = [
        s for s in read_examples(ROOT / "sessions.jsonl") if s["split"] == args.split
    ]
    if args.limit:
        cases = cases[: args.limit]
    rows = []
    for case in cases:
        actual, oracle = (
            CanvasSession(case["initial_canvas"]),
            CanvasSession(case["initial_canvas"]),
        )
        for turn, step in enumerate(case["turns"]):
            actual.external(step["before"])
            oracle.external(step["before"])
            before = actual.snapshot()
            result = predict(
                model, tokenizer, step["command"], actual.canvas, history=actual.history
            )
            created_id = f"{case['id']}:created-{turn}"
            try:
                actual.execute(step["command"], result["prediction"], created_id)
            except ValueError as error:
                result.update(prediction=None, error=str(error))
                actual.execute(step["command"], None, created_id)
            oracle.execute(step["command"], step["expected"], created_id)
            result.update(
                id=step["id"],
                session_id=case["id"],
                turn=turn,
                command=step["command"],
                expected=step["expected"],
                correct=result["prediction"] == step["expected"],
                mutated=before != actual.snapshot(),
                state_matches=actual.snapshot() == oracle.snapshot(),
            )
            rows.append(result)
        correct = sum(r["correct"] for r in rows[-len(case["turns"]) :])
        print(f"{case['id']}: {correct}/{len(case['turns'])} turns", flush=True)
    report = {
        **_metadata(),
        **summarize_sessions(rows),
        "split": args.split,
        "adapter": str(args.adapter),
        "mode": "closed_loop_predicted_state",
        "examples": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "examples"}, indent=2))


def _summarize(rows):
    valid = [r for r in rows if r["prediction"] is not None]
    selections = [
        re.match(r"\s*<start_function_call>call:(\w+)", r["raw_output"]) for r in rows
    ]
    per_action = {}
    for name in sorted({r["expected"]["name"] for r in rows}):
        subset = [r for r in rows if r["expected"]["name"] == name]
        per_action[name] = {
            "correct": sum(r["correct"] for r in subset),
            "total": len(subset),
        }
    return {
        "total": len(rows),
        "correct": sum(r["correct"] for r in rows),
        "exact_action_accuracy": sum(r["correct"] for r in rows) / len(rows),
        "valid_call_rate": len(valid) / len(rows),
        "tool_selection_accuracy": sum(
            selection is not None and selection[1] == row["expected"]["name"]
            for row, selection in zip(rows, selections, strict=True)
        )
        / len(rows),
        "median_seconds": statistics.median(r["seconds"] for r in rows),
        "max_peak_model_memory_gb": max(r["peak_model_memory_gb"] for r in rows),
        "per_action": per_action,
    }


def compare(args):
    before, after = [json.loads(path.read_text()) for path in (args.before, args.after)]
    before_examples, after_examples = before["examples"], after["examples"]
    if [(e["id"], e["command"], e["expected"]) for e in before_examples] != [
        (e["id"], e["command"], e["expected"]) for e in after_examples
    ]:
        raise ValueError("Reports must evaluate the same examples in the same order.")
    if (before["model"], before["revision"]) != (after["model"], after["revision"]):
        raise ValueError("Reports must use the same base model revision.")
    if before["dataset_sha256"] != after["dataset_sha256"]:
        raise ValueError("Reports must use the same dataset version.")
    if before["task_sha256"] != after["task_sha256"]:
        raise ValueError("Reports must use the same prompt and action definitions.")
    print("Metric                    Baseline     Adapter")
    for key in (
        "exact_action_accuracy",
        "valid_call_rate",
        "tool_selection_accuracy",
        "median_seconds",
        "max_peak_model_memory_gb",
    ):
        print(f"{key:27} {before[key]:8.3f} {after[key]:11.3f}")


def positive_integer(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return number


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate a local canvas action model."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    commands.add_parser("inspect")
    bundle = commands.add_parser("bundle-colab")
    bundle.add_argument("--resume", type=Path)
    train = commands.add_parser("train")
    train.add_argument("--run", default="first-lora")
    train.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    train.add_argument("--iters", type=positive_integer)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--adapter", type=Path)
    evaluation.add_argument("--split", choices=["valid", "test"], default="test")
    evaluation.add_argument("--limit", type=positive_integer)
    evaluation.add_argument("--output", type=Path, required=True)
    session_evaluation = commands.add_parser("evaluate-sessions")
    session_evaluation.add_argument("--adapter", type=Path, required=True)
    session_evaluation.add_argument(
        "--split", choices=["valid", "test"], default="valid"
    )
    session_evaluation.add_argument("--limit", type=positive_integer)
    session_evaluation.add_argument("--output", type=Path, required=True)
    prediction = commands.add_parser("predict")
    prediction.add_argument("text")
    prediction.add_argument("--canvas", type=Path)
    prediction.add_argument("--adapter", type=Path)
    comparison = commands.add_parser("compare")
    comparison.add_argument("before", type=Path)
    comparison.add_argument("after", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare_data(), indent=2))
    elif args.command == "inspect":
        inspect_data()
    elif args.command == "train":
        train_model(args)
    elif args.command == "bundle-colab":
        bundle_colab(args.resume)
    elif args.command == "evaluate":
        evaluate(args)
    elif args.command == "evaluate-sessions":
        evaluate_sessions(args)
    elif args.command == "compare":
        compare(args)
    else:
        canvas = json.loads(args.canvas.read_text()) if args.canvas else {}
        model, tokenizer = load_model(args.adapter)
        print(json.dumps(predict(model, tokenizer, args.text, canvas), indent=2))


if __name__ == "__main__":
    main()
