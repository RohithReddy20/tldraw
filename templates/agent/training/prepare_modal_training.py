import argparse
import copy
import hashlib
import json
import shutil
import subprocess
import tempfile
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import yaml

from actions import TOOLS, execution_guard, parse_call, validate_call
from build_workflow import (
    accuracy_capability_issue,
    accuracy_fingerprint,
    audit_accuracy_sessions,
)
from dataset import audit_examples, training_row
from lab import MODEL, REVISION, ToolPrefixEncoder, _task_hash
from training import write_training_exposure

TRAINING = Path(__file__).resolve().parent
FROZEN = TRAINING / "runs/v7-accuracy-workflow-revised"
SUPPLEMENT = TRAINING / "runs/quality-next/supplement-revised-v2"
BASE = TRAINING / "runs/base-bb327a9a-float16"
OUTPUT = TRAINING / "runs/v8-quality-modal/input"
WARM = (
    TRAINING
    / "runs/kaggle/spoken/download/refinement/dual-window/adapter/adapters.safetensors"
)
WARM_SHA256 = "08593b33c249f6a4fc0394d6365d034c3665fb325b6a47f6fe0f4d0c069deeb3"
SOURCE_NAMES = (
    "actions.py",
    "dataset.py",
    "sessions.py",
    "lab.py",
    "training.py",
    "gpu_benchmark.py",
    "build_workflow.py",
    "prepare_modal_training.py",
)
COUNTS = {"train": 26400, "valid": 2108, "test": 980}
UPDATES, GLOBAL_BATCH, SEED = 3900, 8, 72


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def bytes_digest(value):
    return hashlib.sha256(value).hexdigest()


def read_records(path):
    records = []
    with path.open("rb") as source:
        for line in source:
            if not line.strip() or not line.endswith(b"\n"):
                raise ValueError(
                    f"Frozen JSONL requires complete nonempty lines: {path}"
                )
            records.append((json.loads(line), line))
    return records


def write_bytes(path, value):
    with path.open("xb") as output:
        output.write(value)


def write_json(path, value):
    write_bytes(path, (json.dumps(value, indent=2) + "\n").encode())


def verify_sources(frozen, supplement, warm):
    source = json.loads((frozen / "source-receipt.json").read_text())
    for name in ("examples.jsonl", "sessions.jsonl", "config.yaml", "lab.py"):
        if digest(frozen / name) != source["sha256"][name]:
            raise ValueError(f"Frozen v7 input changed: {name}")
    tokens = json.loads((frozen / "data/tokens.json").read_text())
    if tokens["dataset_sha256"] != source["sha256"]["examples.jsonl"]:
        raise ValueError("The v7 token cache belongs to another dataset.")
    for name, checksum in tokens["files"].items():
        if digest(frozen / "data" / name) != checksum:
            raise ValueError(f"The v7 token cache changed: {name}")
    summary = json.loads((supplement / "dataset-summary.json").read_text())
    for name, checksum in summary["artifact_sha256"].items():
        if digest(supplement / name) != checksum:
            raise ValueError(f"The reviewed supplement changed: {name}")
    if summary["splits"] != {"train": 2400, "valid": 408}:
        raise ValueError("Use the reviewed 2400/408 supplement export.")
    if digest(warm) != WARM_SHA256:
        raise ValueError("The verified v6 warm-start adapter changed.")
    warm_config = json.loads(warm.with_name("adapter_config.json").read_text())
    if (
        warm_config["model"] != MODEL
        or warm_config["num_layers"] != -1
        or warm_config["lora_parameters"] != {"rank": 32, "dropout": 0.0, "scale": 8.0}
        or warm_config["quality_updates"] != 2000
    ):
        raise ValueError(
            "The warm-start configuration differs from the verified v6 adapter."
        )
    return source, tokens, summary


def normalize_training_canvas(row, raw):
    derived = copy.deepcopy(row)
    changes = []
    for shape in derived["canvas"].get("shapes", []):
        updates = (
            {"color": "black"}
            if shape["kind"] == "frame"
            else {"w": 200, "h": 200}
            if shape["kind"] == "note"
            else {}
        )
        for field, value in updates.items():
            if shape[field] == value:
                continue
            changes.append(
                {
                    "shape_id": shape["id"],
                    "field": field,
                    "original": shape[field],
                    "normalized": value,
                    "reason": (
                        "Native frames expose no color property"
                        if field == "color"
                        else (
                            "Native default notes fit to 200 by 200 "
                            "for these short labels"
                        )
                    ),
                }
            )
            shape[field] = value
    if not changes:
        return row, raw, None
    for field in ("id", "group", "split", "command", "history", "expected"):
        if derived.get(field) != row.get(field):
            raise ValueError("The training overlay changed an instruction or outcome.")
    for field in ("schemas", "selected_ids", "camera", "can_undo", "can_redo"):
        if derived["canvas"].get(field) != row["canvas"].get(field):
            raise ValueError(
                "The training overlay changed canvas references or camera."
            )
    encoded = (json.dumps(derived) + "\n").encode()
    return (
        derived,
        encoded,
        {
            "id": row["id"],
            "original_row_sha256": bytes_digest(raw),
            "derived_row_sha256": bytes_digest(encoded),
            "changes": changes,
            "history_and_expected_unchanged": True,
        },
    )


def combine_data(original, supplemental, original_cases, supplemental_cases):
    original_counts = Counter(row["split"] for row, _ in original)
    if original_counts != Counter(train=24000, valid=1700, test=980):
        raise ValueError("The frozen v7 split sizes changed.")
    if Counter(row["split"] for row, _ in supplemental) != Counter(
        train=2400, valid=408
    ):
        raise ValueError("The supplemental split sizes changed.")
    raw_rows = [row for row, _ in [*original, *supplemental]]
    raw_cases = [case for case, _ in [*original_cases, *supplemental_cases]]
    audit_examples(raw_rows)
    case_ids = [case["id"] for case in raw_cases]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Original and supplemental session IDs must be unique.")
    replay = audit_accuracy_sessions(raw_rows, raw_cases)
    rows, lines, ledger = [], [], []
    unchanged_original_train = set()
    for pool, records in (("original", original), ("supplement", supplemental)):
        for row, raw in records:
            if row["split"] == "train":
                derived, encoded, change = normalize_training_canvas(row, raw)
                if change:
                    ledger.append({"pool": pool, **change})
                elif pool == "original":
                    unchanged_original_train.add(row["id"])
            else:
                derived, encoded = row, raw
            rows.append(derived)
            lines.append(encoded)
    if audit_examples(rows) != COUNTS:
        raise ValueError("The merged data does not match its declared split sizes.")
    fingerprints = [accuracy_fingerprint(row) for row in rows]
    if len(set(fingerprints)) != len(rows):
        raise ValueError("The derived dataset contains duplicate full model inputs.")
    for row in rows:
        if row["split"] == "test":
            continue
        if validate_call(row["expected"], row["canvas"]) != row["expected"]:
            raise ValueError(f"Noncanonical action label: {row['id']}")
        if accuracy_capability_issue(row):
            raise ValueError(f"Unavailable native capability: {row['id']}")
        guarded = execution_guard(
            row["command"], row["expected"], row["canvas"], row.get("history") or {}
        )
        if guarded != row["expected"]:
            raise ValueError(
                f"The current execution guard reverses a label: {row['id']}"
            )
    return rows, lines, ledger, unchanged_original_train, replay


def training_config(
    frozen, original_rows, original_cases, *, micro_batch=2, checkpoint="all"
):
    if micro_batch not in (2, 8) or checkpoint not in ("all", "none"):
        raise ValueError("Use a verified micro-batch/checkpoint profile.")
    config = yaml.safe_load((frozen / "config.yaml").read_text())
    for key, valid, size in (
        (
            "selection_example_ids",
            {row["id"] for row, _ in original_rows if row["split"] == "valid"},
            256,
        ),
        (
            "selection_session_ids",
            {case["id"] for case, _ in original_cases if case["split"] == "valid"},
            6,
        ),
    ):
        chosen = config[key]
        if len(chosen) != size or len(set(chosen)) != size or not set(chosen) <= valid:
            raise ValueError(f"The fixed v7 checkpoint selection IDs changed: {key}")
    config.update(
        seed=SEED,
        iters=UPDATES,
        quality_updates=UPDATES,
        batch_size=GLOBAL_BATCH,
        refinement_micro_batch=micro_batch,
        refinement_checkpoint=checkpoint,
        grad_checkpoint=checkpoint == "all",
        learning_rate=5e-6,
        checkpoint_selection_steps=[650, 1950, 3900],
        save_every=500,
        model_dtype="float16",
    )
    config["lr_schedule"] = {
        "name": "cosine_decay",
        "arguments": [5e-6, 3850, 5e-7],
        "warmup": 50,
        "warmup_init": 5e-7,
    }
    if (
        config["lora_parameters"]["rank"] != 32
        or config["num_layers"] != -1
        or config["loss_mode"] != "completion_tokens_v1"
    ):
        raise ValueError("Keep the audited rank-32 all-layer completion-only contract.")
    return config


def build_training_plan(lengths):
    lengths = np.asarray(lengths)
    if lengths.shape != (26400,) or np.any(lengths <= 0):
        raise ValueError("The plan requires positive lengths for exactly 26400 rows.")
    random = np.random.RandomState(SEED)
    original = np.argsort(lengths[:24000], kind="stable").reshape(-1, GLOBAL_BATCH)
    supplemental = (
        np.argsort(lengths[24000:], kind="stable").reshape(-1, GLOBAL_BATCH) + 24000
    )
    original = original[random.permutation(len(original))]
    supplemental = np.concatenate(
        [supplemental[random.permutation(len(supplemental))] for _ in range(3)]
    )
    plan = np.empty((UPDATES, GLOBAL_BATCH), dtype="<i8")
    for cycle in range(300):
        start = cycle * 13
        plan[start : start + 10] = original[cycle * 10 : cycle * 10 + 10]
        plan[start + 10 : start + 13] = supplemental[cycle * 3 : cycle * 3 + 3]
    counts = np.bincount(plan.ravel(), minlength=26400)
    if not np.all(counts[:24000] == 1) or not np.all(counts[24000:] == 3):
        raise ValueError(
            "The plan must expose original rows once and supplement rows three times."
        )
    cycles = plan.reshape(300, 13, GLOBAL_BATCH)
    if not np.all(cycles[:, :10] < 24000) or not np.all(cycles[:, 10:] >= 24000):
        raise ValueError(
            "Every 13-update block must contain 10 original and 3 supplemental batches."
        )
    padded = (1 + 32 * ((lengths[plan].max(axis=1) + 31) // 32)) * GLOBAL_BATCH
    summary = {
        "seed": SEED,
        "updates": UPDATES,
        "global_batch": GLOBAL_BATCH,
        "selected_positions": int(plan.size),
        "unique_examples": len(counts),
        "original": {"unique_rows": 24000, "positions": 24000, "exposures_per_row": 1},
        "supplement": {"unique_rows": 2400, "positions": 7200, "exposures_per_row": 3},
        "interleave": (
            "10 original batches followed by 3 supplement batches, repeated 300 times"
        ),
        "indices_sha256": bytes_digest(plan.tobytes()),
        "padding_fraction": 1 - float(lengths[plan].sum() / padded.sum()),
        "test_set_used": False,
    }
    return plan, summary


def tokenize(rows, output, frozen, old_tokens, unchanged_ids, base, config):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(base), local_files_only=True)
    encoder = ToolPrefixEncoder(tokenizer)
    task = _task_hash()
    compatible = old_tokens["task_sha256"] == task
    cached = {}
    if compatible:
        for split in COUNTS:
            with np.load(
                frozen / "data" / f"tokens-{split}.npz", allow_pickle=False
            ) as arrays:
                cached[split] = {
                    key: arrays[key] for key in ("tokens", "boundaries", "offsets")
                }
    original_positions = Counter()
    sizes, offsets = {split: [] for split in COUNTS}, {split: [] for split in COUNTS}
    reused, checked_tools = Counter(), set()
    maximum_completion = 0
    data = output / "data"
    data.mkdir()
    with ExitStack() as resources:
        directory = Path(resources.enter_context(tempfile.TemporaryDirectory(dir=data)))
        streams = {
            split: resources.enter_context((directory / split).open("wb"))
            for split in COUNTS
        }
        for index, example in enumerate(rows):
            split = example["split"]
            row = training_row(example)
            full = encoder.encode(row["messages"])
            prompt = encoder.encode(row["messages"][:-1], generation=True)
            if full[: len(prompt)] != prompt:
                raise ValueError(f"Prompt masking mismatch: {example['id']}")
            name = example["expected"]["name"]
            if name not in checked_tools:
                for content, tokens, generation in (
                    (row["messages"], full, False),
                    (row["messages"][:-1], prompt, True),
                ):
                    reference = tokenizer.apply_chat_template(
                        content,
                        tools=TOOLS,
                        add_generation_prompt=generation,
                        return_dict=False,
                    )
                    if tokens != reference:
                        raise ValueError(
                            "The cached tool prefix changes official-template tokens."
                        )
                checked_tools.add(name)
            if (
                parse_call(tokenizer.decode(full[len(prompt) :]), example["canvas"])
                != example["expected"]
            ):
                raise ValueError(f"Function-call round trip failed: {example['id']}")
            completion = len(full) - len(prompt)
            if (
                len(full) > config["max_seq_length"]
                or completion > config["completion_window"]
            ):
                raise ValueError(
                    f"Sequence or completion window would truncate {example['id']}"
                )
            maximum_completion = max(maximum_completion, completion)
            array = np.asarray(full, dtype=np.int32)
            position = original_positions[split]
            original = position < {"train": 24000, "valid": 1700, "test": 980}[split]
            if (
                compatible
                and original
                and (split != "train" or example["id"] in unchanged_ids)
            ):
                cache = cached[split]
                begin, end = cache["boundaries"][position : position + 2]
                previous = cache["tokens"][begin:end]
                # Exact encoded equality binds older caches to the current tokenizer.
                if int(cache["offsets"][position]) == len(prompt) and np.array_equal(
                    array, previous
                ):
                    array = previous
                    reused[split] += 1
            original_positions[split] += 1
            streams[split].write(array.tobytes())
            sizes[split].append(len(full))
            offsets[split].append(len(prompt))
            if (index + 1) % 4000 == 0:
                print(
                    f"CPU tokenizer verified {index + 1}/{len(rows)} rows.", flush=True
                )
        for split, stream in streams.items():
            stream.flush()
            tokens = np.memmap(directory / split, mode="r", dtype=np.int32)
            np.savez_compressed(
                data / f"tokens-{split}.npz",
                tokens=tokens,
                boundaries=np.concatenate(
                    ([0], np.cumsum(sizes[split], dtype=np.int64))
                ),
                offsets=np.asarray(offsets[split], dtype=np.int32),
            )
            del tokens
    result = {
        "dataset_sha256": digest(output / "examples.jsonl"),
        "task_sha256": task,
        "max_tokens": max(max(values) for values in sizes.values()),
        "max_completion_tokens": maximum_completion,
        "roundtrip_verified_examples": len(rows),
        "official_template_verified_tools": sorted(checked_tools),
        "old_cache_task_matches": compatible,
        "reused_unchanged_v7_token_vectors": dict(reused),
        "cache_reuse_gate": (
            "Matching task and exact per-example full-token/prompt-offset equality "
            "with the current local tokenizer"
        ),
        "tokenizer_files": {
            path.name: digest(path)
            for path in sorted(base.iterdir())
            if path.name.startswith("tokenizer") or path.suffix == ".jinja"
        },
        "test_set_used": False,
        "test_handling": (
            "Frozen held-out rows copied and CPU tokenization round-tripped; "
            "no model inference, scoring, or training exposure"
        ),
    }
    return sizes["train"], result


def prepare(
    output=OUTPUT,
    *,
    frozen=FROZEN,
    supplement=SUPPLEMENT,
    base=BASE,
    warm=WARM,
    micro_batch=2,
    checkpoint="all",
):
    if output.exists():
        raise FileExistsError("Keep previously prepared training inputs immutable.")
    if not (base / "model.safetensors").is_file():
        raise ValueError("The pinned local FP16 model/tokenizer is unavailable.")
    source_sha = {name: digest(TRAINING / name) for name in SOURCE_NAMES}
    source_receipt, old_tokens, supplement_summary = verify_sources(
        frozen, supplement, warm
    )
    original, supplemental = (
        read_records(folder / "examples.jsonl") for folder in (frozen, supplement)
    )
    original_cases, supplemental_cases = (
        read_records(folder / "sessions.jsonl") for folder in (frozen, supplement)
    )
    rows, lines, ledger, unchanged, replay = combine_data(
        original, supplemental, original_cases, supplemental_cases
    )
    config = training_config(
        frozen, original, original_cases, micro_batch=micro_batch, checkpoint=checkpoint
    )
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=TRAINING, text=True
    ).strip()
    output.mkdir(parents=True, exist_ok=False)
    for name in SOURCE_NAMES:
        shutil.copyfile(TRAINING / name, output / name)
    write_bytes(output / "examples.jsonl", b"".join(lines))
    write_bytes(
        output / "sessions.jsonl",
        b"".join(line for _, line in [*original_cases, *supplemental_cases]),
    )
    write_bytes(
        output / "training-overlay.jsonl",
        b"".join((json.dumps(entry) + "\n").encode() for entry in ledger),
    )
    write_bytes(
        output / "config.yaml", yaml.safe_dump(config, sort_keys=False).encode()
    )
    shutil.copyfile(warm, output / "warm-start.safetensors")
    shutil.copyfile(
        warm.with_name("adapter_config.json"), output / "warm-start-config.json"
    )
    write_json(
        output / "warm-start.json",
        {
            "initialization": "weights_only",
            "adapter_sha256": digest(warm),
            "source_step": 2000,
        },
    )
    pool = {}
    for label, examples, cases in (
        ("original", original, original_cases),
        ("supplement", supplemental, supplemental_cases),
    ):
        pool[label] = {
            "example_ids": [
                row["id"] for row, _ in examples if row["split"] == "valid"
            ],
            "session_ids": [
                case["id"] for case, _ in cases if case["split"] == "valid"
            ],
        }
        for split in ("valid", "test"):
            write_bytes(
                output / f"{label}-{split}-examples.jsonl",
                b"".join(line for row, line in examples if row["split"] == split),
            )
            write_bytes(
                output / f"{label}-{split}-sessions.jsonl",
                b"".join(line for case, line in cases if case["split"] == split),
            )
    lengths, token_manifest = tokenize(
        rows, output, frozen, old_tokens, unchanged, base, config
    )
    plan, plan_summary = build_training_plan(lengths)
    np.savez_compressed(output / "data/train-plan.npz", indices=plan)
    write_json(output / "training-plan.json", plan_summary)
    token_manifest["files"] = {
        name: digest(output / "data" / name)
        for name in (
            "tokens-train.npz",
            "tokens-valid.npz",
            "tokens-test.npz",
            "train-plan.npz",
        )
    }
    write_json(output / "data/tokens.json", token_manifest)
    exposure = write_training_exposure(
        output / "data/tokens-train.npz", plan.ravel().tolist(), 26400, SEED, 0, UPDATES
    )
    exposure.update(
        training_batch_sha256=plan_summary["indices_sha256"],
        dataset_sha256=token_manifest["dataset_sha256"],
        task_sha256=token_manifest["task_sha256"],
        test_set_used=False,
    )
    write_json(output / "planned-training-exposure.json", exposure)
    if any(
        digest(TRAINING / name) != checksum for name, checksum in source_sha.items()
    ):
        raise ValueError(
            "Training source changed during preparation; do not submit this input."
        )
    manifest = {
        "source_commit": source_commit,
        "configuration": config,
        "source_sha256": source_sha,
        "dataset_sha256": token_manifest["dataset_sha256"],
        "task_sha256": token_manifest["task_sha256"],
        "model": MODEL,
        "model_revision": REVISION,
        "local_converted_dtype": "float16",
        "model_files": {
            path.name: digest(path) for path in sorted(base.iterdir()) if path.is_file()
        },
        "warm_start_sha256": digest(warm),
        "warm_start_config_sha256": digest(warm.with_name("adapter_config.json")),
        "warm_start_source_step": 2000,
        "training_examples": 26400,
        "splits": COUNTS,
        "development_pools": pool,
        "training_pools": {
            label: {
                "example_ids": [
                    row["id"] for row, _ in examples if row["split"] == "train"
                ],
                "exposures_per_row": 1 if label == "original" else 3,
            }
            for label, examples in (
                ("original", original),
                ("supplement", supplemental),
            )
        },
        "training_plan": {
            **plan_summary,
            "archive_sha256": token_manifest["files"]["train-plan.npz"],
        },
        "source_data": {
            "original": {
                "path": str(frozen),
                "source_commit": source_receipt["source_commit"],
                "examples_sha256": digest(frozen / "examples.jsonl"),
                "sessions_sha256": digest(frozen / "sessions.jsonl"),
            },
            "supplement": {
                "path": str(supplement),
                "examples_sha256": supplement_summary["artifact_sha256"][
                    "examples.jsonl"
                ],
                "sessions_sha256": supplement_summary["artifact_sha256"][
                    "sessions.jsonl"
                ],
            },
        },
        "overlay": {
            "train_only": True,
            "changed_rows": len(ledger),
            "changed_rows_by_pool": dict(Counter(entry["pool"] for entry in ledger)),
            "changed_fields": dict(
                Counter(
                    change["field"] for entry in ledger for change in entry["changes"]
                )
            ),
            "ledger_sha256": digest(output / "training-overlay.jsonl"),
            "commands_ids_history_labels_preserved": True,
        },
        "session_fixture_replay_before_overlay": replay,
        "overlay_geometry_limit": (
            "Training canvas normalization is a derived input overlay; "
            "raw session fixtures remain unchanged. No exact native geometry "
            "or normalized-session replay equivalence is claimed."
        ),
        "promotion_gate": (
            "Fixed original256+6 checkpoint selection; full original1700/8 and "
            "supplemental408/3 development pools must pass separate regression "
            "gates before any test scoring"
        ),
        "selection_example_ids": config["selection_example_ids"],
        "selection_session_ids": config["selection_session_ids"],
        "test_set_used": False,
        "test_handling": token_manifest["test_handling"],
        "files": {
            str(path.relative_to(output)): digest(path)
            for path in sorted(output.rglob("*"))
            if path.is_file()
        },
    }
    write_json(output / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "input": str(output),
                "splits": COUNTS,
                "updates": UPDATES,
                "training_positions": int(plan.size),
                "training_plan_sha256": plan_summary["indices_sha256"],
                "normalized_training_rows": len(ledger),
                "max_tokens": token_manifest["max_tokens"],
                "reused_v7_vectors": token_manifest[
                    "reused_unchanged_v7_token_vectors"
                ],
                "test_set_used": False,
                "cloud_calls": 0,
            },
            indent=2,
        )
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the audited H100 training input using CPU tokenization only."
        )
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--frozen", type=Path, default=FROZEN)
    parser.add_argument("--supplement", type=Path, default=SUPPLEMENT)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--warm-start", type=Path, default=WARM)
    parser.add_argument("--micro-batch", type=int, choices=(2, 8), default=2)
    parser.add_argument("--checkpoint", choices=("all", "none"), default="all")
    args = parser.parse_args()
    prepare(
        args.output,
        frozen=args.frozen,
        supplement=args.supplement,
        base=args.base,
        warm=args.warm_start,
        micro_batch=args.micro_batch,
        checkpoint=args.checkpoint,
    )


if __name__ == "__main__":
    main()
