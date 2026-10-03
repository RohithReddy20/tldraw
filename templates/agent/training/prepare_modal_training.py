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
    audit_quality_supplement,
)
from dataset import audit_examples, training_row
from lab import MODEL, REVISION, ToolPrefixEncoder, _task_hash
from training import action_stratum, write_training_exposure

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
LEXICAL = TRAINING / "runs/quality-next/lexical-refinement-v1"
LEXICAL_OUTPUT = TRAINING / "runs/v9-english-modal/input"
V8 = TRAINING / "runs/v8-quality-modal"
LEXICAL_WARM = V8 / "download/refinement/step-1950/adapter/adapters.safetensors"
LEXICAL_WARM_SHA256 = "804ba6abcec7580513a04638d88f07029b2c082165516745b8315de2e758d896"
LEXICAL_COUNTS = {"train": 2000, "valid": 2308, "test": 980}
LEXICAL_UPDATES, LEXICAL_SEED = 400, 73
LEXICAL_SOURCE_NAMES = (*SOURCE_NAMES, "modal_training.py", "modal_training_worker.py")


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


def committed_training_sources():
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=TRAINING, text=True
    ).strip()
    checksums = {name: digest(TRAINING / name) for name in LEXICAL_SOURCE_NAMES}
    for name, checksum in checksums.items():
        committed = subprocess.check_output(
            ["git", "show", f"{commit}:templates/agent/training/{name}"], cwd=TRAINING
        )
        if bytes_digest(committed) != checksum:
            raise ValueError(
                f"Commit the final training source before preparation: {name}"
            )
    return commit, checksums


def command_sentence(command):
    return " ".join(command.casefold().split()).rstrip(".!?")


def select_lexical_replay(records, development, *, count=800, seed=LEXICAL_SEED):
    reserved = {command_sentence(row["command"]) for row, _ in development}
    training = [(row, raw) for row, raw in records if row["split"] == "train"]
    if len({row["id"] for row, _ in training}) != len(training):
        raise ValueError("Replay source has duplicate training IDs.")
    strata = {action_stratum(row["expected"]) for row, _ in training}
    buckets = {}
    for row, raw in training:
        if command_sentence(row["command"]) not in reserved:
            buckets.setdefault(action_stratum(row["expected"]), []).append((row, raw))
    if set(buckets) != strata or sum(map(len, buckets.values())) < count:
        raise ValueError(
            "Replay cannot cover every action stratum without dev sentences."
        )
    random = np.random.RandomState(seed)
    keys = sorted(buckets)
    for key in keys:
        ordered = sorted(buckets[key], key=lambda pair: pair[0]["id"])
        buckets[key] = [ordered[index] for index in random.permutation(len(ordered))]
    keys = [keys[index] for index in random.permutation(len(keys))]
    positions = Counter()
    selected = []
    while len(selected) < count:
        for key in keys:
            if positions[key] < len(buckets[key]):
                selected.append(buckets[key][positions[key]])
                positions[key] += 1
                if len(selected) == count:
                    break
    if len(selected) != len({row["id"] for row, _ in selected}):
        raise ValueError("Replay must use unique old training rows.")
    return selected, {
        "seed": seed,
        "selected_rows": count,
        "stratum_counts": dict(sorted(positions.items())),
        "eligible_old_train_rows": sum(map(len, buckets.values())),
        "excluded_dev_sentence_rows": len(training) - sum(map(len, buckets.values())),
        "development_sentences_used": False,
    }


def build_lexical_training_plan(lengths):
    lengths = np.asarray(lengths)
    if lengths.shape != (2000,) or np.any(lengths <= 0):
        raise ValueError("The lexical plan requires positive lengths for 2000 rows.")
    random = np.random.RandomState(LEXICAL_SEED)
    new = np.argsort(lengths[:1200], kind="stable").reshape(-1, GLOBAL_BATCH)
    new = np.concatenate([new[random.permutation(len(new))] for _ in range(2)])
    replay = np.argsort(lengths[1200:], kind="stable").reshape(-1, GLOBAL_BATCH) + 1200
    replay = replay[random.permutation(len(replay))]
    plan = np.empty((LEXICAL_UPDATES, GLOBAL_BATCH), dtype="<i8")
    for cycle in range(100):
        plan[cycle * 4 : cycle * 4 + 3] = new[cycle * 3 : cycle * 3 + 3]
        plan[cycle * 4 + 3] = replay[cycle]
    counts = np.bincount(plan.ravel(), minlength=2000)
    if not np.all(counts[:1200] == 2) or not np.all(counts[1200:] == 1):
        raise ValueError("Lexical rows must be seen twice and old replay rows once.")
    padded = (1 + 32 * ((lengths[plan].max(axis=1) + 31) // 32)) * GLOBAL_BATCH
    return plan, {
        "seed": LEXICAL_SEED,
        "updates": LEXICAL_UPDATES,
        "global_batch": GLOBAL_BATCH,
        "selected_positions": int(plan.size),
        "unique_examples": len(counts),
        "lexical": {"unique_rows": 1200, "positions": 2400, "exposures_per_row": 2},
        "replay": {"unique_rows": 800, "positions": 800, "exposures_per_row": 1},
        "interleave": "3 lexical batches then 1 replay batch, repeated 100 times",
        "indices_sha256": bytes_digest(plan.tobytes()),
        "padding_fraction": 1 - float(lengths[plan].sum() / padded.sum()),
        "test_set_used": False,
    }


def lexical_training_config(frozen, original_rows, original_cases):
    config = training_config(frozen, original_rows, original_cases)
    config.update(
        seed=LEXICAL_SEED,
        iters=LEXICAL_UPDATES,
        quality_updates=LEXICAL_UPDATES,
        learning_rate=2e-6,
        checkpoint_selection_steps=[LEXICAL_UPDATES],
        save_every=100,
        warm_start_source_step=1950,
        refinement_mode="lexical",
    )
    config["lr_schedule"] = {
        "name": "cosine_decay",
        "arguments": [2e-6, 380, 2e-7],
        "warmup": 20,
        "warmup_init": 2e-7,
    }
    return config


def verify_lexical_source(folder):
    summary = json.loads((folder / "dataset-summary.json").read_text())
    required = {
        "examples.jsonl",
        "sessions.jsonl",
        "counterfactual-setups.jsonl",
        "contrastive-pairs.jsonl",
        "data-audit.json",
        "tokenization-audit.json",
    }
    if not required <= set(summary["artifact_sha256"]):
        raise ValueError("Lexical export lacks required replay or audit artifacts.")
    for name, checksum in summary["artifact_sha256"].items():
        if Path(name).name != name or digest(folder / name) != checksum:
            raise ValueError(f"Reviewed lexical artifact changed: {name}")
    if not summary.get("source_sha256"):
        raise ValueError("Lexical export must bind its source implementation.")
    for name, checksum in summary["source_sha256"].items():
        if name not in SOURCE_NAMES or digest(TRAINING / name) != checksum:
            raise ValueError(f"Reviewed lexical source changed: {name}")
    records = read_records(folder / "examples.jsonl")
    cases = read_records(folder / "sessions.jsonl")
    if (
        summary["splits"] != {"train": 1200, "valid": 200}
        or Counter(row["split"] for row, _ in records) != Counter(train=1200, valid=200)
        or Counter(case["split"] for case, _ in cases) != Counter(train=20, valid=4)
        or Counter(case["split"] for case, _ in cases for _ in case["turns"])
        != Counter(train=240, valid=48)
    ):
        raise ValueError("Use the reviewed lexical 1200/200 and 20/4 session export.")
    audit = audit_quality_supplement(
        [row for row, _ in records],
        [case for case, _ in cases],
        [row for row, _ in read_records(folder / "counterfactual-setups.jsonl")],
        [row for row, _ in read_records(folder / "contrastive-pairs.jsonl")],
    )
    if audit["guard_reversal_count"]:
        raise ValueError("The current guard reverses a lexical supervised label.")
    return summary, records, cases, audit


def combine_lexical_data(
    original,
    supplemental,
    lexical,
    original_cases,
    supplemental_cases,
    lexical_cases,
    *,
    replay_count=800,
):
    pools = (("original", original), ("supplement", supplemental), ("lexical", lexical))
    development = [
        pair for _, records in pools for pair in records if pair[0]["split"] == "valid"
    ]
    reserved = {command_sentence(row["command"]) for row, _ in development}
    new = [pair for pair in lexical if pair[0]["split"] == "train"]
    if any(command_sentence(row["command"]) in reserved for row, _ in new):
        raise ValueError("Lexical training must not contain development sentences.")
    replay, replay_summary = select_lexical_replay(
        [*original, *supplemental], development, count=replay_count
    )
    rows, lines, ledger = [], [], []
    for pool, records in (("lexical", new), ("replay", replay)):
        for row, raw in records:
            if pool == "replay":
                row, raw, change = normalize_training_canvas(row, raw)
                if change:
                    ledger.append({"pool": pool, **change})
            rows.append(row)
            lines.append(raw)
    for _, records in pools:
        for row, raw in records:
            if row["split"] != "train":
                rows.append(row)
                lines.append(raw)
    cases = [
        pair
        for pair in [*original_cases, *supplemental_cases]
        if pair[0]["split"] != "train"
    ] + lexical_cases
    audit_examples(rows)
    if len({accuracy_fingerprint(row) for row in rows}) != len(rows):
        raise ValueError("Lexical preparation contains duplicate full model inputs.")
    for row in rows:
        if row["split"] == "test":
            continue
        if (
            validate_call(row["expected"], row["canvas"]) != row["expected"]
            or accuracy_capability_issue(row)
            or execution_guard(
                row["command"], row["expected"], row["canvas"], row.get("history") or {}
            )
            != row["expected"]
        ):
            raise ValueError(f"Lexical/native action audit failed: {row['id']}")
    replay_audit = audit_accuracy_sessions(
        rows, [case for case, _ in cases if case["split"] != "test"]
    )
    return rows, lines, cases, replay, replay_summary, ledger, replay_audit


def tokenize(
    rows,
    output,
    frozen,
    old_tokens,
    unchanged_ids,
    base,
    config,
    *,
    cache_reuse=True,
):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(base), local_files_only=True)
    encoder = ToolPrefixEncoder(tokenizer)
    task = _task_hash()
    compatible = cache_reuse and old_tokens["task_sha256"] == task
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


def verify_lexical_warm_start(v8, warm, base):
    manifest = json.loads((v8 / "input/manifest.json").read_text())
    launch = json.loads((v8 / "run.json").read_text())
    receipt = json.loads((v8 / "task-receipts/selection-step-1950.json").read_text())
    report_path = v8 / "download/selection-step-1950.json"
    report = json.loads(report_path.read_text())
    if (
        launch["input_manifest_sha256"] != digest(v8 / "input/manifest.json")
        or receipt["input_manifest_sha256"] != launch["input_manifest_sha256"]
        or receipt["report_sha256"] != digest(report_path)
        or report["mode"] != "selection"
        or report["case"] != "step-1950"
        or report["step"] != 1950
        or report["metrics"]["adapter_sha256"] != LEXICAL_WARM_SHA256
        or digest(warm) != LEXICAL_WARM_SHA256
    ):
        raise ValueError(
            "Lexical warm-start must be the hash-verified v8 step1950 adapter."
        )
    for name in ("adapters.safetensors", "adapter_config.json"):
        expected = receipt["case_artifacts"][f"refinement/step-1950/adapter/{name}"]
        if digest(warm.with_name(name)) != expected:
            raise ValueError(f"Retained v8 starting adapter changed: {name}")
    config = json.loads(warm.with_name("adapter_config.json").read_text())
    if (
        config["model"] != MODEL
        or config["model_dtype"] != "float16"
        or config["num_layers"] != -1
        or config["lora_parameters"] != {"rank": 32, "dropout": 0.0, "scale": 8.0}
    ):
        raise ValueError("Retain the v8 rank32 all-layer FP16 starting adapter.")
    for name, checksum in manifest["model_files"].items():
        if digest(base / name) != checksum:
            raise ValueError(f"Pinned v8 FP16 base changed: {name}")
    return {
        "source_step": 1950,
        "adapter_sha256": digest(warm),
        "adapter_config_sha256": digest(warm.with_name("adapter_config.json")),
        "v8_input_manifest_sha256": launch["input_manifest_sha256"],
        "selection_receipt_sha256": digest(report_path),
        "retained_v6_promotion_baseline": True,
    }


def prepare_lexical(
    output=LEXICAL_OUTPUT,
    *,
    frozen=FROZEN,
    supplement=SUPPLEMENT,
    lexical=LEXICAL,
    base=BASE,
    warm=LEXICAL_WARM,
    baseline=WARM,
    v8=V8,
):
    if output.exists():
        raise FileExistsError("Keep previously prepared lexical inputs immutable.")
    source_commit, source_sha = committed_training_sources()
    source_receipt, old_tokens, supplement_summary = verify_sources(
        frozen, supplement, baseline
    )
    warm_receipt = verify_lexical_warm_start(v8, warm, base)
    lexical_summary, lexical_rows, lexical_cases, lexical_audit = verify_lexical_source(
        lexical
    )
    original, supplemental = (
        read_records(folder / "examples.jsonl") for folder in (frozen, supplement)
    )
    original_cases, supplemental_cases = (
        read_records(folder / "sessions.jsonl") for folder in (frozen, supplement)
    )
    if (
        Counter(row["split"] for row, _ in original)
        != Counter(train=24000, valid=1700, test=980)
        or Counter(row["split"] for row, _ in supplemental)
        != Counter(train=2400, valid=408)
        or sum(case["split"] == "valid" for case, _ in original_cases) != 8
        or sum(case["split"] == "valid" for case, _ in supplemental_cases) != 3
    ):
        raise ValueError("Retain original and supplemental development/test pools.")
    rows, lines, cases, replay, replay_summary, ledger, replay_audit = (
        combine_lexical_data(
            original,
            supplemental,
            lexical_rows,
            original_cases,
            supplemental_cases,
            lexical_cases,
        )
    )
    if audit_examples(rows) != LEXICAL_COUNTS:
        raise ValueError("Lexical preparation does not match the declared split sizes.")
    config = lexical_training_config(frozen, original, original_cases)
    output.mkdir(parents=True, exist_ok=False)
    for name in LEXICAL_SOURCE_NAMES:
        shutil.copyfile(TRAINING / name, output / name)
    write_bytes(output / "examples.jsonl", b"".join(lines))
    write_bytes(output / "sessions.jsonl", b"".join(raw for _, raw in cases))
    write_bytes(
        output / "config.yaml", yaml.safe_dump(config, sort_keys=False).encode()
    )
    write_bytes(
        output / "training-overlay.jsonl",
        b"".join((json.dumps(entry) + "\n").encode() for entry in ledger),
    )
    for source, name in (
        (warm, "warm-start.safetensors"),
        (warm.with_name("adapter_config.json"), "warm-start-config.json"),
        (baseline, "baseline.safetensors"),
        (baseline.with_name("adapter_config.json"), "baseline-config.json"),
    ):
        shutil.copyfile(source, output / name)
    write_json(
        output / "warm-start.json", {"initialization": "weights_only", **warm_receipt}
    )
    write_json(
        output / "baseline.json",
        {"source_step": 2000, "adapter_sha256": digest(baseline)},
    )
    pools, preservation = {}, {}
    for label, examples, sessions in (
        ("original", original, original_cases),
        ("supplement", supplemental, supplemental_cases),
        ("lexical", lexical_rows, lexical_cases),
    ):
        pools[label] = {
            "example_ids": [
                row["id"] for row, _ in examples if row["split"] == "valid"
            ],
            "session_ids": [
                case["id"] for case, _ in sessions if case["split"] == "valid"
            ],
        }
        for split in ("valid", "test"):
            for kind, records in (("examples", examples), ("sessions", sessions)):
                raw = b"".join(line for row, line in records if row["split"] == split)
                name = f"{label}-{split}-{kind}.jsonl"
                write_bytes(output / name, raw)
                preservation[name] = bytes_digest(raw)
    for name in (
        "counterfactual-setups.jsonl",
        "contrastive-pairs.jsonl",
        "data-audit.json",
        "tokenization-audit.json",
        "dataset-summary.json",
    ):
        shutil.copyfile(lexical / name, output / f"lexical-{name}")
    lengths, token_manifest = tokenize(
        rows, output, frozen, old_tokens, set(), base, config, cache_reuse=False
    )
    plan, plan_summary = build_lexical_training_plan(lengths)
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
        output / "data/tokens-train.npz",
        plan.ravel().tolist(),
        2000,
        LEXICAL_SEED,
        0,
        LEXICAL_UPDATES,
    )
    exposure.update(
        training_batch_sha256=plan_summary["indices_sha256"],
        dataset_sha256=token_manifest["dataset_sha256"],
        task_sha256=token_manifest["task_sha256"],
        test_set_used=False,
    )
    write_json(output / "planned-training-exposure.json", exposure)
    final_commit, final_source_sha = committed_training_sources()
    if final_commit != source_commit or final_source_sha != source_sha:
        raise ValueError("Committed source changed during lexical preparation.")
    manifest = {
        "refinement_mode": "lexical",
        "source_commit": source_commit,
        "source_sha256": source_sha,
        "configuration": config,
        "dataset_sha256": token_manifest["dataset_sha256"],
        "task_sha256": token_manifest["task_sha256"],
        "model": MODEL,
        "model_revision": REVISION,
        "local_converted_dtype": "float16",
        "model_files": {
            path.name: digest(path) for path in sorted(base.iterdir()) if path.is_file()
        },
        "warm_start_sha256": warm_receipt["adapter_sha256"],
        "warm_start_config_sha256": warm_receipt["adapter_config_sha256"],
        "warm_start_source_step": 1950,
        "baseline_sha256": digest(baseline),
        "baseline_config_sha256": digest(baseline.with_name("adapter_config.json")),
        "baseline_source_step": 2000,
        "training_examples": 2000,
        "splits": LEXICAL_COUNTS,
        "development_pools": pools,
        "heldout_bytes_sha256": preservation,
        "training_pools": {
            "lexical": {
                "example_ids": [
                    row["id"] for row, _ in lexical_rows if row["split"] == "train"
                ],
                "exposures_per_row": 2,
            },
            "replay": {
                "example_ids": [row["id"] for row, _ in replay],
                "exposures_per_row": 1,
            },
        },
        "training_plan": {
            **plan_summary,
            "archive_sha256": token_manifest["files"]["train-plan.npz"],
        },
        "replay_selection": replay_summary,
        "source_data": {
            "original": {
                "path": str(frozen),
                "source_commit": source_receipt["source_commit"],
                "examples_sha256": digest(frozen / "examples.jsonl"),
                "sessions_sha256": digest(frozen / "sessions.jsonl"),
            },
            "supplement": {
                "path": str(supplement),
                "artifact_sha256": supplement_summary["artifact_sha256"],
            },
            "lexical": {
                "path": str(lexical),
                "summary_sha256": digest(lexical / "dataset-summary.json"),
                "artifact_sha256": lexical_summary["artifact_sha256"],
                "source_sha256": lexical_summary["source_sha256"],
            },
            "warm_start": warm_receipt,
        },
        "overlay": {
            "train_only": True,
            "changed_rows": len(ledger),
            "ledger_sha256": digest(output / "training-overlay.jsonl"),
            "commands_ids_history_labels_preserved": True,
        },
        "lexical_data_audit": lexical_audit,
        "session_fixture_replay": replay_audit,
        "promotion_gate": (
            "Original256/6 selection at400; separate original1700/8, supplement408/3 "
            "and lexical200/4 full gates, then native checks before fresh test"
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
                "splits": LEXICAL_COUNTS,
                "updates": LEXICAL_UPDATES,
                "training_positions": int(plan.size),
                "test_set_used": False,
                "cloud_calls": 0,
            },
            indent=2,
        )
    )
    return manifest


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
    parser.add_argument("--output", type=Path)
    parser.add_argument("--frozen", type=Path, default=FROZEN)
    parser.add_argument("--supplement", type=Path, default=SUPPLEMENT)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument("--micro-batch", type=int, choices=(2, 8), default=2)
    parser.add_argument("--checkpoint", choices=("all", "none"), default="all")
    parser.add_argument("--lexical-refinement", action="store_true")
    parser.add_argument("--lexical-data", type=Path, default=LEXICAL)
    parser.add_argument("--baseline", type=Path, default=WARM)
    args = parser.parse_args()
    if args.lexical_refinement:
        if args.micro_batch != 2 or args.checkpoint != "all":
            parser.error(
                "Lexical refinement preserves micro-batch2 and checkpoint all."
            )
        prepare_lexical(
            args.output or LEXICAL_OUTPUT,
            frozen=args.frozen,
            supplement=args.supplement,
            lexical=args.lexical_data,
            base=args.base,
            warm=args.warm_start or LEXICAL_WARM,
            baseline=args.baseline,
        )
    else:
        prepare(
            args.output or OUTPUT,
            frozen=args.frozen,
            supplement=args.supplement,
            base=args.base,
            warm=args.warm_start or WARM,
            micro_batch=args.micro_batch,
            checkpoint=args.checkpoint,
        )


if __name__ == "__main__":
    main()
