import hashlib
import json
from collections import Counter
from pathlib import Path

from actions import TOOLS, messages_for, model_call, validate_call

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def training_row(example):
    messages = messages_for(
        example["command"], example["canvas"], example.get("history")
    )
    messages.append(
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "function": model_call(example["expected"], example["canvas"]),
                }
            ],
        }
    )
    return {"messages": messages, "tools": TOOLS}


def read_examples(path=ROOT / "examples.jsonl"):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def dataset_hash():
    return hashlib.sha256((ROOT / "examples.jsonl").read_bytes()).hexdigest()


def audit_examples(examples):
    if not examples:
        raise ValueError("Dataset is empty.")
    ids, prompts, groups, labels = set(), {}, {}, {}
    for example in examples:
        split = example["split"]
        if split not in {"train", "valid", "test"} or example["id"] in ids:
            raise ValueError("Invalid split or duplicate example ID.")
        ids.add(example["id"])
        validate_call(example["expected"], example["canvas"])
        fingerprint = json.dumps(
            messages_for(example["command"], example["canvas"], example.get("history")),
            sort_keys=True,
        )
        label = json.dumps(
            model_call(example["expected"], example["canvas"]), sort_keys=True
        )
        if fingerprint in labels and labels[fingerprint] != label:
            raise ValueError("The same input has conflicting action labels.")
        labels[fingerprint] = label
        for key, seen in ((fingerprint, prompts), (example["group"], groups)):
            if key in seen and seen[key] != split:
                raise ValueError(
                    "The same prompt or scenario group occurs across splits."
                )
            seen[key] = split
    counts = Counter(e["split"] for e in examples)
    if set(counts) != {"train", "valid", "test"}:
        raise ValueError("Training, validation and test sets must all be present.")
    return dict(counts)


def prepare_data():
    examples = read_examples()
    counts = audit_examples(examples)
    DATA.mkdir(exist_ok=True)
    for split in ("train", "valid", "test"):
        rows = [training_row(e) for e in examples if e["split"] == split]
        (DATA / f"{split}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
    return counts
