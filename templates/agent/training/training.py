import argparse
import hashlib
import json
import zipfile
from itertools import count
from pathlib import Path


def completion_loss(model, batch, lengths, window=192):
    import mlx.core as mx
    import mlx.nn as nn

    hidden = model.model(batch[:, :-1])
    positions = lengths[:, :1] + mx.arange(window)[None, :]
    rows = mx.arange(batch.shape[0])[:, None]
    selected = hidden[rows, mx.clip(positions - 1, 0, hidden.shape[1] - 1)]
    logits = (
        model.model.embed_tokens.as_linear(selected)
        if model.tie_word_embeddings
        else model.lm_head(selected)
    )
    targets = batch[rows, mx.minimum(positions, batch.shape[1] - 1)]
    mask = (positions > 0) & (positions < lengths[:, 1:])
    count = mask.sum()
    return (
        nn.losses.cross_entropy(logits, targets).astype(mx.float32) * mask
    ).sum() / count, count


def completion_indices(lengths, width, pad_to=32):
    import numpy as np

    indices = []
    for row, (offset, end) in enumerate(np.asarray(lengths)):
        if not 1 <= offset < end <= width:
            raise ValueError("Completion bounds must refer to untruncated targets.")
        indices.extend(row * (width - 1) + np.arange(offset - 1, end - 1))
    count = len(indices)
    padded = pad_to * ((count + pad_to - 1) // pad_to)
    return (
        np.pad(np.asarray(indices, dtype=np.int32), (0, padded - count)),
        np.arange(padded) < count,
    )


def packed_completion_loss(model, batch, indices, mask):
    import mlx.core as mx
    import mlx.nn as nn

    hidden = model.model(batch[:, :-1])
    selected = hidden.reshape(-1, hidden.shape[-1])[indices]
    logits = (
        model.model.embed_tokens.as_linear(selected)
        if model.tie_word_embeddings
        else model.lm_head(selected)
    )
    targets = batch[:, 1:].reshape(-1)[indices]
    count = mask.sum()
    return (
        nn.losses.cross_entropy(logits, targets).astype(mx.float32) * mask
    ).sum() / count, count


def resumed_batches(iterator, resume_step, seed, **kwargs):
    import numpy as np

    if not kwargs.get("loop"):
        yield from iterator(**kwargs)
        return
    state = np.random.RandomState(seed).get_state()
    batches = iterator(**kwargs)
    for index in count():
        previous = np.random.get_state()
        try:
            np.random.set_state(state)
            batch = next(batches)
            state = np.random.get_state()
        finally:
            np.random.set_state(previous)
        if index >= resume_step:
            yield batch


def restore_random_state(key):
    import mlx.core as mx

    if key.shape != (2,) or key.dtype != mx.uint32:
        raise ValueError("Retained random state must be a two-word uint32 key.")
    high, low = key.tolist()
    # MLX 0.32.3 makes random.state read-only; seed encodes the same two key words.
    mx.random.seed((high << 32) | low)


def save_training_checkpoint(model, optimizer, adapter, step, schedule_offset):
    import mlx.core as mx
    from mlx.utils import tree_flatten

    checkpoint = adapter / "checkpoints" / f"{step:07d}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(checkpoint / "adapters.safetensors"),
        dict(tree_flatten(model.trainable_parameters())),
    )
    mx.save_safetensors(
        str(checkpoint / "optimizer.safetensors"), dict(tree_flatten(optimizer.state))
    )
    mx.save_safetensors(
        str(checkpoint / "random.safetensors"), {"key": mx.random.state[0]}
    )
    task = json.loads((adapter.parent / "data" / "tokens.json").read_text())
    (checkpoint / "progress.json").write_text(
        json.dumps(
            {
                "step": step,
                "schedule_offset": schedule_offset,
                "dataset_sha256": task["dataset_sha256"],
                "task_sha256": task["task_sha256"],
            }
        )
    )
    archive_path = checkpoint / "checkpoint.zip"
    with zipfile.ZipFile(
        archive_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for name in (
            "adapters.safetensors",
            "optimizer.safetensors",
            "random.safetensors",
            "progress.json",
        ):
            archive.write(checkpoint / name, f"adapter/{name}")
        archive.write(adapter / "adapter_config.json", "adapter/adapter_config.json")
        archive.write(adapter.parent / "config.yaml", "config.yaml")
    parts = []
    with archive_path.open("rb") as source:
        while chunk := source.read(16 * 1024 * 1024):
            path = checkpoint / f"checkpoint.zip.part-{len(parts):03d}"
            path.write_bytes(chunk)
            parts.append(
                {"name": path.name, "sha256": hashlib.sha256(chunk).hexdigest()}
            )
    (checkpoint / "checkpoint.parts.json").write_text(json.dumps(parts))
    print(f"Global step {step}: Saved resumable checkpoint.", flush=True)


def run_training(settings):
    import mlx.core as mx
    import mlx.optimizers as optim
    import numpy as np
    from mlx.utils import tree_unflatten
    from mlx_lm import load
    from mlx_lm.tuner.callbacks import TrainingCallback
    from mlx_lm.tuner.datasets import CacheDataset, load_dataset
    from mlx_lm.tuner.trainer import TrainingArgs, iterate_batches, train
    from mlx_lm.tuner.utils import (
        build_schedule,
        linear_to_lora_layers,
        print_trainable_parameters,
    )
    from mlx_lm.utils import save_config

    class CompactDataset(CacheDataset):
        def __getitem__(self, index):
            if self._proc_data[index] is None:
                tokens, offset = self._data.process(self._data[index])
                if len(tokens) - offset > settings["completion_window"]:
                    raise ValueError(
                        "Completion exceeds the loss window; refusing truncation."
                    )
                self._proc_data[index] = (np.asarray(tokens, dtype=np.int32), offset)
            return self._proc_data[index]

        def itemlen(self, index):
            return len(self[index][0])

    class PreparedDataset:
        def __init__(self, path):
            with np.load(path) as arrays:
                self.tokens = arrays["tokens"]
                self.boundaries = arrays["boundaries"]
                self.offsets = arrays["offsets"]

        def __len__(self):
            return len(self.offsets)

        def __getitem__(self, index):
            begin, end = self.boundaries[index : index + 2]
            return self.tokens[begin:end], int(self.offsets[index])

        def itemlen(self, index):
            return self.boundaries[index + 1] - self.boundaries[index]

    args = argparse.Namespace(**settings)
    np.random.seed(args.seed)
    mx.random.seed(args.seed)
    model, tokenizer = load(args.model)
    if model.model_type != "gemma3_text" or not args.mask_prompt:
        raise ValueError(
            "Completion loss requires the pinned Gemma model and masked prompts."
        )
    model.freeze()
    linear_to_lora_layers(model, args.num_layers, args.lora_parameters)
    if args.resume_adapter_file:
        model.load_weights(args.resume_adapter_file, strict=False)
        print(f"Loaded warm-start weights from {args.resume_adapter_file}", flush=True)
    print_trainable_parameters(model)
    if (Path(args.data) / "tokens.json").exists():
        train_set, valid_set = [
            PreparedDataset(Path(args.data) / f"tokens-{split}.npz")
            for split in ("train", "valid")
        ]
    else:
        train_set, valid_set, _ = load_dataset(args, tokenizer)
        train_set, valid_set = CompactDataset(train_set), CompactDataset(valid_set)
    adapter = Path(args.adapter_path)
    adapter.mkdir(parents=True, exist_ok=True)
    save_config(settings, adapter / "adapter_config.json")
    resume_step = settings.get("resume_step", 0)
    if not 0 <= resume_step < args.iters:
        raise ValueError("Resume step must be smaller than the total update target.")
    schedule_offset = resume_step
    restored = None
    if state_path := settings.get("resume_training_state"):
        state_path = Path(state_path)
        restored = json.loads((state_path / "progress.json").read_text())
        task = json.loads((Path(args.data) / "tokens.json").read_text())
        if restored["step"] != resume_step or any(
            restored[key] != task[key] for key in ("dataset_sha256", "task_sha256")
        ):
            raise ValueError(
                "Resume state does not match the retained step, dataset or task."
            )
        schedule_offset = restored["schedule_offset"]
    schedule = build_schedule(args.lr_schedule)
    optimizer = optim.Adam(learning_rate=lambda step: schedule(step + schedule_offset))
    if restored is not None:
        optimizer.state = tree_unflatten(
            list(mx.load(str(state_path / "optimizer.safetensors")).items())
        )
        restore_random_state(mx.load(str(state_path / "random.safetensors"))["key"])
        print("Restored optimizer and random state.", flush=True)
    elif resume_step:
        print(
            "Retained checkpoint has weights only; optimizer restarts "
            "at the retained schedule position.",
            flush=True,
        )

    class Checkpoints(TrainingCallback):
        def on_train_loss_report(self, info):
            step = resume_step + info["iteration"]
            if step % args.save_every == 0 or step == args.iters:
                save_training_checkpoint(
                    model, optimizer, adapter, step, schedule_offset
                )

    if args.save_every % args.steps_per_report:
        raise ValueError(
            "Checkpoint interval must be a multiple of the report interval."
        )
    training_args = TrainingArgs(
        batch_size=args.batch_size,
        iters=args.iters - resume_step,
        val_batches=args.val_batches,
        steps_per_report=args.steps_per_report,
        steps_per_eval=args.steps_per_eval,
        steps_per_save=args.save_every,
        max_seq_length=args.max_seq_length,
        adapter_file=str(adapter / "adapters.safetensors"),
        grad_checkpoint=args.grad_checkpoint,
        clear_cache_threshold=settings.get("clear_cache_threshold", 0),
    )
    print("Training loss uses completion tokens only; padding is excluded.", flush=True)
    train(
        model,
        optimizer,
        train_set,
        valid_set,
        args=training_args,
        loss=lambda model, batch, lengths: completion_loss(
            model, batch, lengths, args.completion_window
        ),
        iterate_batches=lambda **kwargs: resumed_batches(
            iterate_batches, resume_step, args.seed, **kwargs
        ),
        training_callback=Checkpoints(),
    )
