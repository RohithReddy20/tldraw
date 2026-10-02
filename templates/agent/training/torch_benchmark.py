import json
import statistics
import time


def worker(root, output, case, verify=False):
    import numpy as np
    import torch
    import torch.distributed as distributed
    import torch.nn.functional as functional
    import yaml
    from safetensors.torch import load_file, save_file
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.utils.checkpoint import checkpoint

    from gpu_benchmark import CASES, representative_batches
    from training import completion_indices

    settings = CASES[case]
    workers = settings["workers"]
    if workers > 1:
        distributed.init_process_group("nccl")
    rank = distributed.get_rank() if workers > 1 else 0
    if not torch.cuda.is_available():
        raise RuntimeError("The PyTorch benchmark requires CUDA.")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    base = root / "runs/base-bb327a9a-float16"
    config = json.loads((base / "config.json").read_text())
    training = yaml.safe_load((root / "config.yaml").read_text())
    if training["lora_parameters"]["dropout"] != 0:
        raise ValueError("This numerical comparison requires zero LoRA dropout.")
    weights = {}
    for path in sorted(base.glob("*.safetensors")):
        weights.update(load_file(str(path), device="cuda"))
    compute_dtype = (
        torch.float32 if settings.get("dtype") == "float32" else torch.float16
    )
    weights = {key: value.to(compute_dtype) for key, value in weights.items()}
    adapter = load_file(str(root / "warm-start.safetensors"), device="cuda")
    use_xformers = settings.get("attention") == "xformers"
    if use_xformers:
        import xformers.ops as xops
        from xformers.ops.fmha.attn_bias import BlockDiagonalMask

        attention_biases = {}

    class Gemma(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.adapter = torch.nn.ParameterDict(
                {
                    key.replace(".", "__"): torch.nn.Parameter(value.float())
                    for key, value in adapter.items()
                }
            )
            self.scale = training["lora_parameters"]["scale"]

        def linear(self, value, name):
            result = functional.linear(value, weights[name + ".weight"])
            key = (name + ".lora_a").replace(".", "__")
            if key in self.adapter:
                low = value.float() @ self.adapter[key]
                low = low @ self.adapter[(name + ".lora_b").replace(".", "__")]
                result = result + (self.scale * low).to(value.dtype)
            return result

        def norm(self, value, name):
            # MLX rounds 1 + weight in the base dtype before its fused RMS norm.
            weight = (weights[name + ".weight"] + 1).float()
            return functional.rms_norm(
                value.float(), (value.shape[-1],), weight, config["rms_norm_eps"]
            ).to(value.dtype)

        def layer(self, hidden, index, positions, global_mask, sliding_mask):
            prefix = f"model.layers.{index}"
            attention = prefix + ".self_attn"
            normed = self.norm(hidden, prefix + ".input_layernorm")
            batch, length, _ = hidden.shape
            head = config["head_dim"]
            query, key, value = [
                self.linear(normed, attention + "." + projection)
                .reshape(batch, length, -1, head)
                .transpose(1, 2)
                for projection in ("q_proj", "k_proj", "v_proj")
            ]
            query = self.norm(query, attention + ".q_norm")
            key = self.norm(key, attention + ".k_norm")
            global_layer = (index + 1) % config["_sliding_window_pattern"] == 0
            cosine, sine = positions[int(global_layer)]

            def rotate(value):
                first, second = value.float().chunk(2, dim=-1)
                return torch.cat(
                    (first * cosine - second * sine, second * cosine + first * sine),
                    dim=-1,
                ).to(value.dtype)

            query, key = rotate(query), rotate(key)
            repeats = config["num_attention_heads"] // config["num_key_value_heads"]
            key = key.repeat_interleave(repeats, dim=1)
            value = value.repeat_interleave(repeats, dim=1)
            if use_xformers:
                # CUTLASS backward supports block-local masks on T4, but rejects
                # LowerTriangularFromBottomRightLocalAttentionMask.
                attended = (
                    xops.memory_efficient_attention(
                        query.transpose(1, 2).reshape(1, batch * length, -1, head),
                        key.transpose(1, 2).reshape(1, batch * length, -1, head),
                        value.transpose(1, 2).reshape(1, batch * length, -1, head),
                        attn_bias=attention_biases[(batch, length)][int(global_layer)],
                        scale=config["query_pre_attn_scalar"] ** -0.5,
                    )
                    .reshape(batch, length, -1, head)
                    .transpose(1, 2)
                )
            else:
                attended = functional.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=None if global_layer else sliding_mask,
                    is_causal=global_layer,
                    scale=config["query_pre_attn_scalar"] ** -0.5,
                )
            attended = attended.transpose(1, 2).reshape(batch, length, -1)
            attended = self.linear(attended, attention + ".o_proj")

            def residual(left, right):
                if left.dtype != torch.float16:
                    return left + right
                return (
                    (left.float() + right.float()).clamp(-65504, 65504).to(left.dtype)
                )

            hidden = residual(
                hidden, self.norm(attended, prefix + ".post_attention_layernorm")
            )
            normed = self.norm(hidden, prefix + ".pre_feedforward_layernorm")
            gated = functional.gelu(
                self.linear(normed, prefix + ".mlp.gate_proj"), approximate="tanh"
            )
            value = gated * self.linear(normed, prefix + ".mlp.up_proj")
            value = self.linear(value, prefix + ".mlp.down_proj")
            return residual(
                hidden, self.norm(value, prefix + ".post_feedforward_layernorm")
            )

        def forward(self, batch, indices, mask, positions, global_mask, sliding_mask):
            hidden = functional.embedding(
                batch[:, :-1], weights["model.embed_tokens.weight"]
            )
            scale = torch.tensor(
                config["hidden_size"] ** 0.5, dtype=torch.bfloat16, device="cuda"
            )
            hidden = hidden * scale.to(hidden.dtype)
            for index in range(config["num_hidden_layers"]):
                if settings["checkpoint"] == "all":
                    hidden = checkpoint(
                        self.layer,
                        hidden,
                        index,
                        positions,
                        global_mask,
                        sliding_mask,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    hidden = self.layer(
                        hidden, index, positions, global_mask, sliding_mask
                    )
            hidden = self.norm(hidden, "model.norm")
            selected = hidden.reshape(-1, hidden.shape[-1])[indices]
            embedding = weights.get(
                "lm_head.weight", weights["model.embed_tokens.weight"]
            )
            logits = functional.linear(selected, embedding)
            targets = batch[:, 1:].reshape(-1)[indices]
            loss = functional.cross_entropy(
                logits.float(), targets, reduction="none"
            ).to(logits.dtype)
            return (loss.float() * mask).sum() / mask.sum()

    model = Gemma().cuda().train()
    batches = []
    raw_batches = representative_batches(root / "data/tokens-train.npz", verify)
    for raw, bounds in raw_batches:
        rows = np.arange(rank, 8, workers)
        batch, lengths = raw[rows], bounds[rows]
        count = int((lengths[:, 1] - lengths[:, 0]).sum())
        total = int((bounds[:, 1] - bounds[:, 0]).sum())
        indices, mask = completion_indices(lengths, batch.shape[1])
        length = batch.shape[1] - 1
        sequence = torch.arange(length, device="cuda", dtype=torch.float32)
        positions = []
        for theta in (config["rope_local_base_freq"], config["rope_theta"]):
            frequencies = theta ** (
                -torch.arange(0, config["head_dim"], 2, device="cuda").float()
                / config["head_dim"]
            )
            angles = sequence[:, None] * frequencies[None, :]
            positions.append((angles.cos(), angles.sin()))
        causal = sequence[:, None] >= sequence[None, :]
        sliding = causal & (
            sequence[:, None] < sequence[None, :] + config["sliding_window"]
        )
        global_mask = torch.zeros(
            (length, length), dtype=compute_dtype, device="cuda"
        ).masked_fill(~causal, float("-inf"))
        sliding_mask = torch.zeros_like(global_mask).masked_fill(
            ~sliding, float("-inf")
        )
        if use_xformers:
            batch_size = len(rows)
            diagonal = BlockDiagonalMask.from_seqlens(
                [length] * batch_size, device=torch.device("cuda")
            )
            biases = [
                diagonal.make_local_attention(config["sliding_window"]),
                diagonal.make_causal(),
            ]
            attention_biases[(batch_size, length)] = biases
            for bias, expected in zip(biases, (sliding_mask, global_mask), strict=True):
                combined = torch.full(
                    (batch_size * length, batch_size * length),
                    float("-inf"),
                    dtype=compute_dtype,
                    device="cuda",
                )
                for row in range(batch_size):
                    begin, end = row * length, (row + 1) * length
                    combined[begin:end, begin:end] = expected
                if not torch.equal(
                    bias.materialize(
                        combined.shape, dtype=compute_dtype, device="cuda"
                    ),
                    combined,
                ):
                    raise ValueError(
                        "Structured attention changes the allowed context."
                    )
                del combined
        batches.append(
            (
                (
                    torch.as_tensor(batch, dtype=torch.long, device="cuda"),
                    torch.as_tensor(indices, dtype=torch.long, device="cuda"),
                    torch.as_tensor(mask, device="cuda"),
                    positions,
                    global_mask,
                    sliding_mask,
                ),
                workers * count / total,
            )
        )

    def gradients():
        return {
            key.replace("__", "."): value.grad.detach().contiguous()
            for key, value in model.adapter.items()
        }

    def synchronize_gradients():
        if workers == 1:
            return
        flat = torch.cat([value.grad.reshape(-1) for value in model.parameters()])
        distributed.all_reduce(flat)
        flat /= workers
        offset = 0
        for value in model.parameters():
            value.grad.copy_(flat[offset : offset + value.numel()].view_as(value))
            offset += value.numel()

    output.mkdir(parents=True, exist_ok=True)
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        forward = (
            torch.compile(model, dynamic=False) if settings.get("compile") else model
        )
        data, weight = batches[-1]
        initial = forward(*data) * weight
        initial.backward()
        synchronize_gradients()
        if workers > 1:
            distributed.all_reduce(initial.detach())
            initial = initial / workers
        initial_loss = initial.item()
        if rank == 0:
            save_file(gradients(), str(output / "gradients.safetensors"))
        model.zero_grad(set_to_none=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-5, fused=True)
        updates, interval, warmup = (80, 8, 16) if verify else (24, 4, 12)
        reports, rates = [], []
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        steady_started = None
        elapsed, losses = 0.0, 0.0
        optimizer_error = None
        for step in range(1, updates + 1):
            torch.cuda.synchronize()
            tick = time.perf_counter()
            data, weight = batches[(step - 1) % len(batches)]
            loss = forward(*data) * weight
            loss.backward()
            synchronize_gradients()
            # MLX Adam disables bias correction; undo PyTorch's correction exactly.
            correction = (1 - 0.999**step) ** 0.5
            for group in optimizer.param_groups:
                group["lr"] = 3e-5 * (1 - 0.9**step) / correction
                group["eps"] = 1e-8 / correction
            if step == 1:
                expected = [
                    value.detach()
                    - 3e-5
                    * (0.1 * value.grad)
                    / ((0.001 * value.grad.square()).sqrt() + 1e-8)
                    for value in model.parameters()
                ]
            optimizer.step()
            if step == 1:
                optimizer_error = max(
                    (value.detach() - reference).abs().max().item()
                    for value, reference in zip(
                        model.parameters(), expected, strict=True
                    )
                )
                if optimizer_error > 1e-7:
                    raise ValueError("PyTorch Adam update does not match MLX Adam.")
                del expected
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            elapsed += time.perf_counter() - tick
            losses += loss.item()
            if step % interval == 0:
                report = {
                    "iteration": step,
                    "train_loss": losses / interval,
                    "iterations_per_second": interval / elapsed,
                }
                reports.append(report)
                print(json.dumps(report), flush=True)
                if step > warmup:
                    rates.append(report["iterations_per_second"])
                if step == warmup:
                    steady_started = time.time()
                elapsed, losses = 0.0, 0.0
        if rank == 0:
            save_file(
                {
                    key.replace("__", "."): value.detach()
                    for key, value in model.adapter.items()
                },
                str(output / "benchmark-adapters.safetensors"),
            )
            rate = statistics.median(rates)
            result = {
                "case": case,
                "settings": settings,
                "verification": verify,
                "torch_version": torch.__version__,
                "optimizer": "adam_without_bias_correction",
                "optimizer_check_max_error": optimizer_error,
                "cuda_version": torch.version.cuda,
                "global_batch": 8,
                "updates": updates,
                "batch_shapes": [list(batch.shape) for batch, _ in raw_batches],
                "initial_loss": initial_loss,
                "updates_per_second": rate,
                "examples_per_second": 8 * rate,
                "min_updates_per_second": min(rates),
                "max_updates_per_second": max(rates),
                "peak_memory_gb_per_worker": torch.cuda.max_memory_allocated() / 1e9,
                "training_seconds_including_warmup": time.monotonic() - started,
                "steady_started_at": steady_started,
                "finished_at": time.time(),
                "reports": reports,
            }
            (output / "result.json").write_text(json.dumps(result, indent=2))
            print("RESULT " + json.dumps(result), flush=True)
    if workers > 1:
        distributed.destroy_process_group()
