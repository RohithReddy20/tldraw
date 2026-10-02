import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from actions import messages_for, model_call, parse_call, validate_call
from dataset import audit_examples, read_examples
from lab import _summarize, verify_token_cache
from sessions import CanvasSession, summarize_sessions


class ActionValidationTests(unittest.TestCase):
    def test_short_ids_round_trip_to_real_canvas_ids(self):
        canvas = {
            "schemas": [
                {"id": "schema:opaque-first", "name": "Other"},
                {"id": "shape:opaque-target", "name": "User"},
            ],
            "selected_ids": ["shape:opaque-target"],
        }
        action = {
            "name": "rename_schema",
            "arguments": {"schema_id": "shape:opaque-target", "new_name": "Member"},
        }
        self.assertEqual(
            model_call(action, canvas),
            {
                "name": "rename_schema",
                "arguments": {"schema_id": "box2", "new_name": "Member"},
            },
        )
        text = (
            "<start_function_call>call:rename_schema{schema_id:<escape>box2<escape>,"
            "new_name:<escape>Member<escape>}<end_function_call>"
        )
        self.assertEqual(parse_call(text, canvas), action)
        self.assertIn(
            '"selection": "single", "selected_ids": ["box2"]',
            messages_for("Rename it to Member.", canvas)[1]["content"],
        )

    def test_invalid_selection_and_duplicate_ids_fail_before_inference(self):
        for canvas in (
            {"selected_ids": ["unknown"]},
            {"schemas": [{"id": "same", "name": "A"}, {"id": "same", "name": "B"}]},
            {"schemas": [{"id": "one", "name": "A"}], "selected_ids": ["one", "one"]},
        ):
            with self.subTest(canvas=canvas), self.assertRaises(ValueError):
                messages_for("Rename it to B.", canvas)

    def test_preserves_punctuation_inside_strings(self):
        call = (
            "<start_function_call>call:create_schema_box{"
            "name:<escape>Account {draft}<escape>,"
            "fields:[<escape>lastName, firstName<escape>],"
            "methods:[<escape>getName()<escape>]}<end_function_call>"
        )
        self.assertEqual(
            parse_call(call, {}),
            {
                "name": "create_schema_box",
                "arguments": {
                    "name": "Account {draft}",
                    "fields": ["lastName, firstName"],
                    "methods": ["getName()"],
                },
            },
        )

    def test_rejects_explanation_multiple_calls_and_truncation(self):
        call = (
            "<start_function_call>call:no_action{"
            "reason:<escape>missing_target<escape>}<end_function_call>"
        )
        for text in (f"Sure. {call}", call + call, call[:-5]):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_call(text, {})

    def test_rejects_unknown_ids_and_extra_arguments(self):
        for arguments in (
            {"schema_id": "schema:invented", "property_name": "age"},
            {"schema_id": "schema:invented", "property_name": "age", "code": "run()"},
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                validate_call({"name": "add_property", "arguments": arguments}, {})

    def test_rejects_duplicate_arguments(self):
        text = (
            "<start_function_call>call:no_action{"
            "reason:<escape>missing_target<escape>,"
            "reason:<escape>ambiguous_target<escape>}<end_function_call>"
        )
        with self.assertRaises(ValueError):
            parse_call(text, {})

    def test_rejects_removing_a_missing_property(self):
        canvas = {
            "schemas": [{"id": "schema:42", "name": "User", "properties": ["name"]}]
        }
        with self.assertRaises(ValueError):
            validate_call(
                {
                    "name": "remove_property",
                    "arguments": {"schema_id": "schema:42", "property_name": "age"},
                },
                canvas,
            )


class DatasetTests(unittest.TestCase):
    def test_tool_selection_is_separate_from_argument_validation(self):
        report = _summarize(
            [
                {
                    "raw_output": "<start_function_call>call:add_property{broken}",
                    "prediction": None,
                    "expected": {"name": "add_property", "arguments": {}},
                    "correct": False,
                    "seconds": 1.0,
                    "peak_model_memory_gb": 1.0,
                }
            ]
        )
        self.assertEqual(report["tool_selection_accuracy"], 1.0)
        self.assertEqual(report["valid_call_rate"], 0.0)
        self.assertEqual(report["exact_action_accuracy"], 0.0)

    def test_starter_labels_and_splits_are_valid(self):
        counts = audit_examples(read_examples())
        self.assertEqual(set(counts), {"train", "valid", "test"})
        self.assertTrue(all(count > 0 for count in counts.values()))

    def test_rejects_paraphrase_group_leakage(self):
        examples = read_examples()
        leaked = copy.deepcopy(examples[0])
        leaked.update(id="leaked", split="test", command="A different paraphrase.")
        with self.assertRaises(ValueError):
            audit_examples([*examples, leaked])

    def test_rejects_duplicate_input_across_splits(self):
        examples = read_examples()
        leaked = copy.deepcopy(examples[0])
        leaked.update(id="leaked", split="test", group="different-group")
        with self.assertRaises(ValueError):
            audit_examples([*examples, leaked])

    def test_rejects_conflicting_labels_for_the_same_input(self):
        examples = read_examples()
        conflict = copy.deepcopy(examples[0])
        conflict.update(
            id="conflicting-label",
            expected={
                "name": "no_action",
                "arguments": {"reason": "unsupported_request"},
            },
        )
        with self.assertRaisesRegex(ValueError, "conflicting"):
            audit_examples([*examples, conflict])


class SessionTests(unittest.TestCase):
    def test_reordered_aliases_keep_the_same_history_target(self):
        session = CanvasSession({"schemas": [{"id": "other", "name": "Other"}]})
        session.execute(
            "Create User.",
            {
                "name": "create_schema_box",
                "arguments": {"name": "User", "fields": ["name"], "methods": []},
            },
            "user",
        )
        for index in range(8):
            session.execute(
                f"Add p{index}.",
                {
                    "name": "add_property",
                    "arguments": {"schema_id": "user", "property_name": f"p{index}"},
                },
                "unused",
            )
        session.external([{"kind": "reorder", "ids": ["user", "other"]}])
        prompt = messages_for(
            "Rename the last created box.", session.canvas, session.history
        )[1]["content"]
        self.assertEqual(len(session.history["turns"]), 3)
        self.assertIn('"last_created_id": "box1"', prompt)
        self.assertIn('"last_edited_id": "box1"', prompt)
        session.external([{"kind": "delete", "id": "user"}])
        self.assertIn(
            '"last_created_id": null',
            messages_for("Rename it.", session.canvas, session.history)[1]["content"],
        )

    def test_rejected_and_idempotent_edits_preserve_the_canvas(self):
        session = CanvasSession(
            {
                "schemas": [{"id": "user", "name": "User", "properties": ["name"]}],
                "selected_ids": ["user"],
            }
        )
        before = session.snapshot()
        with self.assertRaises(ValueError):
            session.execute(
                "Remove age.",
                {
                    "name": "remove_property",
                    "arguments": {"schema_id": "user", "property_name": "age"},
                },
                "unused",
            )
        self.assertEqual(session.snapshot(), before)
        session.execute(
            "Add name.",
            {
                "name": "add_property",
                "arguments": {"schema_id": "user", "property_name": "name"},
            },
            "unused",
        )
        self.assertEqual(session.canvas["schemas"][0]["properties"], ["name"])
        session.execute(
            "Explain.",
            {"name": "no_action", "arguments": {"reason": "unsupported_request"}},
            "unused",
        )
        self.assertEqual(session.history["last_edited_id"], "user")

    def test_session_report_counts_compounding_errors_and_recovery(self):
        rows = [
            {
                "session_id": "one",
                "turn": i,
                "correct": correct,
                "state_matches": state,
                "mutated": mutated,
                "expected": {"name": "no_action" if i == 1 else "add_property"},
            }
            for i, (correct, state, mutated) in enumerate(
                [(True, True, True), (False, False, True), (True, True, True)]
            )
        ]
        report = summarize_sessions(rows)
        self.assertEqual(report["perfect_sessions"], 0)
        self.assertEqual(report["unwanted_mutations"], 1)
        self.assertEqual(report["recoveries"], 1)
        self.assertEqual(report["first_error_turns"], {"one": 2})


class RecoveryTests(unittest.TestCase):
    def test_recovery_uses_the_newest_complete_verified_checkpoint(self):
        from colab_follow import NAME, newest_checkpoint

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for step in (500, 1000, 1500):
                path = root / "runs" / f"{NAME}-checkpoint-{step}"
                adapter = path / "adapter"
                adapter.mkdir(parents=True)
                weights = f"weights-{step}".encode()
                (adapter / "adapters.safetensors").write_bytes(weights)
                (path / "checkpoint.json").write_text(
                    json.dumps(
                        {
                            "step": step,
                            "adapter_sha256": hashlib.sha256(weights).hexdigest(),
                        }
                    )
                )
                if step > 500:
                    (adapter / "progress.json").write_text(
                        json.dumps(
                            {
                                "step": step,
                                "dataset_sha256": "dataset",
                                "task_sha256": "task",
                            }
                        )
                    )
                    (adapter / "optimizer.safetensors").write_bytes(b"optimizer")
                    if step == 1000:
                        (adapter / "random.safetensors").write_bytes(b"random")
            with (
                patch("colab_follow.ROOT", root),
                patch("lab.dataset_hash", return_value="dataset"),
                patch("lab._task_hash", return_value="task"),
            ):
                self.assertEqual(newest_checkpoint().name, f"{NAME}-checkpoint-1000")
                (
                    root
                    / "runs"
                    / f"{NAME}-checkpoint-1000"
                    / "adapter"
                    / "adapters.safetensors"
                ).write_bytes(b"corrupted")
                self.assertEqual(newest_checkpoint().name, f"{NAME}-checkpoint-500")

    def test_automatic_recovery_stops_at_the_configured_limit(self):
        from colab_follow import RuntimeDisappeared, main

        with (
            tempfile.TemporaryDirectory() as directory,
            patch("colab_follow.ROOT", Path(directory)),
            patch(
                "sys.argv",
                ["colab_follow.py", "--session", "original", "--max-recoveries", "1"],
            ),
            patch("colab_follow.cli", return_value="Session 'training' not found."),
            patch("colab_follow.recover", return_value="replacement") as recover,
        ):
            with self.assertRaises(RuntimeDisappeared):
                main()
            recover.assert_called_once()

    def test_missing_runtime_is_a_failure_even_when_cli_exits_successfully(self):
        from colab_follow import require_live_session

        require_live_session("Kernel: BUSY")
        for output in (
            "[colab] Session 'training' not found.",
            "No active sessions found on server.",
        ):
            with (
                self.subTest(output=output),
                self.assertRaisesRegex(RuntimeError, "interrupted"),
            ):
                require_live_session(output)

    def test_resumed_batches_continue_without_repeating_data(self):
        import itertools

        import numpy as np

        from training import resumed_batches

        def iterator(**kwargs):
            while True:
                yield from np.random.permutation(20)

        expected = list(
            itertools.islice(resumed_batches(iterator, 0, 42, loop=True), 40)
        )
        resumed = resumed_batches(iterator, 17, 42, loop=True)
        actual = []
        for _ in range(23):
            np.random.random(10)
            actual.append(next(resumed))
        self.assertEqual(actual, expected[17:])

    def test_checkpoint_random_state_continues_the_same_draws(self):
        import mlx.core as mx

        from training import restore_random_state

        mx.random.seed(42)
        mx.eval(mx.random.normal((8,)))
        retained = mx.random.state[0]
        expected = [mx.random.normal((8,)), mx.random.uniform(shape=(8,))]
        mx.eval(expected)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "random.safetensors"
            mx.save_safetensors(str(path), {"key": retained})
            mx.random.seed(999)
            restore_random_state(mx.load(str(path))["key"])
            actual = [mx.random.normal((8,)), mx.random.uniform(shape=(8,))]
            mx.eval(actual)
            for original, continued in zip(expected, actual, strict=True):
                self.assertTrue(mx.array_equal(original, continued).item())

    def test_checkpoint_preserves_optimizer_and_verifies_archive_parts(self):
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        from mlx.utils import tree_flatten, tree_unflatten

        from training import save_training_checkpoint

        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            adapter = run / "adapter"
            adapter.mkdir()
            (run / "data").mkdir()
            (run / "data" / "tokens.json").write_text(
                json.dumps({"dataset_sha256": "dataset", "task_sha256": "task"})
            )
            (run / "config.yaml").write_text("iters: 15000\n")
            (adapter / "adapter_config.json").write_text("{}")
            model = nn.Linear(2, 2)
            optimizer = optim.Adam(learning_rate=0.001)
            gradients = model.trainable_parameters()
            optimizer.update(model, gradients)
            mx.eval(model.parameters(), optimizer.state)
            save_training_checkpoint(model, optimizer, adapter, 500, 100)
            checkpoint = adapter / "checkpoints" / "0000500"
            restored = tree_unflatten(
                list(mx.load(str(checkpoint / "optimizer.safetensors")).items())
            )
            expected = dict(tree_flatten(optimizer.state))
            self.assertEqual(set(dict(tree_flatten(restored))), set(expected))
            for key, actual in tree_flatten(restored):
                self.assertTrue(mx.array_equal(actual, expected[key]).item(), key)
            continued_model = nn.Linear(2, 2)
            continued_model.update(mx.load(str(checkpoint / "adapters.safetensors")))
            continued_optimizer = optim.Adam(learning_rate=0.001)
            continued_optimizer.state = restored
            gradients = model.trainable_parameters()
            optimizer.update(model, gradients)
            continued_optimizer.update(continued_model, gradients)
            mx.eval(model.parameters(), continued_model.parameters())
            continued = dict(tree_flatten(continued_model.parameters()))
            for key, actual in tree_flatten(model.parameters()):
                self.assertTrue(mx.array_equal(actual, continued[key]).item(), key)
            self.assertEqual(
                json.loads((checkpoint / "progress.json").read_text())[
                    "schedule_offset"
                ],
                100,
            )
            chunks = []
            for part in json.loads((checkpoint / "checkpoint.parts.json").read_text()):
                content = (checkpoint / part["name"]).read_bytes()
                self.assertEqual(hashlib.sha256(content).hexdigest(), part["sha256"])
                chunks.append(content)
            self.assertEqual(
                b"".join(chunks), (checkpoint / "checkpoint.zip").read_bytes()
            )


class CompletionLossTests(unittest.TestCase):
    def test_prepared_tokens_must_match_the_task_and_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokens = root / "tokens-train.npz"
            tokens.write_bytes(b"validated-cache")
            manifest = {
                "dataset_sha256": "dataset",
                "task_sha256": "task",
                "max_tokens": 1444,
                "max_completion_tokens": 100,
                "files": {tokens.name: hashlib.sha256(tokens.read_bytes()).hexdigest()},
            }
            (root / "tokens.json").write_text(json.dumps(manifest))
            config = {"max_seq_length": 2048, "completion_window": 192}
            with (
                patch("lab.DATA", root),
                patch("lab.dataset_hash", return_value="dataset"),
                patch("lab._task_hash", return_value="task"),
            ):
                self.assertTrue(verify_token_cache(config))
                tokens.write_bytes(b"modified-cache")
                with self.assertRaisesRegex(ValueError, "checksum"):
                    verify_token_cache(config)
                with self.assertRaisesRegex(ValueError, "limit"):
                    verify_token_cache({**config, "completion_window": 99})
            with (
                patch("lab.DATA", root),
                patch("lab.dataset_hash", return_value="new-data"),
                patch("lab._task_hash", return_value="task"),
            ):
                self.assertFalse(verify_token_cache(config))

    def test_answer_only_projection_matches_full_loss_and_gradients(self):
        import mlx.core as mx
        import mlx.nn as nn
        from mlx.utils import tree_flatten
        from mlx_lm.models.gemma3_text import Model, ModelArgs

        from training import completion_indices, completion_loss, packed_completion_loss

        batch = mx.array([[1, 2, 3, 4, 5, 0], [2, 3, 4, 5, 0, 0]])
        lengths = mx.array([[2, 5], [2, 4]])
        model = Model(
            ModelArgs(
                model_type="gemma3_text",
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=2,
                num_attention_heads=2,
                head_dim=4,
                num_key_value_heads=1,
                vocab_size=16,
                sliding_window_pattern=2,
            )
        )

        def full_loss(model, batch, lengths):
            positions = mx.arange(1, batch.shape[1])
            mask = (positions >= lengths[:, :1]) & (positions < lengths[:, 1:])
            loss = nn.losses.cross_entropy(model(batch[:, :-1]), batch[:, 1:])
            return (loss.astype(mx.float32) * mask).sum() / mask.sum(), mask.sum()

        for tied in (False, True):
            if tied:
                model.pop("lm_head")
                model.tie_word_embeddings = True
            full, full_grad = nn.value_and_grad(model, full_loss)(model, batch, lengths)
            compact, compact_grad = nn.value_and_grad(
                model, lambda m, b, lengths: completion_loss(m, b, lengths, 4)
            )(model, batch, lengths)
            indices, mask = completion_indices(lengths.tolist(), batch.shape[1])
            packed, packed_grad = nn.value_and_grad(model, packed_completion_loss)(
                model, batch, mx.array(indices), mx.array(mask)
            )
            for result, gradients in ((compact, compact_grad), (packed, packed_grad)):
                self.assertTrue(mx.allclose(full[0], result[0], atol=1e-5).item())
                self.assertEqual(result[1].item(), 5)
                for (key, expected), (other_key, actual) in zip(
                    tree_flatten(full_grad), tree_flatten(gradients), strict=True
                ):
                    self.assertEqual(key, other_key)
                    self.assertTrue(
                        mx.allclose(expected, actual, atol=1e-5).item(), key
                    )

    def test_packed_targets_reject_truncated_or_empty_answers(self):
        from training import completion_indices

        for bounds in ([[0, 5]], [[5, 5]], [[3, 7]]):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                completion_indices(bounds, 6)


if __name__ == "__main__":
    unittest.main()
