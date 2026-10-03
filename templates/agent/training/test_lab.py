import copy
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from actions import messages_for, model_call, model_canvas, parse_call, validate_call
from dataset import audit_examples, read_examples
from lab import _summarize, verify_token_cache
from sessions import CanvasSession, summarize_sessions


class RefinementDataTests(unittest.TestCase):
    def test_refinement_preserves_holdouts_and_pairs_missing_fields_with_present_fields(
        self,
    ):
        from build_dataset import build_refinement

        original = [
            {
                "id": split,
                "group": split,
                "split": split,
                "command": f"Explain {split}.",
                "canvas": {},
                "expected": {
                    "name": "no_action",
                    "arguments": {"reason": "unsupported_request"},
                },
            }
            for split in ("train", "valid", "test")
        ]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "examples.jsonl"
            source.write_text("".join(json.dumps(row) + "\n" for row in original))
            rows = build_refinement(source, seed=54, replay_count=1, pairs=2)
        self.assertEqual([row for row in rows if row["split"] != "train"], original[1:])
        generated = {row["id"]: row for row in rows if row["id"].startswith("v5-hard-")}
        for index in range(2):
            present, absent = (
                generated[f"v5-hard-{index}:2"],
                generated[f"v5-hard-{index}:3"],
            )
            self.assertEqual(present["command"], absent["command"])
            self.assertEqual(present["expected"]["name"], "remove_property")
            field = present["expected"]["arguments"]["property_name"]
            self.assertIn(field, present["canvas"]["schemas"][0]["properties"])
            self.assertNotIn(field, absent["canvas"]["schemas"][0]["properties"])
            self.assertEqual(
                absent["expected"],
                {"name": "no_action", "arguments": {"reason": "missing_target"}},
            )


class LexicalPreparationTests(unittest.TestCase):
    def record(self, identifier, split="train", command=None, expected=None):
        row = {
            "id": identifier,
            "group": identifier,
            "split": split,
            "command": command or f"Explain the concept {identifier}.",
            "canvas": {},
            "expected": expected
            or {
                "name": "no_action",
                "arguments": {"reason": "unsupported_request"},
            },
        }
        return row, (json.dumps(row, sort_keys=True) + " \n").encode()

    def test_balanced_replay_is_unique_train_only_and_excludes_development_sentences(
        self,
    ):
        from prepare_modal_training import command_sentence, select_lexical_replay

        actions = (
            {"name": "no_action", "arguments": {"reason": "missing_target"}},
            {"name": "canvas_command", "arguments": {"operation": "zoom_in"}},
            {
                "name": "canvas_command",
                "arguments": {"operation": "pan", "dx": 20, "dy": 0},
            },
        )
        records = [
            self.record(f"train-{action}-{index}", expected=expected)
            for action, expected in enumerate(actions)
            for index in range(5)
        ]
        forbidden = self.record(
            "forbidden", command="  Explain the dev topic! ", expected=actions[0]
        )
        development = [self.record("dev", "valid", "explain the dev topic.")]
        selected, summary = select_lexical_replay(
            [*records, forbidden, *development], development, count=6
        )
        repeated, _ = select_lexical_replay(
            [*records, forbidden, *development], development, count=6
        )
        self.assertEqual(selected, repeated)
        self.assertEqual(len({row["id"] for row, _ in selected}), 6)
        self.assertEqual(set(summary["stratum_counts"].values()), {2})
        self.assertTrue(all(row["split"] == "train" for row, _ in selected))
        self.assertFalse(
            any(
                command_sentence(row["command"])
                == command_sentence(development[0][0]["command"])
                for row, _ in selected
            )
        )
        only_reserved = [self.record("unavailable", command="Explain the dev topic.")]
        with self.assertRaisesRegex(ValueError, "every action stratum"):
            select_lexical_replay(only_reserved, development, count=1)

    def test_plan_has_exact_exposures_interleaving_and_unchanged_index_digest(self):
        import numpy as np

        from gpu_benchmark import quality_batches
        from prepare_modal_training import build_lexical_training_plan

        lengths = np.asarray([2 + index % 4 for index in range(2000)])
        plan, summary = build_lexical_training_plan(lengths)
        repeated, _ = build_lexical_training_plan(lengths)
        np.testing.assert_array_equal(plan, repeated)
        self.assertEqual(plan.shape, (400, 8))
        counts = np.bincount(plan.ravel(), minlength=2000)
        np.testing.assert_array_equal(counts[:1200], np.full(1200, 2))
        np.testing.assert_array_equal(counts[1200:], np.ones(800))
        blocks = plan.reshape(100, 4, 8)
        self.assertTrue(np.all(blocks[:, :3] < 1200))
        self.assertTrue(np.all(blocks[:, 3] >= 1200))
        self.assertTrue(all(len(set(batch)) == 8 for batch in plan))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            tokens = data / "tokens-train.npz"
            samples = [
                np.full(length, index + 100, np.int32)
                for index, length in enumerate(lengths)
            ]
            np.savez(
                tokens,
                tokens=np.concatenate(samples),
                boundaries=np.cumsum([0, *lengths]),
                offsets=np.ones(2000, np.int32),
            )
            frozen_plan = data / "train-plan.npz"
            np.savez(frozen_plan, indices=plan)
            (data / "tokens.json").write_text(
                json.dumps(
                    {
                        "dataset_sha256": "data",
                        "task_sha256": "task",
                        "files": {
                            "train-plan.npz": hashlib.sha256(
                                frozen_plan.read_bytes()
                            ).hexdigest()
                        },
                    }
                )
            )
            (root / "examples.jsonl").write_text(
                "".join(
                    json.dumps(
                        {
                            **self.record(f"train-{index}")[0],
                            "provenance": "lexical" if index < 1200 else "replay",
                        }
                    )
                    + "\n"
                    for index in range(2000)
                )
            )
            exposure_path = root / "exposure.json"
            batches, digest = quality_batches(
                tokens, 999, 0, 400, exposure_path=exposure_path
            )
            self.assertEqual(digest, summary["indices_sha256"])
            np.testing.assert_array_equal(
                np.asarray([batch[:, 0] - 100 for batch, _ in batches]), plan
            )
            exposure = json.loads(exposure_path.read_text())
            self.assertEqual(exposure["selected_positions"], 3200)
            self.assertEqual(exposure["unique_examples"], 2000)
            self.assertEqual(exposure["coverage_fraction"], 1.0)
            self.assertEqual(
                exposure["provenance_counts"], {"lexical": 2400, "replay": 800}
            )
        with self.assertRaisesRegex(ValueError, "2000 rows"):
            build_lexical_training_plan(lengths[:-1])

    def test_combining_keeps_every_development_and_test_row_byte_for_byte(self):
        from prepare_modal_training import combine_lexical_data

        original = [
            *[self.record(f"old-{index}") for index in range(6)],
            self.record("original-valid", "valid"),
            self.record("original-test", "test"),
        ]
        supplemental = [
            self.record("supplement-train"),
            self.record("supplement-valid", "valid"),
        ]
        lexical = [
            self.record("new-train-a"),
            self.record("new-train-b"),
            self.record("new-valid", "valid"),
        ]

        def case(identifier, split):
            row = {"id": identifier, "split": split, "initial_canvas": {}, "turns": []}
            return row, (json.dumps(row) + " \n").encode()

        original_cases = [
            case("old-practice", "train"),
            case("original-session", "valid"),
            case("original-test-session", "test"),
        ]
        supplemental_cases = [case("supplement-session", "valid")]
        lexical_cases = [case("new-practice", "train"), case("new-session", "valid")]
        rows, lines, cases, replay, _, ledger, _ = combine_lexical_data(
            original,
            supplemental,
            lexical,
            original_cases,
            supplemental_cases,
            lexical_cases,
            replay_count=2,
        )
        expected = [
            pair
            for pair in [*original, *supplemental, *lexical]
            if pair[0]["split"] != "train"
        ]
        actual = [
            (row, raw)
            for row, raw in zip(rows, lines, strict=True)
            if row["split"] != "train"
        ]
        self.assertEqual(actual, expected)
        self.assertEqual(
            cases, [*original_cases[1:], *supplemental_cases, *lexical_cases]
        )
        self.assertEqual(len([row for row in rows if row["split"] == "train"]), 4)
        self.assertEqual(len(replay), 2)
        self.assertEqual(ledger, [])
        leaked = self.record("leaked", command=lexical[-1][0]["command"])
        with self.assertRaisesRegex(ValueError, "development sentences"):
            combine_lexical_data(
                original,
                supplemental,
                [*lexical, leaked],
                original_cases,
                supplemental_cases,
                lexical_cases,
                replay_count=2,
            )

    def test_config_is_conservative_and_preserves_the_original_selection_pool(self):
        import yaml

        from prepare_modal_training import lexical_training_config, training_config

        examples = [self.record(f"valid-{index}", "valid") for index in range(256)]
        cases = [
            ({"id": f"session-{index}", "split": "valid"}, b"") for index in range(6)
        ]
        config = {
            "selection_example_ids": [row["id"] for row, _ in examples],
            "selection_session_ids": [row["id"] for row, _ in cases],
            "num_layers": -1,
            "lora_parameters": {"rank": 32, "dropout": 0.0, "scale": 8.0},
            "loss_mode": "completion_tokens_v1",
            "completion_window": 160,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.yaml").write_text(yaml.safe_dump(config))
            refined = lexical_training_config(root, examples, cases)
            legacy = training_config(root, examples, cases)
        self.assertEqual(refined["quality_updates"], 400)
        self.assertEqual(refined["checkpoint_selection_steps"], [400])
        self.assertEqual(refined["batch_size"], 8)
        self.assertEqual(refined["refinement_micro_batch"], 2)
        self.assertEqual(refined["refinement_checkpoint"], "all")
        self.assertEqual(refined["model_dtype"], "float16")
        self.assertEqual(refined["completion_window"], 160)
        self.assertEqual(refined["lr_schedule"]["arguments"], [2e-6, 380, 2e-7])
        self.assertEqual(refined["lr_schedule"]["warmup"], 20)
        self.assertEqual(
            refined["selection_example_ids"], config["selection_example_ids"]
        )
        self.assertEqual(
            refined["selection_session_ids"], config["selection_session_ids"]
        )
        self.assertEqual(legacy["quality_updates"], 3900)
        self.assertEqual(legacy["checkpoint_selection_steps"], [650, 1950, 3900])

    def test_source_freeze_requires_the_committed_head_and_preparation_never_overwrites(
        self,
    ):
        import prepare_modal_training as preparation

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code = root / "actions.py"
            code.write_bytes(b"committed source\n")
            with (
                patch.object(preparation, "TRAINING", root),
                patch.object(preparation, "LEXICAL_SOURCE_NAMES", ("actions.py",)),
            ):
                with patch.object(
                    preparation.subprocess,
                    "check_output",
                    side_effect=["commit-id\n", b"committed source\n"],
                ):
                    commit, hashes = preparation.committed_training_sources()
                self.assertEqual(commit, "commit-id")
                self.assertEqual(
                    hashes["actions.py"], hashlib.sha256(code.read_bytes()).hexdigest()
                )
                code.write_bytes(b"uncommitted source\n")
                with patch.object(
                    preparation.subprocess,
                    "check_output",
                    side_effect=["commit-id\n", b"committed source\n"],
                ):
                    with self.assertRaisesRegex(
                        ValueError, "Commit the final training source"
                    ):
                        preparation.committed_training_sources()
            with self.assertRaises(FileExistsError):
                preparation.prepare_lexical(root)


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
        _, identifiers = model_canvas(session.canvas)
        self.assertIn(f'"last_created_id": "{identifiers["user"]}"', prompt)
        self.assertIn(f'"last_edited_id": "{identifiers["user"]}"', prompt)
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


class CheckpointSelectionTests(unittest.TestCase):
    def config(self):
        return {
            "quality_updates": 3000,
            "checkpoint_selection_steps": [500, 1500, 3000],
            "selection_example_ids": [f"valid-{index}" for index in range(256)],
            "selection_session_ids": [f"session-{index}" for index in range(6)],
        }

    def metrics(self):
        return {
            "command_accuracy": 0.5,
            "session_action_accuracy": 0.5,
            "state_agreement_rate": 0.25,
            "perfect_sessions": 0,
            "final_state_matches": 0,
            "unwanted_mutations": 0,
            "joint_score": 1.25 / 3,
            "per_stratum": {
                "arrange_shapes:pack": {"correct": 1, "total": 2},
                "move_shapes": {"correct": 3, "total": 6},
            },
        }

    def test_selection_rejects_state_and_rare_operation_regressions(self):
        from gpu_benchmark import choose_checkpoint

        baseline = self.metrics()
        state = {
            **copy.deepcopy(baseline),
            "command_accuracy": 0.9,
            "state_agreement_rate": 0.1,
            "joint_score": 0.5,
        }
        operation = copy.deepcopy(state)
        operation["state_agreement_rate"] = 0.5
        operation["per_stratum"]["arrange_shapes:pack"]["correct"] = 0
        improved = {
            **copy.deepcopy(baseline),
            "command_accuracy": 0.75,
            "joint_score": 1.5 / 3,
        }
        result = choose_checkpoint(
            baseline, {500: state, 1500: operation, 3000: improved}
        )
        self.assertEqual(result["chosen_step"], 3000)
        self.assertEqual(result["eligible_steps"], [3000])
        self.assertIn("state_agreement_rate", result["rejected_steps"]["500"])
        self.assertIn("stratum:arrange_shapes:pack", result["rejected_steps"]["1500"])
        self.assertEqual(choose_checkpoint(baseline, {500: baseline})["chosen_step"], 0)
        self.assertEqual(
            choose_checkpoint(baseline, {1500: improved, 500: improved})["chosen_step"],
            500,
        )

    def test_exposure_records_exact_consumed_rows_without_changing_batches(self):
        import numpy as np

        from gpu_benchmark import quality_batches

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            path = root / "data/tokens-train.npz"
            np.savez(
                path,
                tokens=np.arange(16 * 4, dtype=np.int32),
                boundaries=np.arange(17) * 4,
                offsets=np.ones(16, dtype=np.int32),
            )
            rows = [
                {
                    "id": f"train-{index}",
                    "split": "train",
                    "expected": {
                        "name": "canvas_command",
                        "arguments": {"operation": "undo" if index % 2 else "redo"},
                    },
                    "provenance": "deliberate coverage",
                }
                for index in range(16)
            ]
            (root / "examples.jsonl").write_text(
                "".join(
                    json.dumps(row) + "\n"
                    for row in [{"id": "valid", "split": "valid"}, *rows]
                )
            )
            plain, digest = quality_batches(path, 77, 0, updates=2)
            manifest = root / "training-exposure.json"
            recorded, after = quality_batches(
                path, 77, 0, updates=2, exposure_path=manifest
            )
            self.assertEqual(digest, after)
            for before, actual in zip(plain, recorded, strict=True):
                for a, b in zip(before, actual, strict=True):
                    np.testing.assert_array_equal(a, b)
            exposure = json.loads(manifest.read_text())
            self.assertEqual(exposure["unique_examples"], 16)
            self.assertEqual(exposure["coverage_fraction"], 1)
            self.assertEqual(exposure["epoch_equivalents"], 1)
            self.assertEqual(
                exposure["selected_ids"],
                [rows[index]["id"] for index in exposure["selected_indices"]],
            )
            self.assertEqual(
                exposure["operation_counts"],
                {"canvas_command:redo": 8, "canvas_command:undo": 8},
            )
            quality_batches(path, 77, 1, updates=1, exposure_path=manifest)
            resumed = json.loads(manifest.read_text())
            self.assertEqual(
                resumed["selected_indices"], exposure["selected_indices"][8:]
            )
            self.assertEqual(resumed["coverage_fraction"], 0.5)

    def test_opt_in_selection_runs_after_training_without_scoring_test(self):
        import yaml

        from gpu_benchmark import refinement_experiment

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "root"
            work = Path(directory) / "working"
            root.mkdir()
            work.mkdir()
            (root / "config.yaml").write_text(yaml.safe_dump(self.config()))
            (root / "warm-start.json").write_text("{}")
            (root / "warm-start.safetensors").write_bytes(b"warm")
            (root / "warm-adapter-config.json").write_text("{}")
            with (
                patch(
                    "gpu_benchmark.run_case", return_value={"end_step": 3000}
                ) as train,
                patch("gpu_benchmark.select_refinement_checkpoint") as select,
                patch("gpu_benchmark.evaluate_pair") as evaluate,
            ):
                refinement_experiment(root, work, training_only=True)
            train.assert_called_once_with(
                root, work / "refinement", "dual-window", quality=True
            )
            select.assert_called_once_with(root, work, self.config())
            evaluate.assert_not_called()
            self.assertEqual(
                json.loads((work / "refinement-training.json").read_text()),
                {"end_step": 3000},
            )

    def test_export_rejects_another_task_and_preserves_final_checkpoint(self):
        from gpu_benchmark import export_checkpoint

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "dual-window"
            checkpoint = source / "adapter/checkpoints/0000500"
            checkpoint.mkdir(parents=True)
            (source / "config.yaml").write_text("model: same\n")
            (source / "adapter/adapter_config.json").write_text("{}")
            (source / "adapter/adapters.safetensors").write_bytes(b"final-3000")
            (checkpoint / "adapters.safetensors").write_bytes(b"selected-500")
            metadata = {"dataset_sha256": "dataset", "task_sha256": "task"}
            (checkpoint / "progress.json").write_text(
                json.dumps({"step": 500, **metadata})
            )
            snapshot = {
                str(path.relative_to(source)): path.read_bytes()
                for path in source.rglob("*")
                if path.is_file()
            }
            selected = Path(directory) / "selected"
            export_checkpoint(source, selected, 500, metadata)
            self.assertEqual(
                (selected / "adapter/adapters.safetensors").read_bytes(),
                b"selected-500",
            )
            self.assertEqual(
                {
                    str(path.relative_to(source)): path.read_bytes()
                    for path in source.rglob("*")
                    if path.is_file()
                },
                snapshot,
            )
            rejected = Path(directory) / "rejected"
            with self.assertRaisesRegex(ValueError, "differs"):
                export_checkpoint(
                    source, rejected, 500, {**metadata, "task_sha256": "another"}
                )
            self.assertFalse(rejected.exists())

    def test_selection_exports_the_best_checkpoint_after_full_validation(self):
        import yaml

        from gpu_benchmark import select_refinement_checkpoint

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "root"
            work = Path(directory) / "working"
            (root / "data").mkdir(parents=True)
            metadata = {"dataset_sha256": "dataset", "task_sha256": "task"}
            (root / "data/tokens.json").write_text(json.dumps(metadata))
            config = self.config()
            for case in ("baseline", "dual-window"):
                folder = work / "refinement" / case
                (folder / "adapter").mkdir(parents=True)
                (folder / "config.yaml").write_text(yaml.safe_dump(config))
                (folder / "adapter/adapter_config.json").write_text("{}")
                (folder / "adapter/adapters.safetensors").write_text(
                    "baseline" if case == "baseline" else "3000"
                )
            final = work / "refinement/dual-window"
            for step in config["checkpoint_selection_steps"]:
                checkpoint = final / "adapter/checkpoints" / f"{step:07d}"
                checkpoint.mkdir(parents=True)
                (checkpoint / "adapters.safetensors").write_text(str(step))
                (checkpoint / "progress.json").write_text(
                    json.dumps({"step": step, **metadata})
                )
            snapshot = {
                str(path.relative_to(final)): path.read_bytes()
                for path in final.rglob("*")
                if path.is_file()
            }
            force_regression = False

            def score(*args, folders, selection=False):
                for folder in folders.values():
                    weight = (folder / "adapter/adapters.safetensors").read_text()
                    action = {
                        "name": "no_action",
                        "arguments": {"reason": "missing_target"},
                    }

                    def correct(index, model_weight=weight):
                        if force_regression and not selection:
                            return model_weight == "baseline"
                        return model_weight in ("1500", "3000") or (
                            model_weight == "500" and index % 2 == 0
                        )

                    command_rows = [
                        {
                            "id": identifier,
                            "command": "Move the missing shape.",
                            "expected": action,
                            "correct": correct(index),
                        }
                        for index, identifier in enumerate(
                            config["selection_example_ids"]
                        )
                    ]
                    session_rows = [
                        {
                            "id": f"{identifier}:{turn}",
                            "session_id": identifier,
                            "turn": turn,
                            "command": "Move the missing shape.",
                            "expected": action,
                            "correct": correct(turn),
                            "state_matches": correct(turn),
                            "mutated": False,
                        }
                        for identifier in config["selection_session_ids"]
                        for turn in range(2)
                    ]
                    prefix = "selection-" if selection else ""
                    base = {
                        "model": "same-model",
                        "revision": "same-revision",
                        **metadata,
                        "split": "valid",
                        "execution_guards": True,
                        "adapter_sha256": hashlib.sha256(weight.encode()).hexdigest(),
                    }
                    command = {
                        **base,
                        "examples": command_rows,
                        "per_action": {
                            "no_action": {
                                "total": len(command_rows),
                                "correct": sum(row["correct"] for row in command_rows),
                            }
                        },
                    }
                    session = {**base, "examples": session_rows}
                    (folder / f"{prefix}valid-guarded.json").write_text(
                        json.dumps(command)
                    )
                    (folder / f"{prefix}sessions-valid-guarded.json").write_text(
                        json.dumps(session)
                    )

            with patch("gpu_benchmark.evaluate_pair", side_effect=score) as evaluate:
                result = select_refinement_checkpoint(root, work, config)
            self.assertEqual(evaluate.call_count, 3)
            self.assertEqual(result["chosen_step"], 1500)
            self.assertTrue(result["promotion_passed"])
            self.assertFalse(result["test_set_used"])
            selected = work / "refinement/selected/adapter"
            self.assertEqual((selected / "adapters.safetensors").read_text(), "1500")
            self.assertEqual(
                result["chosen_adapter_sha256"], hashlib.sha256(b"1500").hexdigest()
            )
            self.assertTrue((selected / "checkpoints/0001500/progress.json").exists())
            self.assertEqual(
                {
                    str(path.relative_to(final)): path.read_bytes()
                    for path in final.rglob("*")
                    if path.is_file()
                },
                snapshot,
            )
            self.assertEqual(
                json.loads((work / "checkpoint-selection.json").read_text())[
                    "chosen_step"
                ],
                1500,
            )
            force_regression = True
            rejected = Path(directory) / "rejected"
            for case in ("baseline", "dual-window"):
                shutil.copytree(
                    work / "refinement" / case, rejected / "refinement" / case
                )
            with patch("gpu_benchmark.evaluate_pair", side_effect=score):
                failed = select_refinement_checkpoint(root, rejected, config)
            self.assertEqual(failed["chosen_step"], 1500)
            self.assertFalse(failed["promotion_passed"])
            self.assertIn(
                "state_agreement_rate", failed["full_validation"]["regressions"]
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
