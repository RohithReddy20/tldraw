import argparse
import copy
import json
import re
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from actions import (
    ACTION_MODELS,
    TOOLS,
    messages_for,
    model_canvas,
    parse_call,
    validate_call,
)
from build_workflow import build_spoken_refinement, build_workflow, call
from dataset import audit_examples, training_row
from sessions import CanvasSession


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.canvas = {
            "schemas": [{"id": "box", "name": "User", "methods": ["getName"]}],
            "shapes": [
                {"id": "a", "name": "Class", "kind": "rectangle"},
                {"id": "b", "name": "Subject", "kind": "ellipse", "x": 300.0},
            ],
            "selected_ids": ["a", "b"],
        }

    def test_select_all_keeps_large_selections_available_for_canvas_commands(self):
        shapes = [
            {"id": f"shape-{index}", "name": f"Item {index}", "kind": "rectangle"}
            for index in range(40)
        ]
        canvas = {"shapes": shapes, "selected_ids": [shape["id"] for shape in shapes]}
        context, identifiers = model_canvas(canvas)
        self.assertEqual(
            context["selected_ids"], [identifiers[shape["id"]] for shape in shapes]
        )
        action = {
            "name": "canvas_command",
            "arguments": {"operation": "clear_selection"},
        }
        self.assertEqual(validate_call(action, canvas), action)
        self.assertEqual(
            validate_call(
                {"name": "canvas_command", "arguments": {"operation": "zoom_to_fit"}},
                canvas,
            )["name"],
            "canvas_command",
        )

    def test_array_aliases_and_optional_coordinates_round_trip(self):
        text = (
            "<start_function_call>call:move_shapes{shape_ids:[<escape>shape1<escape>,"
            "<escape>shape2<escape>],dx:-100,dy:50}<end_function_call>"
        )
        self.assertEqual(
            parse_call(text, self.canvas),
            call("move_shapes", shape_ids=["a", "b"], dx=-100.0, dy=50.0),
        )
        text = (
            "<start_function_call>call:create_shape{kind:<escape>rectangle<escape>,"
            "text:<escape><escape>,x:None,y:None,width:160,height:100}"
            "<end_function_call>"
        )
        self.assertEqual(parse_call(text, self.canvas)["arguments"]["x"], None)

    def test_delete_undo_redo_and_selection_keep_actual_references(self):
        session = CanvasSession(self.canvas)
        session.execute(
            "Move them.",
            call("move_shapes", shape_ids=["a", "b"], dx=100.0, dy=50.0),
            "unused",
        )
        session.execute(
            "Delete Class.", call("delete_shapes", shape_ids=["a"]), "unused"
        )
        self.assertNotIn("a", session.objects())
        session.execute("Undo.", call("canvas_command", operation="undo"), "unused")
        self.assertEqual(session.objects()["a"]["x"], 100)
        session.execute(
            "Select Subject.", call("select_shapes", shape_ids=["b"]), "unused"
        )
        self.assertTrue(session.canvas["can_redo"])
        session.execute("Redo.", call("canvas_command", operation="redo"), "unused")
        self.assertNotIn("a", session.objects())
        session.execute(
            "Add a method.",
            call("add_method", schema_id="box", method_name="getClass"),
            "unused",
        )
        session.execute("Undo.", call("canvas_command", operation="undo"), "unused")
        self.assertEqual(session.objects()["box"]["methods"], ["getName"])

    def test_manual_edits_after_a_failed_creation_keep_scoring_predicted_state(self):
        actual, oracle = CanvasSession({}), CanvasSession({})
        oracle.execute(
            "Create Class.",
            call("create_shape", kind="rectangle", text="Class", x=100, y=100),
            "created",
        )
        events = [
            {"kind": "move", "id": "created", "dx": 35, "dy": 25},
            {"kind": "text", "id": "created", "text": "Course"},
        ]
        before = actual.snapshot()
        actual.external(events)
        oracle.external(events)
        self.assertEqual(actual.snapshot(), before)
        self.assertFalse(actual.canvas["can_undo"])
        self.assertNotEqual(actual.snapshot(), oracle.snapshot())
        self.assertEqual(oracle.objects()["created"]["x"], 135)
        self.assertEqual(oracle.objects()["created"]["text"], "Course")

    def test_spoken_refinement_keeps_evaluation_frozen_and_teaches_vertical_motion(
        self,
    ):
        canvas = CanvasSession(
            {
                "shapes": [{"id": "group", "name": "Group", "kind": "group"}],
                "selected_ids": ["group"],
                "can_redo": True,
            }
        ).canvas
        actions = [
            call("move_shapes", shape_ids=["group"], dx=100, dy=0),
            call(
                "style_shapes",
                shape_ids=["group"],
                color="blue",
                fill=None,
                opacity=None,
            ),
            call("arrange_shapes", shape_ids=["group"], operation="ungroup"),
            call("canvas_command", operation="redo"),
        ]
        original = [
            {
                "id": f"source-{i}",
                "group": f"source-{i}",
                "split": "train",
                "command": f"Original instruction {i}.",
                "canvas": canvas,
                "expected": action,
            }
            for i, action in enumerate(actions)
        ]
        held_out = {
            "id": "held-out",
            "group": "held-out",
            "split": "test",
            "command": "Explain this untouched drawing.",
            "canvas": {},
            "expected": call("no_action", reason="unsupported_request"),
        }
        validation = {
            **held_out,
            "id": "validation",
            "group": "validation",
            "split": "valid",
            "command": "Explain the validation drawing.",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "examples.jsonl"
            source.write_text(
                "".join(
                    json.dumps(row) + "\n" for row in [*original, validation, held_out]
                )
            )
            sessions = '[{"id":"frozen-session"}]\n'
            (root / "sessions.jsonl").write_text(sessions)
            output = root / "spoken"
            build_spoken_refinement(
                source, output, count=200, replay=4, seed=65, practice_sessions=0
            )
            rows = [
                json.loads(line)
                for line in (output / "examples.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                [row for row in rows if row["split"] == "test"], [held_out]
            )
            self.assertEqual((output / "sessions.jsonl").read_text(), sessions)
        spoken = [row for row in rows if row["id"].startswith("spoken-train-")]
        self.assertEqual(len(spoken), 200)
        self.assertTrue(any(row["command"].startswith("Make ") for row in spoken))
        for direction, sign in (("up", -1), ("down", 1)):
            moves = [
                row
                for row in spoken
                if row["expected"]["name"] == "move_shapes"
                and re.search(rf"\b{direction}\b", row["command"])
            ]
            self.assertTrue(moves)
            for row in moves:
                args = row["expected"]["arguments"]
                self.assertEqual(args["dx"], 0)
                self.assertGreater(args["dy"] * sign, 0)

    def test_group_parent_alias_and_failed_action_are_consistent(self):
        session = CanvasSession(self.canvas)
        session.execute(
            "Group them.",
            call("arrange_shapes", shape_ids=["a", "b"], operation="group"),
            "group",
        )
        self.assertIn(
            '"parent_id": "shape3"',
            messages_for("Move the group.", session.canvas)[1]["content"],
        )
        before = session.snapshot()
        with self.assertRaises(ValueError):
            session.execute(
                "Move a missing shape.",
                call("move_shapes", shape_ids=["a", "missing"], dx=100.0, dy=0.0),
                "unused",
            )
        self.assertEqual(before, session.snapshot())
        session.execute(
            "Move the group.",
            call("move_shapes", shape_ids=["group", "a"], dx=100.0, dy=0.0),
            "unused",
        )
        self.assertEqual(session.objects()["a"]["x"], 100)

    def test_new_data_removes_obsolete_labels_and_keeps_split_groups_separate(self):
        original = [
            {
                "id": "obsolete",
                "group": "obsolete",
                "split": "train",
                "command": "Move User to the left.",
                "canvas": {},
                "expected": call("no_action", reason="unsupported_request"),
            },
            {
                "id": "keep",
                "group": "keep",
                "split": "train",
                "command": "Explain User.",
                "canvas": {},
                "expected": call("no_action", reason="unsupported_request"),
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in original))
            rows, cases = build_workflow(path, independent=1000, sessions=10, replay=10)
        audit_examples(rows)
        self.assertNotIn("obsolete", {row["id"] for row in rows})
        self.assertIn("keep", {row["id"] for row in rows})
        self.assertEqual(set(ACTION_MODELS), {row["expected"]["name"] for row in rows})
        self.assertTrue(all(24 <= len(case["turns"]) <= 80 for case in cases))

    def test_cached_tool_prefix_matches_the_official_template_for_every_tool(self):
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer

        from lab import MODEL, REVISION, ToolPrefixEncoder

        try:
            checkpoint = Path(
                hf_hub_download(
                    MODEL,
                    "tokenizer_config.json",
                    revision=REVISION,
                    local_files_only=True,
                )
            ).parent
        except FileNotFoundError:
            self.skipTest("Pinned tokenizer is not cached.")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        encoder = ToolPrefixEncoder(tokenizer)
        examples = [
            call(
                "create_schema_box", name="User", fields=["name"], methods=["getName"]
            ),
            call("add_property", schema_id="box", property_name="email"),
            call("remove_property", schema_id="box", property_name="name"),
            call("rename_schema", schema_id="box", new_name="Member"),
            call("connect_schemas", source_id="box", target_id="a", label="owns"),
            call("no_action", reason="ambiguous_target"),
            call("create_shape", kind="text", text="Class", x=None, y=None),
            call("select_shapes", shape_ids=["a", "b"]),
            call("move_shapes", shape_ids=["a", "b"], dx=-100.0, dy=50.0),
            call("delete_shapes", shape_ids=["a"]),
            call("resize_shape", shape_id="a", width=400.0, height=300.0),
            call("set_text", shape_id="a", text="New label"),
            call("style_shapes", shape_ids=["a"], color="red", fill=None, opacity=None),
            call("arrange_shapes", shape_ids=["a", "b"], operation="group"),
            call("canvas_command", operation="zoom_in"),
            call("pan_canvas", dx=100.0, dy=-50.0),
            call("add_method", schema_id="box", method_name="getClass"),
            call("remove_method", schema_id="box", method_name="getName"),
        ]
        canvas = copy.deepcopy(self.canvas)
        canvas["schemas"][0]["properties"] = ["name"]
        for action in examples:
            action = validate_call(action, canvas)
            row = training_row(
                {
                    "command": "Quoted label, brackets [v2], apostrophe's.",
                    "canvas": canvas,
                    "expected": action,
                }
            )
            with self.subTest(tool=action["name"]):
                full = encoder.encode(row["messages"])
                prompt = encoder.encode(row["messages"][:-1], generation=True)
                self.assertEqual(
                    full,
                    tokenizer.apply_chat_template(
                        row["messages"], tools=TOOLS, return_dict=False
                    ),
                )
                self.assertEqual(
                    prompt,
                    tokenizer.apply_chat_template(
                        row["messages"][:-1],
                        tools=TOOLS,
                        add_generation_prompt=True,
                        return_dict=False,
                    ),
                )
                self.assertEqual(
                    parse_call(tokenizer.decode(full[len(prompt) :]), canvas), action
                )

    def test_remote_observer_survives_an_outage_and_retries_result_download(self):
        from kaggle_follow import main

        failure = subprocess.CalledProcessError(1, ["kaggle", "kernels", "status"])
        complete = 'owner/kernel has status "KernelWorkerStatus.COMPLETE"\n'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch(
                    "sys.argv",
                    ["follow", "--kernel", "owner/kernel", "--workflow", str(root)],
                ),
                patch("kaggle_follow.time.sleep") as sleep,
                patch(
                    "kaggle_follow.cli",
                    side_effect=[*[failure] * 4, complete, "logs", complete, "logs"],
                ),
                patch(
                    "kaggle_follow.finish_workflow", side_effect=[failure, None]
                ) as finish,
            ):
                main()
            self.assertEqual((root / "kernel.log").read_text(), "logs")
            self.assertEqual(sleep.call_count, 5)
            self.assertEqual(finish.call_count, 2)

    def test_native_failure_preserves_the_verified_adapter_and_releases_its_service(
        self,
    ):
        from kaggle_follow import verify_native

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            launch = {"status": "training"}
            (root / "run.json").write_text(json.dumps(launch))
            with patch("kaggle_follow.subprocess.Popen") as start:
                with self.assertRaisesRegex(ValueError, "completed workflow"):
                    verify_native(root)
                start.assert_not_called()
            launch.update(
                status="complete",
                adapter_sha256="verified",
                downloaded_adapter="candidate",
                test_set_used=False,
            )
            (root / "run.json").write_text(json.dumps(launch))
            with (
                patch("kaggle_follow.subprocess.Popen") as start,
                patch("kaggle_follow.urlopen") as health,
                patch(
                    "kaggle_follow.subprocess.run",
                    return_value=subprocess.CompletedProcess([], 1),
                ),
            ):
                start.return_value.poll.return_value = None
                health.return_value.__enter__.return_value.status = 200
                verify_native(root)
                start.return_value.terminate.assert_called_once()
                start.return_value.wait.assert_called_once()
            completed = json.loads((root / "run.json").read_text())
            self.assertEqual(completed["status"], "complete")
            self.assertEqual(completed["downloaded_adapter"], "candidate")
            self.assertFalse(completed["test_set_used"])
            self.assertEqual(completed["native_verification"]["status"], "failed")

    def test_remote_results_reject_a_different_final_adapter(self):
        import numpy as np
        from safetensors.numpy import save_file

        from kaggle_follow import finish_workflow

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            launch = {
                "updates": 6000,
                "dataset_sha256": "data",
                "task_sha256": "task",
                "warm_start_sha256": "weights",
            }
            (root / "run.json").write_text(json.dumps(launch))
            download = root / "download"
            checkpoint = download / "refinement/dual-window/adapter/checkpoints/0006000"
            checkpoint.mkdir(parents=True)
            (checkpoint / "progress.json").write_text(
                json.dumps(
                    {"step": 6000, "dataset_sha256": "data", "task_sha256": "task"}
                )
            )
            (download / "refinement-training.json").write_text(
                json.dumps(
                    {
                        **launch,
                        "end_step": 6000,
                        "initialization": "weights_only",
                        "training_finite": True,
                    }
                )
            )
            saved = {"weight": np.asarray([1.0], dtype=np.float32)}
            final = checkpoint.parent.parent / "adapters.safetensors"
            save_file(saved, str(checkpoint / "adapters.safetensors"))
            save_file({"weight": np.asarray([2.0], dtype=np.float32)}, str(final))
            with (
                patch("kaggle_follow.cli"),
                patch("gpu_benchmark.quality_comparison", return_value={}) as compare,
            ):
                with self.assertRaisesRegex(ValueError, "weights differ"):
                    finish_workflow("owner/kernel", root)
                compare.assert_not_called()
                save_file(saved, str(final))
                finish_workflow("owner/kernel", root, training_only=True)
                compare.assert_not_called()
                recovered = json.loads((root / "run.json").read_text())
                self.assertEqual(recovered["status"], "trained")
                finish_workflow("owner/kernel", root)
            completed = json.loads((root / "run.json").read_text())
            self.assertEqual(completed["status"], "complete")
            self.assertFalse(completed["test_set_used"])

    def test_validation_recovery_reuses_finished_weights_without_training(self):
        import hashlib

        import numpy as np
        from safetensors.numpy import save_file

        from gpu_benchmark import resume_refinement_evaluation

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source", root / "output"
            output.mkdir()
            expected = {
                "updates": 6000,
                "dataset_sha256": "data",
                "task_sha256": "task",
            }
            checkpoint = source / "refinement/dual-window/adapter/checkpoints/0006000"
            checkpoint.mkdir(parents=True)
            saved = {"weight": np.asarray([1.0], dtype=np.float32)}
            for case in ("baseline", "dual-window"):
                run = source / "refinement" / case
                (run / "adapter").mkdir(parents=True, exist_ok=True)
                save_file(saved, str(run / "adapter/adapters.safetensors"))
                (run / "adapter/adapter_config.json").write_text("{}")
                (run / "config.yaml").write_text("evaluate_guards: true\n")
            weights = source / "refinement/dual-window/adapter/adapters.safetensors"
            digest = hashlib.sha256(weights.read_bytes()).hexdigest()
            expected.update(warm_start_sha256=digest, adapter_sha256=digest)
            save_file(saved, str(checkpoint / "adapters.safetensors"))
            (checkpoint / "progress.json").write_text(
                json.dumps(
                    {"step": 6000, "dataset_sha256": "data", "task_sha256": "task"}
                )
            )
            (source / "refinement-training.json").write_text(
                json.dumps(
                    {
                        **expected,
                        "end_step": 6000,
                        "initialization": "weights_only",
                        "training_finite": True,
                    }
                )
            )
            metadata = {"dataset_sha256": "data", "task_sha256": "task"}
            (source / "refinement/baseline/valid.json").write_text(
                json.dumps({**metadata, "split": "valid", "execution_guards": False})
            )
            with (
                patch(
                    "gpu_benchmark.Path.rglob",
                    return_value=[source / "refinement-training.json"],
                ),
                patch("lab._metadata", return_value=metadata),
                patch("gpu_benchmark.evaluate_pair") as evaluate,
                patch("gpu_benchmark.run_case") as train,
                patch("gpu_benchmark.quality_comparison", return_value={}),
            ):
                resume_refinement_evaluation(root, output, expected)
                train.assert_not_called()
                evaluate.assert_called_once_with(
                    root, output / "refinement", selected=False
                )
            report = json.loads((output / "refinement/baseline/valid.json").read_text())
            self.assertEqual(report["adapter_sha256"], digest)
            self.assertEqual(
                (
                    output / "refinement/dual-window/adapter/adapters.safetensors"
                ).read_bytes(),
                weights.read_bytes(),
            )

    def test_final_scoring_preserves_the_holdout_after_failed_native_checks(self):
        from gpu_benchmark import resume_refinement_evaluation
        from kaggle_follow import await_final_scoring, launch_final_scoring

        with patch("kaggle_follow.verify_workflow_checkpoint") as verify:
            with self.assertRaisesRegex(ValueError, "passed native checks"):
                resume_refinement_evaluation(
                    Path("unused"),
                    Path("unused"),
                    {"native_verification": "failed"},
                    selected=True,
                )
            verify.assert_not_called()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run.json").write_text(
                json.dumps(
                    {
                        "adapter_sha256": "model",
                        "test_set_used": False,
                        "native_verification": {
                            "status": "failed",
                            "adapter_sha256": "model",
                        },
                    }
                )
            )
            with patch("kaggle_follow.cli") as cli:
                with self.assertRaisesRegex(ValueError, "passed native checks"):
                    launch_final_scoring(root)
                with patch("kaggle_follow.launch_final_scoring") as score:
                    await_final_scoring(root)
                    score.assert_not_called()
                cli.assert_not_called()
            self.assertFalse((root / "final-score").exists())
            launch = json.loads((root / "run.json").read_text())
            self.assertEqual(launch["final_scoring"]["status"], "needs_native_fix")
            launch["native_verification"].update(
                status="passed", adapter_sha256="other"
            )
            (root / "run.json").write_text(json.dumps(launch))
            with self.assertRaisesRegex(ValueError, "passed native checks"):
                launch_final_scoring(root)
            launch["native_verification"]["adapter_sha256"] = "model"
            launch["test_set_started"] = True
            (root / "run.json").write_text(json.dumps(launch))
            with self.assertRaisesRegex(ValueError, "already been used"):
                launch_final_scoring(root)

    def test_final_reports_require_the_full_holdout_and_verified_adapter(self):
        from kaggle_follow import finish_selected_workflow

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            launch = {
                "adapter_sha256": "model",
                "dataset_sha256": "data",
                "task_sha256": "task",
                "test_set_started": True,
            }
            result = {
                "adapter_sha256": "model",
                "test_set_used": True,
                "reports": {
                    name: {**launch, count: total}
                    for name, count, total in (
                        ("valid", "total", 1200),
                        ("valid-guarded", "total", 1200),
                        ("sessions-valid", "turns", 480),
                        ("sessions-valid-guarded", "turns", 480),
                        ("test-guarded", "total", 2880),
                        ("sessions-test-guarded", "turns", 1440),
                    )
                },
            }
            (root / "run.json").write_text(json.dumps(launch))
            (root / "download").mkdir()
            report = root / "download/refinement-selected-evaluation.json"
            with patch("kaggle_follow.finish_workflow"):
                for changes in ({"turns": 100}, {"adapter_sha256": "other"}):
                    invalid = copy.deepcopy(result)
                    invalid["reports"]["sessions-test-guarded"].update(changes)
                    report.write_text(json.dumps(invalid))
                    with self.assertRaisesRegex(ValueError, "sessions-test-guarded"):
                        finish_selected_workflow("owner/kernel", root)
                report.write_text(json.dumps(result))
                finish_selected_workflow("owner/kernel", root)
            saved = json.loads((root / "run.json").read_text())
            self.assertEqual(saved["status"], "evaluated")
            self.assertTrue(saved["test_set_used"])
            self.assertTrue(saved["test_set_started"])

    def test_checkpoint_readiness_retries_transient_errors_and_requires_ready(self):
        from kaggle_follow import wait_for_dataset

        failure = subprocess.CalledProcessError(1, ["kaggle", "datasets", "status"])
        with (
            patch(
                "kaggle_follow.cli", side_effect=[failure, "not ready", "ready\n"]
            ) as cli,
            patch("kaggle_follow.time.sleep") as sleep,
        ):
            wait_for_dataset("owner/checkpoint")
            self.assertEqual(cli.call_count, 3)
            self.assertEqual(sleep.call_count, 2)

    def test_prompt_cache_reuses_prefill_without_reusing_changed_canvas_state(self):
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache

        from lab import PromptPrefixCache

        tokenizer = Mock(bos_token="<bos>")
        prefix = list(range(2132))
        tokenizer.encode.side_effect = [prefix + [5], prefix + [7]]
        encoder = Mock(prefix=prefix)

        def prefill(tokens, cache):
            values = tokens[:, None, :, None].astype(mx.float32)
            cache[0].update_and_fetch(values, values)

        model = Mock(side_effect=prefill)
        with (
            patch("lab.ToolPrefixEncoder", return_value=encoder),
            patch("mlx_lm.models.cache.make_prompt_cache", return_value=[KVCache()]),
        ):
            retained = PromptPrefixCache(model, tokenizer)
            first, first_cache = retained.prepare(model, tokenizer, "<bos>first")
            first_cache[0].update_and_fetch(
                mx.full((1, 1, 1, 1), 99), mx.full((1, 1, 1, 1), 99)
            )
            second, second_cache = retained.prepare(model, tokenizer, "<bos>second")
        model.assert_called_once()
        self.assertEqual(first, prefix[2048:] + [5])
        self.assertEqual(second, prefix[2048:] + [7])
        self.assertEqual(
            (retained.cache[0].offset, first_cache[0].offset, second_cache[0].offset),
            (2048, 2049, 2048),
        )
        self.assertEqual(second_cache[0].keys[0, 0, -1, 0].item(), 2047)

    def test_prompt_cache_rejects_another_model_or_changed_tool_prefix(self):
        from lab import PromptPrefixCache

        model, tokenizer = Mock(), Mock(bos_token="<bos>")
        with patch(
            "lab.ToolPrefixEncoder", return_value=Mock(prefix=list(range(2132)))
        ):
            retained = PromptPrefixCache(model, tokenizer)
        with self.assertRaisesRegex(ValueError, "different model"):
            retained.prepare(Mock(), tokenizer, "<bos>other model")
        tokenizer.encode.return_value = [-1] + list(range(1, 2228))
        with self.assertRaisesRegex(ValueError, "retained tool prefix"):
            retained.prepare(model, tokenizer, "<bos>changed tools")
        model.assert_not_called()

    def test_batch_predictions_keep_aliases_separate_with_out_of_order_tokens(self):
        from lab import predict_batch

        canvases = [
            {
                "schemas": [],
                "shapes": [{"id": identifier, "kind": "rectangle", "name": "Box"}],
            }
            for identifier in ("first-canvas-shape", "second-canvas-shape")
        ]
        examples = [
            {"command": "Move Box right by 100.", "canvas": canvas}
            for canvas in canvases
        ]
        text = (
            "<start_function_call>call:move_shapes{shape_ids:[<escape>shape1<escape>],"
            "dx:100,dy:0}<end_function_call>"
        )
        tokenizer = Mock(bos_token="<bos>", eos_token_ids=[0])
        tokenizer.apply_chat_template.return_value = "<bos>prompt"
        tokenizer.encode.return_value = [98]
        tokenizer.decode.return_value = text
        generator = Mock()
        generator.insert.return_value = [7, 3]
        generator.next_generated.side_effect = [
            [
                SimpleNamespace(uid=3, token=12, finish_reason=None),
                SimpleNamespace(uid=7, token=11, finish_reason=None),
            ],
            [
                SimpleNamespace(uid=3, token=0, finish_reason="stop"),
                SimpleNamespace(uid=7, token=0, finish_reason="stop"),
            ],
            [],
        ]
        with patch("mlx_lm.generate.BatchGenerator", return_value=generator):
            results = predict_batch(Mock(), tokenizer, examples, guarded=True)
        self.assertEqual(
            [row["prediction"] for row in results],
            [
                {
                    "name": "move_shapes",
                    "arguments": {"shape_ids": [identifier], "dx": 100.0, "dy": 0.0},
                }
                for identifier in ("first-canvas-shape", "second-canvas-shape")
            ],
        )
        self.assertEqual(
            [call.args[0] for call in tokenizer.decode.call_args_list], [[11], [12]]
        )
        self.assertEqual([row["error"] for row in results], [None, None])
        generator.close.assert_called_once()

    def test_batch_prediction_closes_generator_if_a_command_never_finishes(self):
        from lab import predict_batch

        tokenizer = Mock(bos_token="<bos>", eos_token_ids=[0])
        tokenizer.apply_chat_template.return_value = "<bos>prompt"
        tokenizer.encode.return_value = [98]
        generator = Mock()
        generator.insert.return_value = [7]
        generator.next_generated.return_value = []
        with patch("mlx_lm.generate.BatchGenerator", return_value=generator):
            with self.assertRaisesRegex(RuntimeError, "did not finish"):
                predict_batch(Mock(), tokenizer, [{"command": "Undo", "canvas": {}}])
        generator.close.assert_called_once()

    def test_batched_evaluation_keeps_the_last_partial_batch_and_report_order(self):
        from lab import evaluate

        expected = {"name": "no_action", "arguments": {"reason": "unsupported"}}
        examples = [
            {
                "id": f"case-{index}",
                "split": "valid",
                "command": "Explain this diagram.",
                "canvas": {},
                "expected": expected,
            }
            for index in range(5)
        ]
        results = [
            {
                "raw_output": "<start_function_call>call:no_action",
                "prediction": copy.deepcopy(expected) if index != 3 else None,
                "error": None if index != 3 else "Incomplete function call.",
                "seconds": 0.1,
                "peak_model_memory_gb": 1.0,
            }
            for index in range(5)
        ]
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("lab.prepare_data"),
            patch("lab.load_model", return_value=(Mock(), Mock())),
            patch("lab.read_examples", return_value=examples),
            patch("lab._metadata", return_value={}),
            patch("lab.predict_batch", side_effect=[results[:4], results[4:]]),
        ):
            output = Path(directory) / "valid.json"
            evaluate(
                argparse.Namespace(
                    adapter=None,
                    split="valid",
                    limit=None,
                    output=output,
                    batch_size=4,
                )
            )
            report = json.loads(output.read_text())
        self.assertEqual(
            [row["id"] for row in report["examples"]],
            [f"case-{index}" for index in range(5)],
        )
        self.assertEqual((report["total"], report["correct"]), (5, 4))
        self.assertEqual(report["exact_action_accuracy"], 0.8)


if __name__ == "__main__":
    unittest.main()
