import json
import tempfile
import threading
import unittest
from http.server import HTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from actions import execution_guard
from sessions import CanvasSession, compare_session_state, summarize_sessions
from voice_server import CommandRequest, handler_for


class VoiceGuardTests(unittest.TestCase):
    def setUp(self):
        self.canvas = {
            "schemas": [
                {"id": "first", "name": "User", "properties": ["name"]},
                {"id": "second", "name": "User", "properties": []},
            ],
            "selected_ids": ["first"],
        }
        self.action = {
            "name": "add_property",
            "arguments": {"schema_id": "first", "property_name": "email"},
        }

    def test_duplicate_named_target_is_rejected_but_single_selection_works(self):
        self.assertEqual(
            execution_guard("Add email to User.", self.action, self.canvas, {}),
            {"name": "no_action", "arguments": {"reason": "ambiguous_target"}},
        )
        self.assertEqual(
            execution_guard("Add email to it.", self.action, self.canvas, {}),
            self.action,
        )

    def test_partial_execution_of_compound_edit_is_rejected(self):
        self.assertEqual(
            execution_guard(
                "Add email to it and move the box.", self.action, self.canvas, {}
            ),
            {"name": "no_action", "arguments": {"reason": "unsupported_request"}},
        )

    def test_compound_request_has_priority_over_a_wrong_no_action_reason(self):
        self.assertEqual(
            execution_guard(
                "Add email to User and move the box.",
                {"name": "no_action", "arguments": {"reason": "ambiguous_target"}},
                self.canvas,
                {},
            ),
            {"name": "no_action", "arguments": {"reason": "unsupported_request"}},
        )

    def test_rejected_request_keeps_its_reason_when_history_target_is_absent(self):
        action = {"name": "no_action", "arguments": {"reason": "unsupported_request"}}
        for kind in ("created", "edited"):
            with self.subTest(kind=kind):
                self.assertEqual(
                    execution_guard(
                        f"Change the last {kind} shape. Then explain why.",
                        action,
                        self.canvas,
                        {f"last_{kind}_id": "removed"},
                    ),
                    action,
                )

    def test_selected_target_requires_one_selection_and_named_targets_override_it(self):
        self.canvas["schemas"][1]["name"] = "Account"
        for selection in ([], ["first", "second"]):
            self.canvas["selected_ids"] = selection
            with self.subTest(selection=selection):
                self.assertEqual(
                    execution_guard(
                        "Append email to the selected class.",
                        self.action,
                        self.canvas,
                        {},
                    ),
                    {"name": "no_action", "arguments": {"reason": "ambiguous_target"}},
                )
                self.assertEqual(
                    execution_guard("Add email to User.", self.action, self.canvas, {}),
                    self.action,
                )
        self.assertEqual(
            execution_guard(
                "Add email to the selected box.", self.action, {"schemas": []}, {}
            ),
            {"name": "no_action", "arguments": {"reason": "missing_target"}},
        )

    def test_method_lists_are_not_mistaken_for_a_second_edit(self):
        create = {
            "name": "create_schema_box",
            "arguments": {
                "name": "User",
                "fields": ["name"],
                "methods": ["addSubject", "removeSubject"],
            },
        }
        self.assertEqual(
            execution_guard(
                "Create User with fields name and methods "
                "addSubject and removeSubject.",
                create,
                self.canvas,
                {},
            ),
            create,
        )
        self.assertEqual(
            execution_guard(
                "Create User with fields name and move it right.",
                create,
                self.canvas,
                {},
            ),
            {"name": "no_action", "arguments": {"reason": "unsupported_request"}},
        )

    def test_deleted_last_created_box_is_not_replaced_with_a_current_box(self):
        for last_id in (None, "deleted"):
            with self.subTest(last_id=last_id):
                self.assertEqual(
                    execution_guard(
                        "Add email to the last created box.",
                        self.action,
                        self.canvas,
                        {"last_created_id": last_id},
                    ),
                    {"name": "no_action", "arguments": {"reason": "missing_target"}},
                )

    def test_history_retains_deleted_ids_and_rejects_unknown_actions(self):
        request = CommandRequest.model_validate(
            {
                "command": "Create User.",
                "canvas": {"schemas": []},
                "history": {
                    "turns": [{"command": "Add email.", "action": self.action}],
                    "last_edited_id": "first",
                },
            }
        )
        self.assertEqual(request.history.turns[0].action, self.action)
        for action_name in ("delete_everything", ["add_property"]):
            with self.subTest(action_name=action_name), self.assertRaises(ValueError):
                CommandRequest.model_validate(
                    {
                        "command": "Create User.",
                        "canvas": {},
                        "history": {
                            "turns": [
                                {
                                    "command": "Bad action.",
                                    "action": {"name": action_name, "arguments": {}},
                                }
                            ]
                        },
                    }
                )


class SessionContractTests(unittest.TestCase):
    def test_group_styles_apply_only_supported_leaf_properties(self):
        session = CanvasSession(
            {
                "schemas": [{"id": "schema", "name": "Schema", "parent_id": "group"}],
                "shapes": [
                    {"id": "group", "name": "Group", "kind": "group"},
                    *[
                        {"id": kind, "name": kind, "kind": kind, "parent_id": "group"}
                        for kind in ("text", "note", "frame", "rectangle")
                    ],
                ],
            }
        )
        session.execute(
            "Style the group.",
            {
                "name": "style_shapes",
                "arguments": {
                    "shape_ids": ["group"],
                    "color": "blue",
                    "fill": "pattern",
                    "opacity": 0.5,
                },
            },
            "unused",
        )
        objects = session.objects()
        self.assertEqual(
            {key: objects["group"][key] for key in ("color", "fill", "opacity")},
            {"color": "black", "fill": "none", "opacity": 1.0},
        )
        for kind in ("text", "note", "frame", "rectangle", "schema"):
            with self.subTest(kind=kind):
                self.assertEqual(objects[kind]["opacity"], 0.5)
                self.assertEqual(
                    objects[kind]["color"], "black" if kind == "frame" else "blue"
                )
                self.assertEqual(
                    objects[kind]["fill"],
                    "pattern" if kind in ("rectangle", "schema") else "none",
                )

    def test_serialized_context_matches_native_order_and_unnamed_geo_names(self):
        session = CanvasSession(
            {
                "shapes": [
                    {"id": "z", "name": "Last", "kind": "rectangle", "order": 0},
                    {"id": "b", "name": "First", "kind": "rectangle", "order": 1},
                ],
                "selected_ids": ["z", "b"],
            }
        )
        self.assertEqual(
            [shape["id"] for shape in session.canvas["shapes"]], ["b", "z"]
        )
        self.assertEqual(session.canvas["selected_ids"], ["b", "z"])
        self.assertEqual([shape["order"] for shape in session.canvas["shapes"]], [1, 0])
        session.execute(
            "Create a rectangle.",
            {"name": "create_shape", "arguments": {"kind": "rectangle"}},
            "a",
        )
        self.assertEqual(session.canvas["shapes"][0]["id"], "a")
        self.assertEqual(session.objects()["a"]["name"], "geo")
        session.external([{"kind": "text", "id": "a", "text": "Label"}])
        session.execute(
            "Clear its label.",
            {"name": "set_text", "arguments": {"shape_id": "a", "text": ""}},
            "unused",
        )
        self.assertEqual(session.objects()["a"]["name"], "geo")

    def test_connection_is_visible_deduplicated_and_restored_by_undo(self):
        session = CanvasSession(
            {
                "schemas": [
                    {"id": "a", "name": "Source"},
                    {"id": "b", "name": "Destination", "x": 400},
                ]
            }
        )
        action = {
            "name": "connect_schemas",
            "arguments": {"source_id": "a", "target_id": "b", "label": "owns"},
        }
        session.execute("Connect Source to Destination.", action, "link")
        self.assertEqual(session.objects()["link"]["kind"], "arrow")
        self.assertEqual(session.objects()["link"]["text"], "owns")
        self.assertEqual(session.history["last_created_id"], None)
        self.assertEqual(session.history["last_edited_id"], "a")
        connected = session.snapshot()
        session.execute("Connect Source to Destination.", action, "unused")
        self.assertEqual(session.snapshot(), connected)
        session.execute(
            "Delete the arrow.",
            {"name": "delete_shapes", "arguments": {"shape_ids": ["link"]}},
            "unused",
        )
        self.assertNotIn("link", session.objects())
        self.assertEqual(session.connections, [])
        session.execute(
            "Undo.",
            {"name": "canvas_command", "arguments": {"operation": "undo"}},
            "unused",
        )
        self.assertIn("link", session.objects())
        self.assertEqual(session.connections, connected["connections"])
        session.external([{"kind": "delete", "id": "a"}])
        self.assertIn("link", session.objects())
        self.assertEqual(session.connections, [])

    def test_component_metrics_distinguish_viewport_from_document_errors(self):
        session = CanvasSession(
            {"shapes": [{"id": "a", "name": "A", "kind": "rectangle"}]}
        )
        expected = session.snapshot()
        session.execute(
            "Pan right.",
            {"name": "pan_canvas", "arguments": {"dx": 100, "dy": 0}},
            "unused",
        )
        components = compare_session_state(session.snapshot(), expected)
        self.assertEqual(
            components,
            {
                "state_matches": False,
                "document_matches": True,
                "selection_matches": True,
                "camera_matches": False,
            },
        )
        row = {
            "session_id": "session",
            "turn": 0,
            "correct": False,
            "mutated": True,
            "expected": {"name": "pan_canvas"},
            **components,
        }
        report = summarize_sessions([row])
        self.assertEqual(report["state_agreement_rate"], 0)
        self.assertEqual(report["document_agreement_rate"], 1)
        self.assertEqual(report["selection_agreement_rate"], 1)
        self.assertEqual(report["camera_agreement_rate"], 0)
        legacy = {
            key: value
            for key, value in row.items()
            if key not in components or key == "state_matches"
        }
        self.assertEqual(summarize_sessions([legacy])["document_agreement_rate"], None)


class EvaluationSelectionTests(unittest.TestCase):
    def test_id_selection_keeps_dataset_order_and_rejects_invalid_ids(self):
        from lab import filter_examples

        examples = [{"id": identifier} for identifier in ("third", "first", "second")]
        self.assertIs(filter_examples(examples, None), examples)
        self.assertEqual(
            filter_examples(examples, ["second", "third"]),
            [{"id": "third"}, {"id": "second"}],
        )
        for requested in ([], "third", [None], ["third", "third"], ["missing"]):
            with self.subTest(requested=requested), self.assertRaises(ValueError):
                filter_examples(examples, requested)

    def test_filtered_evaluation_rejects_wrong_split_before_loading_model(self):
        from lab import evaluate, evaluate_sessions

        args = SimpleNamespace(
            adapter=None,
            split="valid",
            limit=None,
            example_ids=["other-split"],
            session_ids=["other-split"],
        )
        with (
            patch("lab.prepare_data"),
            patch(
                "lab.read_examples",
                return_value=[{"id": "other-split", "split": "train"}],
            ),
            patch("lab.load_model") as load,
        ):
            for evaluate_fn in (evaluate, evaluate_sessions):
                with (
                    self.subTest(evaluate_fn=evaluate_fn),
                    self.assertRaises(ValueError),
                ):
                    evaluate_fn(args)
            load.assert_not_called()

    def test_session_subset_reports_camera_divergence(self):
        from lab import evaluate_sessions

        case = {
            "id": "selected",
            "split": "valid",
            "initial_canvas": {},
            "turns": [
                {
                    "id": "selected:0",
                    "before": [],
                    "command": "Pan right.",
                    "expected": {
                        "name": "pan_canvas",
                        "arguments": {"dx": 100, "dy": 0},
                    },
                }
            ],
        }
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("lab.read_examples", return_value=[{**case, "id": "excluded"}, case]),
            patch("lab.load_model", return_value=(object(), object())),
            patch("lab._metadata", return_value={}),
            patch(
                "lab.predict",
                return_value={
                    "prediction": {
                        "name": "canvas_command",
                        "arguments": {"operation": "zoom_to_fit"},
                    }
                },
            ) as predict,
        ):
            args = SimpleNamespace(
                adapter=None,
                split="valid",
                limit=None,
                session_ids=["selected"],
                output=Path(directory) / "report.json",
            )
            evaluate_sessions(args)
            report = json.loads(args.output.read_text())
            self.assertEqual(predict.call_count, 1)
            self.assertEqual([row["id"] for row in report["examples"]], ["selected:0"])
            self.assertEqual(report["turns"], 1)
            self.assertEqual(report["state_agreement_rate"], 0)
            self.assertEqual(report["document_agreement_rate"], 1)
            self.assertEqual(report["camera_agreement_rate"], 0)
            self.assertEqual(report["simulator_version"], "native-visible-shapes-v2")


class VoiceHttpTests(unittest.TestCase):
    def test_only_local_origins_and_valid_requests_reach_inference(self):
        calls = []

        class Engine:
            def command(self, request):
                calls.append(request.command)
                return {
                    "command": request.command,
                    "action": {
                        "name": "no_action",
                        "arguments": {"reason": "unsupported_request"},
                    },
                }

        server = HTTPServer(("127.0.0.1", 0), handler_for(Engine()))
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        url = f"http://127.0.0.1:{server.server_port}/command"

        def send(body, origin):
            request = Request(
                url,
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json", "Origin": origin},
            )
            try:
                return urlopen(request, timeout=5)
            except HTTPError as error:
                return error

        try:
            body = {"command": "Explain User.", "canvas": {}}
            with send(body, "https://untrusted.example") as response:
                self.assertEqual(response.code, 403)
            with send(
                {**body, "audio": "anything"}, "http://localhost:5173"
            ) as response:
                self.assertEqual(response.code, 422)
            with send(body, "http://localhost:5173") as response:
                self.assertEqual(response.code, 200)
                self.assertEqual(
                    response.headers["Access-Control-Allow-Origin"],
                    "http://localhost:5173",
                )
            self.assertEqual(calls, ["Explain User."])
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
