import json
import threading
import unittest
from http.server import HTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from actions import execution_guard
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
