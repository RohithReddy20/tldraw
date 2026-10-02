import copy
from collections import defaultdict

from actions import Canvas, validate_call


class CanvasSession:
    def __init__(self, canvas):
        self.canvas = Canvas.model_validate(canvas).model_dump()
        self.connections = []
        self.history = {"turns": [], "last_created_id": None, "last_edited_id": None}

    def external(self, events):
        for event in events:
            ids = {box["id"] for box in self.canvas["schemas"]}
            if event["kind"] == "select":
                self.canvas["selected_ids"] = [i for i in event["ids"] if i in ids]
            elif event["kind"] == "reorder":
                order = {identifier: i for i, identifier in enumerate(event["ids"])}
                self.canvas["schemas"].sort(key=lambda box: order.get(box["id"], 999))
            elif event["kind"] == "delete":
                target = event["id"]
                self.canvas["schemas"] = [
                    box for box in self.canvas["schemas"] if box["id"] != target
                ]
                self.canvas["selected_ids"] = [
                    i for i in self.canvas["selected_ids"] if i != target
                ]
                self.connections = [
                    link
                    for link in self.connections
                    if target not in (link["source_id"], link["target_id"])
                ]
            else:
                raise ValueError(f"Unknown external event: {event['kind']}")

    def execute(self, command, action, created_id):
        previous = copy.deepcopy((self.canvas, self.connections, self.history))
        try:
            self._execute(command, action, created_id)
        except ValueError:
            self.canvas, self.connections, self.history = previous
            raise

    def _execute(self, command, action, created_id):
        action = validate_call(action, self.canvas) if action else None
        target = None
        if action and action["name"] != "no_action":
            name, args = action["name"], action["arguments"]
            boxes = {box["id"]: box for box in self.canvas["schemas"]}
            if name == "create_schema_box":
                if created_id in boxes or len(boxes) >= 30:
                    raise ValueError("Creation exceeds canvas limits or reuses an ID.")
                self.canvas["schemas"].append(
                    {
                        "id": created_id,
                        "name": args["name"],
                        "properties": args["fields"],
                        "methods": args["methods"],
                    }
                )
                self.history["last_created_id"] = target = created_id
            elif name == "connect_schemas":
                if args not in self.connections:
                    self.connections.append(copy.deepcopy(args))
                target = args["source_id"]
            else:
                target = args["schema_id"]
                box = boxes[target]
                if (
                    name == "add_property"
                    and args["property_name"] not in box["properties"]
                ):
                    box["properties"].append(args["property_name"])
                elif name == "remove_property":
                    box["properties"].remove(args["property_name"])
                elif name == "rename_schema":
                    box["name"] = args["new_name"]
            self.canvas["selected_ids"] = [target]
            self.history["last_edited_id"] = target
        entry = {"command": command, "action": copy.deepcopy(action)}
        if action and action["name"] == "create_schema_box":
            entry["created_id"] = created_id
        self.history["turns"] = [*self.history["turns"], entry][-3:]
        Canvas.model_validate(self.canvas)

    def snapshot(self):
        ids = {box["id"] for box in self.canvas["schemas"]}
        return {
            "schemas": sorted(
                copy.deepcopy(self.canvas["schemas"]), key=lambda b: b["id"]
            ),
            "selected_ids": sorted(self.canvas["selected_ids"]),
            "connections": sorted(
                copy.deepcopy(self.connections), key=lambda a: tuple(a.values())
            ),
            "references": {
                key: value if value in ids else None
                for key, value in self.history.items()
                if key != "turns"
            },
        }


def summarize_sessions(rows):
    groups, positions = defaultdict(list), defaultdict(list)
    for row in rows:
        groups[row["session_id"]].append(row)
        positions[
            f"{(row['turn'] // 10) * 10 + 1}-{(row['turn'] // 10 + 1) * 10}"
        ].append(row)
    negatives = [r for r in rows if r["expected"]["name"] == "no_action"]
    recovered = sum(
        not before["state_matches"] and after["state_matches"]
        for group in groups.values()
        for before, after in zip(group, group[1:], strict=False)
    )
    return {
        "sessions": len(groups),
        "turns": len(rows),
        "exact_action_accuracy": sum(r["correct"] for r in rows) / len(rows),
        "state_agreement_rate": sum(r["state_matches"] for r in rows) / len(rows),
        "perfect_sessions": sum(
            all(r["correct"] for r in group) for group in groups.values()
        ),
        "final_state_matches": sum(
            group[-1]["state_matches"] for group in groups.values()
        ),
        "unsupported_or_missing_requests": len(negatives),
        "unwanted_mutations": sum(r["mutated"] for r in negatives),
        "recoveries": recovered,
        "by_turn_position": {
            key: {
                "total": len(group),
                "correct": sum(r["correct"] for r in group),
                "state_matches": sum(r["state_matches"] for r in group),
            }
            for key, group in sorted(
                positions.items(), key=lambda pair: int(pair[0].split("-")[0])
            )
        },
        "first_error_turns": {
            key: next((r["turn"] + 1 for r in group if not r["correct"]), None)
            for key, group in groups.items()
        },
    }
