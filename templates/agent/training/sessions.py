import copy
from collections import defaultdict

from actions import Canvas, CanvasShape, SchemaBox, validate_call


class CanvasSession:
    def __init__(self, canvas):
        self.canvas = Canvas.model_validate(canvas).model_dump()
        self.connections = []
        self.history = {"turns": [], "last_created_id": None, "last_edited_id": None}
        self.undo_stack, self.redo_stack = [], []

    def objects(self):
        return {
            shape["id"]: shape
            for shape in [*self.canvas["schemas"], *self.canvas["shapes"]]
        }

    def document(self):
        return copy.deepcopy(
            (
                self.canvas["schemas"],
                self.canvas["shapes"],
                self.canvas["selected_ids"],
                self.connections,
            )
        )

    def restore(self, snapshot):
        (
            self.canvas["schemas"],
            self.canvas["shapes"],
            self.canvas["selected_ids"],
            self.connections,
        ) = copy.deepcopy(snapshot)

    def sync_history(self):
        self.canvas["can_undo"] = bool(self.undo_stack)
        self.canvas["can_redo"] = bool(self.redo_stack)

    def descendants(self, targets):
        ids = set(targets)
        while True:
            expanded = ids | {
                s["id"] for s in self.objects().values() if s["parent_id"] in ids
            }
            if expanded == ids:
                return ids
            ids = expanded

    def delete(self, ids):
        ids = self.descendants(ids)
        for kind in ("schemas", "shapes"):
            self.canvas[kind] = [s for s in self.canvas[kind] if s["id"] not in ids]
        self.canvas["selected_ids"] = [
            i for i in self.canvas["selected_ids"] if i not in ids
        ]
        self.connections = [
            link
            for link in self.connections
            if not ids.intersection((link["source_id"], link["target_id"]))
        ]

    def external(self, events):
        for event in events:
            before = self.document()
            ids = self.objects()
            kind = event["kind"]
            if kind == "select":
                self.canvas["selected_ids"] = [i for i in event["ids"] if i in ids]
            elif kind == "reorder":
                order = {identifier: i for i, identifier in enumerate(event["ids"])}
                self.canvas["schemas"].sort(key=lambda box: order.get(box["id"], 999))
                self.canvas["shapes"].sort(key=lambda box: order.get(box["id"], 999))
            elif kind == "delete":
                self.delete([event["id"]])
            elif kind == "move":
                for target in self.descendants([event["id"]]):
                    ids[target]["x"] += event["dx"]
                    ids[target]["y"] += event["dy"]
            elif kind == "text":
                shape = ids[event["id"]]
                shape["name"] = event["text"][:100]
                if "text" in shape:
                    shape["text"] = event["text"]
            else:
                raise ValueError(f"Unknown external event: {kind}")
            if kind in ("delete", "move", "text") and self.document() != before:
                self.undo_stack.append(before)
                self.redo_stack.clear()
            self.sync_history()

    def execute(self, command, action, created_id):
        previous = copy.deepcopy(
            (
                self.canvas,
                self.connections,
                self.history,
                self.undo_stack,
                self.redo_stack,
            )
        )
        try:
            self._execute(command, action, created_id)
        except ValueError:
            (
                self.canvas,
                self.connections,
                self.history,
                self.undo_stack,
                self.redo_stack,
            ) = previous
            raise

    def _execute(self, command, action, created_id):
        action = validate_call(action, self.canvas) if action else None
        before = self.document()
        target, made, document_edit = None, False, False
        if action and action["name"] != "no_action":
            name, args = action["name"], action["arguments"]
            objects = self.objects()
            boxes = {box["id"]: box for box in self.canvas["schemas"]}
            if name in ("create_schema_box", "create_shape"):
                if created_id in objects:
                    raise ValueError("Creation reuses an ID.")
                if name == "create_schema_box":
                    self.canvas["schemas"].append(
                        SchemaBox(
                            id=created_id,
                            name=args["name"],
                            properties=args["fields"],
                            methods=args["methods"],
                            h=schema_height(args["fields"], args["methods"]),
                            order=len(objects),
                        ).model_dump()
                    )
                else:
                    self.canvas["shapes"].append(
                        CanvasShape(
                            id=created_id,
                            name=args["text"][:100] or args["kind"],
                            kind=args["kind"],
                            text=args["text"],
                            x=args["x"] or 0,
                            y=args["y"] or 0,
                            w=args["width"],
                            h=args["height"],
                            order=len(objects),
                        ).model_dump()
                    )
                target, made, document_edit = created_id, True, True
                self.canvas["selected_ids"] = [created_id]
            elif name == "connect_schemas":
                if args not in self.connections:
                    self.connections.append(copy.deepcopy(args))
                target, document_edit = args["source_id"], True
                self.canvas["selected_ids"] = [target]
            elif name == "select_shapes":
                self.canvas["selected_ids"] = args["shape_ids"]
            elif name == "pan_canvas":
                self.canvas["camera"]["x"] -= args["dx"]
                self.canvas["camera"]["y"] -= args["dy"]
            elif name == "canvas_command":
                operation = args["operation"]
                if operation in ("undo", "redo"):
                    take, give = (
                        (self.undo_stack, self.redo_stack)
                        if operation == "undo"
                        else (self.redo_stack, self.undo_stack)
                    )
                    give.append(self.document())
                    self.restore(take.pop())
                elif operation == "select_all":
                    self.canvas["selected_ids"] = [
                        s["id"] for s in objects.values() if s["parent_id"] is None
                    ]
                elif operation == "clear_selection":
                    self.canvas["selected_ids"] = []
                elif operation == "zoom_in":
                    self.canvas["camera"]["z"] = min(
                        8.0, self.canvas["camera"]["z"] * 2
                    )
                elif operation == "zoom_out":
                    self.canvas["camera"]["z"] = max(
                        0.125, self.canvas["camera"]["z"] / 2
                    )
                elif operation == "reset_zoom":
                    self.canvas["camera"]["z"] = 1
                elif operation == "zoom_to_fit":
                    self.canvas["camera"] = {"x": 0, "y": 0, "z": 1.0}
            elif name in ("move_shapes", "delete_shapes", "style_shapes"):
                ids = args["shape_ids"]
                if name == "delete_shapes":
                    self.delete(ids)
                elif name == "move_shapes":
                    for identifier in self.descendants(ids):
                        objects[identifier]["x"] += args["dx"]
                        objects[identifier]["y"] += args["dy"]
                    target = ids[0]
                else:
                    for identifier in self.descendants(ids):
                        for style in ("color", "fill", "opacity"):
                            if args[style] is not None:
                                objects[identifier][style] = args[style]
                    target = ids[0]
                document_edit = True
                if name != "delete_shapes":
                    self.canvas["selected_ids"] = ids
            elif name in ("resize_shape", "set_text"):
                target = args["shape_id"]
                shape = objects[target]
                if name == "resize_shape":
                    shape["w"], shape["h"] = args["width"], args["height"]
                elif target in boxes:
                    shape["name"] = args["text"]
                else:
                    shape["text"] = args["text"]
                    shape["name"] = args["text"][:100] or shape["kind"]
                self.canvas["selected_ids"] = [target]
                document_edit = True
            elif name == "arrange_shapes":
                target, made = self.arrange(args, created_id)
                document_edit = True
            else:
                target = args["schema_id"]
                box = boxes[target]
                if name == "rename_schema":
                    box["name"] = args["new_name"]
                else:
                    method = name in ("add_method", "remove_method")
                    values = box["methods" if method else "properties"]
                    value = args["method_name" if method else "property_name"]
                    if name.startswith("remove"):
                        values.remove(value)
                    elif value not in values:
                        values.append(value)
                    box["h"] = schema_height(box["properties"], box["methods"])
                self.canvas["selected_ids"] = [target]
                document_edit = True
        if document_edit:
            if self.document() != before:
                self.undo_stack.append(before)
                self.redo_stack.clear()
            if made:
                self.history["last_created_id"] = target
            if target:
                self.history["last_edited_id"] = target
        self.sync_history()
        entry = {"command": command, "action": copy.deepcopy(action)}
        if made:
            entry["created_id"] = target
        self.history["turns"] = [*self.history["turns"], entry][-3:]
        self.canvas = Canvas.model_validate(self.canvas).model_dump()

    def arrange(self, args, created_id):
        objects = self.objects()
        ids, operation = args["shape_ids"], args["operation"]
        shapes = [objects[i] for i in ids]
        selected = ids[:]
        made = False
        if operation == "duplicate":
            descendants = self.descendants(ids)
            mapping = {
                identifier: created_id
                if identifier == ids[0]
                else f"{created_id}:{index}"
                for index, identifier in enumerate(sorted(descendants))
            }
            for original in sorted(descendants):
                shape = copy.deepcopy(objects[original])
                shape.update(
                    id=mapping[original],
                    x=shape["x"] + 24,
                    y=shape["y"] + 24,
                    parent_id=mapping.get(shape["parent_id"], shape["parent_id"]),
                    order=len(self.objects()),
                )
                self.canvas[
                    "schemas"
                    if original in {s["id"] for s in self.canvas["schemas"]}
                    else "shapes"
                ].append(shape)
            selected, made = [mapping[i] for i in ids], True
        elif operation == "group":
            x, y = min(s["x"] for s in shapes), min(s["y"] for s in shapes)
            self.canvas["shapes"].append(
                CanvasShape(
                    id=created_id,
                    name="group",
                    kind="group",
                    x=x,
                    y=y,
                    w=max(s["x"] + s["w"] for s in shapes) - x,
                    h=max(s["y"] + s["h"] for s in shapes) - y,
                    order=len(objects),
                ).model_dump()
            )
            for shape in shapes:
                shape["parent_id"] = created_id
            selected, made = [created_id], True
        elif operation == "ungroup":
            selected = []
            for shape in self.objects().values():
                if shape["parent_id"] in ids:
                    shape["parent_id"] = objects[shape["parent_id"]]["parent_id"]
                    selected.append(shape["id"])
            self.canvas["shapes"] = [
                shape for shape in self.canvas["shapes"] if shape["id"] not in ids
            ]
        elif operation in ("front", "back", "forward", "backward"):
            ordered = sorted(
                objects, key=lambda identifier: objects[identifier]["order"]
            )
            if operation == "front":
                ordered = [i for i in ordered if i not in ids] + ids
            elif operation == "back":
                ordered = ids + [i for i in ordered if i not in ids]
            else:
                direction = 1 if operation == "forward" else -1
                for identifier in reversed(ids) if direction > 0 else ids:
                    index = ordered.index(identifier)
                    adjacent = max(0, min(len(ordered) - 1, index + direction))
                    ordered[index], ordered[adjacent] = (
                        ordered[adjacent],
                        ordered[index],
                    )
            for index, identifier in enumerate(ordered):
                objects[identifier]["order"] = index
        else:
            axis = "y" if operation.endswith(("top", "bottom", "vertical")) else "x"
            dimension = "h" if axis == "y" else "w"
            low = min(s[axis] for s in shapes)
            high = max(s[axis] + s[dimension] for s in shapes)
            if operation.startswith("align"):
                for shape in shapes:
                    shape[axis] = (
                        low
                        if operation.endswith(("left", "top"))
                        else high - shape[dimension]
                        if operation.endswith(("right", "bottom"))
                        else (low + high - shape[dimension]) / 2
                    )
            elif operation.startswith(("distribute", "stack")):
                ordered = sorted(shapes, key=lambda shape: shape[axis])
                gap = (
                    16
                    if operation.startswith("stack")
                    else (high - low - sum(s[dimension] for s in shapes))
                    / (len(shapes) - 1)
                )
                position = low
                for shape in ordered:
                    shape[axis] = position
                    position += shape[dimension] + gap
            elif operation.startswith("flip"):
                for shape in shapes:
                    shape[axis] = low + high - shape[axis] - shape[dimension]
            elif operation == "pack":
                for index, shape in enumerate(shapes):
                    shape["x"] = low + index * (shape["w"] + 16)
        self.canvas["selected_ids"] = selected
        return selected[0] if selected else None, made

    def snapshot(self):
        ids = self.objects()
        return {
            "schemas": sorted(
                copy.deepcopy(self.canvas["schemas"]), key=lambda b: b["id"]
            ),
            "shapes": sorted(
                copy.deepcopy(self.canvas["shapes"]), key=lambda b: b["id"]
            ),
            "selected_ids": sorted(self.canvas["selected_ids"]),
            "connections": sorted(
                copy.deepcopy(self.connections), key=lambda a: tuple(a.values())
            ),
            "camera": copy.deepcopy(self.canvas["camera"]),
            "can_undo": self.canvas["can_undo"],
            "can_redo": self.canvas["can_redo"],
            "references": {
                key: value if value in ids else None
                for key, value in self.history.items()
                if key != "turns"
            },
        }


def schema_height(fields, methods):
    return 104 + max(1, len(fields)) * 24 + max(1, len(methods)) * 24


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
