import argparse
import copy
import json
import random
import re
from collections import Counter
from pathlib import Path

from actions import (
    Canvas,
    CanvasShape,
    SchemaBox,
    messages_for,
    no_action,
    validate_call,
)
from dataset import audit_examples, iter_examples, read_examples
from sessions import CanvasSession

NAMES = {
    "train": [
        a + b
        for a in (
            "User",
            "Student",
            "Account",
            "Garden",
            "Billing",
            "Audio",
            "Client",
            "Order",
        )
        for b in ("Profile", "Record", "Plan", "Entry", "Queue", "Session")
    ],
    "valid": [
        "MuseumPass",
        "MealPlan",
        "StudyGroup",
        "DeliveryRoute",
        "ServiceQueue",
        "AuditEntry",
    ],
    "test": [
        "TelescopeRun",
        "FestivalPass",
        "LabSample",
        "TrailMarker",
        "GrantProposal",
        "WaterMeter",
    ],
}
FIELDS = [
    "name",
    "class",
    "subjects",
    "email",
    "age",
    "status",
    "user_id",
    "URL",
    "APIKey",
    "is_verified",
    "addresses",
    "retry_count",
    "permissions",
]
METHODS = [
    "getName",
    "getClass",
    "getSubjects",
    "addSubject",
    "removeSubject",
    "getURL",
    "validate",
    "save",
    "setAPIKey",
]
KINDS = [
    "rectangle",
    "ellipse",
    "diamond",
    "triangle",
    "text",
    "note",
    "frame",
    "arrow",
]
COLORS = [
    "black",
    "grey",
    "light-violet",
    "violet",
    "blue",
    "light-blue",
    "yellow",
    "orange",
    "green",
    "light-green",
    "light-red",
    "red",
    "white",
]
ARRANGEMENTS = [
    "duplicate",
    "group",
    "ungroup",
    "front",
    "back",
    "forward",
    "backward",
    "align_left",
    "align_right",
    "align_top",
    "align_bottom",
    "align_center_horizontal",
    "align_center_vertical",
    "distribute_horizontal",
    "distribute_vertical",
    "flip_horizontal",
    "flip_vertical",
    "stack_horizontal",
    "stack_vertical",
    "pack",
]
COMMANDS = [
    "undo",
    "redo",
    "select_all",
    "clear_selection",
    "zoom_in",
    "zoom_out",
    "zoom_to_fit",
    "reset_zoom",
]
TOOLS = [
    "create_schema_box",
    "add_property",
    "remove_property",
    "rename_schema",
    "connect_schemas",
    "create_shape",
    "select_shapes",
    "move_shapes",
    "delete_shapes",
    "resize_shape",
    "set_text",
    "style_shapes",
    "arrange_shapes",
    "canvas_command",
    "pan_canvas",
    "add_method",
    "remove_method",
]
VERBS = {
    "train": {
        "create": ["Create", "Draw", "Make", "um draw"],
        "move": ["Move", "Drag", "Shift", "uh move"],
        "delete": ["Delete", "Remove", "Erase"],
        "select": ["Select", "Pick", "Highlight"],
    },
    "valid": {
        "create": ["Please sketch"],
        "move": ["Reposition"],
        "delete": ["Get rid of"],
        "select": ["Choose"],
    },
    "test": {
        "create": ["Put down"],
        "move": ["Slide"],
        "delete": ["Take away"],
        "select": ["Mark"],
    },
}


def call(tool_name, **arguments):
    return {"name": tool_name, "arguments": arguments}


def initial_canvas(rng, split, prefix, count=5):
    labels = rng.sample(NAMES[split], count)
    return Canvas(
        schemas=[
            SchemaBox(
                id=f"{prefix}:box",
                name=labels[0],
                properties=rng.sample(FIELDS, 3),
                methods=rng.sample(METHODS, 2),
                x=-200.0,
                order=0,
            )
        ],
        shapes=[
            CanvasShape(
                id=f"{prefix}:shape-{index}",
                name=label,
                kind=rng.choice(KINDS),
                text=label,
                x=float(index * 220),
                y=float(rng.randint(-3, 3) * 100),
                w=float(rng.choice([100, 160, 200, 280])),
                h=float(rng.choice([80, 100, 160])),
                order=index + 1,
            )
            for index, label in enumerate(labels[1:])
        ],
    ).model_dump()


def target_reference(session, rng, split, candidates):
    objects = session.objects()
    if not candidates:
        return "MissingEntity", None, "missing_target"
    shape = rng.choice(candidates)
    kind = rng.choices(
        ["name", "selected", "last_created", "last_edited"], [55, 30, 8, 7]
    )[0]
    if kind == "name":
        matches = [s for s in objects.values() if s["name"] == shape["name"]]
        return (
            f'"{shape["name"]}"',
            shape["id"],
            "ambiguous_target" if len(matches) != 1 else None,
        )
    if kind == "selected":
        reference = rng.choice(
            {
                "train": ["it", "the selected shape"],
                "valid": ["the chosen shape"],
                "test": ["that shape"],
            }[split]
        )
        ids = session.canvas["selected_ids"]
        target = ids[0] if len(ids) == 1 else None
        error = (
            "missing_target"
            if not objects
            else "ambiguous_target"
            if target is None
            else None
        )
    else:
        reference = (
            "the last created shape"
            if kind == "last_created"
            else "the last edited shape"
        )
        target = session.history[f"{kind}_id"]
        error = None if target in objects else "missing_target"
    if target is not None and target not in {s["id"] for s in candidates}:
        error = error or "unsupported_request"
    return reference, target, error


def instruction(session, rng, split, sequence, tool=None):
    tool = tool or rng.choice(TOOLS)
    if tool == "create_shape" and len(session.canvas["shapes"]) >= 8:
        tool = "delete_shapes"
    if tool == "create_schema_box" and len(session.canvas["schemas"]) >= 5:
        tool = "remove_property"
    objects = session.objects()
    roots = [s for s in objects.values() if s["parent_id"] is None]
    schema_tools = {
        "add_property",
        "remove_property",
        "rename_schema",
        "add_method",
        "remove_method",
    }
    candidates = (
        list(session.canvas["schemas"])
        if tool in schema_tools
        else list(objects.values())
    )
    target, identifier, error = target_reference(session, rng, split, candidates)
    label = rng.choice(NAMES[split])
    field, method = rng.choice(FIELDS), rng.choice(METHODS)

    def verb(kind):
        return rng.choice(VERBS[split][kind])

    if tool == "create_shape":
        kind = rng.choice(KINDS)
        if len(session.canvas["shapes"]) >= 8:
            tool = "delete_shapes"
        else:
            x, y = (
                (None, None)
                if rng.random() < 0.5
                else (float(rng.randint(-5, 5) * 100), float(rng.randint(-5, 5) * 100))
            )
            width, height = (
                (160.0, 100.0)
                if kind in ("text", "note")
                else (
                    float(rng.choice([100, 160, 200, 300])),
                    float(rng.choice([80, 100, 160, 240])),
                )
            )
            text = label if rng.random() < 0.7 else ""
            command = f"{verb('create')} a {kind}" + (
                f' labelled "{text}"' if text else ""
            )
            if kind not in ("text", "note") and (
                width != 160 or height != 100 or rng.random() < 0.5
            ):
                command += f", width {width:g} and height {height:g}"
            if x is not None:
                command += f" at x {x:g}, y {y:g}"
            return command + ".", call(
                tool, kind=kind, text=text, x=x, y=y, width=width, height=height
            )
    if tool == "create_schema_box":
        if len(session.canvas["schemas"]) >= 5:
            tool = "remove_property"
        else:
            fields, methods = (
                rng.sample(FIELDS, rng.randint(0, 4)),
                rng.sample(METHODS, rng.randint(0, 3)),
            )
            command = (
                f"{verb('create')} a schema named {label}, "
                f"fields [{', '.join(fields)}], methods [{', '.join(methods)}]."
            )
            return command, call(tool, name=label, fields=fields, methods=methods)
    if tool == "pan_canvas":
        dx, dy = float(rng.randint(-4, 4) * 100), float(rng.randint(-4, 4) * 100)
        phrasing = {
            "train": f"Pan the view by x {dx:g}, y {dy:g}.",
            "valid": (
                f"Scroll the canvas viewport {dx:g} horizontally and {dy:g} vertically."
            ),
            "test": f"Shift my view by {dx:g} on x and {dy:g} on y.",
        }
        return phrasing[split], call(tool, dx=dx, dy=dy)
    if tool == "canvas_command":
        operation = rng.choice(COMMANDS)
        words = {
            "train": [
                "Undo that.",
                "Redo that.",
                "Select all shapes.",
                "Clear the selection.",
                "Zoom in.",
                "Zoom out.",
                "Fit the canvas to view.",
                "Reset zoom.",
            ],
            "valid": [
                "Reverse the last edit.",
                "Restore the undone edit.",
                "Choose everything.",
                "Deselect everything.",
                "Zoom closer.",
                "Zoom further out.",
                "Show all the drawing.",
                "Set zoom to 100 percent.",
            ],
            "test": [
                "Take back my last change.",
                "Repeat the change I undid.",
                "Mark every object.",
                "Unselect all objects.",
                "Increase the zoom.",
                "Decrease the zoom.",
                "Fit everything on screen.",
                "Return to normal zoom.",
            ],
        }
        expected = call(tool, operation=operation)
        if operation in ("undo", "redo") and not session.canvas[f"can_{operation}"]:
            expected = no_action("missing_target")
        return words[split][COMMANDS.index(operation)], expected
    plural = (
        tool
        in (
            "move_shapes",
            "delete_shapes",
            "select_shapes",
            "style_shapes",
            "arrange_shapes",
        )
        and rng.random() < 0.5
    )
    ids = [identifier] if identifier else []
    if plural:
        target = {
            "train": "the selected shapes",
            "valid": "the chosen objects",
            "test": "all selected objects",
        }[split]
        ids = session.canvas["selected_ids"][:]
        error = None if ids else "ambiguous_target" if objects else "missing_target"
    if tool == "arrange_shapes":
        operation = rng.choice(ARRANGEMENTS)
        if operation == "duplicate" and len(objects) > 14:
            operation = "front"
        required = (
            3
            if operation.startswith("distribute")
            else 2
            if operation == "group"
            else 1
        )
        if operation == "ungroup":
            groups = [s for s in roots if s.get("kind") == "group"]
            if groups:
                ids, error = [groups[0]["id"]], None
                target = "the group"
            else:
                error = "missing_target"
        elif len(ids) < required and len(roots) >= required:
            shapes = rng.sample(roots, required)
            ids = [s["id"] for s in shapes]
            target = " and ".join(f'"{s["name"]}"' for s in shapes)
            error = (
                "ambiguous_target"
                if any(
                    sum(
                        s["name"] == candidate["name"] for candidate in objects.values()
                    )
                    > 1
                    for s in shapes
                )
                else None
            )
        elif len(ids) < required:
            error = error or "unsupported_request"
        words = {
            "duplicate": "Duplicate",
            "group": "Group",
            "ungroup": "Ungroup",
            "front": "Bring to front",
            "back": "Send to back",
            "forward": "Bring forward",
            "backward": "Send backward",
            "pack": "Pack together",
        }
        operation_words = words.get(operation, operation.replace("_", " ").capitalize())
        command = f"{operation_words} {target}."
        if split == "valid":
            command = "Please " + command[0].lower() + command[1:]
        elif split == "test":
            command = command[:-1] + ", please."
        expected = call(tool, shape_ids=ids, operation=operation)
    elif tool == "select_shapes":
        command, expected = f"{verb('select')} {target}.", call(tool, shape_ids=ids)
    elif tool == "move_shapes":
        dx, dy = (
            float(rng.choice([-200, -100, -50, 0, 50, 100, 200])),
            float(rng.choice([-200, -100, -50, 0, 50, 100, 200])),
        )
        direction = "left" if dx < 0 else "right"
        if dy == 0 and dx:
            distance = (
                "" if abs(dx) == 100 and rng.random() < 0.5 else f"{abs(dx):g} units "
            )
            command = f"{verb('move')} {target} {distance}{direction}."
        else:
            command = f"{verb('move')} {target} by x {dx:g}, y {dy:g}."
        expected = call(tool, shape_ids=ids, dx=dx, dy=dy)
    elif tool == "delete_shapes":
        command, expected = f"{verb('delete')} {target}.", call(tool, shape_ids=ids)
    elif tool == "resize_shape":
        width, height = (
            float(rng.choice([80, 100, 200, 320, 400])),
            float(rng.choice([80, 100, 160, 240, 300])),
        )
        command = f"Resize {target} to width {width:g} and height {height:g}."
        expected = call(tool, shape_id=identifier, width=width, height=height)
    elif tool == "set_text":
        command, expected = (
            f'Replace the text of {target} with "{label}".',
            call(tool, shape_id=identifier, text=label),
        )
    elif tool == "style_shapes":
        color, fill, opacity = None, None, None
        style = rng.choice(["color", "fill", "opacity", "all"])
        if style in ("color", "all"):
            color = rng.choice(COLORS)
        if style in ("fill", "all"):
            fill = rng.choice(["none", "semi", "solid", "pattern"])
        if style in ("opacity", "all"):
            opacity = rng.choice([0.25, 0.5, 0.75, 1.0])
        styles = [
            *(["color " + color] if color else []),
            *(["fill " + fill] if fill else []),
            *([f"opacity {opacity * 100:g} percent"] if opacity is not None else []),
        ]
        command = f"Set {target} to {', '.join(styles)}."
        expected = call(tool, shape_ids=ids, color=color, fill=fill, opacity=opacity)
    elif tool == "connect_schemas":
        if len(roots) < 2:
            return f"Connect {target} to MissingEntity.", no_action("missing_target")
        first, second = rng.sample(roots, 2)
        if any(
            sum(s["name"] == value["name"] for s in objects.values()) > 1
            for value in (first, second)
        ):
            error = "ambiguous_target"
        else:
            error = None
        command = f'Connect "{first["name"]}" to "{second["name"]}" labelled "{field}".'
        expected = call(
            tool, source_id=first["id"], target_id=second["id"], label=field
        )
    elif tool == "rename_schema":
        command, expected = (
            f"Rename {target} to {label}.",
            call(tool, schema_id=identifier, new_name=label),
        )
    elif tool in ("add_property", "remove_property", "add_method", "remove_method"):
        method_action = tool.endswith("method")
        value = method if method_action else field
        collection = "methods" if method_action else "properties"
        if (
            tool.startswith("remove")
            and identifier in objects
            and objects[identifier].get(collection)
            and rng.random() < 0.7
        ):
            value = rng.choice(objects[identifier][collection])
        adding = tool.startswith("add")
        command = (
            f"{'Add' if adding else 'Remove'} "
            f"{'method' if method_action else 'property'} {value} "
            f"{'to' if adding else 'from'} {target}."
        )
        expected = call(
            tool,
            schema_id=identifier,
            **{"method_name" if method_action else "property_name": value},
        )
        if (
            not error
            and tool.startswith("remove")
            and value not in objects[identifier][collection]
        ):
            error = "missing_target"
    else:
        raise ValueError(f"Unknown generator tool: {tool}")
    if not error:
        try:
            expected = validate_call(expected, session.canvas)
        except ValueError:
            error = "unsupported_request"
    if rng.random() < 0.12 and tool not in ("canvas_command", "pan_canvas"):
        command += " Then explain why."
        return command, no_action("unsupported_request")
    return command, no_action(error) if error else expected


def generate_session(rng, split, index, turns):
    group = f"v5-{split}-workflow-{index}"
    initial = initial_canvas(rng, split, group)
    session = CanvasSession(initial)
    steps, examples = [], []
    for turn in range(turns):
        before = []
        objects = list(session.objects())
        if objects and rng.random() < 0.5:
            before.append(
                {
                    "kind": "select",
                    "ids": rng.sample(
                        objects, min(len(objects), rng.choice([0, 1, 1, 2, 3]))
                    ),
                }
            )
        if objects and rng.random() < 0.1:
            before.append(
                {
                    "kind": "move",
                    "id": rng.choice(objects),
                    "dx": float(rng.choice([-40, 40])),
                    "dy": 20.0,
                }
            )
        if objects and rng.random() < 0.05:
            before.append({"kind": "delete", "id": rng.choice(objects)})
        session.external(before)
        command, expected = instruction(session, rng, split, turn)
        expected = validate_call(expected, session.canvas)
        example = {
            "id": f"{group}:{turn}",
            "group": group,
            "split": split,
            "command": command,
            "canvas": copy.deepcopy(session.canvas),
            "history": copy.deepcopy(session.history),
            "expected": expected,
            "provenance": "seeded general workflow session; simulated document state",
        }
        examples.append(example)
        steps.append(
            {
                "id": example["id"],
                "command": command,
                "expected": expected,
                "before": before,
            }
        )
        session.execute(command, expected, f"{group}:created-{turn}")
    return {
        "id": group,
        "split": split,
        "initial_canvas": initial,
        "turns": steps,
    }, examples


def build_workflow(source, *, seed=64, independent=80000, sessions=4000, replay=12000):
    rng = random.Random(seed)
    old = read_examples(source)
    # Earlier unsupported labels contradict the expanded tool vocabulary.
    replay_pool = [
        row
        for row in old
        if row["split"] == "train"
        and not (
            row["expected"]["name"] == "no_action"
            and row["expected"]["arguments"]["reason"] == "unsupported_request"
            and re.search(
                r"\b(?:move|blue|undo|delete the (?:entire|whole))\b",
                row["command"],
                re.I,
            )
        )
    ]
    rows = copy.deepcopy(rng.sample(replay_pool, min(replay, len(replay_pool))))
    cases = []
    for split, singles, count in (
        ("train", independent, sessions),
        ("valid", 720, 12),
        ("test", 1440, 24),
    ):
        for index in range(singles):
            group = f"v5-{split}-single-{index}"
            session = CanvasSession(
                initial_canvas(rng, split, group, rng.randint(2, 5))
            )
            objects = list(session.objects())
            session.canvas["selected_ids"] = rng.sample(
                objects, rng.randint(0, min(3, len(objects)))
            )
            session.canvas["can_undo"] = rng.random() < 0.5
            session.canvas["can_redo"] = rng.random() < 0.3
            command, expected = instruction(
                session, rng, split, index, TOOLS[index % len(TOOLS)]
            )
            rows.append(
                {
                    "id": group,
                    "group": group,
                    "split": split,
                    "command": command,
                    "canvas": copy.deepcopy(session.canvas),
                    "expected": validate_call(expected, session.canvas),
                    "provenance": "seeded independent general canvas state",
                }
            )
        for index in range(count):
            case, examples = generate_session(
                rng,
                split,
                index,
                rng.randint(24, 80)
                if split == "train"
                else 40
                if split == "valid"
                else 60,
            )
            cases.append(case)
            rows.extend(examples)
    rng.shuffle(rows)
    return augment_rows(rows, seed), cases


def augment_rows(rows, seed, compounds=24000, corrections=6000):
    rng = random.Random(seed + 100)
    supported = [
        row
        for row in rows
        if row["split"] == "train" and row["expected"]["name"] != "no_action"
    ]
    extra = []
    for index in range(compounds + corrections):
        source = rng.choice(supported)
        row = copy.deepcopy(source)
        row["id"] = f"v5-train-counterfactual-{index}"
        if index < compounds:
            row["command"] += rng.choice(
                [
                    " And then move it left.",
                    " Then delete it.",
                    " And rename it to SomethingElse.",
                    " Then undo that.",
                    " Then explain why.",
                ]
            )
            row["expected"] = no_action("unsupported_request")
            row["provenance"] = "paired unsupported compound request"
        else:
            row["command"] = "Explain the drawing. No, actually " + row["command"]
            row["provenance"] = (
                "cancelled request followed by a final supported correction"
            )
        extra.append(row)
    return [*rows, *extra]


def spoken_instruction(row, rng):
    action = copy.deepcopy(row["expected"])
    name, args = action["name"], action["arguments"]
    if name == "canvas_command":
        phrases = {
            "undo": [
                "Undo.",
                "Undo that.",
                "Go back one step.",
                "Cancel my last edit.",
            ],
            "redo": [
                "Redo.",
                "Redo that.",
                "Put that change back.",
                "Do the undone action again.",
            ],
            "select_all": ["Select all.", "Select everything.", "Select every shape."],
            "clear_selection": ["Deselect.", "Clear selection.", "Select nothing."],
            "zoom_in": ["Zoom in.", "Zoom in a bit.", "Make the view closer."],
            "zoom_out": ["Zoom out.", "Zoom out a bit.", "Make the view wider."],
            "zoom_to_fit": [
                "Fit the drawing.",
                "Show everything.",
                "Zoom to fit all shapes.",
            ],
            "reset_zoom": [
                "Reset the zoom.",
                "Reset zoom.",
                "Zoom to one hundred percent.",
            ],
        }
        return rng.choice(phrases[args["operation"]]), action
    ids = args["shape_ids"]
    canvas, history = row["canvas"], row.get("history", {})
    objects = {
        shape["id"]: shape for shape in [*canvas["schemas"], *canvas.get("shapes", [])]
    }
    references = []
    if canvas["selected_ids"] == ids:
        references.extend(
            ["it", "that one", "the selected shape", "this shape"]
            if len(ids) == 1
            else ["them", "the selected shapes", "these shapes", "the selection"]
        )
    if len(ids) == 1:
        for kind in ("created", "edited"):
            if history.get(f"last_{kind}_id") == ids[0]:
                references.append(f"the last {kind} shape")
    if all(
        sum(
            shape["name"].casefold() == objects[target]["name"].casefold()
            for shape in objects.values()
        )
        == 1
        for target in ids
    ):
        references.append(
            " and ".join(json.dumps(objects[target]["name"]) for target in ids)
        )
    if not references:
        return None
    target = rng.choice(references)
    if name == "move_shapes":
        direction = rng.choice(["left", "right", "up", "down"])
        distance = rng.choice([10, 25, 35, 40, 50, 75, 80, 100, 125, 200])
        args.update(
            dx=-distance
            if direction == "left"
            else distance
            if direction == "right"
            else 0,
            dy=-distance
            if direction == "up"
            else distance
            if direction == "down"
            else 0,
        )
        command = rng.choice(
            [
                f"Move {target} {direction} by {distance}.",
                f"Move {target} {distance} pixels {direction}.",
                f"Shift {target} {direction} {distance} units.",
                f"Nudge {target} {direction} by {distance}.",
                f"Drag {target} {direction} {distance} pixels.",
            ]
        )
    elif name == "style_shapes":
        color = args["color"]
        command = rng.choice(
            [
                f"Make {target} {color}.",
                f"Color {target} {color}.",
                f"Paint {target} {color}.",
                f"Set {target} to {color}.",
                f"Change the color of {target} to {color}.",
                f"Turn {target} {color}.",
            ]
        )
    else:
        operation = args["operation"]
        command = rng.choice(
            {
                "group": [
                    f"Group {target}.",
                    f"Put {target} in a group.",
                    f"Group {target} together.",
                ],
                "ungroup": [
                    f"Ungroup {target}.",
                    f"Break {target} out of the group.",
                    f"Remove the grouping from {target}.",
                ],
                "duplicate": [
                    f"Duplicate {target}.",
                    f"Make a copy of {target}.",
                    f"Copy {target}.",
                ],
            }[operation]
        )
    return command, validate_call(action, canvas)


def workflow_practice(rng, index):
    group = f"spoken-session-{index}"
    initial = initial_canvas(rng, "train", group, count=3)
    targets = [shape["id"] for shape in initial["shapes"]]
    initial["selected_ids"] = targets
    session = CanvasSession(initial)
    rows, turns = [], []
    for cycle in range(3):
        group_id = f"{group}:group-{cycle}"
        actions = [
            call("arrange_shapes", shape_ids=targets, operation="group"),
            call("move_shapes", shape_ids=[group_id], dx=0, dy=40),
            call("arrange_shapes", shape_ids=[group_id], operation="ungroup"),
            call("move_shapes", shape_ids=targets, dx=0, dy=-25),
            call(
                "style_shapes",
                shape_ids=targets,
                color=rng.choice(COLORS),
                fill=None,
                opacity=None,
            ),
            call("delete_shapes", shape_ids=targets),
            call("canvas_command", operation="undo"),
            call("canvas_command", operation="redo"),
            call("canvas_command", operation="undo"),
        ]
        for position, action in enumerate(actions):
            before = (
                [{"kind": "move", "id": targets[0], "dx": 35, "dy": 25}]
                if position == 3
                else []
            )
            session.external(before)
            row = {
                "canvas": copy.deepcopy(session.canvas),
                "history": copy.deepcopy(session.history),
                "expected": action,
            }
            if action["name"] == "delete_shapes":
                command, expected = (
                    "Delete the selected shapes.",
                    validate_call(action, session.canvas),
                )
            else:
                command, expected = spoken_instruction(row, rng)
            identifier = f"{group}:{len(turns)}"
            rows.append(
                {
                    **row,
                    "id": identifier,
                    "group": group,
                    "split": "train",
                    "command": command,
                    "expected": expected,
                    "provenance": "editing cycles: drag, group, delete, undo, redo",
                }
            )
            turns.append(
                {
                    "id": identifier,
                    "command": command,
                    "expected": expected,
                    "before": before,
                }
            )
            session.execute(command, expected, group_id)
    return rows, {
        "id": group,
        "group": group,
        "split": "train",
        "initial_canvas": initial,
        "turns": turns,
    }


def build_spoken_refinement(
    source, output, *, count=48000, replay=48000, seed=65, practice_sessions=800
):
    rng = random.Random(seed)
    pools, retained, evaluation, reserved = {}, [], [], set()
    seen = 0
    for row in iter_examples(source):
        if row["split"] != "train":
            evaluation.append(row)
            reserved.add(
                json.dumps(
                    messages_for(row["command"], row["canvas"], row.get("history")),
                    sort_keys=True,
                )
            )
            continue
        seen += 1
        if len(retained) < replay:
            retained.append(row)
        else:
            index = rng.randrange(seen)
            if index < replay:
                retained[index] = row
        name, args = row["expected"]["name"], row["expected"]["arguments"]
        key = name
        if name == "style_shapes":
            if (
                args["color"] is None
                or args["fill"] is not None
                or args["opacity"] is not None
            ):
                continue
        elif name in ("arrange_shapes", "canvas_command"):
            if name == "arrange_shapes" and args["operation"] not in (
                "group",
                "ungroup",
                "duplicate",
            ):
                continue
            key += ":" + args["operation"]
        elif name != "move_shapes":
            continue
        pool = pools.setdefault(key, [])
        if len(pool) < 2000:
            pool.append(row)
    required = {
        "move_shapes",
        "style_shapes",
        "arrange_shapes:ungroup",
        "canvas_command:redo",
    }
    if not required <= pools.keys():
        raise ValueError("Spoken refinement is missing required workflow examples.")
    extra, attempts = [], 0
    while len(extra) < count:
        attempts += 1
        if attempts > count * 20:
            raise ValueError("Could not create enough unambiguous spoken examples.")
        row = copy.deepcopy(rng.choice(pools[rng.choice(sorted(pools))]))
        result = spoken_instruction(row, rng)
        if result is None:
            continue
        row["command"], row["expected"] = result
        if (
            json.dumps(
                messages_for(row["command"], row["canvas"], row.get("history")),
                sort_keys=True,
            )
            in reserved
        ):
            continue
        row.update(
            id=f"spoken-train-{len(extra)}",
            provenance="spoken workflow variation; frozen evaluation preserved",
        )
        extra.append(row)
    practice_rows, cases = [], []
    for index in range(practice_sessions):
        examples, case = workflow_practice(rng, index)
        practice_rows.extend(examples)
        cases.append(case)
    rows = [*retained, *extra, *practice_rows, *evaluation]
    counts = audit_examples(rows)
    rng.shuffle(rows)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "examples.jsonl").open("w") as destination:
        for row in rows:
            destination.write(json.dumps(row) + "\n")
    (output / "sessions.jsonl").write_bytes(
        source.with_name("sessions.jsonl").read_bytes()
    )
    with (output / "sessions.jsonl").open("a") as destination:
        for case in cases:
            destination.write(json.dumps(case) + "\n")
    summary = {
        "seed": seed,
        "splits": counts,
        "spoken_examples": len(extra),
        "replay_examples": len(retained),
        "practice_sessions": len(cases),
        "practice_turns": len(practice_rows),
        "spoken_families": {key: len(pool) for key, pool in pools.items()},
        "evaluation": "Original validation and test examples and sessions preserved",
    }
    (output / "dataset-summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Generate general canvas instruction and editing-session data."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=64)
    parser.add_argument("--independent", type=int, default=80000)
    parser.add_argument("--sessions", type=int, default=4000)
    parser.add_argument("--replay", type=int, default=12000)
    parser.add_argument("--spoken-refinement", type=int, default=0)
    parser.add_argument("--practice-sessions", type=int, default=800)
    args = parser.parse_args()
    if args.output.resolve() == args.source.resolve():
        raise ValueError("Keep the previous dataset frozen.")
    if args.spoken_refinement:
        build_spoken_refinement(
            args.source,
            args.output,
            count=args.spoken_refinement,
            replay=args.replay,
            seed=args.seed,
            practice_sessions=args.practice_sessions,
        )
        return
    rows, sessions = build_workflow(
        args.source,
        seed=args.seed,
        independent=args.independent,
        sessions=args.sessions,
        replay=args.replay,
    )
    counts = audit_examples(rows)
    args.output.mkdir(parents=True, exist_ok=False)
    for filename, values in (("examples.jsonl", rows), ("sessions.jsonl", sessions)):
        with (args.output / filename).open("w") as output:
            for value in values:
                output.write(json.dumps(value) + "\n")
    summary = {
        "seed": args.seed,
        "splits": counts,
        "sessions": dict(Counter(case["split"] for case in sessions)),
        "tools": dict(
            Counter(row["expected"]["name"] for row in rows if row["split"] == "train")
        ),
        "state_evaluation": (
            "Python document simulation; browser tests separately "
            "verify native editor behavior"
        ),
        "obsolete_unsupported_labels_excluded": True,
        "compound_pairs": 24000,
        "corrections": 6000,
    }
    (args.output / "dataset-summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
