import argparse
import copy
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from actions import (
    ACTION_MODELS,
    Canvas,
    CanvasShape,
    SchemaBox,
    messages_for,
    no_action,
    validate_call,
)
from dataset import audit_examples, iter_examples, read_examples
from sessions import CanvasSession, supports_style

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


ACCURACY_NAMES = {
    "train": [
        f"{subject} {suffix}"
        for subject in (
            "Customer",
            "Invoice",
            "Lesson",
            "Library",
            "Project",
            "Team",
            "Recipe",
            "Budget",
            "Booking",
            "Product",
            "Message",
            "Schedule",
        )
        for suffix in (
            "Board",
            "List",
            "Card",
            "Plan",
            "Details",
            "Archive",
            "Draft",
            "Summary",
        )
    ],
    "valid": [
        f"{subject} {suffix}"
        for subject in ("Harbor", "Orchard", "Gallery", "Workshop", "Parcel", "Journal")
        for suffix in (
            "Index",
            "Overview",
            "Register",
            "Checklist",
            "Folder",
            "Tracker",
        )
    ],
    "test": [
        f"{subject} {suffix}"
        for subject in (
            "Observatory",
            "Reservoir",
            "Expedition",
            "Theater",
            "Meadow",
            "Aquarium",
        )
        for suffix in (
            "Ledger",
            "Outline",
            "Catalog",
            "Dashboard",
            "Agenda",
            "Notebook",
        )
    ],
}
ACCURACY_FIELDS = [
    *FIELDS,
    "firstName",
    "lastName",
    "dateOfBirth",
    "courseId",
    "score",
    "owner",
    "createdAt",
]
ACCURACY_METHODS = [
    *METHODS,
    "getEmail",
    "setName",
    "getScore",
    "setOwner",
    "reset",
    "listItems",
]


def accuracy_stratum(action):
    name, arguments = action["name"], action["arguments"]
    detail = arguments.get("operation", arguments.get("reason"))
    return f"{name}:{detail}" if detail else name


def accuracy_families():
    return {
        name: [f"{name}:{detail}" for detail in details] if details else [name]
        for name in ACTION_MODELS
        for details in [
            ARRANGEMENTS
            if name == "arrange_shapes"
            else COMMANDS
            if name == "canvas_command"
            else ["missing_target", "ambiguous_target", "unsupported_request"]
            if name == "no_action"
            else []
        ]
    }


def accuracy_quotas(count):
    families = accuracy_families()
    minimum = sum(map(len, families.values()))
    if count < minimum:
        raise ValueError(
            f"At least {minimum} examples are needed to cover every action family."
        )
    tools = list(families)
    quotas = {}
    remaining = count - minimum
    totals = {name: len(values) for name, values in families.items()}
    for _ in range(remaining):
        name = min(tools, key=lambda tool: (totals[tool], tools.index(tool)))
        totals[name] += 1
    for name, values in families.items():
        base, remainder = divmod(totals[name], len(values))
        quotas.update({value: base + (i < remainder) for i, value in enumerate(values)})
    return quotas


def accuracy_context(rng, split, prefix):
    names = rng.sample(ACCURACY_NAMES[split], rng.randint(6, 10))
    session = CanvasSession(
        Canvas(
            schemas=[
                SchemaBox(
                    id=f"{prefix}:box-{i}",
                    name=label,
                    properties=rng.sample(ACCURACY_FIELDS, 4),
                    methods=rng.sample(ACCURACY_METHODS, 3),
                    x=float(rng.randint(-900, 900)),
                    y=float(rng.randint(-600, 600)),
                    order=i,
                )
                for i, label in enumerate(names[:2])
            ],
            shapes=[
                CanvasShape(
                    id=f"{prefix}:shape-{i}",
                    name=label,
                    text=label,
                    kind=rng.choice(KINDS),
                    x=float(rng.randint(-900, 900)),
                    y=float(rng.randint(-600, 600)),
                    w=float(rng.choice([100, 160, 200, 280])),
                    h=float(rng.choice([80, 100, 160, 240])),
                    color=rng.choice(COLORS),
                    order=i + 2,
                )
                for i, label in enumerate(names[2:])
            ],
            camera={
                "x": float(rng.randint(-500, 500)),
                "y": float(rng.randint(-500, 500)),
                "z": rng.choice([0.5, 1.0, 2.0]),
            },
        ).model_dump()
    )
    for index in range(rng.randint(1, 4)):
        target = rng.choice(list(session.objects()))
        action = call(
            "move_shapes",
            shape_ids=[target],
            dx=rng.choice([-35, 25, 80]),
            dy=rng.choice([-40, 0, 50]),
        )
        command = (
            f'Move "{session.objects()[target]["name"]}" '
            f"by x {action['arguments']['dx']}, y {action['arguments']['dy']}."
        )
        session.execute(command, action, f"{prefix}:setup-{index}")
    if rng.random() < 0.3:
        session.execute(
            "Undo that edit.",
            call("canvas_command", operation="undo"),
            f"{prefix}:setup-undo",
        )
    session.external(
        [
            {
                "kind": "select",
                "ids": rng.sample(list(session.objects()), rng.randint(0, 3)),
            }
        ]
    )
    if rng.random() < 0.15:
        first, second = session.canvas["schemas"][:2]
        session.external([{"kind": "text", "id": second["id"], "text": first["name"]}])
    return session


def accuracy_reference(session, ids, rng, split):
    objects = session.objects()
    references = []
    if all(
        sum(
            s["name"].casefold() == objects[i]["name"].casefold()
            for s in objects.values()
        )
        == 1
        for i in ids
    ):
        references.append(
            (" and ".join(json.dumps(objects[i]["name"]) for i in ids), "named")
        )
    if set(session.canvas["selected_ids"]) == set(ids):
        choices = {
            "train": ["it", "this shape", "the selected shape"]
            if len(ids) == 1
            else ["them", "these shapes", "the selection"],
            "valid": ["the chosen shape"] if len(ids) == 1 else ["the chosen objects"],
            "test": ["that selected object"]
            if len(ids) == 1
            else ["all the selected objects"],
        }
        references.extend((text, "selected") for text in choices[split])
    if len(ids) == 1:
        for kind in ("created", "edited"):
            if session.history[f"last_{kind}_id"] == ids[0]:
                references.append((f"the last {kind} shape", f"last_{kind}"))
    if not references:
        session.external([{"kind": "select", "ids": ids}])
        return (
            "the selected shape" if len(ids) == 1 else "the selected shapes"
        ), "selected"
    return rng.choice(references)


def accuracy_wording(split, key, values, rng):
    # Each split has a separate sentence construction, not merely a politeness prefix.
    templates = {
        "create_schema_box": [
            [
                "Create a schema called {label} with properties {fields} "
                "and methods {methods}.",
                "Draw the class {label}. Fields: {fields}. Functions: {methods}.",
                "Make a {label} schema; attributes {fields}; operations {methods}.",
                "I need a schema box named {label}, listing {fields} as fields "
                "and {methods} as methods.",
            ],
            [
                "For schema {label}, put {fields} under properties "
                "and {methods} under functions.",
                "Build {label} as a class containing fields {fields}, "
                "with these methods: {methods}.",
            ],
            [
                "A new class box please, titled {label}; its properties should be "
                "{fields} and its functions {methods}.",
                "The schema I want is {label}. Give it fields {fields} "
                "and methods {methods}.",
            ],
        ],
        "create_shape": [
            [
                "Draw a {kind} labelled {label}, width {width} "
                "and height {height}{position}.",
                "Create a {kind} with text {label}, "
                "sized {width} by {height}{position}.",
                "Make a {kind}, put {label} on it, "
                "width {width}, height {height}{position}.",
            ],
            [
                "Add a {kind} bearing {label}; its width is {width} "
                "and height is {height}{position}."
            ],
            [
                "I want a {kind} that says {label}, "
                "{width} wide and {height} high{position}."
            ],
        ],
        "add_property": [
            [
                "Add property {value} to {target}.",
                "Give {target} a new field named {value}.",
                "Put an attribute called {value} in {target}.",
            ],
            ["The attributes of {target} need one extra entry, {value}."],
            ["Include {value} in the properties section of {target}."],
        ],
        "remove_property": [
            [
                "Remove property {value} from {target}.",
                "Drop the field {value} from {target}.",
                "Delete the {value} attribute in {target}.",
            ],
            ["Take {value} out of the attributes on {target}."],
            ["{target} should no longer have a property called {value}."],
        ],
        "add_method": [
            [
                "Add method {value} to {target}.",
                "Give {target} a function called {value}.",
                "Include operation {value} in {target}.",
            ],
            ["The methods section on {target} needs {value} added."],
            ["Put {value} among the functions of {target}."],
        ],
        "remove_method": [
            [
                "Remove method {value} from {target}.",
                "Drop the function {value} on {target}.",
                "Delete operation {value} from {target}.",
            ],
            ["Take the method {value} out of {target}."],
            ["{target} should no longer list the function {value}."],
        ],
        "rename_schema": [
            [
                "Rename {target} to {label}.",
                "Call {target} {label} instead.",
                "Change the name of {target} to {label}.",
            ],
            ["Use {label} as the new title for {target}."],
            ["The new name for {target} should be {label}."],
        ],
        "connect_schemas": [
            [
                "Connect {source} to {destination} with label {label}.",
                "Draw an arrow from {source} to {destination} labelled {label}.",
                "Link {source} to {destination}, naming the connection {label}.",
            ],
            ["Join {source} to {destination} using an arrow that says {label}."],
            ["An arrow labelled {label} should go from {source} into {destination}."],
        ],
        "select_shapes": [
            ["Select {target}.", "Pick {target}.", "Highlight {target}."],
            ["Make {target} the active selection."],
            ["I want {target} selected now."],
        ],
        "move_shapes": [
            [
                "Move {target} {direction} by {distance}.",
                "Drag {target} {distance} pixels {direction}.",
                "Shift {target} {direction} {distance} units.",
                "Reposition {target} {distance} units {direction}.",
            ],
            ["Move {target} a distance of {distance} towards the {direction}."],
            ["{target} needs to go {distance} units {direction}."],
        ],
        "delete_shapes": [
            [
                "Delete {target}.",
                "Remove {target} from the canvas.",
                "Erase {target}.",
                "Get rid of {target}.",
            ],
            ["Take {target} off the drawing."],
            ["I do not want {target} on this canvas anymore."],
        ],
        "resize_shape": [
            [
                "Resize {target} to width {width} and height {height}.",
                "Make {target} {width} wide and {height} high.",
                "Set the dimensions of {target} to {width} by {height}.",
            ],
            ["Change the size of {target}: width {width}, height {height}."],
            ["{target} should measure {width} units across and {height} units tall."],
        ],
        "set_text": [
            [
                "Set the text of {target} to {label}.",
                "Replace the text on {target} with {label}.",
                "Change what {target} says to {label}.",
            ],
            ["Use {label} for the text displayed on {target}."],
            ["The words shown on {target} should be {label}."],
        ],
        "style_shapes": [
            [
                "Set {target} to {styles}.",
                "Change the style of {target}: {styles}.",
                "Apply {styles} to {target}.",
            ],
            ["For {target}, use these appearance settings: {styles}."],
            ["The appearance I want for {target} is {styles}."],
        ],
        "pan_canvas": [
            [
                "Pan the view by x {dx}, y {dy}.",
                "Scroll the view {dx} units horizontally and {dy} units vertically.",
                "Move the viewport by {dx} on x and {dy} on y.",
                "Shift the canvas view horizontally {dx}, vertically {dy}.",
            ],
            [
                "Move my view across the canvas: horizontal distance {dx}, "
                "vertical distance {dy}."
            ],
            ["The viewport should travel {dx} horizontally plus {dy} vertically."],
        ],
    }
    index = ("train", "valid", "test").index(split)
    if key == "move_shapes" and "dx" in values:
        choices = [
            [
                "Move {target} by x {dx}, y {dy}.",
                "Reposition {target} by x {dx}, y {dy}.",
                "Drag {target} {dx} horizontally and {dy} vertically.",
            ],
            ["For {target}, change x by {dx} and y by {dy}."],
            ["{target} should travel {dx} horizontally and {dy} vertically."],
        ][index]
    elif key == "create_shape" and values["kind"] == "note":
        choices = [
            [
                "Draw a note labelled {label}{position}.",
                "Create a note with text {label}{position}.",
                "Make a note that says {label}{position}.",
            ],
            ["Add a note bearing {label}{position}."],
            ["I want a note that says {label}{position}."],
        ][index]
    elif key == "create_shape" and values["kind"] == "text":
        choices = [
            [
                "Draw text saying {label}, width {width}{position}.",
                "Create text {label} with width {width}{position}.",
                "Make text that says {label}, {width} units wide{position}.",
            ],
            ["Add text reading {label}; its width is {width}{position}."],
            ["I want text that says {label}, {width} wide{position}."],
        ][index]
    else:
        choices = templates[key][index]
    choice = rng.randrange(len(choices))
    mode = ":vector" if key == "move_shapes" and "dx" in values else ""
    return choices[choice].format(**values), f"accuracy-{split}:{key}{mode}:{choice}"


def accuracy_example(session, rng, split, identifier, family):
    tool, _, operation = family.partition(":")
    objects = session.objects()
    roots = [s for s in objects.values() if s["parent_id"] is None]
    schemas = list(session.canvas["schemas"])
    label = rng.choice(ACCURACY_NAMES[split])
    reference_kind = "none"
    values = {"label": json.dumps(label)}
    if family == "no_action:ambiguous_target" and not objects:
        return accuracy_example(
            session, rng, split, identifier, "no_action:missing_target"
        )
    if tool == "no_action":
        if operation == "missing_target":
            removed = [
                kind
                for kind in ("created", "edited")
                if session.history[f"last_{kind}_id"]
                and session.history[f"last_{kind}_id"] not in objects
            ]
            if removed and rng.random() < 0.5:
                kind = rng.choice(removed)
                command, wording = accuracy_wording(
                    split,
                    "move_shapes",
                    {
                        "target": f"the last {kind} shape",
                        "distance": "100",
                        "direction": "right",
                    },
                    rng,
                )
                reference_kind = f"removed_last_{kind}"
            elif schemas and rng.random() < 0.5:
                target = rng.choice(schemas)
                session.external([{"kind": "select", "ids": [target["id"]]}])
                command, wording = accuracy_wording(
                    split,
                    "remove_property",
                    {
                        "target": "the selected shape",
                        "value": rng.choice(
                            [
                                value
                                for value in ACCURACY_FIELDS
                                if value not in target["properties"]
                            ]
                        ),
                    },
                    rng,
                )
                reference_kind = "missing_field"
            else:
                missing = rng.choice(
                    [
                        value
                        for value in ACCURACY_NAMES[split]
                        if all(
                            shape["name"].casefold() != value.casefold()
                            for shape in objects.values()
                        )
                    ]
                )
                command, wording = accuracy_wording(
                    split, "delete_shapes", {"target": json.dumps(missing)}, rng
                )
                reference_kind = "missing_name"
        elif operation == "ambiguous_target":
            duplicated = [
                shape
                for shape in objects.values()
                if sum(
                    other["name"].casefold() == shape["name"].casefold()
                    for other in objects.values()
                )
                > 1
            ]
            if duplicated and rng.random() < 0.5:
                command, wording = accuracy_wording(
                    split,
                    "delete_shapes",
                    {
                        "target": json.dumps(rng.choice(duplicated)["name"]),
                    },
                    rng,
                )
                reference_kind = "duplicate_name"
            else:
                selection = (
                    rng.sample(list(objects), 2)
                    if len(objects) >= 2 and rng.random() < 0.5
                    else []
                )
                session.external([{"kind": "select", "ids": selection}])
                command, wording = accuracy_wording(
                    split,
                    "move_shapes",
                    {"target": "it", "distance": "100", "direction": "right"},
                    rng,
                )
                reference_kind = (
                    "multiple_selection" if selection else "absent_selection"
                )
        else:
            unsupported_targets = [
                shape
                for shape in objects.values()
                if shape.get("kind") in ("note", "text", "frame")
            ]
            if unsupported_targets and rng.random() < 0.25:
                target = rng.choice(unsupported_targets)
                session.external([{"kind": "select", "ids": [target["id"]]}])
                if target["kind"] == "note" and rng.random() < 0.5:
                    command, wording = accuracy_wording(
                        split,
                        "resize_shape",
                        {
                            "target": "the selected shape",
                            "width": 200,
                            "height": 100,
                        },
                        rng,
                    )
                    reference_kind = "unsupported_note_resize"
                else:
                    style = (
                        "color"
                        if target["kind"] == "frame" and rng.random() < 0.5
                        else "fill"
                    )
                    value = (
                        rng.choice(COLORS)
                        if style == "color"
                        else rng.choice(["semi", "solid", "pattern"])
                    )
                    command, wording = accuracy_wording(
                        split,
                        "style_shapes",
                        {
                            "target": "the selected shape",
                            "styles": f"{style} {value}",
                        },
                        rng,
                    )
                    reference_kind = f"unsupported_{target['kind']}_{style}"
                wording += ":unsupported-capability"
            else:
                command = rng.choice(
                    {
                        "train": [
                            "Explain the drawing.",
                            "What does this schema mean?",
                            "Generate Python code for this diagram.",
                            "Delete the selected shape and then draw a rectangle.",
                            "Move it right and then undo that.",
                        ],
                        "valid": [
                            "Tell me why this drawing is useful.",
                            "Resize this shape and then delete it.",
                        ],
                        "test": [
                            "Can you describe what these boxes represent?",
                            "Group these shapes and then move the group left.",
                        ],
                    }[split]
                )
                wording = f"accuracy-{split}:unsupported"
        expected = no_action(operation)
    elif tool == "canvas_command":
        if operation in ("undo", "redo") and not session.canvas[f"can_{operation}"]:
            return accuracy_example(
                session, rng, split, identifier, "no_action:missing_target"
            )
        phrases = {
            "undo": [
                [
                    "Undo.",
                    "Undo that edit.",
                    "Reverse the last edit.",
                    "Go back one step.",
                ],
                ["Revert my previous edit."],
                ["Take back the change I just made."],
            ],
            "redo": [
                [
                    "Redo.",
                    "Put the undone change back.",
                    "Restore the undone edit.",
                    "Do that undone action again.",
                ],
                ["Reapply the edit I undid."],
                ["Bring back my reversed change."],
            ],
            "select_all": [
                ["Select all shapes.", "Select everything.", "Choose every object."],
                ["Choose the entire drawing."],
                ["Every shape should be selected."],
            ],
            "clear_selection": [
                ["Clear the selection.", "Deselect everything.", "Select nothing."],
                ["Remove all objects from the selection."],
                ["Nothing should remain selected."],
            ],
            "zoom_in": [
                ["Zoom in.", "Make the view closer.", "Increase the zoom."],
                ["Bring the view a step closer."],
                ["I want to see the drawing more closely."],
            ],
            "zoom_out": [
                ["Zoom out.", "Make the view wider.", "Decrease the zoom."],
                ["Pull the view a step further away."],
                ["I want a more distant view of the drawing."],
            ],
            "zoom_to_fit": [
                [
                    "Fit the drawing in the view.",
                    "Show all shapes.",
                    "Zoom to fit everything.",
                ],
                ["Adjust the view so the whole drawing is visible."],
                ["Get the complete drawing onto my screen."],
            ],
            "reset_zoom": [
                ["Reset zoom.", "Set zoom to 100 percent.", "Return to normal zoom."],
                ["Use the original zoom level again."],
                ["Put the view back at one hundred percent zoom."],
            ],
        }
        choices = phrases[operation][("train", "valid", "test").index(split)]
        index = rng.randrange(len(choices))
        command, wording = (
            choices[index],
            f"accuracy-{split}:canvas-{operation}:{index}",
        )
        expected = call(tool, operation=operation)
    else:
        schema_tool = tool in (
            "add_property",
            "remove_property",
            "add_method",
            "remove_method",
            "rename_schema",
        )
        candidates = schemas if schema_tool else roots
        if tool == "connect_schemas":
            candidates = [
                s
                for s in candidates
                if sum(
                    value["name"].casefold() == s["name"].casefold()
                    for value in objects.values()
                )
                == 1
            ]
        if tool == "set_text":
            candidates = [s for s in candidates if s.get("kind") != "group"]
        if tool == "resize_shape":
            candidates = [
                s for s in candidates if s.get("kind") not in ("note", "text")
            ]
        required = (
            3
            if operation.startswith("distribute")
            else 2
            if operation == "group" or tool == "connect_schemas"
            else 1
        )
        if operation == "ungroup":
            candidates = [s for s in roots if s.get("kind") == "group"]
        if (
            tool not in ("create_shape", "create_schema_box", "pan_canvas")
            and len(candidates) < required
        ):
            return accuracy_example(
                session, rng, split, identifier, "no_action:missing_target"
            )
        targets = rng.sample(candidates, required) if candidates else []
        if (
            tool in ("select_shapes", "move_shapes", "delete_shapes", "style_shapes")
            and len(candidates) > 1
            and rng.random() < 0.35
        ):
            targets = rng.sample(candidates, rng.randint(2, min(3, len(candidates))))
        ids = [s["id"] for s in targets]
        if tool in (
            "select_shapes",
            "move_shapes",
            "delete_shapes",
            "style_shapes",
            "arrange_shapes",
        ):
            ids.sort()
        if ids:
            session.external(
                [
                    {
                        "kind": "select",
                        "ids": ids
                        if rng.random() < 0.5
                        else rng.sample(
                            list(objects), rng.randint(0, min(3, len(objects)))
                        ),
                    }
                ]
            )
            values["target"], reference_kind = accuracy_reference(
                session, ids, rng, split
            )
        if tool == "create_schema_box":
            fields, methods = (
                rng.sample(ACCURACY_FIELDS, rng.randint(0, 5)),
                rng.sample(ACCURACY_METHODS, rng.randint(0, 4)),
            )
            values.update(
                fields=", ".join(fields) or "none", methods=", ".join(methods) or "none"
            )
            expected = call(tool, name=label, fields=fields, methods=methods)
        elif tool == "create_shape":
            kind = rng.choice(KINDS)
            width, height = (
                (160, 100)
                if kind == "note"
                else (rng.choice([80, 100, 160, 240, 320]), 100)
                if kind == "text"
                else (
                    rng.choice([80, 100, 160, 240, 320]),
                    rng.choice([80, 100, 160, 240]),
                )
            )
            x, y = (
                (None, None)
                if rng.random() < 0.5
                else (rng.randint(-600, 600), rng.randint(-400, 400))
            )
            values.update(
                kind=kind,
                width=width,
                height=height,
                position="" if x is None else f" at x {x}, y {y}",
            )
            expected = call(
                tool, kind=kind, text=label, x=x, y=y, width=width, height=height
            )
        elif tool in ("add_property", "remove_property", "add_method", "remove_method"):
            method = tool.endswith("method")
            pool = (
                targets[0]["methods" if method else "properties"]
                if tool.startswith("remove")
                else ACCURACY_METHODS
                if method
                else ACCURACY_FIELDS
            )
            if not pool:
                return accuracy_example(
                    session, rng, split, identifier, "no_action:missing_target"
                )
            value = rng.choice(pool)
            values["value"] = value
            expected = call(
                tool,
                schema_id=ids[0],
                **{"method_name" if method else "property_name": value},
            )
        elif tool == "rename_schema":
            expected = call(tool, schema_id=ids[0], new_name=label)
        elif tool == "connect_schemas":
            values.update(
                source=json.dumps(targets[0]["name"]),
                destination=json.dumps(targets[1]["name"]),
                label=json.dumps(
                    rng.choice(
                        ["owns", "uses", "contains", "enrolled in", "depends on"]
                    )
                ),
            )
            expected = call(
                tool,
                source_id=ids[0],
                target_id=ids[1],
                label=json.loads(values["label"]),
            )
            reference_kind = "named_pair"
        elif tool == "move_shapes":
            direction, distance = (
                rng.choice(["left", "right", "up", "down"]),
                rng.choice([10, 25, 35, 40, 50, 75, 80, 100, 125, 200, 12.5]),
            )
            values.update(direction=direction, distance=f"{distance:g}")
            expected = call(
                tool,
                shape_ids=ids,
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
            if rng.random() < 0.3:
                dx, dy = (
                    rng.choice([-200, -100, -25, 0, 40, 75, 200]),
                    rng.choice([-200, -100, -25, 0, 40, 75, 200]),
                )
                values.update(dx=dx, dy=dy)
                expected = call(tool, shape_ids=ids, dx=dx, dy=dy)
        elif tool in ("select_shapes", "delete_shapes"):
            expected = call(tool, shape_ids=ids)
        elif tool == "resize_shape":
            width, height = (
                rng.choice([80, 100, 160, 240, 320]),
                rng.choice([80, 100, 160, 240]),
            )
            values.update(width=width, height=height)
            expected = call(tool, shape_id=ids[0], width=width, height=height)
        elif tool == "set_text":
            expected = call(tool, shape_id=ids[0], text=label)
        elif tool == "style_shapes":
            leaves = session.style_targets(ids)
            color_supported = bool(leaves) and all(
                supports_style(shape, "color") for shape in leaves
            )
            fill_supported = bool(leaves) and all(
                supports_style(shape, "fill") for shape in leaves
            )
            options = ["opacity"]
            if color_supported:
                options.append("color")
            if fill_supported:
                options.append("fill")
            if color_supported and fill_supported:
                options.append("all")
            style = rng.choice(options)
            color, fill, opacity = (
                rng.choice(COLORS) if style in ("color", "all") else None,
                rng.choice(["none", "semi", "solid", "pattern"])
                if style in ("fill", "all")
                else None,
                rng.choice([0.25, 0.5, 0.75, 1.0])
                if style in ("opacity", "all")
                else None,
            )
            styles = [
                *(["color " + color] if color else []),
                *(["fill " + fill] if fill else []),
                *(
                    [f"opacity {opacity * 100:g} percent"]
                    if opacity is not None
                    else []
                ),
            ]
            values["styles"] = ", ".join(styles)
            expected = call(
                tool, shape_ids=ids, color=color, fill=fill, opacity=opacity
            )
        elif tool == "pan_canvas":
            dx, dy = (
                rng.choice([-200, -100, -25, 0, 40, 75, 200]),
                rng.choice([-200, -100, -25, 0, 40, 75, 200]),
            )
            values.update(dx=dx, dy=dy)
            expected = call(tool, dx=dx, dy=dy)
        elif tool == "arrange_shapes":
            words = {
                "duplicate": "Duplicate",
                "group": "Group",
                "ungroup": "Ungroup",
                "front": "Bring to front",
                "back": "Send to back",
                "forward": "Bring forward",
                "backward": "Send backward",
                "align_left": "Align left",
                "align_right": "Align right",
                "align_top": "Align top",
                "align_bottom": "Align bottom",
                "align_center_horizontal": "Align horizontal centers",
                "align_center_vertical": "Align vertical centers",
                "distribute_horizontal": "Distribute horizontally",
                "distribute_vertical": "Distribute vertically",
                "flip_horizontal": "Flip horizontally",
                "flip_vertical": "Flip vertically",
                "stack_horizontal": "Stack horizontally",
                "stack_vertical": "Stack vertically",
                "pack": "Pack together",
            }[operation]
            choices = {
                "train": [
                    f"{words} {{target}}.",
                    f"Please {words.lower()} {{target}}.",
                    f"For {{target}}, {words.lower()}.",
                ],
                "valid": [f"Can you {words.lower()} {{target}}?"],
                "test": [f"I want you to {words.lower()} {{target}} now."],
            }[split]
            choice = rng.randrange(len(choices))
            command, wording = (
                choices[choice].format(**values),
                f"accuracy-{split}:arrange-{operation}:{choice}",
            )
            expected = call(tool, shape_ids=ids, operation=operation)
        else:
            raise ValueError(f"Unknown accuracy family: {family}")
        if tool != "arrange_shapes":
            command, wording = accuracy_wording(split, tool, values, rng)
    if split == "train" and rng.random() < 0.12:
        command = (
            rng.choice(
                ["Explain this. No, actually ", "Wait, cancel that. ", "No, instead "]
            )
            + command
        )
        wording += ":correction"
    expected = validate_call(expected, session.canvas)
    row = {
        "id": identifier,
        "group": identifier.rsplit(":", 1)[0],
        "split": split,
        "command": command,
        "canvas": copy.deepcopy(session.canvas),
        "history": copy.deepcopy(session.history),
        "expected": expected,
        "stratum": accuracy_stratum(expected),
        "wording_family": wording,
        "reference_kind": reference_kind,
        "provenance": "v7 balanced authored grammar; current simulated outcomes",
    }
    probe = copy.deepcopy(session)
    probe.execute(command, expected, identifier + ":verification")
    return row


def accuracy_session(rng, split, index, turns):
    group = f"accuracy-{split}-session-{index}"
    session = accuracy_context(rng, split, group)
    initial = copy.deepcopy(session.canvas)
    initial["can_undo"] = initial["can_redo"] = False
    # The replay starts at an empty outcome history and undo stack, just like runtime.
    session = CanvasSession(initial)
    blocks = [
        ["create_shape", "move_shapes", "resize_shape", "set_text", "style_shapes"],
        ["arrange_shapes:group", "move_shapes", "arrange_shapes:ungroup"],
        [
            "delete_shapes",
            "no_action:missing_target",
            "canvas_command:undo",
            "canvas_command:redo",
            "canvas_command:undo",
        ],
        [
            "add_property",
            "remove_property",
            "add_method",
            "remove_method",
            "rename_schema",
        ],
        [
            "connect_schemas",
            "select_shapes",
            "pan_canvas",
            "canvas_command:zoom_to_fit",
        ],
        [
            "no_action:ambiguous_target",
            "no_action:unsupported_request",
            "canvas_command:select_all",
            "canvas_command:clear_selection",
        ],
    ]
    rng.shuffle(blocks)
    plan = [family for block in blocks for family in block]
    fillers = [
        name
        for name in ACTION_MODELS
        if name not in ("arrange_shapes", "canvas_command", "no_action")
    ]
    rows, steps = [], []
    for turn in range(turns):
        family = plan[turn] if turn < len(plan) else rng.choice(fillers)
        before = []
        objects = list(session.objects())
        if objects and rng.random() < 0.4:
            before.append(
                {
                    "kind": "select",
                    "ids": rng.sample(objects, rng.randint(0, min(3, len(objects)))),
                }
            )
        if objects and rng.random() < 0.15:
            before.append(
                {
                    "kind": "move",
                    "id": rng.choice(objects),
                    "dx": rng.choice([-35, 25]),
                    "dy": rng.choice([-20, 40]),
                }
            )
        text_targets = [
            s for s in session.objects().values() if s.get("kind") != "group"
        ]
        if text_targets and rng.random() < 0.08:
            before.append(
                {
                    "kind": "text",
                    "id": rng.choice(text_targets)["id"],
                    "text": rng.choice(ACCURACY_NAMES[split]),
                }
            )
        if len(objects) > 5 and rng.random() < 0.06:
            before.append({"kind": "delete", "id": rng.choice(objects)})
        session.external(before)
        row = accuracy_example(session, rng, split, f"{group}:{turn}", family)
        # Target preparation only changes selection; record it for exact session replay.
        before.append({"kind": "select", "ids": row["canvas"]["selected_ids"][:]})
        row["group"] = group
        row["provenance"] = (
            "v7 mixed sequential workflow with manual edits and recovery"
        )
        rows.append(row)
        steps.append(
            {
                "id": row["id"],
                "command": row["command"],
                "expected": row["expected"],
                "before": before,
            }
        )
        session.execute(row["command"], row["expected"], f"{group}:created-{turn}")
    return rows, {
        "id": group,
        "split": split,
        "initial_canvas": initial,
        "turns": steps,
    }


def accuracy_independent(rng, split, index, family):
    identifier = f"accuracy-{split}-single-{index}:0"
    session = accuracy_context(rng, split, identifier)
    if family == "arrange_shapes:ungroup":
        ids = rng.sample(list(session.objects()), 2)
        session.execute(
            "Group these two shapes.",
            call("arrange_shapes", shape_ids=ids, operation="group"),
            identifier + ":setup-group",
        )
    elif family == "canvas_command:redo" and not session.canvas["can_redo"]:
        session.execute(
            "Undo that edit.",
            call("canvas_command", operation="undo"),
            identifier + ":setup-undo",
        )
    return accuracy_example(session, rng, split, identifier, family)


def accuracy_fingerprint(row):
    return hashlib.sha256(
        json.dumps(
            messages_for(row["command"], row["canvas"], row.get("history")),
            sort_keys=True,
        ).encode()
    ).hexdigest()


def accuracy_capability_issue(row):
    action = row["expected"]
    if action["name"] not in ("resize_shape", "style_shapes"):
        return None
    session = CanvasSession(row["canvas"])
    arguments = action["arguments"]
    if action["name"] == "resize_shape":
        target = session.objects()[arguments["shape_id"]]
        if target.get("kind") == "note":
            return "Notes cannot be resized by the default native shape utility."
        if (
            target.get("kind") == "text"
            and abs(
                arguments["width"] * target["h"] - arguments["height"] * target["w"]
            )
            > 0.000001
        ):
            return (
                "Native text resizing scales uniformly, not to independent dimensions."
            )
    else:
        leaves = session.style_targets(arguments["shape_ids"])
        failures = [
            f"{style} is unavailable on {shape.get('kind', 'schema')}"
            for style in ("color", "fill")
            if arguments[style] is not None
            for shape in leaves
            if not supports_style(shape, style)
        ]
        if failures:
            return "; ".join(sorted(set(failures)))
    return None


def audit_accuracy_sessions(rows, cases):
    by_id = {row["id"]: row for row in rows}
    turns = Counter()
    for case in cases:
        session = CanvasSession(case["initial_canvas"])
        for turn, step in enumerate(case["turns"]):
            session.external(step["before"])
            row = by_id[step["id"]]
            if session.canvas != row["canvas"] or session.history != row["history"]:
                raise ValueError(
                    f"Session replay differs from its captured input: {row['id']}"
                )
            session.execute(
                step["command"], step["expected"], f"{case['id']}:created-{turn}"
            )
            turns[case["split"]] += 1
    return dict(turns)


def build_accuracy_refinement(
    source,
    output,
    *,
    count=24000,
    seed=66,
    train_sessions=48,
    train_turns=40,
    dev_count=256,
    dev_sessions=8,
    dev_turns=32,
    test_count=500,
    test_sessions=12,
    test_turns=40,
):
    if count % 8:
        raise ValueError("The training count must be divisible by global batch eight.")
    quotas = accuracy_quotas(count)
    output.mkdir(parents=True, exist_ok=False)
    historical_valid, historical_sessions = [], []
    historical_valid_lines, historical_validation_exclusions = [], {}
    historical_test_rows, historical_test_sessions = [], []
    with source.open() as input_rows:
        for line in input_rows:
            row = json.loads(line)
            if row["split"] == "valid":
                historical_valid_lines.append(line)
                issue = accuracy_capability_issue(row)
                if issue:
                    historical_validation_exclusions[row["id"]] = issue
                else:
                    historical_valid.append(row)
            elif row["split"] == "test":
                historical_test_rows.append(line)
    with source.with_name("sessions.jsonl").open() as input_sessions:
        for line in input_sessions:
            split = json.loads(line)["split"]
            if split == "valid":
                historical_sessions.append(line)
            elif split == "test":
                historical_test_sessions.append(line)
    (output / "historical-valid-sessions.jsonl").write_text(
        "".join(historical_sessions)
    )
    (output / "historical-valid-examples.jsonl").write_text(
        "".join(historical_valid_lines)
    )
    (output / "historical-test-examples.jsonl").write_text(
        "".join(historical_test_rows)
    )
    (output / "historical-test-sessions.jsonl").write_text(
        "".join(historical_test_sessions)
    )
    reserved = {accuracy_fingerprint(row) for row in historical_valid}

    def evaluation(split, independent, sessions, turns, offset):
        rng = random.Random(seed + offset)
        rows, cases = [], []
        for index in range(sessions):
            examples, case = accuracy_session(rng, split, index, turns)
            rows.extend(examples)
            cases.append(case)
        targets = accuracy_quotas(independent)
        for index, family in enumerate(
            family for family, n in targets.items() for _ in range(n)
        ):
            rows.append(accuracy_independent(rng, split, index, family))
        for row in rows:
            fingerprint = accuracy_fingerprint(row)
            if fingerprint in reserved:
                raise ValueError("A reserved evaluation input overlaps another split.")
            reserved.add(fingerprint)
        return rows, cases

    test_rows, test_cases = evaluation(
        "test", test_count, test_sessions, test_turns, 20000
    )
    for filename, values in (
        ("reserved-test-examples.jsonl", test_rows),
        ("reserved-test-sessions.jsonl", test_cases),
    ):
        (output / filename).write_text(
            "".join(json.dumps(row) + "\n" for row in values)
        )
    reservation = {
        "seed": seed + 20000,
        "reserved_before_training_generation": True,
        "examples": len(test_rows),
        "independent_examples": test_count,
        "sessions": len(test_cases),
        "session_turns": sum(len(case["turns"]) for case in test_cases),
        "examples_sha256": hashlib.sha256(
            (output / "reserved-test-examples.jsonl").read_bytes()
        ).hexdigest(),
        "sessions_sha256": hashlib.sha256(
            (output / "reserved-test-sessions.jsonl").read_bytes()
        ).hexdigest(),
        "provenance": (
            "New names, grammar families and scenario groups reserved before "
            "training; previous test archived"
        ),
    }
    (output / "test-reservation.json").write_text(json.dumps(reservation, indent=2))
    dev_rows, dev_cases = evaluation("valid", dev_count, dev_sessions, dev_turns, 10000)
    rng = random.Random(seed)
    train_rows, train_cases = [], []
    used = Counter()
    for index in range(train_sessions):
        examples, case = accuracy_session(rng, "train", index, train_turns)
        counts = Counter(row["stratum"] for row in examples)
        fingerprints = [accuracy_fingerprint(row) for row in examples]
        if (
            any(used[family] + n > quotas[family] for family, n in counts.items())
            or len(set(fingerprints)) != len(examples)
            or any(value in reserved for value in fingerprints)
        ):
            continue
        train_rows.extend(examples)
        train_cases.append(case)
        used.update(counts)
        reserved.update(fingerprints)
    index = 0
    for family, target in quotas.items():
        attempts = 0
        while used[family] < target:
            row = accuracy_independent(rng, "train", index, family)
            index += 1
            attempts += 1
            if attempts > target * 20:
                raise ValueError(
                    f"Could not generate enough unique examples for {family}."
                )
            fingerprint = accuracy_fingerprint(row)
            if row["stratum"] != family or fingerprint in reserved:
                continue
            train_rows.append(row)
            used[family] += 1
            reserved.add(fingerprint)
    rows = [*train_rows, *historical_valid, *dev_rows, *test_rows]
    counts = audit_examples(rows)
    current_cases = [*train_cases, *dev_cases, *test_cases]
    session_audit = audit_accuracy_sessions(rows, current_cases)
    for row in [*train_rows, *dev_rows, *test_rows]:
        if accuracy_capability_issue(row):
            raise ValueError(
                f"Generated action uses an unavailable native capability: {row['id']}"
            )
    if len(train_rows) != count or used != Counter(quotas):
        raise ValueError(
            "The training corpus does not match its declared balanced quotas."
        )
    rng.shuffle(rows)
    for filename, values in (
        ("examples.jsonl", rows),
        (
            "sessions.jsonl",
            current_cases,
        ),
    ):
        (output / filename).write_text(
            "".join(json.dumps(row) + "\n" for row in values)
        )
    summary = {
        "seed": seed,
        "splits": counts,
        "unique_training_inputs": len(train_rows),
        "tools": dict(Counter(row["expected"]["name"] for row in train_rows)),
        "strata": dict(used),
        "wording_families": dict(Counter(row["wording_family"] for row in train_rows)),
        "reference_kinds": dict(Counter(row["reference_kind"] for row in train_rows)),
        "provenance": dict(Counter(row["provenance"] for row in train_rows)),
        "training_sessions": len(train_cases),
        "training_session_turns": sum(len(case["turns"]) for case in train_cases),
        "historical_validation_examples": len(historical_valid),
        "historical_validation_original_examples": len(historical_valid_lines),
        "historical_validation_exclusions": historical_validation_exclusions,
        "historical_validation_sessions": len(historical_sessions),
        "historical_sessions_excluded_from_current_evaluation": (
            "Previous simulator omitted visible arrows and used insertion-order aliases"
        ),
        "historical_artifact_sha256": {
            name: hashlib.sha256((output / name).read_bytes()).hexdigest()
            for name in (
                "historical-valid-examples.jsonl",
                "historical-valid-sessions.jsonl",
                "historical-test-examples.jsonl",
                "historical-test-sessions.jsonl",
            )
        },
        "fresh_development_examples": len(dev_rows),
        "fresh_development_sessions": len(dev_cases),
        "fresh_test": reservation,
        "legacy_training_rows_reused": 0,
        "session_replay_verified_turns": session_audit,
        "native_capability_negatives": dict(
            Counter(
                row["reference_kind"]
                for row in train_rows
                if row["reference_kind"].startswith("unsupported_")
            )
        ),
        "semantic_checks": (
            "Each generated action validated and executed against its captured "
            "current state; full session replay and native capability audit passed"
        ),
        "minimum_updates_for_one_pass": count // 8,
    }
    (output / "dataset-summary.json").write_text(json.dumps(summary, indent=2))
    print(
        json.dumps(
            {
                key: value
                for key, value in summary.items()
                if key not in ("wording_families", "reference_kinds")
            },
            indent=2,
        )
    )
    return summary


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
    parser.add_argument("--accuracy-refinement", type=int, default=0)
    parser.add_argument("--practice-sessions", type=int, default=800)
    args = parser.parse_args()
    if args.output.resolve() == args.source.resolve():
        raise ValueError("Keep the previous dataset frozen.")
    if args.accuracy_refinement:
        build_accuracy_refinement(
            args.source, args.output, count=args.accuracy_refinement, seed=args.seed
        )
        return
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
