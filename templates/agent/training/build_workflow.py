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
    execution_guard,
    messages_for,
    model_call,
    no_action,
    validate_call,
)
from dataset import audit_examples, iter_examples, read_examples
from sessions import SIMULATOR_VERSION, CanvasSession, supports_style

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


QUALITY_NAMES = {
    "train": (
        "Trip plan",
        "Passenger record",
        "Packing card",
        "Route tile",
        "Gate label",
        "Ticket note",
        "Departure frame",
    ),
    "valid": (
        "Studio schedule",
        "Supply record",
        "Exhibit card",
        "Outline tile",
        "Gallery label",
        "Artist note",
        "Opening frame",
    ),
}

# Each final pair is a separate development sentence construction.
QUALITY_PAIRS = {
    "named_selection": [
        (
            "Move {name} right {distance} units.",
            "Move the selected shape right {distance} units.",
        ),
        (
            "Could you drag {name} to the right by {distance}?",
            "Could you drag this selected shape to the right by {distance}?",
        ),
        (
            "Um, shift {name} horizontally {distance} units right.",
            "Um, shift it horizontally {distance} units right.",
        ),
        (
            "Reposition the object labelled {name} {distance} units to the right.",
            "Reposition the current selection {distance} units to the right.",
        ),
    ],
    "duplicate_name": [
        (
            "Move {name} down {distance} units.",
            "Move the selected shape down {distance} units.",
        ),
        (
            "Please drag the shape named {name} down by {distance}.",
            "Please drag this selected one down by {distance}.",
        ),
        (
            "Uh, shift {name} vertically {distance} units down.",
            "Uh, shift it vertically {distance} units down.",
        ),
        (
            "Reposition the object labelled {name} {distance} units lower.",
            "Reposition the chosen object {distance} units lower.",
        ),
    ],
    "stale_name": [
        ("Rename schema {schema} to {final}.", "Rename schema {renamed} to {final}."),
        ("Call the {schema} schema {final}.", "Call the {renamed} schema {final}."),
        (
            "Can you change the name of schema {schema} to {final}?",
            "Can you change the name of schema {renamed} to {final}?",
        ),
        (
            "Replace the title on schema {schema} with {final}.",
            "Replace the title on schema {renamed} with {final}.",
        ),
    ],
    "deleted_name": [
        ("Move {name} left {distance} units.", "Move {other} left {distance} units."),
        (
            "Drag {name} to the left by {distance}.",
            "Drag {other} to the left by {distance}.",
        ),
        (
            "Could you shift {name} horizontally {distance} units left?",
            "Could you shift {other} horizontally {distance} units left?",
        ),
        (
            "Reposition the item labelled {name} {distance} units leftward.",
            "Reposition the item labelled {other} {distance} units leftward.",
        ),
    ],
    "empty_selection": [
        (
            "Change the selected shape's text to {payload}.",
            "Change the selected shape's text to {payload}.",
        ),
        (
            "Can you put {payload} on the selected shape?",
            "Can you put {payload} on the selected shape?",
        ),
        (
            "Uh, replace the words on it with {payload}.",
            "Uh, replace the words on it with {payload}.",
        ),
        (
            "Give the currently chosen object the text {payload}.",
            "Give the currently chosen object the text {payload}.",
        ),
    ],
    "plural_selection": [
        (
            "Move it right {distance} units.",
            "Move all selected shapes right {distance} units.",
        ),
        (
            "Drag this shape right by {distance}.",
            "Drag these selected shapes right by {distance}.",
        ),
        (
            "Could you shift the selected shape right {distance} units?",
            "Could you shift the whole selection right {distance} units?",
        ),
        (
            "Reposition that selected object {distance} units rightward.",
            "Reposition both chosen objects {distance} units rightward.",
        ),
    ],
    "deleted_last_created": [
        (
            "Move the last created shape right {distance} units.",
            "Move the last created shape right {distance} units.",
        ),
        (
            "Could you drag the last created shape right by {distance}?",
            "Could you drag the last created shape right by {distance}?",
        ),
        (
            "Um, shift the last created shape {distance} units right.",
            "Um, shift the last created shape {distance} units right.",
        ),
        (
            "Reposition the last created shape {distance} units to the right.",
            "Reposition the last created shape {distance} units to the right.",
        ),
    ],
    "last_edited": [
        ("Make the last edited shape blue.", "Make the selected shape blue."),
        (
            "Color the last edited shape blue, please.",
            "Color this selected shape blue, please.",
        ),
        (
            "Could you give the last edited shape a blue outline?",
            "Could you give the selected shape a blue outline?",
        ),
        (
            "Set the color of the last edited object to blue.",
            "Set the color of the chosen object to blue.",
        ),
    ],
    "missing_field": [
        (
            "Remove field {absent_field} from schema {schema}.",
            "Remove field name from schema {schema}.",
        ),
        (
            "Delete the {absent_field} property on {schema}.",
            "Delete the name property on {schema}.",
        ),
        (
            "Can you take {absent_field} out of schema {schema}'s fields?",
            "Can you take name out of schema {schema}'s fields?",
        ),
        (
            "Drop property {absent_field} from the {schema} class.",
            "Drop property name from the {schema} class.",
        ),
    ],
    "missing_method": [
        (
            "Remove function {absent_method} from schema {schema}.",
            "Remove function getName from schema {schema}.",
        ),
        (
            "Delete method {absent_method} on {schema}.",
            "Delete method getName on {schema}.",
        ),
        (
            "Could you take {absent_method} out of schema {schema}'s functions?",
            "Could you take getName out of schema {schema}'s functions?",
        ),
        (
            "Drop the {absent_method} operation from {schema}.",
            "Drop the getName operation from {schema}.",
        ),
    ],
    "target_correction": [
        (
            "Move {name} right {distance}; no, "
            "move {other} right {distance} units instead.",
            "Move {name} right {distance} units "
            "and then move {other} right {distance} units.",
        ),
        (
            "Drag {name} right {distance} units. Wait, cancel that; "
            "drag {other} right {distance} units.",
            "Drag {name} right {distance} units "
            "and then move {other} right {distance} units.",
        ),
        (
            "Shift {name} right {distance} units; actually, "
            "only move {other} right {distance} units.",
            "Shift {name} right {distance} units "
            "and move {other} right {distance} units as well.",
        ),
        (
            "Reposition {name} right by {distance}. "
            "Scratch that: reposition {other} right by {distance}.",
            "Reposition {name} right by {distance}, "
            "then move {other} right by {distance}.",
        ),
    ],
    "direction_correction": [
        (
            "Move {name} left 100 units; no, right {distance} units instead.",
            "Move {name} left 100 units and then move it right {distance} units.",
        ),
        (
            "Drag {name} down 100 units. Wait, only move it right {distance} units.",
            "Drag {name} down 100 units and then move it right {distance} units.",
        ),
        (
            "Shift {name} left 100 units; actually, cancel that "
            "and move it right {distance} units.",
            "Shift {name} left 100 units and move it right {distance} units too.",
        ),
        (
            "Reposition {name} leftward by 100. "
            "Scratch that; rightward by {distance} only.",
            "Reposition {name} leftward by 100, then move it rightward by {distance}.",
        ),
    ],
    "cancel": [
        ("Delete {name}. Wait, cancel that; do nothing.", "Delete {name}."),
        ("Remove {name}; no, leave it as it is.", "Remove {name}, please."),
        (
            "Could you erase {name}? Actually, never mind, cancel the request.",
            "Could you erase {name}?",
        ),
        (
            "Get rid of {name}. Scratch that; leave the canvas unchanged.",
            "Get rid of {name}.",
        ),
    ],
    "literal_conjunction": [
        (
            "Set the text of {name} to {literal}.",
            "Set the text of {name} to {payload} and delete {other}.",
        ),
        (
            "Replace the words on {name} with the exact text {literal}.",
            "Replace the words on {name} with {payload} and then delete {other}.",
        ),
        (
            "Could you write {literal} inside {name}?",
            "Could you write {payload} inside {name} and delete {other}?",
        ),
        (
            "Give {name} this literal label: {literal}.",
            "Give {name} the label {payload}, then delete {other}.",
        ),
    ],
    "pan_move": [
        (
            "Pan the canvas right {distance} units.",
            "Move {name} right {distance} units.",
        ),
        (
            "Slide the viewport to the right by {distance}.",
            "Slide the shape named {name} to the right by {distance}.",
        ),
        (
            "Um, shift the camera horizontally {distance} units right.",
            "Um, shift {name} horizontally {distance} units right.",
        ),
        (
            "Move the view {distance} page units rightward.",
            "Reposition the object {name} {distance} page units rightward.",
        ),
    ],
    "pan_zoom": [
        ("Pan the viewport down {distance} units.", "Zoom to fit all shapes."),
        (
            "Move the camera down by {distance} page units.",
            "Fit the whole drawing in the view.",
        ),
        (
            "Can you scroll the canvas down {distance} units?",
            "Can you zoom to fit the entire diagram?",
        ),
        (
            "Shift the view {distance} units downward.",
            "Frame every object in the viewport.",
        ),
    ],
    "zoom_opacity": [
        ("Set zoom to 100 percent.", "Set the opacity of {name} to 100 percent."),
        ("Reset the canvas zoom to 100%.", "Make {name} fully opaque."),
        (
            "Could you return the view to one-to-one zoom?",
            "Could you set {name}'s opacity to one?",
        ),
        (
            "Restore a zoom factor of one for the camera.",
            "Restore full opacity for {name}.",
        ),
    ],
    "clear_delete": [
        ("Clear the selection.", "Delete all selected shapes."),
        ("Deselect everything.", "Erase everything selected."),
        ("Can you unselect these shapes?", "Can you remove these selected shapes?"),
        ("Release the current selection.", "Get rid of the currently chosen objects."),
    ],
    "undo": [
        ("Undo that last edit.", "Undo that last edit."),
        ("Can you undo the previous change?", "Can you undo the previous change?"),
        ("Uh, undo.", "Uh, undo."),
        ("Revert the latest canvas edit.", "Revert the latest canvas edit."),
    ],
    "redo": [
        ("Redo the last undone edit.", "Redo the last undone edit."),
        ("Can you redo that change?", "Can you redo that change?"),
        ("Uh, redo.", "Uh, redo."),
        (
            "Reapply the most recently undone canvas edit.",
            "Reapply the most recently undone canvas edit.",
        ),
    ],
    "note_resize": [
        ("Resize {note} to 240 by 150.", "Resize {name} to 240 by 150."),
        (
            "Make {note} 240 units wide and 150 units tall.",
            "Make {name} 240 units wide and 150 units tall.",
        ),
        (
            "Could you set the dimensions of {note} to 240 by 150?",
            "Could you set the dimensions of {name} to 240 by 150?",
        ),
        (
            "Give {note} a width of 240 and height of 150.",
            "Give {name} a width of 240 and height of 150.",
        ),
    ],
    "text_fill": [
        ("Set the fill of {text} to solid.", "Set the fill of {name} to solid."),
        ("Give {text} a solid fill.", "Give {name} a solid fill."),
        (
            "Could you apply solid fill to {text}?",
            "Could you apply solid fill to {name}?",
        ),
        ("Use a solid interior on {text}.", "Use a solid interior on {name}."),
    ],
    "frame_color": [
        ("Make {frame} blue.", "Set {frame}'s opacity to 50 percent."),
        ("Set the color of {frame} to blue.", "Make {frame} half opaque."),
        (
            "Could you give {frame} a blue outline?",
            "Could you set {frame}'s opacity to 0.5?",
        ),
        (
            "Apply blue as the color of {frame}.",
            "Apply an opacity of one half to {frame}.",
        ),
    ],
    "text_uniform_resize": [
        (
            "Resize {text} uniformly to 240 by 150.",
            "Resize {text} to 240 by 200 without changing its aspect ratio.",
        ),
        (
            "Scale {text} to one and a half times its size: width 240, height 150.",
            "Make {text} width 240 and height 200 independently.",
        ),
        (
            "Could you resize {text} proportionally to width 240 and height 150?",
            "Could you stretch {text} to width 240 and height 200?",
        ),
        (
            "Increase {text} uniformly by 50 percent, to dimensions 240 by 150.",
            "Change {text}'s dimensions independently to 240 by 200.",
        ),
    ],
    "rejected_creation": [
        (
            "The drawing failed. Move the last created shape right {distance} units.",
            "The drawing failed. Create a rectangle labelled {payload}.",
        ),
        (
            "It did not create anything. Drag the last created shape right {distance}.",
            "It did not create anything. Please draw a rectangle saying {payload}.",
        ),
        (
            "The previous call was rejected; "
            "shift the last created shape right {distance}.",
            "The previous call was rejected; make a rectangle with the text {payload}.",
        ),
        (
            "After that failed creation, "
            "reposition the last created shape right {distance}.",
            "After that failed creation, sketch a rectangle containing {payload}.",
        ),
    ],
    "wrong_target_recovery": [
        (
            "You moved {other}, not {name}. Move {name} right {distance} units.",
            "You moved {other}. Undo the last edit.",
        ),
        (
            "That moved the wrong card. Please move {name} right by {distance}.",
            "That moved the wrong card. Please undo it.",
        ),
        (
            "No, the target was {name}; shift {name} right {distance} units.",
            "No, the wrong target moved; undo that change.",
        ),
        (
            "The last move affected {other}. "
            "Reposition {name} rightward by {distance}.",
            "The last move affected {other}. Revert that canvas edit.",
        ),
    ],
    "wrong_pan_recovery": [
        (
            "That zoomed the view. Pan the canvas right {distance} units.",
            "That zoomed the view. Move {name} right {distance} units.",
        ),
        (
            "I meant camera movement; shift the viewport right {distance} units.",
            "I meant moving the card; shift {name} right {distance} units.",
        ),
        (
            "The zoom changed by mistake. Can you pan right by {distance} units?",
            "The zoom changed by mistake. "
            "Can you move {name} right by {distance} units?",
        ),
        (
            "Following that incorrect zoom, "
            "translate the view rightward by {distance} units.",
            "Following that incorrect zoom, "
            "translate the object {name} rightward by {distance} units.",
        ),
    ],
    "rejected_rename": [
        (
            "The rename failed. Add property email to schema {schema}.",
            "The rename failed. Add property email to schema {renamed}.",
        ),
        (
            "It never changed the name. Put an email field on {schema}.",
            "It never changed the name. Put an email field on {renamed}.",
        ),
        (
            "After the rejected rename, add the email property to {schema}.",
            "After the rejected rename, add the email property to {renamed}.",
        ),
        (
            "The attempted title change did not apply. Append field email to {schema}.",
            "The attempted title change did not apply. "
            "Append field email to {renamed}.",
        ),
    ],
    "default_move": [
        ("Move {name} right.", "Move {name} right {distance} units."),
        ("Drag {name} to the right.", "Drag {name} to the right by {distance}."),
        ("Uh, shift {name} right.", "Uh, shift {name} right by {distance} units."),
        (
            "Reposition {name} rightward.",
            "Reposition {name} rightward by {distance} page units.",
        ),
    ],
    "default_pan": [
        ("Pan the canvas down.", "Pan the canvas down {distance} units."),
        ("Move the viewport downward.", "Move the viewport downward by {distance}."),
        (
            "Um, shift the camera down.",
            "Um, shift the camera down {distance} page units.",
        ),
        (
            "Translate the view downward.",
            "Translate the view downward by {distance} page units.",
        ),
    ],
    "polite_question": [
        ("Could you delete {name}?", "Could you explain what {name} means?"),
        (
            "Can you remove {name} from the canvas?",
            "Can you describe {name} without changing it?",
        ),
        (
            "Would you move {name} right {distance} units?",
            "Would you tell me where {name} is?",
        ),
        (
            "Please erase the object labelled {name}.",
            "Please explain the purpose of the object labelled {name}.",
        ),
    ],
    "select_delete": [
        ("Select {name}.", "Delete {name}."),
        ("Pick the shape named {name}.", "Remove the shape named {name}."),
        ("Can you highlight {name}?", "Can you erase {name}?"),
        (
            "Choose the object labelled {name}.",
            "Get rid of the object labelled {name}.",
        ),
    ],
    "zoom_out_in": [
        ("Zoom out.", "Zoom in."),
        ("Can you zoom the view out?", "Can you zoom the view in?"),
        ("Uh, decrease the canvas zoom.", "Uh, increase the canvas zoom."),
        ("Reduce the camera zoom.", "Increase the camera zoom."),
    ],
    "mixed_group_child_style": [
        ("Make the selected group blue.", "Make {name} blue."),
        (
            "Give the selected group a blue outline.",
            "Give the shape named {name} a blue outline.",
        ),
        (
            "Could you color the selected group blue?",
            "Could you color the child named {name} blue?",
        ),
        (
            "Apply blue to the chosen group.",
            "Apply blue directly to the object labelled {name}.",
        ),
    ],
    "drawing_kind": [
        (
            "Create {drawing_a} with text {payload}.",
            "Create {drawing_b} with text {payload}.",
        ),
        (
            "Can you draw {drawing_a} saying {payload}?",
            "Can you draw {drawing_b} saying {payload}?",
        ),
        (
            "Um, make {drawing_a} labelled {payload}.",
            "Um, make {drawing_b} labelled {payload}.",
        ),
        (
            "Sketch {drawing_a} containing {payload}.",
            "Sketch {drawing_b} containing {payload}.",
        ),
    ],
}


def quality_context(rng, split, group):
    names = [f"{name} {group.rsplit('-', 1)[-1]}" for name in QUALITY_NAMES[split]]
    ids = {
        key: f"{group}:{key}"
        for key in ("box", "other-box", "geo", "other-geo", "text", "note", "frame")
    }
    canvas = Canvas(
        schemas=[
            SchemaBox(
                id=ids["box"],
                name=names[0],
                properties=["name", "subjects"],
                methods=["getName", "getSubjects"],
                x=-400,
                order=0,
            ),
            SchemaBox(
                id=ids["other-box"],
                name=names[1],
                properties=["id"],
                methods=["save"],
                x=-80,
                order=1,
            ),
        ],
        shapes=[
            CanvasShape(
                id=ids[key],
                name=names[i + 2],
                text=names[i + 2],
                kind=kind,
                x=240 + i * 200,
                y=rng.choice([-160, 0, 160]),
                w=200 if kind == "note" else 160,
                h=200 if kind == "note" else 100,
                order=i + 2,
            )
            for i, (key, kind) in enumerate(
                (
                    ("geo", "rectangle"),
                    ("other-geo", "diamond"),
                    ("text", "text"),
                    ("note", "note"),
                    ("frame", "frame"),
                )
            )
        ],
        selected_ids=[ids["other-geo"]],
        camera={
            "x": rng.choice([-240, 0, 240]),
            "y": rng.choice([-160, 0, 160]),
            "z": rng.choice([0.5, 1.0, 2.0]),
        },
    ).model_dump()
    return CanvasSession(canvas), ids


def quality_setup_outcome(session, receipt, command, action):
    identifier = f"{receipt['id']}:setup-{len(receipt['events'])}"
    session.execute(command, action, identifier)
    receipt["events"].append(
        {
            "kind": "outcome",
            "command": command,
            "action": action,
            "created_id": identifier,
        }
    )
    return identifier


def quality_setup_manual(session, receipt, events):
    session.external(events)
    receipt["events"].append({"kind": "manual", "events": events})


def quality_row(
    session,
    command,
    expected,
    *,
    identifier,
    group,
    split,
    family,
    wording,
    reference,
    rationale,
):
    action = validate_call(expected, session.canvas)
    row = {
        "id": identifier,
        "group": group,
        "split": split,
        "command": command,
        "canvas": copy.deepcopy(session.canvas),
        "history": copy.deepcopy(session.history),
        "expected": action,
        "stratum": accuracy_stratum(action),
        "wording_family": wording,
        "reference_kind": reference,
        "scenario": family,
        "rationale": rationale,
        "provenance": (
            "Authored quality supplement; simulated current canvas "
            "and actual recorded outcomes"
        ),
    }
    issue = accuracy_capability_issue(row)
    if issue:
        raise ValueError(f"Unsupported native capability in {identifier}: {issue}")
    copy.deepcopy(session).execute(command, action, identifier + ":verification")
    return row


def quality_pair(rng, split, index, family):
    group = f"quality-{split}-pair-{index}"
    base, ids = quality_context(rng, split, group)
    receipts = [
        {
            "id": f"{group}:{i}",
            "initial_canvas": copy.deepcopy(base.canvas),
            "events": [],
        }
        for i in range(2)
    ]
    sessions = [copy.deepcopy(base), copy.deepcopy(base)]
    names = base.objects()
    distance = rng.choice([25, 40, 60, 80, 120])
    values = {
        "name": json.dumps(names[ids["geo"]]["name"]),
        "other": json.dumps(names[ids["other-geo"]]["name"]),
        "schema": json.dumps(names[ids["box"]]["name"]),
        "renamed": json.dumps(f"Updated {names[ids['box']]['name']}"),
        "final": json.dumps(f"Final {names[ids['box']]['name']}"),
        "text": json.dumps(names[ids["text"]]["name"]),
        "note": json.dumps(names[ids["note"]]["name"]),
        "frame": json.dumps(names[ids["frame"]]["name"]),
        "distance": distance,
        "payload": json.dumps(f"Ready for {group.rsplit('-', 1)[-1]}"),
        "literal": json.dumps(
            rng.choice(["Save and delete", "Move and rename", "Select and remove"])
        ),
        "absent_field": "passportNumber" if split == "train" else "galleryCode",
        "absent_method": "getPassport" if split == "train" else "getGallery",
    }
    kinds = (
        KINDS[(index // len(QUALITY_PAIRS)) % len(KINDS)],
        KINDS[((index // len(QUALITY_PAIRS)) + 4) % len(KINDS)],
    )
    drawings = {
        "rectangle": "a rectangle",
        "ellipse": "an ellipse",
        "diamond": "a diamond",
        "triangle": "a triangle",
        "text": "a text label",
        "note": "a note",
        "frame": "a frame",
        "arrow": "an arrow",
    }
    values.update(drawing_a=drawings[kinds[0]], drawing_b=drawings[kinds[1]])
    templates = QUALITY_PAIRS[family]
    slot = rng.randrange(len(templates) - 1) if split == "train" else len(templates) - 1
    commands = [text.format(**values) for text in templates[slot]]
    a, b, box = ids["geo"], ids["other-geo"], ids["box"]
    missing, ambiguous, unsupported = (
        no_action(reason)
        for reason in ("missing_target", "ambiguous_target", "unsupported_request")
    )
    actions, reference = None, family

    def move(target, dx=distance, dy=0):
        return call(
            "move_shapes",
            shape_ids=[target] if isinstance(target, str) else target,
            dx=dx,
            dy=dy,
        )

    def both_outcome(command, action):
        return [
            quality_setup_outcome(session, receipt, command, action)
            for session, receipt in zip(sessions, receipts, strict=True)
        ]

    def manual(which, events):
        quality_setup_manual(sessions[which], receipts[which], events)

    if family == "named_selection":
        actions = [move(a), move(b)]
    elif family == "duplicate_name":
        for i in range(2):
            manual(
                i,
                [
                    {"kind": "text", "id": b, "text": names[a]["name"]},
                    {"kind": "select", "ids": [a]},
                ],
            )
        actions = [ambiguous, move(a, 0, distance)]
    elif family == "stale_name":
        both_outcome(
            f"Rename schema {values['schema']} to {values['renamed']}.",
            call(
                "rename_schema", schema_id=box, new_name=json.loads(values["renamed"])
            ),
        )
        actions = [
            missing,
            call("rename_schema", schema_id=box, new_name=json.loads(values["final"])),
        ]
    elif family == "deleted_name":
        both_outcome(f"Delete {values['name']}.", call("delete_shapes", shape_ids=[a]))
        actions = [missing, move(b, -distance)]
    elif family == "empty_selection":
        manual(0, [{"kind": "select", "ids": []}])
        manual(1, [{"kind": "select", "ids": [a]}])
        actions = [
            ambiguous,
            call("set_text", shape_id=a, text=json.loads(values["payload"])),
        ]
    elif family == "plural_selection":
        for i in range(2):
            manual(i, [{"kind": "select", "ids": [a, b]}])
        actions = [ambiguous, move(sorted([a, b]))]
    elif family == "deleted_last_created":
        created = both_outcome(
            f"Create a rectangle labelled {values['payload']}.",
            call("create_shape", kind="rectangle", text=json.loads(values["payload"])),
        )
        manual(
            1, [{"kind": "delete", "id": created[1]}, {"kind": "select", "ids": [b]}]
        )
        actions = [move(created[0]), missing]
    elif family == "last_edited":
        both_outcome(
            f"Move {values['name']} down {distance} units.", move(a, 0, distance)
        )
        for i in range(2):
            manual(i, [{"kind": "select", "ids": [b]}])
        actions = [
            call("style_shapes", shape_ids=[a], color="blue"),
            call("style_shapes", shape_ids=[b], color="blue"),
        ]
    elif family == "missing_field":
        actions = [
            missing,
            call("remove_property", schema_id=box, property_name="name"),
        ]
    elif family == "missing_method":
        actions = [missing, call("remove_method", schema_id=box, method_name="getName")]
    elif family == "target_correction":
        actions = [move(b), unsupported]
    elif family == "direction_correction":
        actions = [move(a), unsupported]
    elif family == "cancel":
        actions = [unsupported, call("delete_shapes", shape_ids=[a])]
    elif family == "literal_conjunction":
        actions = [
            call("set_text", shape_id=a, text=json.loads(values["literal"])),
            unsupported,
        ]
    elif family in ("pan_move", "wrong_pan_recovery"):
        if family == "wrong_pan_recovery":
            both_outcome(
                f"Pan the canvas right {distance} units.",
                call("canvas_command", operation="zoom_in"),
            )
        actions = [call("pan_canvas", dx=distance, dy=0), move(a)]
    elif family == "pan_zoom":
        actions = [
            call("pan_canvas", dx=0, dy=distance),
            call("canvas_command", operation="zoom_to_fit"),
        ]
    elif family == "zoom_opacity":
        actions = [
            call("canvas_command", operation="reset_zoom"),
            call("style_shapes", shape_ids=[a], opacity=1.0),
        ]
    elif family == "clear_delete":
        for i in range(2):
            manual(i, [{"kind": "select", "ids": [a, b]}])
        actions = [
            call("canvas_command", operation="clear_selection"),
            call("delete_shapes", shape_ids=sorted([a, b])),
        ]
    elif family in ("undo", "redo"):
        quality_setup_outcome(
            sessions[0],
            receipts[0],
            f"Move {values['name']} right {distance} units.",
            move(a),
        )
        if family == "redo":
            quality_setup_outcome(
                sessions[0],
                receipts[0],
                "Undo that edit.",
                call("canvas_command", operation="undo"),
            )
        actions = [call("canvas_command", operation=family), missing]
    elif family == "note_resize":
        actions = [unsupported, call("resize_shape", shape_id=a, width=240, height=150)]
    elif family == "text_fill":
        actions = [unsupported, call("style_shapes", shape_ids=[a], fill="solid")]
    elif family == "frame_color":
        actions = [
            unsupported,
            call("style_shapes", shape_ids=[ids["frame"]], opacity=0.5),
        ]
    elif family == "text_uniform_resize":
        actions = [
            call("resize_shape", shape_id=ids["text"], width=240, height=150),
            unsupported,
        ]
    elif family == "rejected_creation":
        both_outcome(f"Create a rectangle labelled {values['payload']}.", None)
        actions = [
            missing,
            call("create_shape", kind="rectangle", text=json.loads(values["payload"])),
        ]
    elif family == "wrong_target_recovery":
        both_outcome(f"Move {values['name']} right {distance} units.", move(b))
        actions = [move(a), call("canvas_command", operation="undo")]
    elif family == "rejected_rename":
        both_outcome(f"Rename schema {values['schema']} to {values['renamed']}.", None)
        actions = [call("add_property", schema_id=box, property_name="email"), missing]
    elif family == "default_move":
        actions = [move(a, 100), move(a)]
    elif family == "default_pan":
        actions = [
            call("pan_canvas", dx=0, dy=100),
            call("pan_canvas", dx=0, dy=distance),
        ]
    elif family == "polite_question":
        actions = [
            move(a) if slot == 2 else call("delete_shapes", shape_ids=[a]),
            unsupported,
        ]
    elif family == "select_delete":
        actions = [
            call("select_shapes", shape_ids=[a]),
            call("delete_shapes", shape_ids=[a]),
        ]
    elif family == "zoom_out_in":
        actions = [
            call("canvas_command", operation="zoom_out"),
            call("canvas_command", operation="zoom_in"),
        ]
    elif family == "mixed_group_child_style":
        both_outcome(
            f"Group {values['name']} and {values['frame']}.",
            call(
                "arrange_shapes", shape_ids=sorted([a, ids["frame"]]), operation="group"
            ),
        )
        actions = [unsupported, call("style_shapes", shape_ids=[a], color="blue")]
    elif family == "drawing_kind":
        actions = [
            call("create_shape", kind=kind, text=json.loads(values["payload"]))
            for kind in kinds
        ]
    else:
        raise ValueError(f"Unknown quality pair family: {family}")
    rows = [
        quality_row(
            session,
            command,
            action,
            identifier=f"{group}:{i}",
            group=group,
            split=split,
            family=family,
            wording=f"quality-{split}-{family}-{slot}",
            reference=reference,
            rationale=(
                f"Authored {family} contrast, side {i + 1}; "
                "label grounded in captured current canvas and recorded setup outcomes."
            ),
        )
        for i, (session, command, action) in enumerate(
            zip(sessions, commands, actions, strict=True)
        )
    ]
    pair = {
        "id": group,
        "split": split,
        "scenario": family,
        "example_ids": [row["id"] for row in rows],
        "same_canvas_and_history": rows[0]["canvas"] == rows[1]["canvas"]
        and rows[0]["history"] == rows[1]["history"],
        "same_model_canvas_and_history": messages_for(
            "", rows[0]["canvas"], rows[0]["history"]
        )
        == messages_for("", rows[1]["canvas"], rows[1]["history"]),
        "same_command": commands[0] == commands[1],
        "wording_family": rows[0]["wording_family"],
    }
    return rows, receipts, pair


def quality_session(rng, split, index, turns=72):
    if turns < 60 or turns > 80:
        raise ValueError("Quality sessions must contain 60 to 80 coherent turns.")
    group = f"quality-{split}-session-{index}"
    session, ids = quality_context(rng, split, group)
    session.external([{"kind": "select", "ids": []}])
    initial = copy.deepcopy(session.canvas)
    rows, steps = [], []

    def emit(command, action, reference="named", before=None):
        turn = len(rows)
        before = before or []
        session.external(before)
        row = quality_row(
            session,
            command,
            action,
            identifier=f"{group}:{turn}",
            group=group,
            split=split,
            family="coherent_editing_session",
            wording=f"quality-{split}-session-{turn % 36}",
            reference=reference,
            rationale=(
                "One edit in a coherent create/refine/schema/link/group/manual-edit/"
                "undo workflow; current visible state determines its target."
            ),
        )
        rows.append(row)
        steps.append(
            {
                "id": row["id"],
                "command": command,
                "expected": row["expected"],
                "before": before,
            }
        )
        created_id = f"{group}:created-{turn}"
        session.execute(command, row["expected"], created_id)
        return created_id

    def words(train, valid):
        return train if split == "train" else valid

    for block in range(3):
        start = len(rows)
        suffix = f"{index}-{block}"
        card = f"{'Packing task' if split == 'train' else 'Exhibit task'} {suffix}"
        refined = (
            f"{'Ready parcel' if split == 'train' else 'Prepared exhibit'} {suffix}"
        )
        duplicate = (
            f"{'Backup parcel' if split == 'train' else 'Spare exhibit'} {suffix}"
        )
        schema = f"{'Shipment order' if split == 'train' else 'Gallery order'} {suffix}"
        renamed = (
            f"{'Reviewed shipment' if split == 'train' else 'Reviewed gallery'} "
            f"{suffix}"
        )
        manual_name = (
            "Manually checked parcel"
            if split == "train"
            else "Manually checked exhibit"
        )
        manual = f"{manual_name} {suffix}"
        quoted = {
            key: json.dumps(value)
            for key, value in (
                ("card", card),
                ("refined", refined),
                ("duplicate", duplicate),
                ("schema", schema),
                ("renamed", renamed),
                ("manual", manual),
            )
        }
        card_id = emit(
            words(
                f"Draw a rectangle saying {quoted['card']}.",
                f"Sketch a rectangular card containing {quoted['card']}.",
            ),
            call("create_shape", kind="rectangle", text=card),
            "new",
        )
        emit(
            words("Move it right.", "Shift the chosen card rightward."),
            call("move_shapes", shape_ids=[card_id], dx=100, dy=0),
            "selected",
        )
        emit(
            words(
                "Resize the selected card to 240 by 150.",
                "Give the chosen card a width of 240 and a height of 150.",
            ),
            call("resize_shape", shape_id=card_id, width=240, height=150),
            "selected",
        )
        emit(
            words(
                f"Set its text to {quoted['refined']}.",
                f"Replace the chosen card's label with {quoted['refined']}.",
            ),
            call("set_text", shape_id=card_id, text=refined),
            "selected",
        )
        emit(
            words(
                f"Make {quoted['refined']} blue.",
                f"Apply blue to the object labelled {quoted['refined']}.",
            ),
            call("style_shapes", shape_ids=[card_id], color="blue"),
            before=[{"kind": "move", "id": card_id, "dx": -35, "dy": 20}],
        )
        emit(
            words(
                "Move the last created shape down 40 units.",
                "Reposition the last created shape 40 units downward.",
            ),
            call("move_shapes", shape_ids=[card_id], dx=0, dy=40),
            "last_created",
            [{"kind": "select", "ids": [ids["other-geo"]]}],
        )
        emit(
            words("Clear the selection.", "Release the current selection."),
            call("canvas_command", operation="clear_selection"),
            "canvas",
        )
        emit(
            words(
                "Move it left 25 units.",
                "Shift that selected object leftward by 25 units.",
            ),
            no_action("ambiguous_target"),
            "no_selection",
        )
        emit(
            words(
                f"Select {quoted['refined']}.",
                f"Choose the object labelled {quoted['refined']}.",
            ),
            call("select_shapes", shape_ids=[card_id]),
        )
        copy_id = emit(
            words("Duplicate the selected card.", "Make a copy of the chosen card."),
            call("arrange_shapes", shape_ids=[card_id], operation="duplicate"),
            "selected",
        )
        emit(
            words(
                f"Move {quoted['refined']} right 25 units.",
                f"Reposition {quoted['refined']} rightward by 25 units.",
            ),
            no_action("ambiguous_target"),
            "duplicate_name",
        )
        emit(
            words(
                "Move the selected copy right 60 units.",
                "Shift this chosen copy 60 units rightward.",
            ),
            call("move_shapes", shape_ids=[copy_id], dx=60, dy=0),
            "selected",
        )
        emit(
            words(
                f"Change its text to {quoted['duplicate']}.",
                f"Give the selected copy the label {quoted['duplicate']}.",
            ),
            call("set_text", shape_id=copy_id, text=duplicate),
            "selected",
        )
        box_id = emit(
            words(
                f"Create schema {quoted['schema']} with fields name, class, subjects "
                "and methods getName, getClass, getSubjects.",
                f"Build the {quoted['schema']} class: "
                "properties name, class, subjects; "
                "functions getName, getClass, getSubjects.",
            ),
            call(
                "create_schema_box",
                name=schema,
                fields=["name", "class", "subjects"],
                methods=["getName", "getClass", "getSubjects"],
            ),
            "new",
        )
        emit(
            words(
                f"Add property email to {quoted['schema']}.",
                f"Append an email field to the {quoted['schema']} class.",
            ),
            call("add_property", schema_id=box_id, property_name="email"),
        )
        emit(
            words(
                f"Remove field class from {quoted['schema']}.",
                f"Drop the class property on {quoted['schema']}.",
            ),
            call("remove_property", schema_id=box_id, property_name="class"),
        )
        emit(
            words(
                f"Add method addSubject to {quoted['schema']}.",
                f"Append the addSubject operation to {quoted['schema']}.",
            ),
            call("add_method", schema_id=box_id, method_name="addSubject"),
        )
        emit(
            words(
                f"Remove method getClass from {quoted['schema']}.",
                f"Drop the getClass function on {quoted['schema']}.",
            ),
            call("remove_method", schema_id=box_id, method_name="getClass"),
        )
        emit(
            words(
                f"Rename schema {quoted['schema']} to {quoted['renamed']}.",
                f"Replace the title on the {quoted['schema']} class "
                f"with {quoted['renamed']}.",
            ),
            call("rename_schema", schema_id=box_id, new_name=renamed),
        )
        emit(
            words(
                f"Add property ownerId to {quoted['schema']}.",
                f"Append ownerId to the {quoted['schema']} class.",
            ),
            no_action("missing_target"),
            "stale_name",
        )
        emit(
            words(
                f"Add property ownerId to {quoted['renamed']}.",
                f"Append ownerId to the {quoted['renamed']} class.",
            ),
            call("add_property", schema_id=box_id, property_name="ownerId"),
        )
        emit(
            words(
                f"Connect {quoted['renamed']} to {quoted['refined']} "
                "with label contains.",
                f"Link the source {quoted['renamed']} to destination "
                f"{quoted['refined']}, with contains as the connection label.",
            ),
            call(
                "connect_schemas", source_id=box_id, target_id=card_id, label="contains"
            ),
        )
        emit(
            words("Pan the canvas down.", "Translate the view downward."),
            call("pan_canvas", dx=0, dy=100),
            "canvas",
        )
        emit(
            words(
                "Zoom to fit all shapes.", "Frame the whole drawing in the viewport."
            ),
            call("canvas_command", operation="zoom_to_fit"),
            "canvas",
        )
        emit(
            words(
                f"Select {quoted['refined']} and {quoted['duplicate']}.",
                f"Choose both {quoted['refined']} and {quoted['duplicate']}.",
            ),
            call("select_shapes", shape_ids=sorted([card_id, copy_id])),
        )
        group_id = emit(
            words(
                "Group the selected cards.", "Combine the chosen cards into a group."
            ),
            call(
                "arrange_shapes",
                shape_ids=sorted([card_id, copy_id]),
                operation="group",
            ),
            "selected",
        )
        emit(
            words(
                "Move the selected group right 40 units.",
                "Shift the chosen group 40 units rightward.",
            ),
            call("move_shapes", shape_ids=[group_id], dx=40, dy=0),
            "selected",
        )
        emit(
            words(
                "Make the selected group green.",
                "Apply green recursively to the chosen group.",
            ),
            call("style_shapes", shape_ids=[group_id], color="green"),
            "selected",
        )
        emit(
            words(
                "Ungroup the selected group.",
                "Separate the members of the chosen group.",
            ),
            call("arrange_shapes", shape_ids=[group_id], operation="ungroup"),
            "selected",
        )
        emit(
            words(
                f"Move {quoted['manual']} left 25 units.",
                f"Reposition the manually relabelled {quoted['manual']} "
                "25 units leftward.",
            ),
            call("move_shapes", shape_ids=[card_id], dx=-25, dy=0),
            "manual_rename",
            [{"kind": "text", "id": card_id, "text": manual}],
        )
        emit(
            words(
                f"Delete {quoted['duplicate']}.",
                f"Erase the object labelled {quoted['duplicate']}.",
            ),
            no_action("missing_target"),
            "manual_delete",
            [{"kind": "delete", "id": copy_id}],
        )
        emit(
            words("Undo that last deletion.", "Revert the latest canvas edit."),
            call("canvas_command", operation="undo"),
            "canvas",
        )
        emit(
            words("Redo that deletion.", "Reapply the previously undone edit."),
            call("canvas_command", operation="redo"),
            "canvas",
        )
        emit(
            words("Undo that deletion again.", "Revert the last edit once more."),
            call("canvas_command", operation="undo"),
            "canvas",
        )
        emit(
            words(
                f"Align {quoted['manual']} and {quoted['duplicate']} on the left.",
                f"Give {quoted['manual']} and {quoted['duplicate']} "
                "the same left edge.",
            ),
            call(
                "arrange_shapes",
                shape_ids=sorted([card_id, copy_id]),
                operation="align_left",
            ),
        )
        emit(
            words("Clear the selection.", "Release the chosen objects."),
            call("canvas_command", operation="clear_selection"),
            "canvas",
        )
        if len(rows) - start != 36:
            raise ValueError("The coherent editing block must contain 36 turns.")
        if len(rows) >= turns:
            break
    rows, steps = rows[:turns], steps[:turns]
    # Truncated sessions must retain the state at their final recorded turn.
    final = CanvasSession(initial)
    for turn, step in enumerate(steps):
        final.external(step["before"])
        final.execute(step["command"], step["expected"], f"{group}:created-{turn}")
    return rows, {
        "id": group,
        "split": split,
        "initial_canvas": initial,
        "turns": steps,
        "final_snapshot": final.snapshot(),
        "provenance": (
            "Authored coherent workflow with recorded manual user edits; "
            "oracle actions, no model rollout"
        ),
    }


def audit_quality_supplement(rows, cases, receipts, pairs):
    ids, fingerprints, labels, groups = {}, {}, {}, {}
    for row in rows:
        if row["split"] not in ("train", "valid") or row["id"] in ids:
            raise ValueError(
                "Quality data has an invalid split or duplicate example ID."
            )
        action = validate_call(row["expected"], row["canvas"])
        if action != row["expected"]:
            raise ValueError("Quality labels must use canonical action arguments.")
        fingerprint = accuracy_fingerprint(row)
        label = json.dumps(model_call(action, row["canvas"]), sort_keys=True)
        if fingerprint in labels and labels[fingerprint] != label:
            raise ValueError("The same quality input has conflicting action labels.")
        if fingerprint in fingerprints:
            raise ValueError(
                "Quality data must not contain duplicate full model inputs."
            )
        if row["group"] in groups and groups[row["group"]] != row["split"]:
            raise ValueError("A quality scenario group occurs across splits.")
        if accuracy_capability_issue(row):
            raise ValueError(
                "A quality action requires an unsupported native capability."
            )
        ids[row["id"]], fingerprints[fingerprint], labels[fingerprint] = (
            row,
            row["split"],
            label,
        )
        groups[row["group"]] = row["split"]
    for receipt in receipts:
        session = CanvasSession(receipt["initial_canvas"])
        for event in receipt["events"]:
            if event["kind"] == "outcome":
                session.execute(event["command"], event["action"], event["created_id"])
            else:
                session.external(event["events"])
        row = ids[receipt["id"]]
        if row["canvas"] != session.canvas or row["history"] != session.history:
            raise ValueError(
                f"Counterfactual setup replay differs from captured input: {row['id']}"
            )
        session.execute(row["command"], row["expected"], row["id"] + ":verification")
    for case in cases:
        session = CanvasSession(case["initial_canvas"])
        for turn, step in enumerate(case["turns"]):
            session.external(step["before"])
            row = ids[step["id"]]
            if row["canvas"] != session.canvas or row["history"] != session.history:
                raise ValueError(
                    f"Coherent session replay differs from captured input: {row['id']}"
                )
            if row["command"] != step["command"] or row["expected"] != step["expected"]:
                raise ValueError("Session turns and supervised labels must agree.")
            session.execute(
                step["command"], step["expected"], f"{case['id']}:created-{turn}"
            )
        if session.snapshot() != case["final_snapshot"]:
            raise ValueError(
                "The replayed session does not reach its recorded final state."
            )
    for pair in pairs:
        a, b = (ids[identifier] for identifier in pair["example_ids"])
        if model_call(a["expected"], a["canvas"]) == model_call(
            b["expected"], b["canvas"]
        ):
            raise ValueError("A contrastive pair must teach different action labels.")
    covered = [receipt["id"] for receipt in receipts] + [
        step["id"] for case in cases for step in case["turns"]
    ]
    if Counter(covered) != Counter(ids.keys()):
        raise ValueError(
            "Every supervised quality input requires exactly one replay receipt."
        )
    reversals = [
        {
            "id": row["id"],
            "scenario": row["scenario"],
            "command": row["command"],
            "expected": row["expected"],
            "guarded": guarded,
        }
        for row in rows
        if (
            guarded := execution_guard(
                row["command"], row["expected"], row["canvas"], row["history"]
            )
        )
        != row["expected"]
    ]
    return {
        "splits": dict(Counter(row["split"] for row in rows)),
        "unique_full_inputs": len(fingerprints),
        "duplicate_inputs": 0,
        "conflicting_labels": 0,
        "native_capability_failures": 0,
        "independent_replay_inputs": len(receipts),
        "session_replay_turns": dict(
            Counter(case["split"] for case in cases for _ in case["turns"])
        ),
        "guard_reversals": reversals,
        "guard_reversal_count": len(reversals),
    }


def build_quality_supplement(
    output,
    *,
    seed=72,
    train_pairs=768,
    dev_pairs=96,
    train_sessions=12,
    dev_sessions=3,
    session_turns=72,
):
    if (
        min(train_pairs, dev_pairs) < len(QUALITY_PAIRS)
        or min(train_sessions, dev_sessions) < 1
    ):
        raise ValueError(
            "Both quality splits must include every contrast family "
            "and a coherent session."
        )
    if session_turns < 60 or session_turns > 80:
        raise ValueError("Quality sessions must contain 60 to 80 coherent turns.")
    output.mkdir(parents=True, exist_ok=False)
    rows, cases, receipts, pairs = [], [], [], []
    for split, pair_count, session_count, offset in (
        ("train", train_pairs, train_sessions, 0),
        ("valid", dev_pairs, dev_sessions, 10000),
    ):
        rng = random.Random(seed + offset)
        for index in range(pair_count):
            family = list(QUALITY_PAIRS)[index % len(QUALITY_PAIRS)]
            examples, setup, pair = quality_pair(rng, split, index, family)
            rows.extend(examples)
            receipts.extend(setup)
            pairs.append(pair)
        for index in range(session_count):
            examples, case = quality_session(rng, split, index, session_turns)
            rows.extend(examples)
            cases.append(case)
    audit = audit_quality_supplement(rows, cases, receipts, pairs)
    for filename, values in (
        ("examples.jsonl", rows),
        ("sessions.jsonl", cases),
        ("counterfactual-setups.jsonl", receipts),
        ("contrastive-pairs.jsonl", pairs),
    ):
        (output / filename).write_text(
            "".join(json.dumps(value) + "\n" for value in values)
        )
    (output / "data-audit.json").write_text(json.dumps(audit, indent=2))
    summary = {
        "seed": seed,
        "status": "Generated and audited; no model trained or accuracy measured",
        "splits": audit["splits"],
        "unique_full_inputs": audit["unique_full_inputs"],
        "contrastive_pairs": dict(Counter(pair["split"] for pair in pairs)),
        "pair_scenarios": {
            split: dict(
                Counter(pair["scenario"] for pair in pairs if pair["split"] == split)
            )
            for split in ("train", "valid")
        },
        "sessions": dict(Counter(case["split"] for case in cases)),
        "session_turns": audit["session_replay_turns"],
        "manual_events": dict(
            Counter(
                event["kind"]
                for case in cases
                for step in case["turns"]
                for event in step["before"]
            )
        ),
        "tools": {
            split: dict(
                Counter(
                    row["expected"]["name"] for row in rows if row["split"] == split
                )
            )
            for split in ("train", "valid")
        },
        "generic_drawing_kinds": {
            split: dict(
                Counter(
                    row["expected"]["arguments"]["kind"]
                    for row in rows
                    if row["split"] == split and row["scenario"] == "drawing_kind"
                )
            )
            for split in ("train", "valid")
        },
        "guard_reversals": audit["guard_reversal_count"],
        "simulator_version": SIMULATOR_VERSION,
        "split_provenance": {
            "names": QUALITY_NAMES,
            "train_sentence_slots": [0, 1, 2],
            "development_sentence_slots": [3],
            "test_split": "None; existing v7 holdout not read or changed",
        },
        "recovery_provenance": (
            "Independent counterfactual states use injected executed wrong actions "
            "or rejected (null) outcomes; these are simulated, "
            "not real model rollouts or DAgger"
        ),
        "limitations": [
            "State replay uses Python simulation; pixel geometry, text fitting, "
            "connection bounds, group geometry and zoom-to-fit remain approximate",
            "Sustained sessions contain oracle actions and manual edits; "
            "wrong/rejected agent outcomes are covered by independent "
            "replayable counterfactual cases",
            "Canonical labels and native capability checks "
            "do not establish live model accuracy",
        ],
        "artifact_sha256": {
            name: hashlib.sha256((output / name).read_bytes()).hexdigest()
            for name in (
                "examples.jsonl",
                "sessions.jsonl",
                "counterfactual-setups.jsonl",
                "contrastive-pairs.jsonl",
                "data-audit.json",
            )
        },
        "source_sha256": {
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in ("build_workflow.py", "actions.py", "sessions.py")
        },
    }
    (output / "dataset-summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


LEXICAL_NAMES = {
    "train": [
        "Delivery manifest",
        "Inventory ledger",
        "Route marker",
        "Package tile",
        "Dispatch label",
        "Handover note",
        "Dock frame",
    ],
    "valid": [
        "Reading list",
        "Archive catalogue",
        "Shelf marker",
        "Index tile",
        "Research label",
        "Reading note",
        "Study frame",
    ],
}

# Development uses independently authored sentence structures, not filled-in
# versions of training templates; state variations are counted separately.
LEXICAL_PAIRS = {
    "undo_redo": [
        (
            "I preferred it before that edit. Could you revert the last canvas change?",
            (
                "I have changed my mind about the undo. Could you reapply that "
                "canvas change?"
            ),
        ),
        (
            "Um, please take the drawing back by one edit.",
            "Um, please put back the edit that I just undid.",
        ),
        (
            "That latest change was a mistake; would you reverse it for me?",
            "The change I undid was fine; would you restore that edit for me?",
        ),
        (
            "Please redo the edit. No, instead, undo the most recent edit.",
            "Please undo the edit. No, instead, redo the edit I reversed.",
        ),
        (
            "Return the document to its preceding editing state, please.",
            (
                "Advance the document to the editing state available after an "
                "undo, please."
            ),
        ),
        (
            "Could we roll back the latest alteration to the drawing?",
            "Could we repeat the alteration that was rolled back?",
        ),
    ],
    "undo_availability": [
        ("Could you revert the most recent alteration, please?",) * 2,
        ("That change is not what I wanted. Please undo it.",) * 2,
        ("Um, can you take back the last edit on the canvas?",) * 2,
        ("I would like the previous version of the drawing back, one undo please.",)
        * 2,
        ("Reverse the newest document modification for me, would you?",) * 2,
        ("Please step back once through the document's editing history.",) * 2,
    ],
    "redo_availability": [
        ("Could you reapply the edit that was undone, please?",) * 2,
        ("Actually, I want that undone change back. Please redo it.",) * 2,
        ("Um, can you restore the edit I reversed?",) * 2,
        ("Go forward one edit in the undo history for me, please.",) * 2,
        ("Repeat the document modification that I rolled back, would you?",) * 2,
        ("Please step forward once through the document's editing history.",) * 2,
    ],
    "zoom_pan": [
        (
            "The view feels too close. Could you move the viewpoint further away?",
            (
                "The view is too far left. Could you pan the viewport right by "
                "{distance} page units?"
            ),
        ),
        (
            "Um, please zoom out so the diagram looks smaller.",
            "Um, please slide the canvas view right by {distance} page units.",
        ),
        (
            "Bring the view closer. No, instead, zoom out one step.",
            "Bring the view closer. No, instead, pan right by {distance} page units.",
        ),
        (
            "Could you pull back the view so I can see more of the diagram?",
            "Could you shift the viewport right by {distance} page units for me?",
        ),
        (
            "Give me a wider view by decreasing the magnification, please.",
            "Translate the camera rightward by {distance} page units, please.",
        ),
        (
            "The drawing should appear smaller on screen; take the view farther back.",
            "The camera should travel {distance} page units to the right.",
        ),
    ],
    "zoom_out_in": [
        (
            "Would you take the view further away from the drawing?",
            "Would you bring the view closer to the drawing?",
        ),
        (
            "I need to see more around the diagram, so please zoom out.",
            "I need to see the diagram in more detail, so please zoom in.",
        ),
        (
            "Um, make the drawing look smaller by reducing the zoom.",
            "Um, make the drawing look larger by increasing the zoom.",
        ),
        (
            "Please pull the viewpoint back one zoom step.",
            "Please push the viewpoint in one zoom step.",
        ),
        (
            "Decrease the view magnification by one step, if you could.",
            "Increase the view magnification by one step, if you could.",
        ),
        (
            "Widen what the screen shows by zooming outward.",
            "Narrow what the screen shows by zooming inward.",
        ),
    ],
    "clear_delete": [
        (
            "I am done choosing those shapes. Could you clear the selection?",
            "I am done using those shapes. Could you delete the selected objects?",
        ),
        (
            "Um, please deselect everything that is highlighted.",
            "Um, please remove everything that is highlighted from the drawing.",
        ),
        (
            "Delete the chosen shapes. No, instead, just unselect them.",
            "Unselect the chosen shapes. No, instead, delete them.",
        ),
        (
            "Could you leave the chosen shapes in place and release the selection?",
            "Could you get rid of all the chosen shapes on the canvas?",
        ),
        (
            "Drop the active selection while retaining the objects, please.",
            "Erase the objects in the active selection, please.",
        ),
        (
            "Remove the selection highlight from those objects for me.",
            "Remove those highlighted objects themselves from the document for me.",
        ),
    ],
    "property_remove_add": [
        (
            "We do not need the {field} attribute in {schema}; could you remove it?",
            "We need a new attribute called {new_field} in {schema}; could you add it?",
        ),
        (
            "Um, please drop the {field} property from {schema}.",
            "Um, please give {schema} an additional property called {new_field}.",
        ),
        (
            "Could you take {field} out of the attributes of {schema}?",
            "Could you include {new_field} among the attributes of {schema}?",
        ),
        (
            (
                "Add a field. No, instead, remove the existing {field} field "
                "from {schema}."
            ),
            "Remove a field. No, instead, add a field named {new_field} to {schema}.",
        ),
        (
            (
                "The schema {schema} should stop listing {field} as a member; "
                "please delete that attribute."
            ),
            (
                "The schema {schema} should list one more member; please insert"
                " the attribute {new_field}."
            ),
        ),
        (
            "Prune the {field} field belonging to {schema}, would you?",
            "Extend {schema} with the field {new_field}, would you?",
        ),
    ],
    "missing_field": [
        (
            "Could you remove the attribute {new_field} from {schema}?",
            "Could you create the attribute {new_field} in {schema}?",
        ),
        (
            "Um, drop the {new_field} field from {schema}, please.",
            "Um, add the {new_field} field to {schema}, please.",
        ),
        (
            "I no longer need {new_field} listed in {schema}; take that attribute out.",
            "I need {new_field} listed in {schema}; put that attribute in.",
        ),
        (
            "Please remove the property named {new_field} belonging to {schema}.",
            "Please introduce a property named {new_field} belonging to {schema}.",
        ),
        (
            "Delete {new_field} from the members of {schema}, if it exists.",
            "Insert {new_field} into the members of {schema}.",
        ),
        (
            "Would you subtract the {new_field} attribute from {schema}?",
            "Would you extend {schema} by the {new_field} attribute?",
        ),
    ],
    "direction_bearing": [
        (
            "Could you move {name} towards the {direction} by {distance} page units?",
            "Could you rotate {name} to a bearing of {bearing} degrees?",
        ),
        (
            "Um, translate {name} {distance} page units due {direction}, please.",
            (
                "Um, turn {name} so its orientation has a bearing of {bearing} "
                "degrees, please."
            ),
        ),
        (
            (
                "I need {name} {distance} page units farther {direction}; "
                "please shift it there."
            ),
            (
                "I need {name} facing a bearing of {bearing} degrees; please "
                "rotate it there."
            ),
        ),
        (
            (
                "Move {name} east. No, instead, move it {direction} by "
                "{distance} page units."
            ),
            (
                "Move {name} east. No, instead, rotate it to a bearing of "
                "{bearing} degrees."
            ),
        ),
        (
            (
                "Displace the shape called {name} by {distance} page units in "
                "the {direction} direction."
            ),
            "Orient the shape called {name} at {bearing} degrees bearing.",
        ),
        (
            (
                "Please shift {name} {direction} by exactly {distance} page "
                "units on the page."
            ),
            "Please change the rotational bearing of {name} to {bearing} degrees.",
        ),
    ],
    "singular_plural": [
        (
            "Could you move the chosen shape right by {distance} page units?",
            "Could you move all the chosen shapes right by {distance} page units?",
        ),
        (
            "Um, shift it right by {distance} page units, please.",
            "Um, shift them all right by {distance} page units, please.",
        ),
        (
            "Please nudge this one to the right by {distance} page units.",
            (
                "Please nudge every highlighted object to the right by "
                "{distance} page units."
            ),
        ),
        (
            (
                "Would you move the single selected object {distance} page "
                "units to the right?"
            ),
            "Would you move every selected object {distance} page units to the right?",
        ),
        (
            "Translate the one shape I have chosen rightward by {distance} page units.",
            (
                "Translate the complete set of shapes I have chosen rightward "
                "by {distance} page units."
            ),
        ),
        (
            "Can you push that one {distance} page units rightward for me?",
            "Can you push all of those {distance} page units rightward for me?",
        ),
    ],
    "duplicate_name": [
        (
            (
                "Could you move the shape named {name} towards the west by "
                "{distance} page units?"
            ),
        )
        * 2,
        ("Um, delete the object called {name}, please.",) * 2,
        ("Would you set the caption of {name} to {payload}?",) * 2,
        ("Please shift {name} north by {distance} page units.",) * 2,
        ("Displace the object labelled {name} leftward by {distance} page units.",) * 2,
        ("Erase the object bearing the name {name} for me.",) * 2,
    ],
    "caption_action": [
        (
            "Could you put the words {literal} on {text}?",
            "Could you revert the last document edit?",
        ),
        (
            "Um, change the caption on {text} to exactly {literal}, please.",
            "Um, take back the most recent document change, please.",
        ),
        (
            "Please write {literal} as the text of {text}.",
            "Please undo the latest canvas modification.",
        ),
        (
            "I want {text} to say {literal}; replace its caption with those words.",
            "I want the previous canvas state back; reverse one document edit.",
        ),
        (
            (
                "Replace the lettering belonging to {text} with the literal "
                "caption {literal}."
            ),
            "Roll back a single document change.",
        ),
        (
            (
                "The displayed wording of {text} should be {literal}; set its "
                "text accordingly."
            ),
            "Restore the document state immediately before its latest change.",
        ),
    ],
}

LEXICAL_MICRO_TEMPLATES = [
    [
        "Could you add a rectangle with the caption {card}, please?",
        "Um, make a rectangular box that says {card}.",
        "I need a rectangle labelled {card}; could you draw one?",
        "Please put a new rectangle on the canvas with the words {card}.",
        "Introduce a rectangular shape bearing the text {card} for me.",
        "A new rectangle should display {card}; please create it.",
    ],
    [
        "Could you move the box you added east by {distance} page units?",
        "Um, shift that newly added box towards the east by {distance} page units.",
        (
            "Please move the rectangle you just created {distance} page "
            "units to the right."
        ),
        "The box you made needs to go east by {distance} page units; move it there.",
        (
            "Displace the rectangle from your preceding reply rightward by "
            "{distance} page units."
        ),
        "Translate the most recently created rectangle {distance} page units eastward.",
    ],
    [
        "Could you replace its caption with the exact words {caption}?",
        "Um, I want it to say {caption}; please change its text.",
        "Please give the chosen rectangle the caption {caption}.",
        "The text on it should be {caption}; could you update that?",
        "Set the lettering of that one to the literal string {caption}.",
        "Use {caption} as the text displayed by the current single selection.",
    ],
    [
        (
            "Move it north by {distance}. No, instead, move it west by "
            "{distance} page units."
        ),
        (
            "Um, shift it right by {distance}. I mean, shift it left by "
            "{distance} page units."
        ),
        (
            "Move the chosen shape east by {distance}. Scratch that; move "
            "it west by {distance} page units."
        ),
        (
            "Push it down by {distance}. No, cancel that; push it left by "
            "{distance} page units."
        ),
        (
            "Translate that one north by {distance}. Actually, displace it "
            "west by {distance} page units."
        ),
        (
            "Move it to the east by {distance}. Forget that; move it to the"
            " west by {distance} page units."
        ),
    ],
    [
        "Could you delete the chosen shape now?",
        "Um, please remove it from the canvas.",
        "I no longer need this rectangle; delete it, please.",
        "Please get rid of the object I have selected.",
        "Erase the current single selection from the document.",
        "The highlighted rectangle should be removed; please delete that object.",
    ],
    [
        "That deletion was a mistake. Could you revert it?",
        "Um, I still need that box; undo the deletion, please.",
        "Please take back the deletion and restore the rectangle.",
        "I changed my mind about removing it; could you undo that?",
        "Reverse the preceding deletion so the removed object returns.",
        "Roll back the removal you just performed.",
    ],
    [
        (
            "Could you move the restored box {caption} towards the south by"
            " {distance} page units?"
        ),
        "Um, shift {caption} down by {distance} page units now that it is back.",
        (
            "Please move {caption}, the rectangle we recovered, south by "
            "{distance} page units."
        ),
        (
            "The recovered rectangle called {caption} needs to go down by "
            "{distance} page units."
        ),
        (
            "Displace the recovered object labelled {caption} southward by "
            "{distance} page units."
        ),
        (
            "Translate the restored object {caption} exactly {distance} "
            "page units downward."
        ),
    ],
    [
        "Could you reapply the edit that we undid?",
        "Um, redo that undone deletion, please.",
        "Please go forward to the edit that was reversed.",
        "I want the undone change back; could you redo it?",
        "Repeat the modification that was rolled back earlier.",
        "Advance the document by redoing the reversed edit.",
    ],
    [
        "Could you make the box you added blue, please?",
        "Um, color the rectangle you created blue.",
        "Please give the box you made a blue color.",
        "The rectangle you added should be blue; update its color.",
        "Apply blue coloring to the most recently created rectangle.",
        "Set the color of your newly created box to blue.",
    ],
    [
        "Could you move {caption} east by {distance} page units?",
        "Um, push the rectangle named {caption} right by {distance} page units.",
        (
            "Please shift the object called {caption} towards the east by "
            "{distance} page units."
        ),
        "I need {caption} {distance} page units farther right; please move it.",
        "Displace the object labelled {caption} eastward by {distance} page units.",
        (
            "Translate the object bearing the name {caption} exactly "
            "{distance} page units rightward."
        ),
    ],
    [
        (
            "I changed its caption myself. Could you move {manual} east by "
            "{distance} page units?"
        ),
        (
            "Um, its current label is {manual}; shift that box right by "
            "{distance} page units."
        ),
        (
            "The box is now called {manual} after my edit; please move it "
            "east by {distance} page units."
        ),
        (
            "Please move {manual}, the rectangle I renamed, right by "
            "{distance} page units."
        ),
        (
            "Following my manual rename, translate {manual} eastward by "
            "{distance} page units."
        ),
        (
            "Displace the rectangle with the current name {manual} "
            "rightward by {distance} page units."
        ),
    ],
    [
        "Could you take the view further away so I can see the whole area better?",
        "Um, please pull the view back by one zoom step.",
        "I need a wider view around the drawing; please zoom out.",
        "Please make the diagram appear smaller by reducing the zoom.",
        "Decrease the camera magnification for a wider view of the document.",
        "Take the viewpoint farther back from the drawing by zooming outward.",
    ],
]


def lexical_context(rng, split, group):
    session, ids = quality_context(rng, split, group)
    for key, name in zip(ids, LEXICAL_NAMES[split], strict=True):
        shape = session.objects()[ids[key]]
        shape["name"] = f"{name} {group.rsplit('-', 1)[-1]}"
        if "text" in shape:
            shape["text"] = shape["name"]
    return session, ids


def lexical_pair(rng, split, index, family):
    group = f"lexical-{split}-pair-{index}"
    base, ids = lexical_context(rng, split, group)
    sessions = [copy.deepcopy(base), copy.deepcopy(base)]
    receipts = [
        {
            "id": f"{group}:{side}",
            "initial_canvas": copy.deepcopy(base.canvas),
            "events": [],
        }
        for side in range(2)
    ]
    cycle = index // len(LEXICAL_PAIRS)
    slot = cycle % 4 if split == "train" else 4 + cycle % 2
    templates = LEXICAL_PAIRS[family][slot]
    distance = rng.choice([25, 40, 60, 80, 120])
    direction, dx, dy, bearing = (
        ("east", distance, 0, 90),
        ("west", -distance, 0, 270),
        ("north", 0, -distance, 0),
        ("south", 0, distance, 180),
    )[(cycle // 4) % 4]
    a, b, schema = ids["geo"], ids["other-geo"], ids["box"]
    objects = base.objects()
    literal = (
        "Revert and reapply; choose it",
        "Add attributes and remove subjects",
        "Move west and delete the chosen shape",
        "The box you added; undo and redo",
        "Rotate to a bearing and remove the selection",
        "Pan towards the east and zoom out",
    )[slot]
    if cycle % 4 == 0:
        literal = objects[a]["name"]
    values = {
        "name": json.dumps(objects[a]["name"]),
        "schema": json.dumps(objects[schema]["name"]),
        "text": json.dumps(objects[ids["text"]]["name"]),
        "field": json.dumps("subjects"),
        "new_field": json.dumps(
            objects[a]["name"]
            if cycle % 4 == 2
            else "deliveryZone"
            if split == "train"
            else "shelfCode"
        ),
        "payload": json.dumps(
            f"{'Departures' if split == 'train' else 'References'} {index}"
        ),
        "literal": json.dumps(literal),
        "direction": direction,
        "distance": distance,
        "bearing": bearing,
    }
    commands = [template.format(**values) for template in templates]
    reference = "named"
    missing, ambiguous, unsupported = (
        call("no_action", reason=reason)
        for reason in ("missing_target", "ambiguous_target", "unsupported_request")
    )

    def outcome(side, command, action, *, wrong=False):
        created = quality_setup_outcome(sessions[side], receipts[side], command, action)
        receipts[side]["events"][-1]["outcome_class"] = (
            "rejected"
            if action is None
            else "simulated_wrong_action"
            if wrong
            else "oracle_setup_action"
        )
        return created

    def both_outcome(command, action, *, wrong=False):
        for side in range(2):
            outcome(side, command, action, wrong=wrong)

    if family == "undo_redo":
        both_outcome(
            f"Move {values['name']} right by {distance} page units.",
            call("move_shapes", shape_ids=[b], dx=distance, dy=0),
            wrong=True,
        )
        both_outcome(
            f"Move {values['name']} down by {distance} page units.",
            call("move_shapes", shape_ids=[a], dx=0, dy=distance),
        )
        both_outcome("Undo the latest edit.", call("canvas_command", operation="undo"))
        actions = [call("canvas_command", operation=op) for op in ("undo", "redo")]
        reference = "actual_history"
    elif family == "undo_availability":
        prior = f"Delete {values['name']}."
        outcome(0, prior, call("delete_shapes", shape_ids=[b]), wrong=True)
        outcome(1, prior, None)
        actions = [call("canvas_command", operation="undo"), missing]
        reference = "actual_history"
    elif family == "redo_availability":
        both_outcome(
            f"Move {values['name']} right by {distance} page units.",
            call("move_shapes", shape_ids=[a], dx=distance, dy=0),
        )
        both_outcome("Undo that edit.", call("canvas_command", operation="undo"))
        quality_setup_manual(
            sessions[1],
            receipts[1],
            [{"kind": "move", "id": b, "dx": 0, "dy": distance}],
        )
        actions = [call("canvas_command", operation="redo"), missing]
        reference = "actual_history"
    elif family == "zoom_pan":
        both_outcome(
            f"Pan the view right by {distance} page units.",
            call("canvas_command", operation="zoom_in"),
            wrong=True,
        )
        actions = [
            call("canvas_command", operation="zoom_out"),
            call("pan_canvas", dx=distance, dy=0),
        ]
        reference = "viewport"
    elif family == "zoom_out_in":
        actions = [
            call("canvas_command", operation=op) for op in ("zoom_out", "zoom_in")
        ]
        reference = "viewport"
    elif family == "clear_delete":
        both_outcome(
            "Select both drawing shapes.",
            call("select_shapes", shape_ids=sorted([a, b])),
        )
        actions = [
            call("canvas_command", operation="clear_selection"),
            call("delete_shapes", shape_ids=sorted([a, b])),
        ]
        reference = "plural_selected"
    elif family in ("property_remove_add", "missing_field"):
        field = json.loads(values["new_field"])
        both_outcome(
            f"Please add the attribute {values['new_field']} to {values['schema']}.",
            None,
        )
        actions = [
            call("remove_property", schema_id=schema, property_name="subjects")
            if family == "property_remove_add"
            else missing,
            call("add_property", schema_id=schema, property_name=field),
        ]
        reference = "current_schema_attributes"
    elif family == "direction_bearing":
        both_outcome(
            f"Move {values['name']} right by {distance} page units.",
            call("move_shapes", shape_ids=[b], dx=distance, dy=0),
            wrong=True,
        )
        if slot == 3:
            # A cancelled named clause does not resolve the final singular 'it'.
            for side in range(2):
                quality_setup_manual(
                    sessions[side], receipts[side], [{"kind": "select", "ids": [a]}]
                )
        actions = [call("move_shapes", shape_ids=[a], dx=dx, dy=dy), unsupported]
    elif family == "singular_plural":
        targets = sorted([a, b])
        both_outcome(
            f"Move {values['name']} right by {distance} page units.",
            call("move_shapes", shape_ids=[b], dx=distance, dy=0),
            wrong=True,
        )
        both_outcome(
            "Choose the two drawing shapes.", call("select_shapes", shape_ids=targets)
        )
        both_outcome("Move the chosen shape east, please.", None)
        selected = targets if cycle % 3 == 0 else [a]
        quality_setup_manual(
            sessions[0], receipts[0], [{"kind": "select", "ids": selected}]
        )
        if cycle % 3:
            templates = (templates[0], templates[0])
            commands = [commands[0], commands[0]]
            quality_setup_manual(
                sessions[1],
                receipts[1],
                [{"kind": "select", "ids": targets if cycle % 3 == 1 else []}],
            )
        actions = [
            call("move_shapes", shape_ids=selected, dx=distance, dy=0)
            if len(selected) == 1
            else ambiguous,
            ambiguous
            if cycle % 3
            else call("move_shapes", shape_ids=targets, dx=distance, dy=0),
        ]
        reference = "singular_vs_plural_selected"
    elif family == "duplicate_name":
        prior = f"Rename {json.dumps(objects[b]['name'])} to {values['payload']}."
        outcome(0, prior, None)
        outcome(
            1, prior, call("set_text", shape_id=b, text=objects[a]["name"]), wrong=True
        )
        action = (
            call("delete_shapes", shape_ids=[a])
            if slot in (1, 5)
            else call("set_text", shape_id=a, text=json.loads(values["payload"]))
            if slot == 2
            else call(
                "move_shapes",
                shape_ids=[a],
                dx=0 if slot == 3 else -distance,
                dy=-distance if slot == 3 else 0,
            )
        )
        actions = [action, ambiguous]
        reference = "unique_vs_duplicate_current_name"
    elif family == "caption_action":
        both_outcome(
            f"Delete {json.dumps(objects[b]['name'])}.",
            call("delete_shapes", shape_ids=[b]),
        )
        actions = [
            call("set_text", shape_id=ids["text"], text=literal),
            call("canvas_command", operation="undo"),
        ]
        reference = "quoted_payload_vs_history"
    else:
        raise ValueError(f"Unknown lexical pair family: {family}")
    rows = []
    for side, (session, command, action, template) in enumerate(
        zip(sessions, commands, actions, templates, strict=True)
    ):
        row = quality_row(
            session,
            command,
            action,
            identifier=f"{group}:{side}",
            group=group,
            split=split,
            family=family,
            wording=f"lexical-{split}-{family}-{slot}-{side}",
            reference=reference,
            rationale=(
                f"Authored {family} semantic contrast; "
                "current state and actual recorded outcomes determine "
                "the single action."
            ),
        )
        row["sentence_template"] = template
        row["provenance"] = (
            "Authored lexical refinement; simulated state and actual recorded outcomes"
        )
        rows.append(row)
    pair = {
        "id": group,
        "split": split,
        "scenario": family,
        "example_ids": [row["id"] for row in rows],
        "same_canvas_and_history": rows[0]["canvas"] == rows[1]["canvas"]
        and rows[0]["history"] == rows[1]["history"],
        "same_model_canvas_and_history": messages_for(
            "", rows[0]["canvas"], rows[0]["history"]
        )
        == messages_for("", rows[1]["canvas"], rows[1]["history"]),
        "same_command": commands[0] == commands[1],
        "wording_families": [row["wording_family"] for row in rows],
    }
    return rows, receipts, pair


def lexical_micro_conversation(rng, split, index):
    group = f"lexical-{split}-conversation-{index}"
    session, ids = lexical_context(rng, split, group)
    session.external([{"kind": "select", "ids": []}])
    initial = copy.deepcopy(session.canvas)
    slot = index % 4 if split == "train" else 4 + index % 2
    distance = rng.choice([25, 40, 60, 80, 120])
    card = f"{'Courier job' if split == 'train' else 'Library job'} {index}"
    caption = (
        f"{'Revert and delete' if split == 'train' else 'Zoom and reapply'} {index}"
    )
    manual_name = "Hand checked delivery" if split == "train" else "Hand indexed volume"
    manual = f"{manual_name} {index}"
    values = {
        "card": json.dumps(card),
        "caption": json.dumps(caption),
        "manual": json.dumps(manual),
        "distance": distance,
    }
    rows, steps = [], []
    created = f"{group}:created-0"
    actions = [
        call("create_shape", kind="rectangle", text=card),
        call("move_shapes", shape_ids=[created], dx=distance, dy=0),
        call("set_text", shape_id=created, text=caption),
        call("move_shapes", shape_ids=[created], dx=-distance, dy=0),
        call("delete_shapes", shape_ids=[created]),
        call("canvas_command", operation="undo"),
        call("move_shapes", shape_ids=[created], dx=0, dy=distance),
        call("no_action", reason="missing_target"),
        call("style_shapes", shape_ids=[created], color="blue"),
        call("no_action", reason="missing_target"),
        call("move_shapes", shape_ids=[created], dx=distance, dy=0),
        call("canvas_command", operation="zoom_out"),
    ]
    references = [
        "creation",
        "last_created",
        "singular_selected",
        "singular_selected",
        "singular_selected",
        "actual_history",
        "restored_current_name",
        "unavailable_redo",
        "last_created_overrides_selection",
        "stale_name",
        "manual_current_name",
        "viewport",
    ]
    for turn, action in enumerate(actions):
        before = (
            [{"kind": "select", "ids": [ids["other-geo"]]}]
            if turn == 8
            else [{"kind": "text", "id": created, "text": manual}]
            if turn == 9
            else []
        )
        session.external(before)
        template = LEXICAL_MICRO_TEMPLATES[turn][slot]
        command = template.format(**values)
        row = quality_row(
            session,
            command,
            action,
            identifier=f"{group}:{turn}",
            group=group,
            split=split,
            family="micro_conversation",
            wording=f"lexical-{split}-conversation-{slot}-{turn}",
            reference=references[turn],
            rationale=(
                "One conversational edit grounded in the reached state: "
                "creation, correction, deletion recovery, cleared redo history "
                "and manual rename are replayed."
            ),
        )
        row["sentence_template"] = template
        row["provenance"] = (
            "Authored twelve-turn conversation; oracle actions and recorded"
            " manual user edits"
        )
        rows.append(row)
        steps.append(
            {
                "id": row["id"],
                "command": command,
                "expected": row["expected"],
                "before": before,
            }
        )
        session.execute(command, row["expected"], f"{group}:created-{turn}")
    return rows, {
        "id": group,
        "split": split,
        "initial_canvas": initial,
        "turns": steps,
        "final_snapshot": session.snapshot(),
        "provenance": rows[0]["provenance"],
    }


def audit_lexical_tokenization(rows, tokenizer_path):
    from transformers import AutoTokenizer

    from actions import TOOLS, parse_call
    from dataset import training_row
    from lab import ToolPrefixEncoder, _task_hash

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), local_files_only=True
    )
    encoder = ToolPrefixEncoder(tokenizer)
    lengths, completions, checked_tools = [], [], set()
    for row in rows:
        messages = training_row(row)["messages"]
        full = encoder.encode(messages)
        prompt = encoder.encode(messages[:-1], generation=True)
        if full[: len(prompt)] != prompt:
            raise ValueError(f"Lexical training prompt boundary differs: {row['id']}")
        parsed = parse_call(tokenizer.decode(full[len(prompt) :]), row["canvas"])
        if parsed != row["expected"]:
            raise ValueError(
                f"Lexical function-call token round trip differs: {row['id']}"
            )
        tool = row["expected"]["name"]
        if tool not in checked_tools:
            for content, cached, generation in (
                (messages, full, False),
                (messages[:-1], prompt, True),
            ):
                official = tokenizer.apply_chat_template(
                    content,
                    tools=TOOLS,
                    add_generation_prompt=generation,
                    tokenize=True,
                    return_dict=False,
                )
                if official != cached:
                    raise ValueError(
                        "Cached lexical encoding differs from the official tool "
                        "template."
                    )
            checked_tools.add(tool)
        if len(full) > 6144 or len(full) - len(prompt) > 160:
            raise ValueError(
                f"Lexical row exceeds the training token limits: {row['id']}"
            )
        lengths.append(len(full))
        completions.append(len(full) - len(prompt))
    return {
        "round_trips": len(rows),
        "failures": 0,
        "max_tokens": max(lengths),
        "max_completion_tokens": max(completions),
        "task_sha256": _task_hash(),
        "official_template_tools_checked": sorted(checked_tools),
        "tokenizer_sha256": {
            filename: hashlib.sha256(
                (tokenizer_path / filename).read_bytes()
            ).hexdigest()
            for filename in (
                "tokenizer.json",
                "tokenizer_config.json",
                "chat_template.jinja",
            )
        },
        "execution": "CPU tokenizer only; local cache, no model load or inference",
    }


def lexical_command_key(command):
    return re.sub(r"[.!?]+$", "", " ".join(command.casefold().split())).rstrip()


def build_lexical_refinement(
    output,
    *,
    seed=72,
    train_pairs=480,
    dev_pairs=76,
    train_conversations=20,
    dev_conversations=4,
    tokenizer_path=None,
):
    if dev_pairs < 6 * len(LEXICAL_PAIRS):
        raise ValueError(
            "Both lexical splits must cover every family with independent "
            "sentence structures."
        )
    if (
        train_pairs < 12 * len(LEXICAL_PAIRS)
        or train_conversations < 4
        or dev_conversations < 2
    ):
        raise ValueError(
            "Lexical data must cover all authored training/development "
            "sentence structures."
        )
    if output.exists():
        raise FileExistsError(output)
    rows, cases, receipts, pairs = [], [], [], []
    for split, pair_count, conversation_count, offset in (
        ("train", train_pairs, train_conversations, 0),
        ("valid", dev_pairs, dev_conversations, 10000),
    ):
        rng = random.Random(seed + offset)
        for index in range(pair_count):
            family = list(LEXICAL_PAIRS)[index % len(LEXICAL_PAIRS)]
            examples, setup, pair = lexical_pair(rng, split, index, family)
            rows.extend(examples)
            receipts.extend(setup)
            pairs.append(pair)
        for index in range(conversation_count):
            examples, case = lexical_micro_conversation(rng, split, index)
            rows.extend(examples)
            cases.append(case)
    templates = {
        split: {row["sentence_template"] for row in rows if row["split"] == split}
        for split in ("train", "valid")
    }
    if templates["train"] & templates["valid"]:
        raise ValueError(
            "Lexical training and development sentence structures must be disjoint."
        )
    commands = {
        split: {
            lexical_command_key(row["command"]) for row in rows if row["split"] == split
        }
        for split in ("train", "valid")
    }
    if commands["train"] & commands["valid"]:
        raise ValueError("Lexical training and development command sentences overlap.")
    audit = audit_quality_supplement(rows, cases, receipts, pairs)
    if audit["guard_reversal_count"]:
        raise ValueError(
            f"Lexical execution guard reversals: {json.dumps(audit['guard_reversals'])}"
        )
    tokenizer_path = (
        tokenizer_path or Path(__file__).parent / "runs/base-bb327a9a-float16"
    )
    token_audit = audit_lexical_tokenization(rows, tokenizer_path)
    output.mkdir(parents=True, exist_ok=False)
    for filename, values in (
        ("examples.jsonl", rows),
        ("sessions.jsonl", cases),
        ("counterfactual-setups.jsonl", receipts),
        ("contrastive-pairs.jsonl", pairs),
    ):
        (output / filename).write_text(
            "".join(json.dumps(value) + "\n" for value in values)
        )
    (output / "data-audit.json").write_text(json.dumps(audit, indent=2))
    (output / "tokenization-audit.json").write_text(json.dumps(token_audit, indent=2))
    instruction_syntax = [
        (row["split"], re.sub(r'"(?:\\.|[^"\\])*"', " ", row["command"]))
        for row in rows
    ]
    summary = {
        "seed": seed,
        "status": (
            "Generated, replayed and CPU-token audited; no model trained or"
            " accuracy measured"
        ),
        "splits": audit["splits"],
        "unique_full_inputs": audit["unique_full_inputs"],
        "unique_command_sentences": {split: len(commands[split]) for split in commands},
        "contrastive_pairs": dict(Counter(pair["split"] for pair in pairs)),
        "micro_conversations": dict(Counter(case["split"] for case in cases)),
        "conversation_turns": audit["session_replay_turns"],
        "pair_scenarios": {
            split: dict(
                Counter(pair["scenario"] for pair in pairs if pair["split"] == split)
            )
            for split in ("train", "valid")
        },
        "authored_sentence_structures": {
            split: len(templates[split]) for split in templates
        },
        "wording_families": {
            split: len({row["wording_family"] for row in rows if row["split"] == split})
            for split in ("train", "valid")
        },
        "tools": {
            split: dict(
                Counter(
                    row["expected"]["name"] for row in rows if row["split"] == split
                )
            )
            for split in ("train", "valid")
        },
        "semantic_terms": {
            term: dict(
                Counter(
                    row["split"]
                    for row in rows
                    if re.search(rf"\b{term}\b", row["command"], re.I)
                )
            )
            for term in (
                "revert",
                "reapply",
                "further away",
                "chosen shape",
                "towards",
                "bearing",
                "attributes",
            )
        },
        "instruction_semantic_terms": {
            term: dict(
                Counter(
                    split
                    for split, syntax in instruction_syntax
                    if re.search(rf"\b{term}\b", syntax, re.I)
                )
            )
            for term in (
                "revert",
                "reapply",
                "further away",
                "chosen shape",
                "towards",
                "bearing",
                "attributes",
            )
        },
        "setup_outcomes": dict(
            Counter(
                event["outcome_class"]
                for receipt in receipts
                for event in receipt["events"]
                if event["kind"] == "outcome"
            )
        ),
        "guard_reversals": audit["guard_reversal_count"],
        "tokenization": token_audit,
        "split_provenance": {
            "names": LEXICAL_NAMES,
            "train_sentence_slots": [0, 1, 2, 3],
            "development_sentence_slots": [4, 5],
            "command_surface_normalization": (
                "Casefold, collapse whitespace and strip terminal .!? punctuation"
            ),
            "test_use": False,
            "reserved_data": (
                "Existing development and test files were not read or changed"
            ),
        },
        "simulator_version": SIMULATOR_VERSION,
        "limitations": [
            (
                "Counterfactual prior outcomes are simulated, not real model "
                "rollouts or DAgger"
            ),
            (
                "Micro-conversations use oracle actions and manual edits; "
                "current-state replay is not live model accuracy"
            ),
            (
                "Geometry uses the existing simulator; pixel fitting, "
                "group/connection bounds and zoom-to-fit remain approximate"
            ),
        ],
        "artifact_sha256": {
            name: hashlib.sha256((output / name).read_bytes()).hexdigest()
            for name in (
                "examples.jsonl",
                "sessions.jsonl",
                "counterfactual-setups.jsonl",
                "contrastive-pairs.jsonl",
                "data-audit.json",
                "tokenization-audit.json",
            )
        },
        "source_sha256": {
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in (
                "build_workflow.py",
                "actions.py",
                "sessions.py",
                "dataset.py",
                "lab.py",
            )
        },
    }
    (output / "dataset-summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Generate general canvas instruction and editing-session data."
    )
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=64)
    parser.add_argument("--independent", type=int, default=80000)
    parser.add_argument("--sessions", type=int, default=4000)
    parser.add_argument("--replay", type=int, default=12000)
    parser.add_argument("--spoken-refinement", type=int, default=0)
    parser.add_argument("--accuracy-refinement", type=int, default=0)
    parser.add_argument("--quality-supplement", action="store_true")
    parser.add_argument("--lexical-refinement", action="store_true")
    parser.add_argument("--practice-sessions", type=int, default=800)
    args = parser.parse_args()
    if args.lexical_refinement:
        if (
            args.quality_supplement
            or args.accuracy_refinement
            or args.spoken_refinement
        ):
            parser.error("Use one refinement mode at a time.")
        build_lexical_refinement(args.output, seed=args.seed)
        return
    if args.quality_supplement:
        if args.accuracy_refinement or args.spoken_refinement:
            parser.error("Use one refinement mode at a time.")
        build_quality_supplement(args.output, seed=args.seed)
        return
    if args.source is None:
        parser.error("--source is required for existing workflow/refinement modes.")
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
