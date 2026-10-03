import json
import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Name = Annotated[str, Field(min_length=1, max_length=100)]
Names = Annotated[list[Name], Field(max_length=30)]
Number = Annotated[float, Field(ge=-100000, le=100000)]
Size = Annotated[float, Field(gt=0, le=10000)]
Color = Literal[
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
Fill = Literal["none", "semi", "solid", "pattern"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class SchemaBox(StrictModel):
    id: Name
    name: Name
    properties: Names = Field(default_factory=list)
    methods: Names = Field(default_factory=list)
    x: Number = 0
    y: Number = 0
    w: Size = 280
    h: Size = 200
    color: Color = "black"
    fill: Fill = "none"
    opacity: float = Field(default=1, ge=0, le=1)
    parent_id: Name | None = None
    rotation: Number = 0
    order: int = 0


class CanvasShape(StrictModel):
    id: Name
    name: Name
    kind: str = Field(max_length=30)
    x: Number = 0
    y: Number = 0
    w: Size = 160
    h: Size = 100
    text: str = Field(default="", max_length=1000)
    color: Color = "black"
    fill: Fill = "none"
    opacity: float = Field(default=1, ge=0, le=1)
    parent_id: Name | None = None
    rotation: Number = 0
    order: int = 0


class Camera(StrictModel):
    x: Number = 0
    y: Number = 0
    z: float = Field(default=1, gt=0, le=16)


class Canvas(StrictModel):
    schemas: list[SchemaBox] = Field(default_factory=list, max_length=30)
    shapes: list[CanvasShape] = Field(default_factory=list, max_length=60)
    selected_ids: list[Name] = Field(default_factory=list, max_length=90)
    can_undo: bool = False
    can_redo: bool = False
    camera: Camera = Field(default_factory=Camera)

    @model_validator(mode="after")
    def check_ids(self):
        ids = {shape.id for shape in [*self.schemas, *self.shapes]}
        if len(ids) != len(self.schemas) + len(self.shapes):
            raise ValueError("Canvas IDs must be unique.")
        if len(set(self.selected_ids)) != len(self.selected_ids):
            raise ValueError("Selected IDs must be unique.")
        if any(identifier not in ids for identifier in self.selected_ids):
            raise ValueError("Selection contains an unknown canvas ID.")
        if any(
            shape.parent_id not in ids
            for shape in [*self.schemas, *self.shapes]
            if shape.parent_id
        ):
            raise ValueError("A shape parent is absent from the canvas.")
        return self


class CreateSchema(StrictModel):
    name: Name = Field(description="Schema name, preserving the user's spelling.")
    fields: Names = Field(
        description="Requested properties in order; empty if omitted."
    )
    methods: Names = Field(description="Requested methods in order; empty if omitted.")


class AddProperty(StrictModel):
    schema_id: Name = Field(description="ID of an existing schema in the canvas.")
    property_name: Name = Field(description="Property to add.")


class RemoveProperty(StrictModel):
    schema_id: Name = Field(description="ID of an existing schema in the canvas.")
    property_name: Name = Field(description="Existing property to remove.")


class RenameSchema(StrictModel):
    schema_id: Name = Field(description="ID of an existing schema in the canvas.")
    new_name: Name = Field(description="Requested new schema name.")


class ConnectSchemas(StrictModel):
    source_id: Name = Field(description="ID of the existing source shape or schema.")
    target_id: Name = Field(
        description="ID of the existing destination shape or schema."
    )
    label: str = Field(
        max_length=100, description="Requested connection label; empty if omitted."
    )


class NoAction(StrictModel):
    reason: Literal["missing_target", "ambiguous_target", "unsupported_request"] = (
        Field(description="Why the command cannot be executed.")
    )


class ShapeTargets(StrictModel):
    shape_ids: Names = Field(min_length=1)


class SelectShapes(StrictModel):
    shape_ids: Names


class MoveShapes(ShapeTargets):
    dx: Number
    dy: Number


class ResizeShape(StrictModel):
    shape_id: Name
    width: Size
    height: Size


class CreateShape(StrictModel):
    kind: Literal[
        "rectangle", "ellipse", "diamond", "triangle", "text", "note", "frame", "arrow"
    ]
    text: str = Field(default="", max_length=1000)
    x: Number | None = None
    y: Number | None = None
    width: Size = 160
    height: Size = 100


class SetText(StrictModel):
    shape_id: Name
    text: str = Field(max_length=1000)


class StyleShapes(ShapeTargets):
    color: Color | None = None
    fill: Fill | None = None
    opacity: float | None = Field(default=None, ge=0, le=1)


class ArrangeShapes(ShapeTargets):
    operation: Literal[
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


class CanvasCommand(StrictModel):
    operation: Literal[
        "undo",
        "redo",
        "select_all",
        "clear_selection",
        "zoom_in",
        "zoom_out",
        "zoom_to_fit",
        "reset_zoom",
    ]


class PanCanvas(StrictModel):
    dx: Number
    dy: Number


class SchemaMethod(StrictModel):
    schema_id: Name
    method_name: Name


ACTION_MODELS = {
    "create_schema_box": (
        CreateSchema,
        "Create a schema box with properties and methods.",
    ),
    "add_property": (AddProperty, "Add one property to an existing schema."),
    "remove_property": (RemoveProperty, "Remove one property from an existing schema."),
    "rename_schema": (RenameSchema, "Rename an existing schema."),
    "connect_schemas": (
        ConnectSchemas,
        "Connect two existing shapes or schemas with a bound arrow.",
    ),
    "no_action": (
        NoAction,
        "Do nothing for an unclear target or an unsupported request.",
    ),
    "create_shape": (
        CreateShape,
        "Create a drawing shape, text, note, frame, or arrow.",
    ),
    "select_shapes": (
        SelectShapes,
        "Select shapes by ID; an empty list clears selection.",
    ),
    "move_shapes": (
        MoveShapes,
        "Move shapes by page-coordinate offsets. Left/up are negative.",
    ),
    "delete_shapes": (ShapeTargets, "Delete existing shapes."),
    "resize_shape": (ResizeShape, "Set a shape's width and height."),
    "set_text": (SetText, "Replace text or the label of a shape."),
    "style_shapes": (
        StyleShapes,
        "Change color, fill, or opacity; null preserves a style.",
    ),
    "arrange_shapes": (
        ArrangeShapes,
        "Duplicate, group, order, align, or arrange shapes.",
    ),
    "canvas_command": (
        CanvasCommand,
        "Undo, redo, select all, clear selection, or zoom.",
    ),
    "pan_canvas": (
        PanCanvas,
        "Pan the viewport; positive offsets move the view right/down.",
    ),
    "add_method": (SchemaMethod, "Add a method to an existing schema."),
    "remove_method": (SchemaMethod, "Remove an existing schema method."),
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": model.model_json_schema(),
        },
    }
    for name, (model, description) in ACTION_MODELS.items()
]

SYSTEM_PROMPT = (
    "You are a model that can do function calling with the following functions. "
    "Return exactly one function call and no explanation. "
    "Use the short IDs shown in Canvas. A named target overrides selection. "
    "For singular 'it', 'this', 'that box', use selected_ids only when single. "
    "No selection or multiple selections means ambiguous_target; "
    "an empty canvas or absent name or field means missing_target. "
    "Never guess a missing or ambiguous target. "
    "Preserve names, field spelling, and list order. "
    "Do not invent unrequested properties or methods. "
    "Canvas is the current state; Recent contains at most three actual outcomes. "
    "For 'last created' use last_created_id; for 'last edited' use last_edited_id. "
    "An old name in Recent is not a current named target. "
    "Use shapes and schemas from Canvas; named targets override selection. "
    "For plural selected shapes use every selected_id; single-shape edits require one. "
    "Default movement and pan distance is 100 page units; right/down are positive. "
    "Default drawing size is 160 by 100; x/y null places a new shape in the view. "
    "Text height and note dimensions are fitted by the editor. "
    "Notes cannot be resized; text resizes uniformly. "
    "Fill is supported only by schemas, geometric shapes, and arrows. "
    "Frames support opacity only; text and notes support color and opacity. "
    "Group styles apply recursively to child shapes. "
    "Requests for unsupported shape capabilities are unsupported_request. "
    "A target name must match exactly one current shape; "
    "duplicate names are ambiguous. "
    "Undo and redo require can_undo and can_redo respectively. "
    "Requests for information, explanations, code execution, or multiple edits are "
    "unsupported_request. A polite question requesting one supported edit is an "
    "action request; execute it. For a correction, execute only the final "
    "requested edit."
)
INPUT_FORMAT_VERSION = "general-canvas-three-outcomes-v3"


def model_canvas(canvas: dict) -> tuple[dict, dict]:
    context = Canvas.model_validate(canvas)
    identifiers = {schema.id: f"box{i + 1}" for i, schema in enumerate(context.schemas)}
    identifiers.update(
        {shape.id: f"shape{i + 1}" for i, shape in enumerate(context.shapes)}
    )
    selected = [identifiers[s] for s in context.selected_ids if s in identifiers]
    status = "single" if len(selected) == 1 else "multiple" if selected else "none"
    return {
        "schemas": [
            {
                "id": identifiers[s.id],
                "name": s.name,
                "fields": s.properties,
                "methods": s.methods,
                "x": s.x,
                "y": s.y,
                "w": s.w,
                "h": s.h,
                "color": s.color,
                "fill": s.fill,
                "opacity": s.opacity,
                "parent_id": identifiers.get(s.parent_id),
                "rotation": s.rotation,
                "order": s.order,
            }
            for s in context.schemas
        ],
        "shapes": [
            {
                **s.model_dump(),
                "id": identifiers[s.id],
                "parent_id": identifiers.get(s.parent_id),
            }
            for s in context.shapes
        ],
        "selection": status,
        "selected_ids": selected,
        "can_undo": context.can_undo,
        "can_redo": context.can_redo,
        "camera": context.camera.model_dump(),
    }, identifiers


def model_call(call: dict, canvas: dict) -> dict:
    _, identifiers = model_canvas(canvas)
    arguments = dict(call["arguments"])
    for key in ("schema_id", "source_id", "target_id", "shape_id"):
        if key in arguments:
            arguments[key] = identifiers[arguments[key]]
    if "shape_ids" in arguments:
        arguments["shape_ids"] = [
            identifiers[value] for value in arguments["shape_ids"]
        ]
    return {"name": call["name"], "arguments": arguments}


def recent_context(history, identifiers):
    if isinstance(history, dict):
        return {
            **recent_context(history.get("turns", []), identifiers),
            "last_created_id": identifiers.get(history.get("last_created_id")),
            "last_edited_id": identifiers.get(history.get("last_edited_id")),
        }
    recent = []
    created, edited = None, None
    for entry in history:
        action = entry.get("action")
        if action and action["name"] not in (
            "no_action",
            "select_shapes",
            "canvas_command",
            "pan_canvas",
            "delete_shapes",
        ):
            target = entry.get("created_id") or action["arguments"].get(
                "schema_id",
                action["arguments"].get(
                    "source_id", action["arguments"].get("shape_id")
                ),
            )
            target = target or next(
                iter(action["arguments"].get("shape_ids", [])), None
            )
            edited = identifiers.get(target)
            if entry.get("created_id"):
                created = identifiers.get(entry["created_id"])
    for entry in history[-3:]:
        action = entry.get("action")
        if action:
            arguments = dict(action["arguments"])
            for key in ("schema_id", "source_id", "target_id", "shape_id"):
                if key in arguments:
                    arguments[key] = identifiers.get(arguments[key], "removed")
            if "shape_ids" in arguments:
                arguments["shape_ids"] = [
                    identifiers.get(value, "removed")
                    for value in arguments["shape_ids"]
                ]
            action = {"name": action["name"], "arguments": arguments}
        recent.append({"command": entry["command"], "outcome": action or "rejected"})
    return {"last_created_id": created, "last_edited_id": edited, "turns": recent}


def messages_for(command: str, canvas: dict, history=None) -> list[dict]:
    context, identifiers = model_canvas(canvas)
    recent = recent_context(history or [], identifiers)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Canvas: {json.dumps(context)}\nRecent: {json.dumps(recent)}"
                f"\nCommand: {command}"
            ),
        },
    ]


def validate_call(call: dict, canvas: dict) -> dict:
    if set(call) != {"name", "arguments"} or call["name"] not in ACTION_MODELS:
        raise ValueError("Expected a known function name and arguments.")
    model = ACTION_MODELS[call["name"]][0]
    arguments = model.model_validate(call["arguments"]).model_dump()
    context = Canvas.model_validate(canvas)
    schemas = {s.id: s for s in context.schemas}
    shapes = {s.id: s for s in [*context.schemas, *context.shapes]}
    for key in ("schema_id", "source_id", "target_id", "shape_id"):
        targets = schemas if key == "schema_id" else shapes
        if key in arguments and arguments[key] not in targets:
            raise ValueError(f"Unknown canvas ID: {arguments[key]}")
    if "shape_ids" in arguments:
        if len(set(arguments["shape_ids"])) != len(arguments["shape_ids"]):
            raise ValueError("Shape IDs must be unique.")
        if any(value not in shapes for value in arguments["shape_ids"]):
            raise ValueError("Unknown canvas ID in shape targets.")
    if call["name"] == "remove_property":
        if arguments["property_name"] not in schemas[arguments["schema_id"]].properties:
            raise ValueError("Cannot remove a property that does not exist.")
    if (
        call["name"] == "remove_method"
        and arguments["method_name"] not in schemas[arguments["schema_id"]].methods
    ):
        raise ValueError("Cannot remove a method that does not exist.")
    if call["name"] == "set_text":
        target = shapes[arguments["shape_id"]]
        if isinstance(target, CanvasShape) and target.kind not in (
            "rectangle",
            "ellipse",
            "diamond",
            "triangle",
            "text",
            "note",
            "frame",
            "arrow",
        ):
            raise ValueError("This shape has no editable text.")
        if (isinstance(target, SchemaBox) or target.kind == "frame") and not 1 <= len(
            arguments["text"]
        ) <= 100:
            raise ValueError("Box names must contain 1 to 100 characters.")
    if call["name"] == "arrange_shapes":
        operation, ids = arguments["operation"], arguments["shape_ids"]
        if (
            operation == "group"
            and len(ids) < 2
            or operation.startswith("distribute")
            and len(ids) < 3
        ):
            raise ValueError("There are too few shapes for this arrangement.")
        if operation == "ungroup" and any(
            getattr(shapes[value], "kind", "schema") != "group" for value in ids
        ):
            raise ValueError("Only groups can be ungrouped.")
    if call["name"] == "canvas_command" and arguments["operation"] in ("undo", "redo"):
        if not getattr(context, f"can_{arguments['operation']}"):
            raise ValueError("No canvas history is available for that operation.")
    return {"name": call["name"], "arguments": arguments}


def parse_call(text: str, canvas: dict) -> dict:
    match = re.fullmatch(
        r"\s*<start_function_call>call:(\w+)(\{.*\})<end_function_call>"
        r"(?:<start_function_response>|<end_of_turn>)?\s*",
        text,
        re.DOTALL,
    )
    if not match:
        raise ValueError("Expected exactly one complete FunctionGemma call.")
    arguments = _parse_arguments(match[2])
    _, identifiers = model_canvas(canvas)
    originals = {alias: identifier for identifier, alias in identifiers.items()}
    for key in ("schema_id", "source_id", "target_id", "shape_id"):
        if key in arguments and isinstance(arguments[key], str):
            arguments[key] = originals.get(arguments[key], arguments[key])
    if "shape_ids" in arguments and isinstance(arguments["shape_ids"], list):
        arguments["shape_ids"] = [
            originals.get(value, value) for value in arguments["shape_ids"]
        ]
    return validate_call({"name": match[1], "arguments": arguments}, canvas)


def _parse_arguments(text: str) -> dict:
    strings = []

    def replace_string(match):
        strings.append(match[1])
        return json.dumps(f"__string_{len(strings) - 1}__")

    encoded = re.sub(r"<escape>(.*?)<escape>", replace_string, text, flags=re.DOTALL)
    encoded = re.sub(r"([\{,]\s*)([A-Za-z_]\w*)(\s*:)", r'\1"\2"\3', encoded)
    encoded = re.sub(
        r"\b(?:None|True|False)\b",
        lambda match: {"None": "null", "True": "true", "False": "false"}[match[0]],
        encoded,
    )
    arguments = json.loads(encoded, object_pairs_hook=_unique_object)

    def restore(value):
        if isinstance(value, str):
            match = re.fullmatch(r"__string_(\d+)__", value)
            if not match or int(match[1]) >= len(strings):
                raise ValueError("FunctionGemma strings must use escape delimiters.")
            return strings[int(match[1])]
        if isinstance(value, list):
            return [restore(item) for item in value]
        if isinstance(value, dict):
            return {key: restore(item) for key, item in value.items()}
        return value

    return restore(arguments)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate argument: {key}")
        result[key] = value
    return result


def no_action(reason):
    return {"name": "no_action", "arguments": {"reason": reason}}


def execution_guard(command, action, canvas, history):
    # Captions and cancelled requests must not become edits or history references.
    syntax = re.sub(
        r'"(?:\\.|[^"\\])*"|“[^”]*”|(?<!\w)\'(?:\\.|[^\'\\])*\'(?!\w)',
        lambda match: " " * len(match.group()),
        command,
    )
    corrections = list(
        re.finditer(
            r"\b(?:(?:no\s*,?\s*instead|i mean|(?:(?:no|wait|actually)\s*,?\s*)?"
            r"(?:cancel that|scratch that|forget that))\b|(?:no|actually)\s*,)"
            r"[\s,;:.!?]*(?:and\b\s*)?",
            syntax,
            re.I,
        )
    )
    antecedent_command, antecedent_syntax = "", ""
    if corrections:
        start = corrections[-1].end()
        antecedent_command, antecedent_syntax = command[:start], syntax[:start]
        command, syntax = command[start:], syntax[start:]
        if not command.strip():
            return no_action("unsupported_request")
    if re.search(
        r"\band\s+(?:then\s+)?(?:move|rotate|color|delete|rename|connect|add|remove|create|draw|resize|group|ungroup|duplicate|undo|redo)\b",
        syntax,
        re.I,
    ):
        # Method lists can contain conjunctions without requesting another edit.
        if not (
            action["name"] == "create_schema_box"
            and re.search(r"\b(?:properties|fields|methods|functions)\b", syntax, re.I)
        ):
            return no_action("unsupported_request")
        if re.search(
            r"\band\s+(?:then\s+)?(?:move|rotate|color|delete|rename|connect)\b",
            syntax,
            re.I,
        ):
            return no_action("unsupported_request")
    if action["name"] == "no_action":
        return action
    schemas = {
        item["id"]: item for item in [*canvas["schemas"], *canvas.get("shapes", [])]
    }
    arguments = action["arguments"]
    targets = [
        arguments[key]
        for key in ("schema_id", "source_id", "target_id", "shape_id")
        if key in arguments
    ] + arguments.get("shape_ids", [])
    shape_edit = action["name"] not in (
        "create_schema_box",
        "create_shape",
        "canvas_command",
        "pan_canvas",
    )
    if not shape_edit:
        return action
    reference_syntax = syntax
    reference_ids = set()
    for kind, expression in (
        (
            "created",
            r"\b(?:(?:the|this|that)\s+)?(?:last (?:created|added|drawn)"
            r"(?: (?:box|schema|class|shape|object|rectangle|circle|note|frame))?|"
            r"(?:schema|class|box|shape|object|rectangle|"
            r"circle|note|frame) (?:we|you) (?:(?:last|just) )?"
            r"(?:created|added|drew|made))\b",
        ),
        (
            "edited",
            r"\b(?:(?:the|this|that)\s+)?(?:last (?:edited|changed|modified)"
            r"(?: (?:box|schema|class|shape|object))?|"
            r"(?:schema|class|box|shape|object) "
            r"(?:we|you) (?:(?:last|just) )?(?:edited|changed|modified))\b",
        ),
    ):
        if re.search(expression, syntax, re.I):
            identifier = history.get(f"last_{kind}_id")
            if identifier not in schemas:
                return no_action("missing_target")
            reference_ids.add(identifier)
            reference_syntax = re.sub(
                expression,
                lambda match: " " * len(match.group()),
                reference_syntax,
                flags=re.I,
            )
    if "schema_id" in arguments and action["name"] != "rename_schema":
        named_prefix = (
            r"to|from|on|in|of|for|within|(?:schema|class|box)(?:\s+(?:named|called))?"
        )
    elif action["name"] == "set_text":
        named_prefix = r"of|on|in|for|give|named|called"
    elif action["name"] == "rename_schema":
        named_prefix = r"rename|call|named|called"
    else:
        named_prefix = (
            r"move|shift|nudge|drag|delete|remove|erase|select|deselect|duplicate|"
            r"copy|rotate|turn|flip|style|color|colour|make|resize|scale|reposition|bring|"
            r"send|put|connect|link|join|to|from|on|in|of|with|and|need|want|named|called"
        )
    named_ids = set()
    antecedent_ids = set()
    qualified_named = False

    def visible_target(match, content, masked, target_name):
        if not masked[match.start("prefix") : match.end("prefix")].strip():
            return False
        return (
            target_name.casefold() not in ("it", "this", "that")
            or content[match.end("prefix")] in "\"'“"
            or re.search(r"\b(?:named|called)\b", match["prefix"], re.I)
        )

    for item in schemas.values():
        name = item["name"]
        named_target = (
            rf"(?P<prefix>\b(?:{named_prefix})\s+(?:the\s+)?"
            r"(?:(?:schema|box|class|shape|object|frame|group)\s+)?)"
            rf"[\"'“]?{re.escape(name)}(?!\w)[\"'”]?"
        )

        matches = [
            match
            for match in re.finditer(named_target, command, re.I)
            if visible_target(match, command, syntax, name)
        ]
        if any(
            visible_target(match, antecedent_command, antecedent_syntax, name)
            for match in re.finditer(named_target, antecedent_command, re.I)
        ):
            antecedent_ids.add(item["id"])
        possessive = re.search(
            rf"(?<!\w){re.escape(name)}(?:['’]s|\s+(?:schema|box|class)\b)",
            syntax,
            re.I,
        )
        if matches or possessive:
            duplicates = sum(
                other["name"].casefold() == name.casefold()
                for other in schemas.values()
            )
            if shape_edit and duplicates > 1:
                return no_action("ambiguous_target")
            named_ids.add(item["id"])
        named_shape = (
            r"\b(?:(?:the|this|that)\s+)?(?:(?:selected|chosen|highlighted)\s+)?"
            r"(?:box|schema|class|shape|object|rectangle|circle|text|note|frame|group) "
            rf"(?:named|called)\s+[\"'“]?{re.escape(name)}(?!\w)[\"'”]?"
        )
        for match in re.finditer(named_shape, command, re.I):
            qualifier = re.search(r"\b(?:named|called)\b", match.group(), re.I)
            if not syntax[match.start() : match.start() + qualifier.end()].strip():
                continue
            if item["id"] not in named_ids:
                continue
            qualified_named = True
            reference_syntax = (
                reference_syntax[: match.start()]
                + " " * len(match.group())
                + reference_syntax[match.end() :]
            )
    reference_ids.update(named_ids)
    reference_syntax = re.sub(
        r"\b(?:every|each)\s+(?:selected|chosen|highlighted)\s+"
        r"(?:box|schema|class|shape|object|rectangle|circle|text|note|frame|group)\b",
        lambda match: " " * len(match.group()),
        reference_syntax,
        flags=re.I,
    )
    singular_selection = re.search(
        r"\b(?:selected|chosen|highlighted)\s+"
        r"(?:box|schema|class|shape|object|rectangle|circle|text|note|frame|group)\b",
        reference_syntax,
        re.I,
    )
    pronoun_syntax = re.sub(
        r"\ball of (?:it|this|that)\b",
        lambda match: " " * len(match.group()),
        reference_syntax,
        flags=re.I,
    )
    singular_pronoun = re.search(
        r"\b(?:"
        r"(?:move|shift|nudge|drag|delete|remove|erase|rename|call|update|change|edit|resize|"
        r"select|deselect|duplicate|copy|rotate|turn|flip|style|color|colour|make|"
        r"set|bring|send|put|connect|link|to|from|on|in|of|with)\s+(?:just\s+)?"
        r"(?:it|this|that)\b(?!\s+(?:selection|collection|set|pair|"
        r"last|previous|earlier)\b)|"
        r"(?:get\s+rid\s+of|give)\s+(?:it|this|that)\b|"
        r"(?:this|that)\s+(?:box|schema|class|shape|object|rectangle|circle|"
        r"text|note|frame|group|one)\b|"
        r"its\s+(?:name|text|label|color|colour|fill|opacity|width|height|size)\b|"
        r"(?:it|this|that)\s+(?:should|could|needs?\s+to|ought\s+to)\s+be\b)",
        pronoun_syntax,
        re.I,
    )
    separate_pronoun = re.search(
        r"\b(?:it|this|that)\s+and\b|\band\s+(?:it|this|that)\b",
        pronoun_syntax,
        re.I,
    )
    if singular_pronoun and not reference_ids and antecedent_ids:
        if len(antecedent_ids) != 1:
            return no_action("ambiguous_target")
        reference_ids.update(antecedent_ids)
    needs_selection = (
        singular_selection
        or singular_pronoun
        and (
            not reference_ids or separate_pronoun or action["name"] == "connect_schemas"
        )
    )
    if shape_edit and needs_selection:
        selection = canvas.get("selected_ids", [])
        if len(selection) != 1:
            return no_action("ambiguous_target" if schemas else "missing_target")
        reference_ids.update(selection)
    # Named and history references may coexist with a singular selected target.
    enforce_targets = (
        singular_selection
        or singular_pronoun
        or qualified_named
        or (reference_ids - named_ids)
    )
    if enforce_targets and reference_ids and set(targets) != reference_ids:
        return no_action("ambiguous_target")
    return action
