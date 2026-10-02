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
    selected_ids: list[Name] = Field(default_factory=list, max_length=30)
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
    "A target name must match exactly one current shape; "
    "duplicate names are ambiguous. "
    "Undo and redo require can_undo and can_redo respectively. "
    "Questions, explanations, code execution, and requests for multiple edits are "
    "unsupported_request. For a correction, execute only the final requested edit."
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
    if re.search(
        r"\band\s+(?:then\s+)?(?:move|rotate|color|delete|rename|connect|add|remove|create|draw|resize|group|ungroup|duplicate|undo|redo)\b",
        command,
        re.I,
    ):
        # Method lists can contain conjunctions without requesting another edit.
        if not (
            action["name"] == "create_schema_box"
            and re.search(r"\b(?:properties|fields|methods|functions)\b", command, re.I)
        ):
            return no_action("unsupported_request")
        if re.search(
            r"\band\s+(?:then\s+)?(?:move|rotate|color|delete|rename|connect)\b",
            command,
            re.I,
        ):
            return no_action("unsupported_request")
    schemas = {
        item["id"]: item for item in [*canvas["schemas"], *canvas.get("shapes", [])]
    }
    if (
        re.search(
            r"\b(?:to|from|on|in|of|rename|call|update)\s+(?:the\s+)?"
            r"selected (?:box|schema|class|shape|rectangle|circle|text|note)\b",
            command,
            re.I,
        )
        and len(canvas.get("selected_ids", [])) != 1
    ):
        return no_action("ambiguous_target" if schemas else "missing_target")
    for kind, expression in (
        (
            "created",
            r"\b(?:last created|(?:schema|class|box) we (?:last |just )?created)\b",
        ),
        ("edited", r"\b(?:last edited|(?:schema|class|box) we (?:last )?edited)\b"),
    ):
        if (
            re.search(expression, command, re.I)
            and history.get(f"last_{kind}_id") not in schemas
        ):
            return no_action("missing_target")
    if action["name"] != "create_schema_box":
        arguments = action["arguments"]
        targets = [
            arguments.get(key)
            for key in ("schema_id", "source_id", "target_id", "shape_id")
        ]
        targets.extend(arguments.get("shape_ids", []))
        for target_id in targets:
            target = schemas.get(target_id)
            if target is None:
                continue
            name = target["name"]
            duplicates = sum(
                item["name"].casefold() == name.casefold() for item in schemas.values()
            )
            named = re.search(
                rf"\b(?:to|from|on|in|of|named|called|rename|connect)\s+(?:the\s+)?(?:(?:schema|box|class)\s+)?{re.escape(name)}(?!\w)|(?<!\w){re.escape(name)}(?:['’]s|\s+(?:schema|box|class)\b)",
                command,
                re.I,
            )
            if duplicates > 1 and named:
                return no_action("ambiguous_target")
    return action
