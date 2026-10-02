import json
import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Name = Annotated[str, Field(min_length=1, max_length=100)]
Names = Annotated[list[Name], Field(max_length=30)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class SchemaBox(StrictModel):
    id: Name
    name: Name
    properties: Names = Field(default_factory=list)
    methods: Names = Field(default_factory=list)


class Canvas(StrictModel):
    schemas: list[SchemaBox] = Field(default_factory=list, max_length=30)
    selected_ids: list[Name] = Field(default_factory=list, max_length=30)

    @model_validator(mode="after")
    def check_ids(self):
        ids = {schema.id for schema in self.schemas}
        if len(ids) != len(self.schemas):
            raise ValueError("Canvas IDs must be unique.")
        if len(set(self.selected_ids)) != len(self.selected_ids):
            raise ValueError("Selected IDs must be unique.")
        if any(identifier not in ids for identifier in self.selected_ids):
            raise ValueError("Selection contains an unknown canvas ID.")
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
    source_id: Name = Field(description="ID of the existing source schema.")
    target_id: Name = Field(description="ID of the existing destination schema.")
    label: str = Field(
        max_length=100, description="Requested connection label; empty if omitted."
    )


class NoAction(StrictModel):
    reason: Literal["missing_target", "ambiguous_target", "unsupported_request"] = (
        Field(description="Why the command cannot be executed.")
    )


ACTION_MODELS = {
    "create_schema_box": (
        CreateSchema,
        "Create a schema box with properties and methods.",
    ),
    "add_property": (AddProperty, "Add one property to an existing schema."),
    "remove_property": (RemoveProperty, "Remove one property from an existing schema."),
    "rename_schema": (RenameSchema, "Rename an existing schema."),
    "connect_schemas": (ConnectSchemas, "Connect two existing schemas with an arrow."),
    "no_action": (
        NoAction,
        "Do nothing for an unclear target or an unsupported request.",
    ),
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
    "Use the short box IDs shown in Canvas. A named target overrides selection. "
    "For 'it', 'this', 'that box', or 'selected', use selected_ids only when single. "
    "No selection or multiple selections means ambiguous_target; "
    "an empty canvas or absent name or field means missing_target. "
    "Never guess a missing or ambiguous target. "
    "Preserve names, field spelling, and list order. "
    "Do not invent unrequested properties or methods. "
    "Canvas is the current state; Recent contains at most three actual outcomes. "
    "For 'last created' use last_created_id; for 'last edited' use last_edited_id. "
    "An old name in Recent is not a current named target. "
    "Questions, explanations, styling, undo, and requests for multiple edits are "
    "unsupported_request. For a correction, execute only the final requested edit."
)
INPUT_FORMAT_VERSION = "current-canvas-three-outcomes-v2"


def model_canvas(canvas: dict) -> tuple[dict, dict]:
    context = Canvas.model_validate(canvas)
    identifiers = {schema.id: f"box{i + 1}" for i, schema in enumerate(context.schemas)}
    selected = [identifiers[s] for s in context.selected_ids if s in identifiers]
    status = "single" if len(selected) == 1 else "multiple" if selected else "none"
    return {
        "schemas": [
            {"id": identifiers[s.id], "name": s.name, "fields": s.properties}
            for s in context.schemas
        ],
        "selection": status,
        "selected_ids": selected,
    }, identifiers


def model_call(call: dict, canvas: dict) -> dict:
    _, identifiers = model_canvas(canvas)
    arguments = dict(call["arguments"])
    for key in ("schema_id", "source_id", "target_id"):
        if key in arguments:
            arguments[key] = identifiers[arguments[key]]
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
        if action and action["name"] != "no_action":
            target = entry.get("created_id") or action["arguments"].get(
                "schema_id", action["arguments"].get("source_id")
            )
            edited = identifiers.get(target)
            if entry.get("created_id"):
                created = identifiers.get(entry["created_id"])
    for entry in history[-3:]:
        action = entry.get("action")
        if action:
            arguments = dict(action["arguments"])
            for key in ("schema_id", "source_id", "target_id"):
                if key in arguments:
                    arguments[key] = identifiers.get(arguments[key], "removed")
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
    schemas = {s.id: s for s in Canvas.model_validate(canvas).schemas}
    for key in ("schema_id", "source_id", "target_id"):
        if key in arguments and arguments[key] not in schemas:
            raise ValueError(f"Unknown canvas ID: {arguments[key]}")
    if call["name"] == "remove_property":
        if arguments["property_name"] not in schemas[arguments["schema_id"]].properties:
            raise ValueError("Cannot remove a property that does not exist.")
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
    for key in ("schema_id", "source_id", "target_id"):
        if key in arguments and isinstance(arguments[key], str):
            arguments[key] = originals.get(arguments[key], arguments[key])
    return validate_call({"name": match[1], "arguments": arguments}, canvas)


def _parse_arguments(text: str) -> dict:
    strings = []

    def replace_string(match):
        strings.append(match[1])
        return json.dumps(f"__string_{len(strings) - 1}__")

    encoded = re.sub(r"<escape>(.*?)<escape>", replace_string, text, flags=re.DOTALL)
    encoded = re.sub(r"([\{,]\s*)([A-Za-z_]\w*)(\s*:)", r'\1"\2"\3', encoded)
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
        r"\band\s+(?:then\s+)?(?:move|rotate|color|delete|rename|connect|add|remove|create|draw)\b",
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
    schemas = {item["id"]: item for item in canvas["schemas"]}
    if (
        re.search(
            r"\b(?:to|from|on|in|of|rename|call|update)\s+(?:the\s+)?"
            r"selected (?:box|schema|class)\b",
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
        for key in ("schema_id", "source_id", "target_id"):
            target = schemas.get(action["arguments"].get(key))
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
