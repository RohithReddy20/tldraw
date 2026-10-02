import argparse
import hashlib
import json
import random
from pathlib import Path

from actions import validate_call
from dataset import ROOT, audit_examples

PROFILES = {
    "train": [
        (
            "User",
            ["name", "class", "subjects"],
            ["getName", "getClass", "getSubjects", "addSubject", "removeSubject"],
        ),
        (
            "Account",
            ["email", "role", "permissions"],
            ["getEmail", "grantAccess", "revokeAccess"],
        ),
        ("Student", ["rollNumber", "name", "grade"], ["enroll", "getGrade"]),
        ("Teacher", ["name", "department", "courses"], ["assignCourse", "getCourses"]),
        ("Course", ["title", "code", "credits"], ["getTitle", "setCredits"]),
        (
            "Subject",
            ["name", "syllabus", "semester"],
            ["getSyllabus", "updateSyllabus"],
        ),
        ("Classroom", ["roomNumber", "capacity", "building"], ["reserve", "release"]),
        ("Invoice", ["number", "total", "status"], ["calculateTotal", "markPaid"]),
        ("Payment", ["amount", "currency", "reference"], ["refund", "getReceipt"]),
        ("Book", ["title", "author", "isbn"], ["borrow", "returnBook"]),
        ("LibraryMember", ["memberId", "email", "loans"], ["getLoans", "renewLoan"]),
        ("Product", ["sku", "price", "stock"], ["getPrice", "restock", "sell"]),
        (
            "Cart",
            ["items", "subtotal", "discount"],
            ["addItem", "removeItem", "checkout"],
        ),
        ("Customer", ["name", "phone", "address"], ["getAddress", "updatePhone"]),
        ("Employee", ["employeeId", "salary", "team"], ["getSalary", "changeTeam"]),
        ("Project", ["title", "deadline", "owner"], ["getDeadline", "assignOwner"]),
        ("Task", ["description", "assignee", "priority"], ["completeTask", "reassign"]),
        ("Playlist", ["title", "tracks", "duration"], ["addTrack", "removeTrack"]),
        ("Message", ["sender", "recipient", "body"], ["send", "archive"]),
        ("Ticket", ["ticketId", "summary", "severity"], ["resolve", "reopen"]),
        ("Device", ["serialNumber", "model", "battery"], ["getBattery", "restart"]),
        ("Folder", ["path", "files", "size"], ["addFile", "deleteFile"]),
        ("Wallet", ["balance", "currency", "transactions"], ["deposit", "withdraw"]),
        ("Team", ["name", "members", "manager"], ["addMember", "removeMember"]),
    ],
    "valid": [
        ("Order", ["orderId", "items", "total"], ["placeOrder", "cancelOrder"]),
        (
            "Document",
            ["title", "content", "version"],
            ["saveVersion", "restoreVersion"],
        ),
        ("Vehicle", ["registration", "speed", "fuel"], ["accelerate", "refuel"]),
    ],
    "test": [
        (
            "Shipment",
            ["trackingId", "destination", "weight"],
            ["dispatch", "trackParcel"],
        ),
        (
            "Reservation",
            ["guestName", "arrivalDate", "roomType"],
            ["confirmBooking", "cancelBooking"],
        ),
        ("SensorReading", ["timestamp", "value", "unit"], ["calibrate", "getValue"]),
        ("GameSession", ["sessionId", "players", "score"], ["startGame", "endGame"]),
    ],
}

PHRASES = {
    "train": {
        "create": [
            "Create a {name} schema with properties {fields} and methods {methods}.",
            "Draw a schema box called {name}. Fields: {fields}. Functions: {methods}.",
            "Make {name} with attributes {fields}; include operations {methods}.",
            "Build a {name} class: properties {fields}; methods {methods}.",
            "I need {name}. Put {fields} in properties and {methods} in methods.",
            "Sketch a {name} box having fields {fields} and functions {methods}.",
            "For {name}, the methods are {methods} "
            "and the properties are {fields}. Draw it.",
            "um create {name} with {fields} as properties and uh {methods} as methods",
        ],
        "fields": [
            "Create {name} with properties {fields}.",
            "Draw a {name} box with fields {fields} and no methods.",
            "Make {name}; its only attributes are {fields}.",
        ],
        "methods": [
            "Create {name} with methods {methods} and no properties.",
            "Draw {name}; leave fields empty, include functions {methods}.",
        ],
        "empty": [
            "Create an empty {name} schema.",
            "Make {name} with no properties or methods.",
        ],
        "add": [
            "Add {field} to {name}.",
            "Give {name} a new field named {field}.",
            "Extend the {name} box with a property called {field}.",
            "Put a {field} property in {name}.",
            "For {name}, include the attribute {field}.",
        ],
        "remove": [
            "Remove {field} from {name}.",
            "Drop {field} from the properties of {name}.",
            "Take the field {field} out of {name}.",
            "Delete the {field} attribute in {name}.",
        ],
        "rename": [
            "Rename {name} to {new_name}.",
            "Retitle {name} as {new_name}.",
            "Use {new_name} as the new title of {name}.",
            "Call the {name} schema {new_name} instead.",
        ],
        "connect": [
            "Connect {name} to {other}.",
            "Draw an arrow from {name} to {other}.",
            "Link {name} to {other} and label the link {label}.",
            "Make a directed link from {name} to {other} named {label}.",
        ],
        "selected_add": [
            "Add {field} to it.",
            "Put {field} in that box.",
            "Give the selected schema a {field} field.",
        ],
        "selected_rename": ["Rename it to {new_name}.", "Call this box {new_name}."],
        "unsupported": [
            "Explain what {name} means.",
            "Translate {name} into French.",
            "Make the {name} box blue.",
            "Delete every box.",
        ],
    },
    "valid": {
        "create": [
            "Build a diagram class named {name}: "
            "attributes {fields}; operations {methods}.",
            "Please draw {name}, listing {fields} as fields and {methods} as methods.",
            "For a new {name} schema, use fields {fields}, with functions {methods}.",
        ],
        "fields": ["Build a {name} schema holding {fields}; leave its methods empty."],
        "methods": ["Build {name}, whose only entries are the methods {methods}."],
        "empty": ["Build {name} with empty property and method sections."],
        "add": [
            "Append the property {field} to {name}.",
            "Include {field} among {name}'s attributes.",
        ],
        "remove": [
            "Omit {field} from {name}.",
            "Erase the attribute {field} in {name}.",
        ],
        "rename": ["Change {name}'s title into {new_name}."],
        "connect": [
            "Point a connection from {name} towards {other}.",
            "Join {name} to {other}, calling the link {label}.",
        ],
        "selected_add": ["Extend this schema by adding {field}."],
        "selected_rename": ["Change the selected schema's title into {new_name}."],
        "unsupported": [
            "Explain the purpose of {name}.",
            "Paint the selected box green.",
        ],
    },
    "test": {
        "create": [
            "Can you draw {name} having fields {fields} and operations {methods}?",
            "Set up a schema titled {name}. Under properties put {fields}; "
            "under functions put {methods}.",
            "okay a new box named {name} please with {fields} for fields "
            "and {methods} for its methods",
        ],
        "fields": ["Sketch {name}, listing {fields} in its properties section only."],
        "methods": ["Sketch {name}, listing {methods} in its methods section only."],
        "empty": ["Sketch an empty schema entitled {name}."],
        "add": [
            "Insert the attribute {field} into {name}.",
            "The {name} schema needs another property: {field}.",
        ],
        "remove": [
            "Strip the {field} property out of {name}.",
            "Exclude the attribute {field} from {name}.",
        ],
        "rename": ["Replace the title {name} with {new_name}."],
        "connect": [
            "Point an arrow from {name} at {other}.",
            "Create an arrow labelled {label} going from {name} into {other}.",
        ],
        "selected_add": ["Insert {field} into the selected box's properties."],
        "selected_rename": ["Replace this box's title with {new_name}."],
        "unsupported": [
            "Summarize the meaning of {name}.",
            "Shift the selected box to the left.",
        ],
    },
}

EXTRA_FIELDS = [
    "age",
    "email",
    "createdAt",
    "updated_at",
    "isActive",
    "tags",
    "description",
    "phoneNumber",
    "category",
    "notes",
    "ownerId",
    "displayName",
]
RELATIONS = ["owns", "has", "references", "contains", "belongs to", "uses"]
DATASET_VERSION = "v3"


def expanded_catalogue(seed):
    rng = random.Random(seed)
    original = [profile for profiles in PROFILES.values() for profile in profiles]
    fields = sorted(
        {field for _, names, _ in original for field in names} | set(EXTRA_FIELDS)
    )
    methods = sorted({method for _, _, names in original for method in names})
    train = list(original)
    for prefix in (
        "Billing",
        "Garden",
        "Audio",
        "Client",
        "Shipping",
        "Fitness",
        "Support",
        "Research",
    ):
        for suffix in (
            "Profile",
            "Entry",
            "Plan",
            "Record",
            "Session",
            "Event",
            "Report",
            "Asset",
        ):
            train.append(
                (
                    prefix + suffix,
                    rng.sample(fields, rng.randint(1, 6)),
                    rng.sample(methods, rng.randint(1, 6)),
                )
            )
    return {
        "train": train,
        "valid": [
            (
                "Appointment",
                ["date", "doctor", "location"],
                ["scheduleVisit", "rescheduleVisit", "getDoctor"],
            ),
            (
                "Recipe",
                ["ingredients", "servings", "instructions"],
                ["cook", "scaleRecipe", "getIngredients"],
            ),
            (
                "Workspace",
                ["owner", "projects", "visibility"],
                ["inviteMember", "archiveProject", "getOwner"],
            ),
            (
                "Subscription",
                ["plan", "expiresAt", "subscriber"],
                ["renewPlan", "pausePlan", "getSubscriber"],
            ),
        ],
        "test": [
            (
                "Warehouse",
                ["address", "inventory", "capacity"],
                ["receiveStock", "dispatchStock", "getInventory"],
            ),
            (
                "PodcastEpisode",
                ["title", "host", "length"],
                ["publishEpisode", "playEpisode", "getHost"],
            ),
            (
                "FlightTicket",
                ["passenger", "seat", "flightNumber"],
                ["checkIn", "changeSeat", "getPassenger"],
            ),
            (
                "Membership",
                ["memberName", "tier", "joinDate"],
                ["upgradeTier", "cancelMembership", "getTier"],
            ),
            (
                "ExpenseClaim",
                ["amount", "receipt", "approvalStatus"],
                ["submitClaim", "approveClaim", "getReceipt"],
            ),
            (
                "Notification",
                ["recipient", "message", "readAt"],
                ["deliver", "markRead", "getRecipient"],
            ),
        ],
    }


def expanded_phrases():
    train = {
        kind: list(
            dict.fromkeys(phrase for bank in PHRASES.values() for phrase in bank[kind])
        )
        for kind in PHRASES["train"]
    }
    train["create"].extend(
        [
            "Create a schema box of {name} with properties {fields} "
            "and functions {methods}.",
            "Draw {name}. Its functions are {methods}. Its properties are {fields}.",
            "Please make a class named {name} with fields {fields} "
            "and methods {methods}.",
            "a {name} box with properties {fields} and methods {methods} please",
            "Create class {name}: attributes [{fields}], methods [{methods}].",
        ]
    )
    train["rename"].extend(
        [
            "Change the name of {name} to {new_name}.",
            "Update {name}'s name to {new_name}.",
            "{name} should now be called {new_name}.",
        ]
    )
    train["selected_add"].extend(
        ["Add property {field} to this.", "Include {field} in the selected schema."]
    )
    train["selected_rename"].extend(
        ["Rename this to {new_name}.", "Change its name to {new_name}."]
    )
    valid = {
        "create": [
            "Draw a class box for {name}, using properties {fields} "
            "and functions {methods}.",
            "Make the schema {name}: its fields should be {fields}, "
            "and its methods {methods}.",
            "I want a {name} class. Add the methods {methods} "
            "and the properties {fields}.",
        ],
        "fields": ["Draw {name} with the properties {fields}; it has no methods."],
        "methods": ["Draw {name} with the functions {methods}; it has no fields."],
        "empty": ["Add a blank class box called {name}."],
        "add": [
            "Add a field called {field} inside {name}.",
            "Append {field} as an attribute of {name}.",
        ],
        "remove": [
            "Delete property {field} from the {name} schema.",
            "Remove the attribute called {field} in {name}.",
        ],
        "rename": [
            "Change the schema name {name} into {new_name}.",
            "Rename the class {name} as {new_name}.",
        ],
        "connect": [
            "Draw a link from the {name} schema to {other}.",
            "Connect the {name} box to {other} with label {label}.",
        ],
        "selected_add": [
            "Please add {field} to this box.",
            "Append {field} to the selected class.",
        ],
        "selected_rename": [
            "Rename the selected class as {new_name}.",
            "Change this schema's name to {new_name}.",
        ],
        "unsupported": ["Explain the fields of {name}.", "Turn {name} purple."],
    }
    test = {
        "create": [
            "Please create the {name} schema with attributes {fields} "
            "and functions {methods}.",
            "I need a box for {name}. Properties should be {fields}. "
            "Methods should be {methods}.",
            "Draw {name} as a class, with operations {methods} and fields {fields}.",
            "um draw a class box named {name} with fields {fields} "
            "and uh methods {methods}",
        ],
        "fields": ["Make a class {name} containing just the fields {fields}."],
        "methods": ["Make a class {name} containing just the functions {methods}."],
        "empty": ["Draw a box called {name} without fields or functions."],
        "add": [
            "Put a new attribute named {field} into {name}.",
            "Include a property called {field} on the {name} class.",
        ],
        "remove": [
            "Erase {field} from the fields of {name}.",
            "Drop the attribute named {field} out of the {name} box.",
        ],
        "rename": [
            "The class named {name} needs to be renamed {new_name}.",
            "Replace the name of {name} with {new_name}.",
        ],
        "connect": [
            "Link the class {name} to the class {other}.",
            "Draw an arrow labelled {label} from the {name} class to {other}.",
        ],
        "selected_add": [
            "Add an attribute called {field} to it.",
            "Put a new field {field} inside this box.",
        ],
        "selected_rename": [
            "Change the name of this box to {new_name}.",
            "The selected box should be named {new_name}.",
        ],
        "unsupported": ["Tell me what {name} does.", "Move {name} upwards."],
    }
    return {"train": train, "valid": valid, "test": test}


def list_text(items, variant):
    if variant % 3 == 0 or len(items) < 2:
        return ", ".join(items)
    if variant % 3 == 1:
        return ", ".join(items[:-1]) + " and " + items[-1]
    return "; ".join(items)


def box(profile, identifier):
    name, fields, methods = profile
    return {"id": identifier, "name": name, "properties": fields, "methods": methods}


def phrases_for(split, kind, name):
    return list(enumerate(expanded_phrases()[split][kind]))


def make_context(profile, other, rng):
    ids = [f"schema:{rng.getrandbits(32):08x}" for _ in range(3)]
    schemas = [
        box(profile, ids[0]),
        box(other, ids[1]),
        box(("Unrelated", ["value"], []), ids[2]),
    ]
    rng.shuffle(schemas)
    return {"schemas": schemas, "selected_ids": [ids[1]]}, ids[0], ids[1]


def record(split, group, index, command, canvas, name, arguments):
    return {
        "id": f"{DATASET_VERSION}-{split}-{group}-{index}",
        "group": f"{DATASET_VERSION}-{split}-{group}",
        "split": split,
        "command": command,
        "canvas": canvas,
        "expected": validate_call({"name": name, "arguments": arguments}, canvas),
        "provenance": "authored templates with deterministic labels",
    }


def create_examples(split, profile):
    name, fields, methods = profile
    examples = []
    cases = [
        ("create", fields, methods),
        ("fields", fields, []),
        ("methods", [], methods),
        ("empty", [], []),
    ]
    for kind, attributes, functions in cases:
        for index, phrase in phrases_for(split, kind, name):
            if kind == "create" and index > 1:
                attributes = fields[: 1 + index % len(fields)]
                functions = methods[: 1 + index % len(methods)]
            command = phrase.format(
                name=name,
                fields=list_text(attributes, index),
                methods=list_text(functions, index),
            )
            expected = {"name": name, "fields": attributes, "methods": functions}
            examples.append(
                record(
                    split,
                    f"{name}-{kind}",
                    index,
                    command,
                    {},
                    "create_schema_box",
                    expected,
                )
            )
    return examples


def edit_examples(split, profile, other, rng):
    name, fields, _ = profile
    examples = []
    for kind in ("add", "remove", "rename", "connect"):
        for index, phrase in phrases_for(split, kind, name):
            canvas, schema_id, other_id = make_context(profile, other, rng)
            field = (
                fields[index % len(fields)]
                if kind == "remove"
                else next(
                    f
                    for f in rng.sample(EXTRA_FIELDS, len(EXTRA_FIELDS))
                    if f not in fields
                )
            )
            values = {
                "name": name,
                "other": other[0],
                "field": field,
                "new_name": f"{name}Details" if index % 2 else f"Archived{name}",
                "label": rng.choice(RELATIONS),
            }
            actions = {
                "add": (
                    "add_property",
                    {"schema_id": schema_id, "property_name": field},
                ),
                "remove": (
                    "remove_property",
                    {"schema_id": schema_id, "property_name": field},
                ),
                "rename": (
                    "rename_schema",
                    {"schema_id": schema_id, "new_name": values["new_name"]},
                ),
                "connect": (
                    "connect_schemas",
                    {
                        "source_id": schema_id,
                        "target_id": other_id,
                        "label": values["label"] if "{label}" in phrase else "",
                    },
                ),
            }
            action, arguments = actions[kind]
            examples.append(
                record(
                    split,
                    f"{name}-{kind}",
                    index,
                    phrase.format(**values),
                    canvas,
                    action,
                    arguments,
                )
            )
    return examples


def selection_examples(split, profile, other, rng):
    examples = []
    for kind in ("selected_add", "selected_rename"):
        for index, phrase in phrases_for(split, kind, profile[0]):
            canvas, target, other_id = make_context(profile, other, rng)
            field = rng.choice([f for f in EXTRA_FIELDS if f not in profile[1]])
            values = {"field": field, "new_name": f"New{profile[0]}"}
            command = phrase.format(**values)
            action = "add_property" if kind == "selected_add" else "rename_schema"
            arguments = (
                {"schema_id": target, "property_name": field}
                if kind == "selected_add"
                else {"schema_id": target, "new_name": values["new_name"]}
            )
            variants = [
                ([target], action, arguments),
                ([], "no_action", {"reason": "ambiguous_target"}),
                ([target, other_id], "no_action", {"reason": "ambiguous_target"}),
            ]
            for variant, (selection, tool, args) in enumerate(variants):
                context = {**canvas, "selected_ids": selection}
                examples.append(
                    record(
                        split,
                        f"{profile[0]}-{kind}",
                        index * 3 + variant,
                        command,
                        context,
                        tool,
                        args,
                    )
                )
            examples.append(
                record(
                    split,
                    f"{profile[0]}-{kind}",
                    index * 3 + 100,
                    command,
                    {},
                    "no_action",
                    {"reason": "missing_target"},
                )
            )
    return examples


def negative_examples(split, profile, other, rng):
    name, fields, _ = profile
    absent_field = rng.choice([field for field in EXTRA_FIELDS if field not in fields])
    canvas, target, _ = make_context(profile, other, rng)
    duplicate = {
        **canvas,
        "schemas": [*canvas["schemas"], box(profile, target + "-copy")],
        "selected_ids": [],
    }
    cases = [
        (
            expanded_phrases()[split]["add"][0].format(name=name, field="age"),
            {},
            "missing_target",
        ),
        (
            expanded_phrases()[split]["rename"][0].format(
                name=name, new_name=f"Next{name}"
            ),
            duplicate,
            "ambiguous_target",
        ),
        (
            expanded_phrases()[split]["remove"][0].format(
                name=name, field=absent_field
            ),
            canvas,
            "missing_target",
        ),
        (
            expanded_phrases()[split]["connect"][0].format(
                name=name, other="MissingSchema"
            ),
            canvas,
            "missing_target",
        ),
    ]
    cases.extend(
        (phrase.format(name=name), canvas, "unsupported_request")
        for _, phrase in phrases_for(split, "unsupported", name)
    )
    return [
        record(
            split,
            f"{name}-negative-{index}",
            0,
            command,
            context,
            "no_action",
            {"reason": reason},
        )
        for index, (command, context, reason) in enumerate(cases)
    ]


def build_examples(seed=42):
    rng = random.Random(seed)
    examples = []
    for split, profiles in expanded_catalogue(seed).items():
        for index, profile in enumerate(profiles):
            other = profiles[(index + 1) % len(profiles)]
            examples.extend(create_examples(split, profile))
            examples.extend(edit_examples(split, profile, other, rng))
            examples.extend(selection_examples(split, profile, other, rng))
            examples.extend(negative_examples(split, profile, other, rng))
    audit_examples(examples)
    return examples


def build_refinement(source, seed=54, replay_count=16000, pairs=6000):
    from build_sessions import FIELDS, NAMES

    original = [
        json.loads(line) for line in source.read_text().splitlines() if line.strip()
    ]
    rng = random.Random(seed)
    train = [row for row in original if row["split"] == "train"]
    replay = rng.sample(train, min(replay_count, len(train)))
    examples = json.loads(json.dumps(replay))
    for row in examples:
        row["id"] = "v5-replay-" + row["id"]
        row["group"] = "v5-replay-" + row["group"]
    add_phrases = (
        "Add {field} to {target}.",
        "Put a property called {field} in {target}.",
        "Include the field {field} on {target}.",
        "Give {target} a new attribute named {field}.",
    )
    remove_phrases = (
        "Remove {field} from {target}.",
        "Delete the property {field} on {target}.",
        "Drop the field {field} from {target}.",
    )
    extra_edits = (
        " and move it to the right",
        " and then rotate the box",
        " and color that schema blue",
        " and rename it to Revised",
        "; then reposition the class above its neighbor",
        "; after that, delete the entire box",
        " and also connect it to another schema",
        " and make the box wider",
    )
    for index in range(pairs):
        names = rng.sample(NAMES["train"], 3)
        identifiers = [f"refine-{index}-box-{j}" for j in range(3)]
        field, missing = rng.sample(FIELDS, 2)
        properties = rng.sample([x for x in FIELDS if x not in (field, missing)], 4)
        canvas = {
            "schemas": [
                {"id": identifier, "name": name, "properties": list(properties)}
                for identifier, name in zip(identifiers, names, strict=True)
            ],
            "selected_ids": [identifiers[1]],
        }
        target = names[0]
        command = rng.choice(add_phrases).format(field=field, target=target)
        compound = command.rstrip(".") + rng.choice(extra_edits) + "."
        remove = rng.choice(remove_phrases).format(field=missing, target=target)
        present = json.loads(json.dumps(canvas))
        present["schemas"][0]["properties"].append(missing)
        selected = rng.choice(add_phrases).format(
            field=field, target="the selected box"
        )
        ambiguous = json.loads(json.dumps(canvas))
        ambiguous["selected_ids"] = [] if index % 2 else identifiers[:2]
        cases = (
            (
                command,
                canvas,
                "add_property",
                {"schema_id": identifiers[0], "property_name": field},
            ),
            (compound, canvas, "no_action", {"reason": "unsupported_request"}),
            (
                remove,
                present,
                "remove_property",
                {"schema_id": identifiers[0], "property_name": missing},
            ),
            (remove, canvas, "no_action", {"reason": "missing_target"}),
            (
                selected,
                canvas,
                "add_property",
                {"schema_id": identifiers[1], "property_name": field},
            ),
            (selected, ambiguous, "no_action", {"reason": "ambiguous_target"}),
        )
        for kind, (command, context, name, arguments) in enumerate(cases):
            row = record(
                "train",
                f"v5-refinement-{index}",
                kind,
                command,
                context,
                name,
                arguments,
            )
            row.update(id=f"v5-hard-{index}:{kind}", group=f"v5-hard-{index}")
            examples.append(row)
    examples.extend(row for row in original if row["split"] != "train")
    audit_examples(examples)
    return examples


def main():
    parser = argparse.ArgumentParser(
        description="Build the expanded canvas action dataset."
    )
    parser.add_argument("--output", type=Path, default=ROOT / "examples.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--train-sessions", type=int, default=2000)
    parser.add_argument("--refine-from", type=Path)
    parser.add_argument("--replay-count", type=int, default=16000)
    parser.add_argument("--hard-pairs", type=int, default=6000)
    args = parser.parse_args()
    if args.output.exists() and not args.replace:
        raise SystemExit("Output already exists; use --replace to regenerate it.")
    if args.refine_from:
        if args.output.resolve() == args.refine_from.resolve():
            raise SystemExit("Refinement must preserve its original source dataset.")
        examples = build_refinement(
            args.refine_from, args.seed, args.replay_count, args.hard_pairs
        )
        session_path = args.refine_from.parent / "sessions.jsonl"
        (args.output.parent / "sessions.jsonl").write_bytes(session_path.read_bytes())
    else:
        from build_sessions import build_large_dataset

        examples, sessions = build_large_dataset(args.seed, args.train_sessions)
        (args.output.parent / "sessions.jsonl").write_text(
            "".join(json.dumps(session) + "\n" for session in sessions)
        )
    contents = "".join(json.dumps(example) + "\n" for example in examples)
    args.output.write_text(contents)
    print(
        json.dumps(
            {
                "splits": audit_examples(examples),
                "sha256": hashlib.sha256(contents.encode()).hexdigest(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
