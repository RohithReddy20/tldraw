# tldraw agent

This starter kit demonstrates how to build an agent that can manipulate the [tldraw](https://github.com/tldraw/tldraw) canvas.

A chat panel on the right side of the screen lets users communicate with the agent, add context, and see chat history.

## Environment setup

Create a `.dev.vars` file in the root directory and add API keys for any model providers you want to use.

```
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

We recommend using Anthropic for best results. Get your API key from the [Anthropic dashboard](https://console.anthropic.com/settings/keys).

## Local development

From the monorepo root, install dependencies with `pnpm install`.

Run the development server with `pnpm --filter tldraw-agent dev`.

Open `http://localhost:5173/` in your browser to see the app.

## Local voice canvas

Open `http://localhost:5173/?mode=local-voice` for silent speech-to-canvas commands.
The browser records audio when you press **Speak**; press **Finish recording** to
apply it. Recordings stop automatically at 60 seconds. A text input uses the same
action model. This mode uses local Parakeet speech recognition and the trained
FunctionGemma 270M adapter, with no paid inference API or spoken replies.

Start the service in a second terminal from the monorepo root:

```sh
uv run --project templates/agent/training --extra voice python templates/agent/training/voice_server.py \
  --adapter templates/agent/training/runs/colab-270m-v4-sessions-checkpoint-5000/adapter
```

The adapter must already exist. The service requires FFmpeg and the locally cached
`mlx-community/parakeet-tdt-0.6b-v2` model. On this machine the Parakeet weights were
already cached. To prepare another machine, download them once:

```sh
uv run --project templates/agent/training --extra voice python -c \
  'from huggingface_hub import snapshot_download; snapshot_download("mlx-community/parakeet-tdt-0.6b-v2", token=False)'
```

Try creating a User schema with properties name, class, subjects and methods
getName, getClass, getSubjects, addSubject, removeSubject. Then add a property,
remove one, rename the box, or connect it to another schema. Successful edits
select their target; the next command can refer to that selection. Current schema
contents, the last three outcomes, and stable last-created/last-edited IDs go to
the model on each request. Canvas contents persist locally; outcome memory resets
on page reload. General drawing shapes, groups, geometry, styles, selection, camera,
and available undo/redo history are now included in the model context.

The service validates every function call. The browser checks target IDs, field
existence, limits, and whether the canvas changed before applying an edit. One edit
forms one undo step. Connections use tldraw arrow bindings. Additional execution
guards reject known duplicate-name, deleted-reference, and compound-edit failure
patterns; these guards do not change the raw model benchmark scores. They do not
guarantee that every misunderstanding is caught. Audio conversion files are temporary.

The executor now accepts 18 tools covering shape and text creation, selection,
movement, resizing, deletion, text replacement, color/fill/opacity, duplication,
grouping, layer order, alignment, distribution, flipping, stacking, packing,
undo/redo, pan/zoom, schema properties/methods, and bound arrow connections.
Text height and note dimensions follow tldraw's layout constraints. Each request
still produces one action; compound edits are rejected. There are at most 30 schema
boxes, 60 other shapes, and 30 properties or methods per box. The expanded prompt
budget is 6,144 tokens, including a 256-token response allowance. Selection context
can include the full canvas; an individual edit can target up to 30 shapes.

The retained 15,000-update adapter was trained on the earlier six-tool contract.
Its reported accuracy does not establish accuracy on the expanded tools. Four
Playwright tests verify the native editor workflows, including nested group movement,
manual pointer dragging, bound-arrow cleanup, redo preservation, and atomic rejection
of stale or missing targets. Run them with `pnpm --filter tldraw-agent test:voice`.
The optional model integration tests send commands through a running voice service
and apply its responses in the native editor. Set `VOICE_MODEL_URL` to the service
URL and `VOICE_SMOKE_AUDIO` to the synthetic speech fixture to enable both tests.

The general-workflow training set contains 329,213 training examples, including
4,000 editing sessions of 24–80 turns, 24,000 compound-request counterexamples, and
6,000 corrected requests. It preserves 1,200 validation examples and 2,880 test
examples, including 12 validation and 24 test sessions. Earlier labels marking
supported move, delete, styling, and undo commands as unsupported are excluded.
The reproducible builder is `training/build_workflow.py`; generated artifacts stay
under `training/runs/v5-general-canvas/`. Session state checks use a Python document
simulation; they do not measure pixel layout or replace the native browser tests.
The private Kaggle run
`rohithresearch/canvas-270m-v5-general-workflow` completed all 6,000 updates on two
Tesla T4 GPUs, with batch eight, micro batch two per GPU, completion window
160, and a 6,144-token limit. It starts from the retained 15,000-update adapter
with a fresh optimizer. The checked-in profile is `training/workflow-config.yaml`.
All generated calls passed round-trip checks; the longest example uses 4,330
tokens and the longest response uses 92. The downloaded weights exactly match the
final checkpoint, with verified dataset, task, and warm-start hashes. Validation
stopped when a simulated manual drag targeted a shape the baseline had failed to
create. The evaluator now retains that state mismatch without crashing; validation
finished on `rohithresearch/canvas-270m-v5-validation` using the saved weights.
Command validation accuracy is 88.42% raw and 87.75% with execution guards. Across
480 closed-loop validation turns, exact action accuracy is 85.00% raw and 85.42%
guarded; exact simulated state agreement is 19.58% in both modes. These results
show that errors can carry into later edits. Four native browser tests pass. With the
6,000-update model, the synthetic speech integration test passes, while the editing
test fails at "Make it blue." Separate development probes also exposed failures
for moving down, ungrouping, and redo; these probes are not a held-out benchmark.
Run metadata and live output are `training/runs/kaggle/workflow/run.json` and
`live.log`. The observer downloads results and verifies final checkpoint weights
and dataset/task hashes; it compares raw and guarded validation separately and
preserves the fresh test set. Connection failures are retried within the monitor's
13-hour runtime without restarting training.

The next run, `rohithresearch/canvas-270m-v6-spoken-training`, completed 2,000
additional updates on two Tesla T4 GPUs. It started from the verified 6,000-update
weights with a fresh optimizer and the profile `training/spoken-config.yaml`.
Its 117,600 training examples comprise 48,000 spoken command variations, 48,000
replayed examples, and 21,600 turns in 800 editing sessions. These sessions repeatedly
group, move, ungroup, manually drag, style, delete, undo, and redo. The builder retains
the original validation and test examples and sessions unchanged. Training artifacts
stay under `training/runs/v6-spoken-workflow/`; remote run metadata and logs are under
`training/runs/kaggle/spoken/`.

The final checkpoint is downloaded and verified. Both native model integration tests
pass: a 12-command editing session with manual dragging and undo/redo, and synthetic
speech through Parakeet to a schema edit. The first attempt hit a transient browser
helper error and an idle-sleep timeout; rerunning the unchanged tests with the Mac
awake passed. The native check now prevents idle sleep only while its browser tests
run.

Final scoring completed and all six reports match the retained adapter, dataset,
and task hashes. Guarded validation is 1,058/1,200 (88.17%). The fresh test set is
2,163/2,880 (75.10%); action accuracy across 1,440 simulated session turns is 67.15%.
Exact canvas state agreement is 10.56%, with no completely correct test sessions.
Passing the short native tests therefore does not establish reliable long-session
editing. The frozen evaluation made 7,680 predictions on two T4s in about 1 hour
55 minutes, without prefix reuse; guarded independent tests had a 1.56-second
median. Reports are retained under `training/runs/kaggle/spoken/final-score/`.

A separate continuation waits for the native tests to pass before
launching selected-model scoring on Kaggle: raw and guarded validation, plus guarded
fresh tests covering 2,880 independent commands and 1,440 session turns. A failed
native check leaves the fresh test set untouched. Final reports must match the same
adapter, dataset, and task hashes and include every expected example. The checkpoint
upload receipt permits resuming after temporary Kaggle readiness errors without
uploading the weights again. Reattach the
training observer with:

```sh
uv run --project templates/agent/training python templates/agent/training/kaggle_follow.py \
  --kernel rohithresearch/canvas-270m-v6-spoken-training \
  --workflow templates/agent/training/runs/kaggle/spoken --training-only --native-check
```

Start the final scoring continuation separately:

```sh
uv run --project templates/agent/training python templates/agent/training/kaggle_follow.py \
  --kernel rohithresearch/canvas-270m-v6-spoken-training \
  --workflow templates/agent/training/runs/kaggle/spoken --await-final-score
```

`--native-check` starts a temporary local service after checkpoint verification,
runs the model integration tests, and saves results under `native-check/`. It
releases the service afterward and retains the verified adapter if a check fails.

### Accuracy refinement

The version-seven data audit found that version six trained on 16,000 example
positions, only 13.6% of its 117,600 generated rows. Some arrangement operations
received seven or eight examples; panning had one training sentence construction
and was omitted from spoken augmentation. The retained dataset also contained
duplicate prompts and a few obsolete unsupported-action labels.

`build_workflow.py --accuracy-refinement 24000 --seed 66` builds a new corpus from
the frozen version-six source. It balances all 18 tools, all 20 arrangement
operations, eight canvas commands, and three rejection reasons. Training inputs
are unique, with separate train, development, and test names and sentence
constructions. Sequential examples include manual edits, selection changes,
undo/redo, stale references, and recovery. Native capability checks reject labels
that request unsupported fills, frame colors, or note resizing; note dimensions
and text height are not requested as controllable dimensions. Actions are checked
against their captured context, and complete sessions are replayed before export.

Historical validation and test artifacts are preserved separately. Twelve
historical command-validation rows with incompatible native capability labels are
excluded from current validation; historical session oracles are excluded because
they omitted visible connection arrows and used different context ordering.
Current development has 1,700 command examples and eight sessions totaling 256
turns. A new test reservation contains 980 command examples, including the oracle
contexts for 12 sessions totaling 480 turns. It is created and hashed before
training examples are generated. These synthetic tests do not establish accuracy
on unstructured human speech or pixel-perfect layouts.

`training/accuracy-config.yaml` retains FunctionGemma 270M and rank-32 LoRA, with a
fresh optimizer, learning rate 0.00001, global batch eight, and 3,000 updates. That
is one complete pass over the 24,000 rows. `training-exposure.json` records the
selected IDs and action counts; checkpoint recovery verifies complete coverage.
Validation selects among updates 500, 1,500, and 3,000 using 256 fixed development
IDs and six development sessions, then compares the chosen model with the retained
baseline on full validation. The original final checkpoint is preserved, and no
candidate replaces the baseline after an observed action or state regression.
Fresh tests run only after that gate and native integration checks pass. State
reports now separate document, selection, and camera agreement.

The private Kaggle job
[`rohithresearch/canvas-270m-v7-accuracy-refinement`](https://www.kaggle.com/code/rohithresearch/canvas-270m-v7-accuracy-refinement)
completed 3,000 updates from source commit `2d383d112`. The frozen corpus and sources are under
`training/runs/v7-accuracy-workflow-revised/`; launch metadata and live logs are
under `training/runs/kaggle/accuracy/`. Preflight passed 73 Python tests, the
function-call token round trips, and a sampler check showing all 24,000 unique
rows selected exactly once. The longest input is 4,242 tokens and response is 68.
The monitor uses the frozen source, prevents idle sleep, retrieves and verifies
checkpoints, runs both native model integration tests, and starts final scoring
only after validation and native checks pass. Desktop completion notifications
are configured.

On the fixed 256-command development subset, the 500-update checkpoint scored
88.67% guarded exact accuracy versus 78.52% for the starting adapter. Across the
six development sessions (192 turns), action accuracy was 67.19% versus 33.33%.
That checkpoint regressed on zoom-out, ambiguous targets, and unsupported
requests, so it failed the per-operation and rejection-reason promotion gate.
Neither later checkpoint passed. The previous adapter was retained; full
validation, native model checks, and the fresh test were not run for a replacement.
These scores use the version-seven development contexts and simulator. Recorded GPU
activity averaged 96.08% and 96.25% on the two T4s; median speed was 0.325 optimizer
updates per second, and training took 9,633 seconds including warmup and saves.

`build_workflow.py --quality-supplement --output training/runs/quality-next/supplement-revised-v2 --seed 72`
creates a separate supplement with 2,400 training and 408 development examples.
It contains 768 training and 96 development contrast pairs across 35 scenarios,
plus 12 training and three development sessions of 72 turns. Same-context pairs
distinguish panning, shape movement, and zoom in/out; other pairs exercise named and
selected targets, stale names, duplicate names, native capability refusals,
literal captions, actual self-corrections, and recovery from simulated wrong or
rejected prior actions. Creation covers all eight drawing kinds. Group examples
distinguish an unsupported mixed-group edit from styling its supported child.
Each input has a replay receipt and a canonical label;
the export checks full-input uniqueness, label conflicts, capability support,
session replay, split separation, and execution-guard changes. The opt-in mode
does not use or change the frozen version-seven data or test reservation.

Supplemental note contexts use default 200-by-200 dimensions, and frame contexts
report black, matching the native shape utilities. The guard now excludes quoted
captions and explicitly cancelled clauses when checking for extra edits or stale
history references. Native browser checks cover note and frame contexts and moves
targeting a child of a group. Full geometry and zoom-to-fit remain simulated
approximations. The completed version-eight run below trained this supplement,
but no checkpoint passed the replacement gate.
Independent semantic review corrected 54 unavailable undo/redo refusal reasons to
`missing_target`. All 2,808 final inputs pass function-call token round trips;
the longest input is 4,326 tokens within the 6,144-token sequence limit.

`build_workflow.py --lexical-refinement --output training/runs/quality-next/lexical-refinement-v1 --seed 72`
creates a separate ordinary-English refinement with 1,200 training and 200
development rows. These are generated state variants from 132 authored training
sentence structures and 66 withheld development structures, not 1,400 independent
human conversations. Twenty training and four development conversations each
contain twelve turns. They create and refine real simulated objects, change
selection, refer to earlier edits, recover a deletion, and distinguish an actual
wrong action from a rejected request. Contrast pairs cover undo versus redo,
zoom versus pan, deselection versus deletion, attributes, movement directions,
rotation, singular and plural references, names, and literal captions.

Full-input uniqueness, canonical labels, native capabilities, setup receipts,
conversation replay, guard equivalence, and function-call token round trips are
checked before export. The development wording is separate from training, and
the existing reserved test is preserved. Current target guards reject singular
references that would edit multiple shapes while preserving named antecedents,
fields, plural requests, corrections, and available undo/redo. These data and
guard checks do not establish model accuracy on spontaneous human speech.

The workflow follows Google's [FunctionGemma fine-tuning guidance](https://ai.google.dev/gemma/docs/functiongemma/finetuning-with-functiongemma).
The [APIGen paper](https://arxiv.org/abs/2406.18518) provides a reference for checking
function-call data through format, execution, and semantic validation. The checks
here are authored rules and simulator replay, rather than independent human
review of every example.

## Local action-model training

The `training/` directory is a Python lab for learning to fine-tune a small instruction
model. It translates text commands and canvas context into one validated function
call. The local voice mode above connects that call to speech and the tldraw editor.

The lab uses Python 3.12, `uv`, MLX-LM, and a pinned MLX conversion of
[FunctionGemma 270M](https://huggingface.co/mlx-community/functiongemma-270m-it-bf16).
Training and inference run on an Apple Silicon Mac without a paid inference API.
The checkpoint retains the Gemma license.

Both `evaluate` and `evaluate-sessions` accept `--cache-prefix` to reuse the first
2,048 tool-instruction tokens. Each request receives its own copy of the cached
state; changing the canvas or recent history still recomputes the remaining prompt.
The original prefill boundaries and greedy decoding are retained. In 36 training
probes covering all 18 tools on Apple Silicon, raw outputs and validated actions
matched exactly; median time fell from 0.92 to 0.55 seconds.

Independent command evaluation also accepts `--batch-size 8`. A same-T4 check of
36 training probes increased throughput from 0.73 to 4.03 commands per second
(5.56 times), with average GPU activity rising from 22.6% to 55.8% and maximum sampled
GPU memory from 1,919 to 4,847 MiB. Batches of four and eight both retained exact raw
outputs, validated actions, and errors across those probes and 64 existing validation
cases covering all 18 actions. New Kaggle validation and selected-model scoring jobs
reuse the prefix and batch eight independent commands. Session turns retain their
sequence and reuse only the prefix. These are speed and output checks, not new
accuracy measurements or a measurement of peak GPU capacity. The completed full
accuracy reports retain the original evaluator.

Run these commands from the monorepo root. Install
[uv](https://docs.astral.sh/uv/getting-started/installation/) if it is missing.

```sh
uv sync --project templates/agent/training --python 3.12
uv run --project templates/agent/training python templates/agent/training/build_dataset.py --replace --seed 43
uv run --project templates/agent/training python templates/agent/training/lab.py inspect
```

`examples.jsonl` contains 123,851 synthetic examples: 122,530 training, 470 validation,
and 851 test examples. Each line contains a command, canvas context, recent outcomes,
the expected function call, and a scenario group. `sessions.jsonl` contains replayable
sessions: 2,000 training sessions of 12–80 turns, eight validation sessions totaling
300 turns, and 16 test sessions totaling 600 turns. Both generated files are ignored
by Git; regenerate them using the command above. Add reviewed scenarios to the generator.
Keep paraphrases of the same scenario in the same split. Training uses examples to
update weights; validation helps choose settings; the test split measures the final
chosen model. These examples are a controlled learning exercise, not a benchmark of
natural speech or general diagram editing. `build_dataset.py` defines the entity
catalogue, phrasing, and deterministic labels; the current seed is 43. Train, validation,
and test sets have separate target entity names and phrasing. Entire sessions stay
in one split, and the audit rejects identical inputs across splits.
Regeneration replaces manual edits, so keep reviewed additions in the generator or
save a copy of the JSONL file first.

Version four retires all version-three examples into training. Fresh held-out
entities and phrasing are used for validation and test. Fields and methods are
sampled independently for each canvas, including plural, camelCase, snake_case,
acronym, and numbered identifiers. Counterfactual removal pairs keep the command
constant while adding or removing the requested field from the canvas. Sessions
include sequential additions, removals, renames, connections, corrections, references
to recent edits, duplicate names, missing targets, selection changes, manual deletions,
and box reordering. Unsupported questions and compound requests must produce `no_action`.

The actions create a schema box, add or remove a property, rename a schema, connect
schemas, or return `no_action` for a missing or ambiguous target or unsupported request.
The `create_schema_box` argument `fields` represents the box's properties. FunctionGemma's
chat template reserves `properties` while formatting schemas, so using `fields` keeps
this argument visible in the tool declaration. Existing canvas boxes still use
`properties`. Methods and field names retain their requested spelling and order.
The model sees temporary IDs such as `box1` and `box2`, plus an explicit selection
status. The parser maps those IDs back to real canvas IDs before validating an action.
This removes long ID copying from the model's task while retaining exact target checks.

`CanvasSession` maintains the current canvas and connections, stable IDs of the last
created and edited boxes, and three recent command outcomes. IDs in this memory are
mapped against the current canvas on every turn; reordering cannot silently change
a reference, and a deleted target becomes missing. Successful edits select their
target. A named target overrides selection, while "it" uses a single current selection.
Explicit "last created" and "last edited" references use session memory.
The model still produces one action per turn. Arbitrary conversation recall, undo,
method edits, styling, and multiple edits in one utterance are outside this action
contract. `CanvasSession` supplies the Python evaluation simulation; the local
voice mode executes the same action contract in the live tldraw editor.

The inspector audits labels, conflicting annotations, and splits, displays the model's actual prompt, measures
token lengths, and checks that prompt masking and function-call parsing agree with
the tokenizer. In training, prompt masking computes loss on the expected answer.

First, measure the original model on validation examples:

```sh
uv run --project templates/agent/training python templates/agent/training/lab.py evaluate \
  --split valid --output templates/agent/training/runs/baseline-valid.json
```

Next, train an adapter. Choose a new run name for every experiment:

```sh
uv run --project templates/agent/training python templates/agent/training/lab.py train \
  --run my-first-lora
```

`training/config.yaml` specifies 15,000 steps, batch size 8, a 200-update learning
rate warmup to 0.00003 followed by cosine decay to 0.000003, LoRA rank 32 with scale
8, gradient checkpointing, and a 2,048-token sequence limit. The longest current
example is 1,444 tokens. This processes 120,000 examples, nearly one pass over training.
The Colab job starts from the final version-three adapter; this initializes weights
but starts a fresh optimizer. Its checksum is saved in the experiment metadata.
The Python trainer projects only completion positions to vocabulary logits and
excludes padding from loss. A numerical test checks loss and gradients against a
full projection. Token arrays use compact storage, and batches are sorted by actual
token lengths. These changes reduce memory needed for the longer session prompts.
The inspector saves token arrays with dataset and task checksums. Colab verifies
those checksums and uses the prepared tokens, avoiding another full tokenization
pass on the GPU runtime. The longest completion has 83 tokens; the loss window is 96.
LoRA freezes the base model and trains adapter matrices across its transformer layers.
Rank controls adapter capacity; learning rate controls update size; steps control how
many updates run. Training and validation loss, memory use, and throughput appear
in `runs/my-first-lora/training.log`. Adapter weights and checkpoints are saved under
`runs/my-first-lora/adapter/`. `--iters 20` gives a shorter practice run.
Use `--config path/to/config.yaml` to run a separate configuration. Each experiment
saves its configuration, dataset, and task definition alongside the adapter.

Evaluate the adapter on the same validation examples and compare:

```sh
uv run --project templates/agent/training python templates/agent/training/lab.py evaluate \
  --split valid --adapter templates/agent/training/runs/my-first-lora/adapter \
  --output templates/agent/training/runs/my-first-lora/valid.json
uv run --project templates/agent/training python templates/agent/training/lab.py compare \
  templates/agent/training/runs/baseline-valid.json \
  templates/agent/training/runs/my-first-lora/valid.json
```

Reports include exact action accuracy, valid call rate, tool selection accuracy,
latency, peak MLX memory, and each raw prediction and error. Exact accuracy checks
the complete action and its arguments, including target IDs and list order. The
memory measurement covers MLX allocations, not total system RAM. Reports also record
the model revision, dependency version, and hashes of the dataset and task definition.
Use raw errors and validation results to improve data or settings. Lower loss alone
does not establish correct actions.

Training's periodic loss check uses 32 validation batches. The `evaluate` command
generates and checks complete actions on every example in the chosen split unless
you pass `--limit`. Choose the adapter using these complete-action results.

When you have chosen an adapter, run both evaluations with `--split test` and new
output paths. Keep test results out of decisions about that experiment's training.

Try your own text command:

```sh
uv run --project templates/agent/training python templates/agent/training/lab.py predict \
  'Create a User schema with properties name, class, subjects and methods getName, getClass, getSubjects, addSubject, removeSubject.' \
  --adapter templates/agent/training/runs/my-first-lora/adapter
```

For edits, pass `--canvas path/to/canvas.json` with a `schemas` array of objects
containing `id`, `name`, `properties`, and `methods`, plus `selected_ids`. The model
receives that context each time. Unknown IDs, extra arguments, incomplete calls,
and attempts to remove a missing property fail validation. A valid call can still
misinterpret the command, which is why semantic evaluation is necessary.

Single-turn evaluation supplies the correct saved canvas and history for every
example. Session evaluation instead executes the model's predictions against its
own evolving state. It never replaces an incorrect canvas with the expected state.
Reports measure exact action accuracy, current state agreement, completely correct
sessions, final state agreement, unwanted mutations on `no_action` requests, recovery,
and accuracy in ten-turn intervals:

```sh
uv run --project templates/agent/training python templates/agent/training/lab.py evaluate-sessions \
  --split valid --adapter templates/agent/training/runs/my-first-lora/adapter \
  --output templates/agent/training/runs/my-first-lora/sessions-valid.json
```

Synthetic results do not establish reliability on real speech or unrestricted
canvas edits. The session tests measure supported actions for up to 80 turns; they
do not establish how long accuracy remains stable beyond that range.

The Python environment, generated training data, reports, and adapters are ignored
by Git. Check the lab without running a model:

```sh
uv run --project templates/agent/training ruff check templates/agent/training
uv run --project templates/agent/training python -m unittest discover \
  -s templates/agent/training -p 'test_*.py'
```

See the [MLX-LM training guide](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)
and Google's [FunctionGemma fine-tuning tutorial](https://ai.google.dev/gemma/docs/functiongemma/finetuning-with-functiongemma)
for the underlying training tools and model formatting.

### Colab training

Install Google's CLI and complete its Google sign-in flow in your terminal:

```sh
uv tool install google-colab-cli
colab --auth=oauth2 sessions
```

Paste Google's authorization code into the terminal prompt. Enter commands without
surrounding shell backticks. GPU allocation depends on your Colab account and quota.

Bundle the current lab and data, allocate a T4, and upload the bundle:

```sh
uv run --project templates/agent/training python templates/agent/training/lab.py bundle-colab
colab --auth=oauth2 new -s canvas-270m-v4-large --gpu T4
for bundle_part in templates/agent/training/runs/canvas-training.zip.part-*; do
  colab --auth=oauth2 upload -s canvas-270m-v4-large \
    "$bundle_part" "/content/${bundle_part##*/}"
done
```

Run the job, which installs CUDA training dependencies in an isolated environment:

```sh
colab --auth=oauth2 exec -s canvas-270m-v4-large --timeout 43200 \
  -f templates/agent/training/colab_job.py
```

The job retains FunctionGemma 270M, converts its checkpoint to FP16 for T4 training,
and saves the adapter, configuration, dataset snapshot, hardware, log, and validation
and session validation reports. Training runs directly in the notebook kernel, with
checkpoints saved every 500 updates. The bundle includes the version-three final
adapter as the warm start. It refuses to train if the MLX GPU backend is unavailable.
The test set is
reserved for the chosen adapter. Extract the results zip into `training/runs/`;
the local prediction command reads the saved dtype and recreates the FP16 base
checkpoint before loading the adapter.
Uploads and final downloads use 16 MB parts because the CLI sends files in an
encoded request. The job reassembles uploads; `colab_follow.py` verifies checksums
while reassembling results, then stops the GPU before local test evaluation.

### Session experiment

The version-four experiment uses the final version-three adapter as a warm start,
the 122,530-example training split, and a target of 15,000 batch-eight updates on
a Tesla T4. Its console is captured in
`training/runs/colab-270m-v4-sessions-console.log`. The job generates both validation
reports before packaging results. The latest complete local backup is update
5,000, including adapter weights, optimizer, random state, and dataset/task
checksums. The runtime reached update 5,500 before disappearing again; its final
backup was incomplete. The last training report measured 0.817 updates/second and
4.038 GB peak MLX allocations. The October 1 recovery attempts returned
`Service Unavailable` for all four replacement T4 allocations. The delayed recovery
window also failed all four allocation attempts; its log is
`runs/v4-recovery-delayed.log`. No Colab GPU remains active. The 15,000-update target
and fresh test evaluation remain pending.
`colab_follow.py` backs up every 500-update checkpoint, including weights, optimizer
and random state. It checks remote liveness every two minutes and permits up to two
automatic recoveries from the newest verified checkpoint. A `Service Unavailable`
allocation response allows three delayed retries; other allocation errors stop
recovery. It downloads final
results, stops the GPU, and runs the fresh tests locally. Run it in
a second terminal while the Colab execution remains active:

```sh
uv run --project templates/agent/training python templates/agent/training/colab_follow.py --session canvas-270m-v4-recover-r1
```

The initial large session reached update 2,400 before its runtime disappeared.
The server reported no active session, without a termination reason. The old monitor
missed that disappearance and retained only update 500; its log is preserved in
`runs/colab-270m-v4-sessions-interrupted-2400-console.log`. The recovery job is
`canvas-270m-v4-recover`, using the verified update-500 weights and the same dataset.
Its target remains 15,000 total updates, with completed batches skipped and the
learning-rate schedule continued from update 500. This older backup has no optimizer
state, so the first recovery restarts the optimizer. Subsequent backups retain it.
The corrected recovery path passed one real optimizer update; checkpoint tests also
verify that restoring optimizer state produces the same next update. All 22 lab
checks passed. The retained update-500 checkpoint scored 292/470 exact validation
actions (62.1%), compared with 213/470 (45.3%) before the larger run. Its valid-call
rate was 461/470 (98.1%). This interim measurement uses the unchanged validation
cases; final test accuracy and closed-loop session reliability remain pending.
The report is `runs/v4-retained-500-valid.json`.

Update 4,000 was evaluated on the unchanged validation split:

| Measurement                                           | Before the large run | Update 4,000    |
| ----------------------------------------------------- | -------------------- | --------------- |
| Exact single-command actions                          | 213/470 (45.3%)      | 361/470 (76.8%) |
| Exact closed-loop actions                             | 120/300 (40.0%)      | 223/300 (74.3%) |
| Matching canvas states                                | 35/300 (11.7%)       | 71/300 (23.7%)  |
| Perfect sessions                                      | 0/8                  | 1/8             |
| Matching final session states                         | 0/8                  | 2/8             |
| Unwanted mutations on 140 expected no-action requests | 28                   | 37              |

All 51 schema-creation validation cases passed. Supported edits scored 210/219
(95.9%), while `no_action` scored 151/251 (60.2%). Long-session state agreement
remains weak: errors accumulate, and unwanted edits increased. These are interim
raw model results, without the serving guards. The reports are
`runs/v4-retained-4000-valid.json` and `runs/v4-retained-4000-sessions-valid.json`.
The fresh 851 single-command test cases and 600 session test turns remain unused.

The local voice integration adds six passing checks for execution guards, deleted
history IDs, request validation, and local origins. Browser checks exercised a
synthetic microphone recording through Parakeet and the adapter into a real canvas,
plus all five editing tools and no-action cases. They verify the integration path,
not general model accuracy. That browser session also exposed model errors: an
unrequested method on one schema and a changed connection label. The model remains
a prototype; natural microphone noise and accents are not benchmarked yet.

To create a recovery bundle from a verified local checkpoint:

```sh
uv run --project templates/agent/training python templates/agent/training/lab.py bundle-colab \
  --resume templates/agent/training/runs/colab-270m-v4-sessions-checkpoint-500
```

The first attempt failed on a helper import before any optimizer updates; its
console is preserved as `runs/colab-270m-v4-sessions-attempt1-console.log`.
The corrected trainer passed a real cached batch-eight update locally with 2.84 GB
peak MLX memory. A NumPy integer conversion error in the first cached attempt was
fixed and rechecked through the exact training path before retrying. CUDA also
required `MLX_CUDA_GRAPH_CACHE_SIZE=4096` for the larger compiled training graph;
the bootstrap and Colab job set it before MLX initialization.

Before training, the version-three adapter was evaluated with the new prompt and
new validation cases: 213/470 exact actions (45.3%). On eight closed-loop validation
sessions it made 120/300 correct actions (40.0%); none of the eight sessions or
their final states was completely correct. It made 28 unwanted mutations across
140 expected `no_action` requests. The reports are `runs/v4-baseline-valid.json`
and `runs/v4-baseline-sessions-valid.json`. These scores measure a harder task with
new session context; they are not directly comparable with version three's old
single-command test score. The same validation cases will measure the new adapter.

### Kaggle recovery

The October 1 Kaggle recovery uses the verified update-5,000 checkpoint and the
unchanged 122,530 training examples. The target remains 15,000 total updates.
Both the input dataset and training notebook are private. Kaggle allocated two
Tesla T4 GPUs with 15,360 MiB each and driver 580.178.04; this trainer uses device
zero. It passed real CUDA forward and backward checks. The first two launcher
attempts failed during environment setup. The third passed CUDA checks but exposed
MLX 0.32.3's read-only random-state API during restoration. The fourth fixes both
library precedence and random-state restoration, and has entered training with the
optimizer and random state restored. A regression check verifies that restoring a
saved random key reproduces the next random draws exactly.
The first progress report confirms update 5,100: 100 new updates completed, with
training loss 0.009 and peak MLX memory of 4.111 GB. The recovered job completed
15,000 updates, and the result monitor verified and retained its final checkpoint.

`training/kaggle_job.py` verifies the input bundle and retained weights, restores
optimizer and random state, and runs the existing MLX trainer in an isolated Python
3.12 environment. Checkpoints are saved every 500 updates in the Kaggle outputs.
`training/kaggle_follow.py` monitors the batch job, verifies the downloaded archive
checksums and completed update count, and checks the frozen source and dataset
before evaluating the fresh tests locally. Training stays on Kaggle.

The completed run is
<https://www.kaggle.com/code/rohithresearch/canvas-270m-v4-training>.
Its local launch record and logs are under `training/runs/kaggle/`. Follow progress:

```sh
kaggle kernels status rohithresearch/canvas-270m-v4-training
kaggle kernels logs rohithresearch/canvas-270m-v4-training --follow
```

The local result monitor runs independently of this terminal. To resume monitoring:

```sh
uv run --project templates/agent/training python templates/agent/training/kaggle_follow.py \
  --kernel rohithresearch/canvas-270m-v4-training
```

Completed reports are saved under
`training/runs/kaggle/download/colab-270m-v4-sessions/`. The run directory retains its
original Colab name for artifact compatibility. The completed model scored
441/470 validation commands (93.8%). It correctly handled 218/219 supported
actions and 223/251 requests requiring `no_action`. In closed-loop validation,
284/300 actions matched (94.7%), but canvas state matched on only 75/300 turns
(25%). Four of eight sessions had no action errors; nine unsupported or missing
requests caused unwanted changes.

On the reserved test set, the completed model matched 765/851 commands (89.9%),
with valid calls on 98.6% and correct tool selection on 95.7%. Its 16 closed-loop
test sessions contained 600 turns: 487 actions matched (81.2%), canvas state
matched on 187 turns (31.2%), three sessions had no action errors, and three ended
with the expected canvas. There were 25 unwanted mutations on 289 unsupported or
missing requests. These are raw model results; runtime guards are not included.
The drop in matching canvas state during longer sessions remains a measured gap.

#### GPU speed measurement

The October 2 speed check ran separately on Kaggle with the same 270M model,
FP16 base, rank-32 LoRA, retained update-5,000 weights, and completion loss. Each
case ran in a fresh process with a four-minute limit; the notebook had a
15-minute limit. It alternated batch-eight inputs of 1,153 and 1,473 tokens for
24 updates, excluded the first 12 updates for compilation warmup, and reported
the median of the remaining three four-update intervals.

| Settings                            | Updates/second | Examples/second | Result                             |
| ----------------------------------- | -------------: | --------------: | ---------------------------------- |
| Current settings                    |          0.703 |            5.63 | Baseline                           |
| Retain up to 4 GiB of memory cache  |          0.686 |            5.49 | No measured speed improvement      |
| Also disable gradient checkpointing |              — |               — | CUDA out of memory during training |

The two successful cases produced identical initial loss and gradients on a
single longest-context example. These short, controlled measurements do not
establish the fastest possible configuration or predict a complete training run.
They do not justify changing the current settings. The active training job
continued throughout the check; neither its configuration nor checkpoint was
replaced. Using both allocated GPUs would require a distributed trainer.

The benchmark is
<https://www.kaggle.com/code/rohithresearch/canvas-270m-t4-throughput>.
The downloaded results and traceback are under
`training/runs/kaggle/throughput-download/`. Reproduce the bounded cases with
`training/gpu_benchmark.py` through `training/kaggle_job.py --task benchmark`.

Subsequent notebook versions expanded the comparison to effective batch eight,
full-batch gradient checks, two GPUs, and an 80-update verification workload with
eight sequence lengths. Version 2 measured 0.743 updates/second for the verified
baseline. Padding accounted for only 1.4% of input tokens. Gathering only actual
answer positions, changing checkpoint coverage, and accumulating smaller batches
did not produce a verified improvement. Some changes produced similar initial
losses but materially different FP16 gradients.

Version 3 repeated the baseline gradient calculation exactly and measured 0.750
updates/second on the longer verification workload. A two-GPU PyTorch trial
measured 0.927 updates/second versus 0.647 for that version's short baseline trial,
but failed the gradient comparison. That trial also used PyTorch's default Adam
bias correction, unlike the production MLX optimizer, so its training trajectory
is not equivalent. The experimental runner now cancels that correction and checks
the first update against the MLX Adam formula. Neither trial changed production.

Version 4 tested the corrected optimizer, full-precision gradient references,
compilation, and structured sliding-window attention. The two-GPU MLX trial ran
at 1.312 updates/second versus 0.719 for its short single-GPU baseline, but its
FP16 gradients differed by 48.2% relative L2. The full-precision PyTorch trial
matched the MLX full-precision reference within 0.0068% relative L2 and checked
the optimizer update within 3.8e-9, but ran at 0.492 updates/second. The longer
baseline verification measured 0.768 updates/second.

On that low-loss longest-context batch, the single- and two-GPU FP16 gradients
differed from the full-precision reference by 92.8% and 70.8% relative L2,
respectively. A difference from the original FP16 calculation therefore does not
establish worse task accuracy. These measurements cover one initial gradient
calculation, not convergence or command accuracy.

Version 5 completed the bounded search with a supported xFormers mask and checks
for finite training losses and final adapter weights. The compiled case checks
gradients through the compiled function itself.

| Version 5 configuration          | Short trial updates/second | Gradient result                            |
| -------------------------------- | -------------------------: | ------------------------------------------ |
| Existing MLX FP16, one GPU       |                      0.692 | Reference                                  |
| MLX FP32, two GPUs               |                      0.435 | 0.0030% relative L2 against FP32 reference |
| xFormers FP16, two GPUs          |                      1.110 | 48.7% relative L2 against FP16 reference   |
| xFormers FP32, two GPUs          |                      0.644 | 0.0074% relative L2 against FP32 reference |
| Compiled xFormers FP32, two GPUs |                      0.600 | 0.0076% relative L2 against FP32 reference |

The retained configuration is the existing trainer. Its final 80-update
verification across eight context lengths measured 0.752 updates/second, or 6.02
examples/second, with finite losses and weights. The gate requires initial loss
error below 1e-5 and gradient relative L2 below 1% against the corresponding
precision reference. This conservative gate preserves the current calculation;
it does not establish that faster FP16 alternatives have worse task accuracy.
The numerical check covers the longest batch, while the throughput and finite
checks cover all eight lengths. None of these measurements is a convergence or
held-out accuracy comparison. The highest measured FP16 speed, 1.312
updates/second in version 4, remains a candidate for a separate accuracy trial.
No faster replacement was adopted, and the active 15,000-update job continued.

The source of the PyTorch experiments is `training/torch_benchmark.py`; it is a
diagnostic implementation, not the production trainer. Versions 2, 3, 4, and 5 are
retained under `training/runs/kaggle/search-download/`, `attention-download/`,
`precision-download/`, and `final-download/` respectively. The active production
run kept its original configuration throughout.

The hardware constraints matter: [MLX 0.32.3's fused CUDA attention kernel](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/cuda/scaled_dot_product_attention.cpp)
requires Ampere or newer and head dimensions at most 128. T4 and FunctionGemma's
256-dimensional attention heads fall outside those limits. The
[xFormers CUTLASS implementation](https://github.com/facebookresearch/xformers/blob/v0.0.32.post2/xformers/ops/fmha/cutlass.py)
supports older GPUs and head dimension 256. Local attention support differs
between its forward and backward kernels: training rejects the bottom-right local
mask, while the block-diagonal causal local mask is listed as supported. The
experimental runner checks the materialized mask, including separation between
examples, before measuring that representation.

#### Bounded Modal H100 measurement

`training/modal_benchmark.py` runs one H100 worker with the frozen version-seven
training contract and retained version-six warm start. It samples 64 training
examples across eight length quantiles, including the longest batch. Development
and test examples are excluded. Each successful stage measures 40 optimizer
updates after eight warmup updates with effective batch eight.

The first stage uses micro batch two, accumulation four, and checkpointing of all
layers. A second stage can use micro batch eight without checkpointing and gather
only actual completion positions. It runs only if the first stage is finite,
uses less than a quarter of GPU memory, and leaves sufficient time. Median and
longest batches must pass initial loss and all-trainable-gradient comparisons
before the second stage trains. Neither stage promotes an adapter or measures
action accuracy.

The preparation command requires the ignored frozen source and token cache under
`training/runs/v7-accuracy-workflow-revised/`, the converted FP16 base under
`training/runs/base-bb327a9a-float16/`, and the verified version-six adapter under
`training/runs/kaggle/spoken/download/refinement/dual-window/adapter/`. It verifies
their checksums before building the payload. The cloud image downloads the pinned
public BF16 base on CPU and reproduces the local FP16 file exactly; CUDA headers
are provided by the NVIDIA development image.

Run from the monorepo root after reviewing the resource limits and current
[Modal prices](https://modal.com/pricing):

```sh
uv tool install modal==1.6.0
modal setup
uv run --project templates/agent/training python templates/agent/training/prepare_modal_benchmark.py
uv run --no-project --isolated --with modal==1.6.0 python templates/agent/training/modal_benchmark.py
```

Modal requires a payment method. The runner permits one container and one
function call, disables application retries, bounds image setup to 300 seconds,
and stops its specific ephemeral app after completion or a 690-second active
deadline. These deadlines and resource estimates are not an account spending
cap; check usage before another run. The original benchmark allowance was $2
across setup attempts and the measurement. Local reports, telemetry, launch and
stop receipts are saved under `training/runs/quality-next/modal/`.

The October 3 measurement completed 48 updates with finite losses and adapter
weights. Its steady median was 1.477 updates/second, or 11.82 examples/second;
elapsed steady throughput was 1.438 updates/second. Average sampled GPU activity
was 55.19%, peak sampled VRAM was 9,411 MiB, and mean power was 351.02 W across
26 steady samples. This does not establish peak GPU utilization. The median rate
was 4.54 times the recent version-seven dual-T4 rate, but the short quantile
workload differs from the full training corpus and does not guarantee that
speedup for a complete run.

The second stage failed its first numerical comparison: initial loss differed by
0.0002383 and gradient relative L2 by 7.28%, above the 1e-5 and 1% limits. It
performed no optimizer updates. The existing trainer remains the retained
configuration; that numerical failure does not measure task accuracy. GPU
function time was 217.74 seconds, with a requested-resource cost estimate of
$0.258 excluding image setup and idle time. The settled ledger before the next
profile recorded about $0.31 of cumulative credit consumption and no cash
charges; billing records can lag.

A separate `--training-profile` measurement retained the 160-token completion
window and tested one larger batch, without the packed output-head experiment.
It permits a finite candidate to be timed after recording numerical differences;
it does not assert numerical equivalence or action accuracy. Both configurations
completed 48 finite updates:

| Measure                       | Micro batch two, checkpoint all | Micro batch eight, no checkpoint |
| ----------------------------- | ------------------------------- | -------------------------------- |
| Steady elapsed updates/second | 1.531                           | 1.371                            |
| Mean sampled GPU activity     | 60.00%                          | 48.21%                           |
| Peak sampled VRAM             | 9,027 MiB                       | 63,655 MiB                       |
| Estimated cost/update         | $0.000774                       | $0.000864                        |

The larger batch was slower and cost about 12% more per update. The quality run
therefore uses micro batch two, accumulation four, and checkpointing of all
layers. This is the faster of the two measured configurations; full H100
utilization and optimal cost across GPU types have not been established. The
second profile consumed about $0.32 of credits, bringing cumulative metered usage
to about $0.63 with no cash billed. Its app is stopped.

#### Bounded Modal quality training

`training/prepare_modal_training.py` creates an immutable version-eight input
using CPU tokenization. It combines the 24,000 original training rows with the
reviewed 2,400-row quality supplement. An exact 3,900-update plan exposes each
original row once and each supplemental row three times, for 31,200 positions at
effective batch eight. Every 13-update block contains ten original batches and
three supplemental batches. Hashes and per-example exposure counts record this
order.

A training-only canvas overlay sets frame colors to black and short note sizes
to 200 by 200, matching the native execution context. It preserves commands,
IDs, history, and action labels, and records every changed field. Original
development and test fixtures remain byte-identical. Session replay is checked
before this overlay; exact native geometry equivalence is not claimed.

The single-H100 trainer starts from the verified version-six adapter, with
rank-32 LoRA on all layers, FP16 weights, completion window 160, and a 5e-6
learning rate. It retains optimizer and random state at periodic and selection
checkpoints. Recovery requires matching data, configuration, training-plan,
checkpoint-step, and state-file hashes. A committed Modal volume preserves the
checkpoints across container exits.

Run from the monorepo root with the same prepared base, frozen version-seven
inputs, and reviewed supplement as above:

```sh
uv run --project templates/agent/training python templates/agent/training/prepare_modal_training.py --micro-batch 2 --checkpoint all
uv run --no-project --isolated --with modal==1.6.0 python templates/agent/training/modal_training.py
```

The local allowance is $8 in credits, including the additional profile. The
runner disables retries, limits the app to one H100 container, bounds setup to
five minutes and active work to at most 105 minutes, and monitors metered usage.
The deadline is preserved during recovery. These controls are estimates and
watchdogs, not an account spending cap; billing can lag. Inputs, live logs,
verified downloads, stop receipts, and billing snapshots live under
`training/runs/v8-quality-modal/`.

Checkpoints 650, 1,950, and 3,900 first face the unchanged 256-command/six-session
selection gate. The chosen checkpoint then faces separate full original and
supplement development gates, including per-action regressions and document,
selection, and camera state. Native SDK checks must pass for the exact downloaded
adapter before scoring the reserved test set. If any gate fails, the retained
baseline remains selected.

The run completed all 3,900 updates from source commit `86496a865` in 3,201 seconds
including warmup and checkpoint saves. All weights remained finite. Median steady
speed was 1.393 optimizer updates per second; elapsed speed was 1.218. Recorded
GPU activity averaged 46.16% and peak VRAM was 13,943 MiB. This is not peak H100
utilization. The GPU app is stopped.

On the fixed original development subset, the frozen runtime reported:

| Adapter              | Commands, 256 | Session actions, 192 | Complete state matches, 192 |
| -------------------- | ------------- | -------------------- | --------------------------- |
| Retained version six | 78.52%        | 33.33%               | 2.60%                       |
| Update 650           | 90.63%        | 48.96%               | 3.65%                       |
| Update 1,950         | 91.41%        | 60.94%               | 17.71%                      |
| Update 3,900         | 90.63%        | 57.81%               | 17.71%                      |

Every candidate regressed on undo, zoom-out, and ambiguous-target cases, so the
retained adapter remains selected. No full development, native replacement, or
fresh test scoring was run. No complete session finished with the expected final
state. These scores use the original version-eight guard; subsequent guard
changes require a new paired evaluation. Billing receipts show approximately
$5.8 total metered usage across the completed Modal experiments, covered by
credits with zero cash charges at the recorded check.

#### Accuracy trial for the faster trainer

Notebook version 6 resumed the same retained update-5,000 adapter and optimizer
for two arms: the existing single-GPU trainer and the 1.312-updates/second
two-GPU MLX candidate. Each trained 500 additional updates with the same 15,000-step
learning-rate schedule and the same 4,000 training-example positions. The job
confirmed matching hashes of the starting weights, optimizer, dataset, task, and
exact batch order before comparing the arms. A local check
verified its sampling against the native resumed iterator across epoch boundaries.

After training, each arm evaluated all 470 validation commands and all eight
validation sessions (300 turns) on its own GPU. Both arms finished with finite
losses and adapter weights. The first 50 updates were excluded from the sustained
speed measurement. The baseline compiled kernels from a cold cache; the second
arm reused that cache, so total elapsed times are not a fair comparison of
compilation costs.

| Measure                                                   | One GPU          | Two GPUs         |
| --------------------------------------------------------- | ---------------- | ---------------- |
| Sustained updates/second                                  | 0.786            | 1.449            |
| Exact validation commands                                 | 411/470 (87.45%) | 416/470 (88.51%) |
| Supported validation actions                              | 217/219          | 218/219          |
| Exact session actions                                     | 241/300 (80.33%) | 249/300 (83.00%) |
| Matching canvas state                                     | 54/300 (18.00%)  | 60/300 (20.00%)  |
| Perfect sessions                                          | 2/8              | 2/8              |
| Matching final canvas state                               | 3/8              | 3/8              |
| Unwanted mutations on 140 unsupported or missing requests | 27               | 29               |

The faster arm gained 84.3% sustained throughput, 1.06 percentage points on
commands, and 2.67 points on session actions. It corrected ten command cases but
regressed five; session actions had 18 improvements and ten regressions.
It failed the conservative observed-accuracy gate because rename accuracy fell
from 41/41 to 40/41 and unwanted mutations increased. The two-GPU configuration
remains an experimental candidate; the completed main model is retained.

This is one paired trial from update 5,000 to 5,500, so it does not establish
accuracy throughout a longer training run. The final test split was evaluated
only for the completed main model. The launch record and live log are
`training/runs/kaggle/quality-run.json` and `quality-live.log`. Downloaded adapters,
evaluation reports, and raw telemetry are under
`training/runs/kaggle/quality-download/`; `quality-comparison.json` was recomputed
locally and verified against those reports.

The GPU sampler recorded one reading per second. Excluding the first 50 updates,
the baseline supplied 562 samples per device over 575.6 seconds; the faster arm
supplied 305 per device over 312.2 seconds. Each T4 had 15,360 MiB of VRAM and a
70 W power limit.

| Setup and device   | Mean GPU busy time | Peak VRAM used | Mean power |
| ------------------ | ------------------ | -------------- | ---------- |
| One GPU, device 0  | 97.45%             | 6,549 MiB      | 65.87 W    |
| One GPU, device 1  | 0.00%              | 0 MiB          | 9.71 W     |
| Two GPUs, device 0 | 95.13%             | 5,057 MiB      | 64.11 W    |
| Two GPUs, device 1 | 95.19%             | 5,149 MiB      | 64.75 W    |

[NVIDIA defines GPU utilization](https://docs.nvidia.com/deploy/nvidia-smi/index.html#utilization)
as the fraction of sampled time with a kernel executing. It does not measure
the fraction of peak FLOPs achieved. Similarly, its memory utilization measures
time spent reading or writing device memory, separately from VRAM allocation.
Peak compute throughput and kernel occupancy were not profiled. The sustained
utilization summary is saved as `quality-download/quality-gpu.json`. The same
parser reproduced the previous retained baseline readings exactly and now adds
GPU summaries to future accuracy trials.

### First local experiment

The original dataset had 195 examples (132 training, 28 validation, 35 test).
Its snapshot is preserved in `runs/first-lora/examples.jsonl`. The first 200-step
run on an M4 Mac with 16 GB RAM took 436 seconds and used a peak
of 1.97 GB of MLX memory. It trained 1.898 million adapter parameters (0.708% of the
base model); the final adapter is 7.62 MB. These are measurements of this run.

| Evaluation                  | Original model | LoRA adapter |
| --------------------------- | -------------- | ------------ |
| Exact validation actions    | 0/28           | 5/28         |
| Exact held-out test actions | 0/35           | 7/35         |
| Valid test calls            | 16/35          | 28/35        |

Validation loss fell from 2.143 to 0.108, but exact test accuracy was only 20%.
The adapter still drops requested fields and methods, chooses incorrect tools, and
handles ambiguous targets poorly. It is an initial training exercise, not a model
ready to execute canvas edits. Inspect the raw errors in `runs/first-lora/valid.json`
and `runs/first-lora/test.json` before building another dataset version. The next
experiment should diversify wording, field names, entity names, and context examples;
choose changes using validation data and reserve a fresh test set once these test
failures influence training. The saved adapter and logs are under `runs/first-lora/`.

### Second local experiment

The previous dataset had 861 examples (672 training, 81 validation, 108 test)
and was used for `runs/data-v2-diverse/`. This run used
600 updates, batch size 2, and no gradient checkpointing, with the same base model,
learning rate, and LoRA rank. It took 19.1 minutes and peaked at 5.17 GB of MLX
memory. Validation loss fell from 2.150 to 0.067. The data and training settings
both changed, so this comparison measures the complete experiment; it does not
isolate the effect of additional examples.

Both adapters were evaluated on the same new validation and test sets. Target
entity names and command phrasing in these sets were excluded from training.
The 600-step adapter was fixed before evaluating the new test set.

| Evaluation                     | First adapter | Second adapter |
| ------------------------------ | ------------- | -------------- |
| Exact validation actions       | 3/81 (3.7%)   | 50/81 (61.7%)  |
| Exact test actions             | 5/108 (4.6%)  | 46/108 (42.6%) |
| Valid test calls               | 74/108        | 99/108         |
| Correct test tool selection    | 38/108        | 73/108         |
| Median test prediction latency | 0.518 s       | 0.497 s        |

Exact test actions by tool were 5/24 for schema creation, 5/12 for adding a
property, 7/8 for removing a property, 0/8 for renaming, 1/8 for connecting schemas,
and 28/48 for `no_action`. Remaining errors include modified schema names, wrong
selected IDs, and ambiguous edits proceeding instead of returning `no_action`.

### Third Colab experiment

`runs/colab-270m-v3-stable/` contains the completed FunctionGemma 270M experiment
with 12,350 training examples. It ran 3,000 updates on a Tesla T4 with 15,360 MiB
of GPU memory. Training took 37.2 minutes and peaked at 5.80 GB of MLX allocations.
The FP16 base checkpoint stayed frozen; rank-32 LoRA trained 7.594 million parameters.
The saved configuration uses scale 8 and a 100-update learning-rate warmup followed
by cosine decay. Training ran directly inside the notebook kernel through the Colab
CLI. The adapter was downloaded, verified by SHA-256, and the GPU session released.

The new validation and test splits exclude their target entity names and phrasing
from training. Both checkpoints below were evaluated on the same 144 validation
examples on the M4 Mac. The final checkpoint was selected before running the test
evaluation. Earlier experiment results use different, retired splits.

| Checkpoint    | Exact validation actions |
| ------------- | ------------------------ |
| 500 updates   | 125/144 (86.8%)          |
| 3,000 updates | 132/144 (91.7%)          |

The selected checkpoint got **199/222 exact test actions (89.6%)**, produced
219/222 valid calls (98.6%), and selected the correct tool in 209/222 cases (94.1%).
Median prediction time on the Mac was 0.474 seconds, with a maximum of 1.175 GB of
MLX allocations. These latency numbers cover text-to-action inference.

| Tool              | Exact test actions |
| ----------------- | ------------------ |
| Create schema box | 32/42              |
| Add property      | 24/24              |
| Remove property   | 12/12              |
| Rename schema     | 24/24              |
| Connect schemas   | 12/12              |
| No action         | 95/108             |

The original `User` command also passes with all three fields and five methods.
It is a seen training example and serves as a local loading and action check.
Remaining errors include list copying and unsupported requests becoming edits;
the parser rejected three invalid test calls. These are controlled, templated
command results. The local voice mode now supplies speech transcription and canvas execution.

The run includes the adapter, frozen source, configuration, dataset, task definition,
hardware, training log, Colab validation (`valid.json`), Mac validation
(`valid-mac.json`), and test report (`test.json`). The downloaded compact bundle is
`runs/colab-270m-v3-stable-final.zip`; intermediate checkpoint backups are also local.
The adapter's SHA-256 is
`b888c9ea52cca9d12041da830d54ec60323d4d1bdd162f9f51825e53543a1856`.

Use the selected adapter:

```sh
uv run --project templates/agent/training python templates/agent/training/lab.py predict \
  'Create a User schema with properties name, class, subjects and methods getName, getClass, getSubjects, addSubject, removeSubject.' \
  --adapter templates/agent/training/runs/colab-270m-v3-stable/adapter
```

The second adapter passes the original User schema command with all three fields
and five methods, but that exact example is in its training set. This demonstrates
learning a seen example, not generalization to an unseen request.

The model still needs work before automatic canvas execution. Reports and raw
predictions are saved in `runs/data-v2-diverse/valid.json` and `test.json`;
`comparison.json` records both adapters' scores on the same inputs. The seen User
example is recorded separately in `seen-user-diagnostic.json`. Keep this test set
as a record of this experiment; reserve another fresh set if these failures guide
the next training changes. The current local voice mode uses the version-four
checkpoint; these older measurements remain historical experiment records.

## Agent overview

With its default configuration, the agent can perform the following actions:

- Create, update and delete shapes.
- Draw freehand pen strokes.
- Use higher-level operations on multiple shapes at once: Rotate, resize, align, distribute, stack and reorder shapes.
- Write out its thinking and send messages to the user.
- Keep track of its task by writing and updating a todo list.
- Move its viewport to look at different parts of the canvas.
- Count shapes matching a given expression.
- Schedule further work and reviews to be carried out in follow-up requests.
- Call example external APIs: Looking up country information.

To make decisions on what to do, we send the agent information from various sources:

- The user's message.
- The user's current selection of shapes.
- What the user can currently see on their screen.
- Any additional context that the user has provided, such as specific shapes or a particular position or area on the canvas.
- Actions the user has recently taken.
- A screenshot of the agent's current view of the canvas.
- A simplified format of all shapes within the agent's viewport.
- Information on clusters of shapes outside the agent's viewport.
- The history of the current session, including the user's messages and all the agent's actions.
- Lints identifying potential issues with shapes on the canvas.

## Use the agent programmatically

Aside from using the chat panel UI, you can also prompt the agent programmatically.

The simplest way is to call the `prompt()` method to start an agentic loop. The agent continues until it finishes the task.

```ts
// Inside a component wrapped by TldrawAgentAppProvider
const agent = useAgent()
agent.prompt('Draw a cat')
```

You can specify further details about the request as an `AgentInput` object:

```ts
agent.prompt({
	message: 'Draw a cat in this area',
	bounds: { x: 0, y: 0, w: 300, h: 400 },
})
```

The `TldrawAgent` class has additional methods:

- `agent.cancel()` - Cancel the agent's current task.
- `agent.reset()` - Reset the agent's chat and memory.
- `agent.request(input)` - Send a single request to the agent and handle its response _without_ entering into an agentic loop.

## Architecture overview

The agent starter is organized into three main areas:

- **`client/`** - React components, agent logic, and utils that run in the browser
- **`worker/`** - Cloudflare Worker that handles model requests and prompt building
- **`shared/`** - Types, schemas, and utilities shared between client and worker

## Customize the agent

The agent's behavior is defined in `client/modes/AgentModeDefinitions.ts`. The `AGENT_MODE_DEFINITIONS` array contains mode definitions. Each mode has two arrays:

- `parts` determine what the agent can **see**.
- `actions` determine what the agent can **do**.

Add, edit or remove an entry in either array to change what the agent can see or do in a given mode.

### Mode system

The agent uses a **mode system** to control what parts and actions it has access to at any given time. Modes are defined in `client/modes/AgentModeDefinitions.ts`.

The default `working` mode includes all standard capabilities. You can create additional modes with different subsets of parts and actions.

Modes can be transitioned between over the course of a prompt depending on the behavior you desire. Call `agent.mode.setMode(modeType)` to change modes. To control the lifecycles of different modes, you can optionally implement any desired mode lifecycle hooks in `client/modes/AgentModeChart.ts`. You have access to:

- `onEnter(agent, fromMode)` - runs when you enter a mode
- `onExit(agent, toMode)` - runs when you exit a mode
- `onPromptStart(agent, request)` - runs when a prompt commences, either because a user has prompted it or because it has entered another step in its agentic loop
- `onPromptEnd(agent, request)` - runs when a prompt ends
- `onPromptCancel(agent, request)` - runs when a prompt is canceled

## Change what the agent can see

**Change what the agent can see by adding, editing or removing a prompt part.**

Prompt parts assemble and build the prompt that we give to the model, with each util adding a different piece of information. This includes the user's message, the model name, the system prompt, chat history and more.

This example shows how to let the model see what the current time is.

First, define a prompt part type in `shared/schema/PromptPartDefinitions.ts`:

```ts
export interface TimePart extends BasePromptPart<'time'> {
	time: string
}
```

Next, create a prompt part util in `client/parts/`:

```ts
export const TimePartUtil = registerPromptPartUtil(
	class TimePartUtil extends PromptPartUtil<TimePart> {
		static override type = 'time' as const

		override getPart(): TimePart {
			return {
				type: 'time',
				time: new Date().toLocaleTimeString(),
			}
		}
	}
)
```

The `getPart` method gather any data needed to construct the prompt. It can take `(request: AgentRequest, helpers: AgentHelpers)` parameters for access to the current request and helper methods.

Then, back in `shared/schema/PromptPartDefinition.ts`, create the definition for that prompt part.

```ts
export const TimePartDefinition: PromptPartDefinition<TimePart> = {
	type: 'time',
	priority: -100,
	buildContent({ time }: TimePart) {
		return [`The user's current time is: ${time}`]
	},
}
```

The prompt part definition is used by the worker to turn prompt parts into messages sent to the model. Override `priority` to control what order the part should be added in the messages. Override `buildContent` to control how the data is turned into a message for the model.

There are other methods available on the `PromptPartDefinition` interface that you can override for more granular control.

- `getModelName` - Determine which AI model to use.
- `buildMessages` - Manually override how prompt messages are constructed from the prompt part.

**Enable the prompt part**

To enable the prompt part, import its util in `client/modes/AgentModeDefinitions.ts` and add its type to a mode's `parts` array. It's important to make sure you import it here and use its `type` field, instead of using the type string literal. This is to ensure the util properly self-registers.

```ts
import { TimePartUtil } from '../parts/TimePartUtil'

// Then in the mode definition:
parts: [
	// ... other parts
	TimePartUtil.type,
]
```

## Change what the agent can do

**Change what the agent can do by adding, editing or removing an agent action.**

Agent action utils define the actions the agent can perform. Each `AgentActionUtil` adds a different capability.

This example shows how to allow the agent to clear the screen.

First, define an agent action schema in `shared/schema/AgentActionSchemas.ts`:

```ts
export const ClearAction = z
	// All agent actions must have a _type field
	// The underscore encourages the model to put this field first
	.object({
		_type: z.literal('clear'),
	})
	// A title and description tell the model what the action does
	.meta({
		title: 'Clear',
		description: 'The agent deletes all shapes on the canvas.',
	})

// Infer the action's type
export type ClearAction = z.infer<typeof ClearAction>
```

Then, create an agent action util in `client/actions/`:

```ts
export const ClearActionUtil = registerActionUtil(
	class ClearActionUtil extends AgentActionUtil<ClearAction> {
		static override type = 'clear' as const

		override applyAction(action: Streaming<ClearAction>) {
			// Don't do anything until the action has finished streaming
			if (!action.complete) return

			// Delete all shapes on the page
			const { editor } = this
			const shapes = editor.getCurrentPageShapes()
			editor.deleteShapes(shapes)
		}
	}
)
```

The `applyAction` method executes the action. It can take a second `helpers: AgentHelpers` parameter for access to helper methods.

Override these methods on `AgentActionUtil` for more control:

- `getInfo` - Determine how the action gets displayed in the chat panel UI.
- `savesToHistory` - Control whether actions get saved to chat history or not.
- `sanitizeAction` - Sanitize the action before saving it to history and applying it. More details on [sanitization](#sanitize-data-received-from-the-model) below.

**Enable the agent action part**

To enable the agent action, import its util in `client/modes/AgentModeDefinitions.ts` and add its type to a mode's `actions` array.

```ts
import { ClearActionUtil } from '../actions/ClearActionUtil'

// Then in the mode definition:
actions: [
	// ... other actions
	ClearActionUtil.type,
]
```

## Change how actions appear in chat history

Configure the icon and description of an action in the chat panel using the `getInfo()` method.

```ts
override getInfo() {
	return {
		icon: 'trash' as const,
		description: 'Cleared the canvas',
	}
}
```

You can make an action collapsible by adding a `summary` property.

```ts
override getInfo() {
	return {
		summary: 'Cleared the canvas',
		description: 'After much consideration, the agent decided to clear the canvas',
	}
}
```

To customize an action's appearance via CSS, you can define style for the `agent-action-type-{TYPE}` class where `{TYPE}` is the type of the action.

```css
.agent-action-type-clear {
	color: red;
}
```

## Managers

Managers are classes that encapsulate specific concerns and extend the functionality of `TldrawAgent` or `TldrawAgentApp`. Each manager handles a single responsibility—like chat history, model selection, or context management—and exposes methods to interact with that state.

Managers are available as properties on the agent instance (e.g., `agent.chat`, `agent.modelName`, `agent.context`). To create a custom manager, extend `BaseAgentManager` or `BaseAgentAppManager` and add it to the agent in `client/agent/TldrawAgent.ts`.

## Registering `PromptPartUtil`s and `AgentActionUtil`s

Utils use a **self-registration pattern**. When you create a new `PromptPartUtil` or `AgentActionUtil`, wrap it with a registration function:

```ts
export const MyPartUtil = registerPromptPartUtil(
	class MyPartUtil extends PromptPartUtil<MyPart> {
		// ...
	}
)
```

This pattern ensures utils are discovered automatically when their modules are imported in `AgentModeDefinitions.ts`.

### Mode-scoped actions

Different modes can implement actions with the same `_type`. This allows modes to have different behavior for the same action type without requiring globally unique action names.

For example, a "team-member" mode and a "solo" mode might both have a `mark-task-done` action, but with different implementations. The system automatically resolves the correct `AgentActionUtil` and schema based on the current mode.

**Registering a mode-specific action util:**

Use the `forModes` option when registering a util:

```ts
// client/actions/MarkSoloTaskDoneActionUtil.ts
// Default implementation (used when no mode-specific binding exists)
export const MarkSoloTaskDoneActionUtil = registerActionUtil(
	class MarkSoloTaskDoneActionUtil extends AgentActionUtil<MarkSoloTaskDoneAction> {
		static override type = 'mark-task-done' as const
		override applyAction(action: Streaming<MarkSoloTaskDoneAction>) {
			// Default implementation
		}
	}
)

// client/actions/MarkTeamMemberTaskDoneActionUtil.ts
// Mode-specific implementation for "drone" mode
export const MarkTeamMemberTaskDoneActionUtil = registerActionUtil(
	class MarkTeamMemberTaskDoneActionUtil extends AgentActionUtil<MarkTeamMemberTaskDoneAction> {
		static override type = 'mark-task-done' as const // Same type as default
		override applyAction(action: Streaming<MarkTeamMemberTaskDoneAction>) {
			// Team member-specific implementation
		}
	},
	{ forModes: ['team-member'] }
)
```

**Registering a mode-specific schema:**

If a mode needs a different schema for an action, register the schema with `forModes`:

```ts
// shared/schema/AgentActionSchemas.ts

// Default schema
export const MarkSoloTaskDoneAction = z
	.object({
		_type: z.literal('mark-task-done'),
		taskId: z.string(),
	})
	.meta({ title: 'Mark Task Done', description: 'Mark a task as complete.' })

// Mode-specific schema with additional fields
export const MarkTeamMemberTaskDoneAction = z
	.object({
		_type: z.literal('mark-task-done'),
		taskId: z.string(),
		teamId: z.string(), // Extra field for this mode
	})
	.meta({ title: 'Mark Task Done', description: 'Mark a task as complete with notes.' })

// Register the mode-specific schema
registerActionSchema('mark-task-done', MarkTeamMemberTaskDoneAction, { forModes: ['team-member'] })
```

Default schemas are auto-registered when exported from `AgentActionSchemas.ts`. Call `registerActionSchema` explicitly only for mode-specific schemas.

The system maintains two registries (default and mode-specific) and resolves the correct util/schema based on the current mode, falling back to the default when no mode-specific binding exists.

## Schedule further work

Let the agent work over multiple turns by scheduling further work using the `schedule` method.

This example shows how to schedule an extra step for adding detail to the canvas.

```ts
override applyAction(action: Streaming<AddDetailAction>) {
	if (!action.complete) return
	this.agent.schedule('Add more detail to the canvas.')
}
```

As with the `prompt` method, you can specify further details about the request.

```ts
agent.schedule({
	message: 'Add more detail in this area.',
	bounds: { x: 0, y: 0, w: 100, h: 100 },
})
```

Schedule multiple items by calling the `schedule` method more than once.

```ts
agent.schedule('Add more detail to the canvas.')
agent.schedule('Check for spelling mistakes.')
```

If you want to interrupt the agent with a new prompt, instead of waiting until the current prompt ends, you can use the agent's `interrupt` method. `interrupt` also lets you specify a mode to transition into.

This example shows how one might use the `interrupt` method to allow the agent to decide to enter a new mode called `'reviewing'` in order to review some work.

```ts
override applyAction(action: Streaming<EnterReviewingModeAction>){
	if (!action.complete) return
	this.agent.interrupt({
		mode: 'reviewing',
		input: {
			message: 'Review the new area thoroughly for any mistakes',
			bounds: action.bounds
		}
	})
}
```

Use this for things like switching modes, or for programatically telling it to correct a mistake it's made.

## Retrieve data from an external API

To retrieve information from an external API, fetch the data within `applyAction` and schedule a follow-up request with the data.

```ts
override async applyAction(action: Streaming<CountryInfoAction>) {
	if (!action.complete) return

	// Fetch from the external API
	const data = await fetchCountryInfo(action.code)

	// Schedule a follow-up request with the data
	this.agent.schedule({ data: [data] })
}
```

## Sanitize data received from the model

The model can make mistakes. Sometimes this is due to hallucinations, sometimes because the canvas changed since the model last saw it. Either way, an incoming action might contain invalid data.

To correct mistakes, apply fixes in the `sanitizeAction` method. The system runs these before applying the action to the editor or saving it to chat history.

For example, use `ensureShapeIdExists` to verify that a shape ID from the model refers to an existing shape.

```ts
override sanitizeAction(action: Streaming<DeleteAction>, helpers: AgentHelpers) {
	if (!action.complete) return action

	// Ensure the shape ID refers to an existing shape
	action.shapeId = helpers.ensureShapeIdExists(action.shapeId)

	// If the shape ID doesn't refer to an existing shape, cancel the action
	if (!action.shapeId) return null

	return action
}
```

`AgentHelpers` provides these sanitization helpers:

- `ensureShapeIdExists` - Ensure that a shape ID refers to a real shape. Useful for interacting with existing shapes.
- `ensureShapeIdsExist` - Ensure that multiple shape IDs refer to real shapes. Useful for bulk operations.
- `ensureShapeIdIsUnique` - Ensure that a shape ID is unique. Useful for creating new shapes.
- `ensureValueIsVec`, `ensureValueIsNumber`, etc - Useful for more complex actions where the model is more likely to make mistakes.

## Send positions to and from the model

By default, every position sent to the model is offset by the starting position of the current chat.

To apply this offset to a position sent to the model, use the `applyOffsetToVec` method.

```ts
override getPart(request: AgentRequest, helpers: AgentHelpers): ViewportCenterPart {
	if (!this.editor) return { part: 'user-viewport-center', center: null, }

	// Get the center of the user's viewport
	const viewportCenter = this.editor.getViewportBounds().center

	// Apply the chat's offset to the vector
	const offsetViewportCenter = helpers.applyOffsetToVec(viewportCenter)

	// Return the prompt part
	return {
		part: 'user-viewport-center',
		center: offsetViewportCenter,
	}
}
```

To remove the offset from a position received from the model, use the `removeOffsetFromVec` method.

```ts
override applyAction(action: Streaming<MoveAction>, helpers: AgentHelpers) {
	if (!action.complete) return

	// Remove the offset from the position
	const position = helpers.removeOffsetFromVec({ x: action.x, y: action.y })

	// Do something with the position...
}
```

Box-level helpers for working with bounds:

- `applyOffsetToBox` / `removeOffsetFromBox` - Apply or remove offset from a `{ x, y, w, h }` box.
- `applyOffsetToShapePartial` / `removeOffsetFromShapePartial` - Apply or remove offset from a partial shape.

Round numbers before sending them to the model. To restore the original number later, use `roundAndSaveNumber` and `unroundAndRestoreNumber`.

```ts
// In `getPart`...
const roundedX = helpers.roundAndSaveNumber(x, 'my_key_x')
const roundedY = helpers.roundAndSaveNumber(y, 'my_key_y')

// In `applyAction`...
const unroundedX = helpers.unroundAndRestoreNumber(x, 'my_key_x')
const unroundedY = helpers.unroundAndRestoreNumber(y, 'my_key_y')
```

To round all the numbers on a shape, use the `roundShape` and `unroundShape` methods. See the [shapes](#send-shapes-to-the-model) section below for more details on sending shapes to the model.

```ts
// In `getPart`...
const roundedShape = helpers.roundShape(shape)

// In `applyAction`...
const unroundedShape = helpers.unroundShape(roundedShape)
```

Additional rounding helpers:

- `roundBox` - Round the coordinates and dimensions of a box.

## Send shapes to the model

The agent converts tldraw shapes to simplified formats to improve model understanding and performance.

Three main formats:

- `BlurryShape` - Format for shapes within the agent's viewport. Contains bounds, id, type, and text. The "blurry" name indicates the agent can't make out shape details—it provides an overview of what the agent sees.
- `FocusedShape` - Format for shapes the agent is focusing on, such as those you've manually added to its context. Contains most shape properties: color, fill, alignment, and shape-specific information. The "focused" name indicates these are shapes the agent is directly examining.
  - This is also the format that the model outputs when creating shapes.
- `PeripheralShapeCluster` - Format for shapes outside the agent's viewport. Groups nearby shapes into clusters with bounds and shape count. The least detailed format—gives the model awareness of shapes elsewhere on the page.

Use conversion functions in `shared/format/` to send shapes in these formats, such as `convertTldrawShapeToFocusedShape`.

This example picks one random shape on the canvas and sends it to the model in the Focused format.

```ts
override getPart(request: AgentRequest, helpers: AgentHelpers): RandomShapePart {
	if (!this.editor) return { type: 'random-shape', shape: null}
	const { editor } = this

	// Get a random shape
	const shapes = editor.getCurrentPageShapes()
	const randomShape = shapes[Math.floor(Math.random() * shapes.length)]

	// Convert the shape to the Focused format
	const focusedShape = convertTldrawShapeToFocusedShape(editor, randomShape)

	// Normalize the shape's position
	const offsetShape = helpers.applyOffsetToShape(focusedShape)
	const roundedShape = helpers.roundShape(offsetShape)

	return {
		type: 'random-shape',
		shape: roundedShape,
	}
}
```

## Change the system prompt

The system prompt lives in `worker/prompt/buildSystemPrompt.ts`. Edit the sections in `worker/prompt/sections/` to change the system prompt.

The system prompt is rebuilt for each step in the agentic loop depending on which actions and parts are available in the agent's current mode. If you add new actions or parts, you can give the model more detailed instructions for how to use them in `worker/prompt/sections/rules-section.ts`.

The schema showing the actions the agent can output is also automatically added to the system prompt.

## Change to a different model

Set an agent's model using the `setModelName` method on the `modelName` manager.

```ts
agent.modelName.setModelName('gemini-3.8-flash')
```

To change the logic for deciding which model to use for a request, you can edit `ModelNamePartUtil`.

## Support a different model

Add the model's definition to `AGENT_MODEL_DEFINITIONS` in `shared/models.ts`.

```ts
'claude-sonnet-5': {
	name: 'claude-sonnet-5',
	id: 'claude-sonnet-5',
	provider: 'anthropic',
	supportsPrefill: false,
	supportsTemperature: false,
	thinking: 'adaptive',
	effort: 'low',
}
```

Add extra setup or configuration for your provider in `worker/do/AgentService.ts`.

## Support custom shapes

If your app includes [custom shapes](https://tldraw.dev/docs/shapes#Custom-shapes-1), the agent can see, move, delete, resize, rotate, and arrange them with no extra setup. However, you might also want to let the agent create and edit them, and read their custom properties.

To support custom shapes, you have two main options:

1. Add an action that lets the agent create your custom shape.
   See the [Let the agent create custom shapes with an action](#let-the-agent-create-custom-shapes-with-an-action) section below.
2. Add your custom shape to the schema so that the agent read, edit and create it like any other shape.
   See the [Add your custom shape to the schema](#add-your-custom-shape-to-the-schema) section below.

### Let the agent create a custom shape with an action

For partial support, let the agent create a custom shape with an [agent action](#change-what-the-agent-can-do). This example creates a custom "sticker" shape:

```ts
// In shared/schema/AgentActionSchemas.ts
export const StickerAction = z
	.object({
		_type: z.literal('sticker'),
		stickerType: z.enum(['heart', 'star']),
		x: z.number(),
		y: z.number(),
	})
	.meta({
		title: 'Sticker',
		description: 'Add a sticker to the canvas.',
	})

export type StickerAction = z.infer<typeof StickerAction>
```

Create an action util to define how the action applies to the canvas:

```ts
// client/actions/StickerActionUtil.ts
export const StickerActionUtil = registerActionUtil(
	class StickerActionUtil extends AgentActionUtil<StickerAction> {
		static override type = 'sticker' as const

		// How to display the action in chat history
		override getInfo(action: Streaming<StickerAction>) {
			return {
				icon: 'pencil' as const,
				description: 'Added a sticker',
			}
		}

		// Execute the action
		override applyAction(action: Streaming<StickerAction>, helpers: AgentHelpers) {
			if (!action.complete) return

			// Normalize the position
			const position = helpers.removeOffsetFromVec({ x: action.x, y: action.y })

			// Create the custom shape
			this.editor.createShape({
				type: 'sticker',
				id: createShapeId(),
				x: position.x,
				y: position.y,
				props: { stickerType: action.stickerType },
			})
		}
	}
)
```

### Add a custom shape to the schema

To let the agent see the custom properties of your custom shape, add it to the schema in `shared/format/FocusedShape.ts`

For example, here's a schema for a custom sticker shape.

```ts
const FocusedStickerShape = z
	.object({
		// Required properties
		_type: z.literal('sticker'),
		note: z.string(),
		shapeId: z.string(),

		// Custom properties
		stickerType: z.enum(['heart', 'star']),
		x: z.number(),
		y: z.number(),
	})
	.meta({
		// Information about the shape to give to the agent
		title: 'Sticker Shape',
		description:
			'A sticker shape is a small symbol stamped onto the canvas. There are two types of stickers: heart and star.',
	})
```

The `_type` and `shapeId` properties are required so that the app can identify your shape. The `note` property is also required. The agent uses it to leave notes for itself.

For optional properties, it's worth considering how the agent should see your custom shape. You might want to leave out some properties and focus on showing the most important ones. It's also best to keep them in alphabetical order for better performance with Gemini models.

Enable your custom shape schema by adding it to the list of `FOCUSED_SHAPES` in the same file to enable it.

```ts
const FOCUSED_SHAPES = [
	FocusedDrawShape,
	FocusedGeoShape,
	FocusedLineShape,
	FocusedTextShape,
	FocusedArrowShape,
	FocusedNoteShape,
	FocusedUnknownShape,

	// Our custom shape
	FocusedStickerShape,
] as const
```

Tell the app how to convert your custom shape into the `FocusedShape` format by adding it as a case in `shared/format/convertTldrawShapeToFocusedShape.ts`.

```ts
export function convertTldrawShapeToFocusedShape(editor: Editor, shape: TLShape): FocusedShape {
	switch (shape.type) {
		// ...
		case 'sticker':
			const bounds = getShapeBounds(shape)
			return {
				_type: 'sticker',
				note: (shape.meta.note as string) ?? '',
				shapeId: convertTldrawIdToSimpleId(shape.id),
				stickerType: shape.props.stickerType,
				x: bounds.x,
				y: bounds.y,
			}
		// ...
	}
}
```

To allow the agent to edit your custom shape's properties, tell the app how to convert your shape from the `FocusedShape` format that the model outputs to the actual format of your shape.

```ts
export function convertFocusedShapeToTldrawShape(
	editor: Editor,
	focusedShape: TLShape
	{ defaultShape }: { defaultShape: Partial<TLShape> }
): {
	switch (focusedShape.type) {
		// ...
		case 'sticker':
			const shapeId = convertSimpleIdToTldrawId(focusedShape.shapeId)
			return {
				shape: {
					id: shapeId
					x: focusedShape.x,
					y: focusedShape.y
					// ...
					props: {
						// ...
						stickerType: focusedShape.stickerType
					},
					meta: {
						note: focusedShape.note ?? ''
					}
				}
			}
		// ...
	}
}
```

## License

This project is part of the tldraw SDK. It is provided under the [tldraw SDK license](https://github.com/tldraw/tldraw/blob/main/LICENSE.md).

You can use the tldraw SDK in commercial or non-commercial projects so long as you preserve the "Made with tldraw" watermark on the canvas. To remove the watermark, you can purchase a [business license](https://tldraw.dev/pricing). Visit [tldraw.dev](https://tldraw.dev) to learn more.

## Trademarks

Copyright (c) 2025-present tldraw Inc. The tldraw name and logo are trademarks of tldraw. Please see our [trademark guidelines](https://github.com/tldraw/tldraw/blob/main/TRADEMARKS.md) for info on acceptable usage.

## Distributions

You can find tldraw on npm [here](https://www.npmjs.com/package/@tldraw/tldraw?activeTab=versions).

## Contribution

Found a bug? Please [submit an issue](https://github.com/tldraw/tldraw/issues/new).

## Community

Have questions, comments or feedback? [Join our discord](https://discord.gg/rhsyWMUJxd). For the latest news and release notes, visit [tldraw.dev](https://tldraw.dev).

## Contact

Find us on Twitter/X at [@tldraw](https://twitter.com/tldraw) or email us at [mailto:hello@tldraw.com](hello@tldraw.com).
