import argparse
import base64
import json
import logging
import shutil
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

from pydantic import Field, model_validator

from actions import ACTION_MODELS, TOOLS, Canvas, Name, StrictModel, messages_for
from lab import load_model, predict


class Outcome(StrictModel):
    command: str = Field(max_length=2000)
    action: dict | None = None
    created_id: Name | None = None

    @model_validator(mode="after")
    def check_action(self):
        if self.action is not None:
            if (
                set(self.action) != {"name", "arguments"}
                or not isinstance(self.action["name"], str)
                or self.action["name"] not in ACTION_MODELS
            ):
                raise ValueError("Unknown history action.")
            ACTION_MODELS[self.action["name"]][0].model_validate(
                self.action["arguments"]
            )
        return self


class History(StrictModel):
    turns: list[Outcome] = Field(default_factory=list, max_length=3)
    last_created_id: Name | None = None
    last_edited_id: Name | None = None


class CommandRequest(StrictModel):
    command: str = Field(default="", max_length=2000)
    audio: str | None = Field(default=None, max_length=8_000_000)
    canvas: Canvas
    history: History = Field(default_factory=History)

    @model_validator(mode="after")
    def check_input(self):
        if bool(self.command.strip()) == bool(self.audio):
            raise ValueError("Provide a command or an audio recording.")
        return self


class VoiceEngine:
    def __init__(self, adapter):
        self.model, self.tokenizer = load_model(adapter)
        self.speech_model = None
        self.lock = threading.Lock()

    def transcribe(self, encoded):
        content = base64.b64decode(encoded, validate=True)
        if not content or len(content) > 6_000_000:
            raise ValueError("Recording is empty or too large.")
        if self.speech_model is None:
            from huggingface_hub import snapshot_download
            from parakeet_mlx import from_pretrained

            checkpoint = snapshot_download(
                "mlx-community/parakeet-tdt-0.6b-v2", local_files_only=True, token=False
            )
            self.speech_model = from_pretrained(checkpoint)
        with tempfile.TemporaryDirectory(prefix="canvas-voice-") as directory:
            source, audio = (
                Path(directory) / "recording",
                Path(directory) / "recording.wav",
            )
            source.write_bytes(content)
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-protocol_whitelist",
                    "file,pipe",
                    "-i",
                    str(source),
                    "-t",
                    "60",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    str(audio),
                ],
                check=True,
                capture_output=True,
                timeout=30,
            )
            return self.speech_model.transcribe(str(audio)).text.strip()

    def command(self, request):
        with self.lock:
            start = time.perf_counter()
            command = (
                self.transcribe(request.audio)
                if request.audio
                else request.command.strip()
            )
            speech_seconds = time.perf_counter() - start if request.audio else 0
            if not command:
                raise ValueError("No speech was detected.")
            if len(command) > 2000:
                raise ValueError("The transcribed command is too long.")
            canvas, history = request.canvas.model_dump(), request.history.model_dump()
            tokens = self.tokenizer.apply_chat_template(
                messages_for(command, canvas, history),
                tools=TOOLS,
                add_generation_prompt=True,
                tokenize=True,
            )
            if len(tokens) > 1792:
                raise ValueError("Canvas context is too large for the current demo.")
            result = predict(
                self.model,
                self.tokenizer,
                command,
                canvas,
                history=history,
                guarded=True,
            )
            if result["prediction"] is None:
                raise ValueError("The model did not return a valid edit.")
            action = result["prediction"]
            return {
                "command": command,
                "action": action,
                "speech_seconds": speech_seconds,
                "action_seconds": result["seconds"],
            }


def allowed_origin(origin):
    if not origin:
        return True
    parsed = urlparse(origin)
    return parsed.scheme in {"http", "https"} and parsed.hostname in {
        "localhost",
        "127.0.0.1",
        "::1",
    }


def handler_for(engine):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, data):
            body = json.dumps(data).encode()
            self.send_response(status)
            origin = self.headers.get("Origin")
            if origin and allowed_origin(origin):
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            if not allowed_origin(self.headers.get("Origin")):
                self.reply(403, {"error": "Origin is not allowed."})
                return
            self.send_response(204)
            self.send_header(
                "Access-Control-Allow-Origin",
                self.headers.get("Origin", "http://localhost"),
            )
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_GET(self):
            if not allowed_origin(self.headers.get("Origin")):
                self.reply(403, {"error": "Origin is not allowed."})
            elif self.path == "/health":
                self.reply(
                    200, {"ready": True, "audio": shutil.which("ffmpeg") is not None}
                )
            else:
                self.reply(404, {"error": "Not found."})

        def do_POST(self):
            if not allowed_origin(self.headers.get("Origin")):
                self.reply(403, {"error": "Origin is not allowed."})
                return
            if self.path != "/command":
                self.reply(404, {"error": "Not found."})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 8_500_000:
                    raise ValueError("Invalid request size.")
                request = CommandRequest.model_validate(
                    json.loads(self.rfile.read(size))
                )
                self.reply(200, engine.command(request))
            except (ValueError, subprocess.SubprocessError) as error:
                self.reply(422, {"error": str(error)})
            except Exception:
                logging.exception("Local canvas inference failed")
                self.reply(500, {"error": "Local inference failed."})

    return Handler


def main():
    parser = argparse.ArgumentParser(
        description="Run local speech-to-canvas inference."
    )
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8097)
    args = parser.parse_args()
    engine = VoiceEngine(args.adapter)
    # MLX-LM streams must stay on the thread that loaded the model.
    server = HTTPServer(("127.0.0.1", args.port), handler_for(engine))
    print(f"Local canvas inference ready at http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
