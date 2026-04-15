"""Audio Flamingo Next inference server.

Loads the model once on startup and serves requests over HTTP,
avoiding the cost of reloading weights on every inference call.

Usage:
    uv run python server.py [--model MODEL_ID] [--device cuda|cpu] \
                            [--host HOST] [--port PORT]

Request (POST /infer):
    {"audio": "/abs/path/to/file.wav", "question": "...", "max_new_tokens": 512}

Response:
    {"answer": "..."}

GET /health returns {"status": "ok"}.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Sequence

import inference as _inf

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

# Module-level model state – populated once at startup.
_model: Any = None
_processor: Any = None


class _Handler(BaseHTTPRequestHandler):
    """Minimal HTTP handler for /infer and /health."""

    # Silence default request logging; uncomment to restore.
    # def log_message(self, fmt, *args):  # noqa: D102
    #     pass

    def _send_json(self, status: int, body: dict) -> None:
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/infer":
            self._send_json(404, {"error": "not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)
            req = json.loads(raw)
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": f"invalid request body: {exc}"})
            return

        audio_str = req.get("audio", "")
        question = req.get("question", _inf.DEFAULT_QUESTION)
        max_new_tokens = int(req.get("max_new_tokens", 512))

        audio_path = Path(audio_str)
        try:
            ns = argparse.Namespace(
                audio=audio_path,
                question=question,
                model=None,
                device=None,
                max_new_tokens=max_new_tokens,
            )
            _inf.validate_runtime_args(ns)
        except (FileNotFoundError, ValueError) as exc:
            self._send_json(400, {"error": str(exc)})
            return

        try:
            answer = _inf.run_inference(
                audio_path=audio_path.resolve(),
                question=question,
                model=_model,
                processor=_processor,
                max_new_tokens=max_new_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            self._send_json(500, {"error": str(exc)})
            return

        self._send_json(200, {"answer": answer})


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audio Flamingo Next inference server")
    parser.add_argument("--model", default=_inf.DEFAULT_MODEL_ID, help="HuggingFace model ID")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    global _model, _processor  # noqa: PLW0603

    args = parse_args(argv)
    device = _inf.resolve_device(args.device)
    _inf.print_device_info(device)

    try:
        _model, _processor = _inf.load_model(model_id=args.model, device=device)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1

    server = HTTPServer((args.host, args.port), _Handler)

    def _shutdown(sig, frame):  # noqa: ANN001, ARG001
        print("\nShutting down server…")
        server.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print(f"Server listening on http://{args.host}:{args.port}")
    print("POST /infer  {\"audio\": \"<path>\", \"question\": \"<text>\"}")
    print("GET  /health")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
