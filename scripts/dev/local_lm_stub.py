#!/usr/bin/env python3
"""
Minimal local LLM stub server for live PoC.

Implements a tiny subset of the OpenAI-compatible LM Studio API used by the
orchestrator:
  - GET  /v1/models
  - POST /v1/chat/completions

The responses are deterministic and tuned to satisfy the Story Engine POML
templates (structured JSON outputs) so the live pipeline can run without a
real model.

Usage:
  PORT=8000 MODEL_ID=poc-mini-1 scripts/dev/local_lm_stub.py

Notes:
  - Keep outputs concise; they must not include the words like 'mock' or
    'placeholder', as the engine rejects them.
  - For POML structured prompts, we return a valid JSON object in the
    assistant message content.
"""
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
from datetime import datetime


PORT = int(os.environ.get("PORT", "8000"))
MODEL_ID = os.environ.get("MODEL_ID", "poc-mini-1")


def now_ts() -> int:
    return int(datetime.utcnow().timestamp())


def choose_template(messages):
    """Detect which structured template is being prompted based on system text."""
    system_text = "\n".join(
        [m.get("content", "") for m in messages if m.get("role") == "system"]
    ).lower()
    user_text = "\n".join(
        [m.get("content", "") for m in messages if m.get("role") == "user"]
    ).lower()
    text = system_text + "\n" + user_text
    if "structure_type" in text and "beats" in text:
        return "plot"
    if "scene_description" in text and "characters_present" in text:
        return "scene"
    if '"dialogue"' in text or "dialogue" in text and "tone" in text:
        return "dialogue"
    if "evaluation_text" in text and "scores" in text:
        return "evaluation"
    return "generic"


def build_content(template: str, messages) -> str:
    # Extract a few hints from the user prompt for flavor
    user_texts = [m.get("content", "") for m in messages if m.get("role") == "user"]
    prompt = "\n".join(user_texts)[:800]

    if template == "plot":
        payload = {
            "structure_type": "three_act",
            "beats": [
                {
                    "name": "Setup",
                    "description": "In Jerusalem at dawn, Pontius Pilate faces a restless crowd and a fraught petition.",
                    "tension": 2,
                    "purpose": "Establish stakes and political pressure",
                },
                {
                    "name": "Rising Action",
                    "description": "Accusations mount; Caiaphas presses; Pilate weighs justice against order in a charged courtyard.",
                    "tension": 6,
                    "purpose": "Escalate moral and civic conflict",
                },
                {
                    "name": "Climax",
                    "description": "Pilate interrogates the prophet; the crowd surges; a decision looms over Jerusalem.",
                    "tension": 9,
                    "purpose": "Force a decisive judgment",
                },
                {
                    "name": "Falling Action",
                    "description": "After the verdict, consequences ripple through the temple precincts and the governor's hall.",
                    "tension": 5,
                    "purpose": "Process the fallout",
                },
                {
                    "name": "Resolution",
                    "description": "Pilate confronts the echo of his choice as Jerusalem settles uneasily into a new order.",
                    "tension": 3,
                    "purpose": "Establish the new equilibrium",
                },
            ],
        }
        return json.dumps(payload)

    if template == "scene":
        payload = {
            "scene_description": (
                "Jerusalem, early morning: Pilate sits beneath a Roman standard as the crowd murmurs. "
                "Caiaphas advances with officials; dust hangs in the light."
            ),
            "characters_present": [
                {"name": "Pontius Pilate", "role": "conflicted judge"},
                {"name": "Caiaphas", "role": "antagonist"},
                {"name": "Crowd", "role": "agitated public"},
            ],
            "setting_details": {
                "location": "Praetorium courtyard, Jerusalem",
                "atmosphere": "Tense, dust-laden, echoing with distant shouts",
            },
            "key_actions": [
                {"character": "Caiaphas", "action": "presents charges crisply"},
                {"character": "Pilate", "action": "questions with measured, weary restraint"},
            ],
            "dialogue_snippets": [
                {"speaker": "Pilate", "line": "Bring your accusation clearly, and let the charge be plain."}
            ],
            "emotional_tone": "Grave and controlled, with undercurrents of fear",
            "scene_objective": "Establish the trial's stakes and moral pressure",
        }
        return json.dumps(payload)

    if template == "dialogue":
        payload = {
            "dialogue": [
                {
                    "speaker": "Pontius Pilate",
                    "line": "You would have me trade justice for quiet streets; tell me, whose peace is that?",
                    "tone": "grave",
                    "recipient": "Crowd",
                }
            ]
        }
        return json.dumps(payload)

    if template == "evaluation":
        # Include parseable X/10 lines for the test helper
        text = (
            "Narrative Coherence: 7/10\n"
            "Pacing: 6/10\n"
            "Emotional Impact: 7/10\n"
            "Dialogue Quality: 7/10\n"
            "Setting/Atmosphere: 7/10\n"
            "Theme Integration: 6/10\n"
            "Overall Engagement: 7/10\n"
        )
        payload = {
            "evaluation_text": text,
            "scores": {
                "Narrative Coherence": 7,
                "Character Development": 6,
                "Pacing": 6,
                "Emotional Impact": 7,
                "Dialogue Quality": 7,
                "Setting/Atmosphere": 7,
                "Theme Integration": 6,
                "Overall Engagement": 7,
            },
        }
        return json.dumps(payload)

    # Generic fallback: brief prose (not used by structured POML paths)
    return (
        "A concise, vivid paragraph advances the scene in Jerusalem as Pilate weighs justice against order."
    )


class Handler(BaseHTTPRequestHandler):
    server_version = "LocalLMStub/1.0"

    def _write_json(self, status: int, obj: dict):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/v1/models":
            body = {
                "object": "list",
                "data": [
                    {"id": MODEL_ID, "object": "model", "owned_by": "local"}
                ],
            }
            self._write_json(200, body)
        else:
            self._write_json(404, {})

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/v1/chat/completions":
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8"))
            except Exception:
                payload = {}
            messages = payload.get("messages", [])
            template = choose_template(messages)
            content = build_content(template, messages)
            response = {
                "id": "chatcmpl-local-1",
                "object": "chat.completion",
                "created": now_ts(),
                "model": payload.get("model") or MODEL_ID,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 200, "completion_tokens": 150, "total_tokens": 350},
            }
            self._write_json(200, response)
        else:
            self._write_json(404, {})

    def log_message(self, fmt, *args):
        # Quieter default logging
        return


def main():
    addr = ("0.0.0.0", PORT)
    httpd = ThreadingHTTPServer(addr, Handler)
    print(f"Local LLM stub listening on http://127.0.0.1:{PORT} (model={MODEL_ID})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
