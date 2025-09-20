from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Tuple

from aiohttp import web
from aiohttp.web import AppKey

@dataclass
class MockLMStudioConfig:
    mode: str
    delay_s: float
    reasoning_chunks: int


def _plain_response(model: str, user_prompt: str) -> Dict[str, Any]:
    text = f"[mock:{model}] {user_prompt[:120]}"
    return {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 12, "total_tokens": 22},
    }


def _reasoning_response(model: str, user_prompt: str) -> Dict[str, Any]:
    thought = f"Thinking about: {user_prompt[:50]}..."
    text = f"[reasoned:{model}] {user_prompt[:120]}"
    return {
        "id": "chatcmpl-mock-reasoning",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text,
                    "reasoning": [{"type": "thought", "text": thought}],
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 18, "completion_tokens": 26, "total_tokens": 44},
    }


MOCK_DEFAULTS_KEY: AppKey[Dict[str, Any]] = AppKey("mock_defaults")


async def _handle_chat(request: web.Request) -> web.StreamResponse:
    defaults = request.app.get(MOCK_DEFAULTS_KEY, {})
    cfg = MockLMStudioConfig(
        mode=request.query.get("mode", defaults.get("mode", "plain")),
        delay_s=float(request.query.get("delay", defaults.get("delay", 0.0))),
        reasoning_chunks=int(request.query.get("chunks", defaults.get("chunks", 2))),
    )

    payload = await request.json()

    if payload.get("response_format"):
        return web.json_response(
            {"error": {"message": "response_format.type not supported"}},
            status=400,
        )

    model = payload.get("model") or "auto"
    messages = payload.get("messages") or []
    user_msg = next((m.get("content") for m in messages if m.get("role") == "user"), "")

    if payload.get("stream"):
        return await _streaming_response(request, cfg, model, user_msg)

    if cfg.delay_s:
        await asyncio.sleep(cfg.delay_s)

    if cfg.mode == "reasoning":
        data = _reasoning_response(model, user_msg)
    else:
        data = _plain_response(model, user_msg)

    return web.json_response(data)


async def _streaming_response(request: web.Request, cfg: MockLMStudioConfig, model: str, user_prompt: str) -> web.StreamResponse:
    response = web.StreamResponse(status=200, reason="OK", headers={"Content-Type": "text/event-stream"})
    await response.prepare(request)

    async def emit(data: Dict[str, Any]) -> None:
        await response.write(f"data: {json.dumps(data)}\n\n".encode("utf-8"))

    if cfg.delay_s:
        await asyncio.sleep(cfg.delay_s)

    if cfg.mode == "reasoning":
        for idx in range(cfg.reasoning_chunks):
            await emit(
                {
                    "id": "chatcmpl-mock-reasoning",
                    "object": "chat.completion.chunk",
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "reasoning": {
                                    "type": "thought",
                                    "text": f"chunk-{idx}: {user_prompt[:40]}",
                                }
                            },
                            "finish_reason": None,
                        }
                    ],
                }
            )
            await asyncio.sleep(0)

        await emit(
            {
                "id": "chatcmpl-mock-reasoning",
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": f"[reasoned:{model}] {user_prompt[:120]}",
                                }
                            ],
                        }
                    }
                ],
            }
        )
        await emit(
            {
                "id": "chatcmpl-mock-reasoning",
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 18, "completion_tokens": 26, "total_tokens": 44},
            }
        )
    else:
        await emit(
            {
                "id": "chatcmpl-mock",
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": f"[mock:{model}] {user_prompt[:120]}",
                                }
                            ],
                        }
                    }
                ],
            }
        )
        await emit(
            {
                "id": "chatcmpl-mock",
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 12, "total_tokens": 22},
            }
        )

    await response.write(b"data: [DONE]\n\n")
    await response.write_eof()
    return response


async def build_mock_app(default_mode: str = "plain", default_delay: float = 0.0, default_chunks: int = 2) -> web.Application:
    app = web.Application()
    app[MOCK_DEFAULTS_KEY] = {"mode": default_mode, "delay": default_delay, "chunks": default_chunks}
    app.add_routes([web.post("/v1/chat/completions", _handle_chat)])
    return app


@asynccontextmanager
async def mock_server_ctx(host: str = "127.0.0.1", port: int = 0, *, mode: str = "plain", delay: float = 0.0, reasoning_chunks: int = 2) -> AsyncIterator[Tuple[str, int]]:
    app = await build_mock_app(default_mode=mode, default_delay=delay, default_chunks=reasoning_chunks)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host=host, port=port)
    await site.start()
    sockets = list(site._server.sockets) if getattr(site, "_server", None) else []
    bound_port = sockets[0].getsockname()[1] if sockets else port
    try:
        yield host, bound_port
    finally:
        await runner.cleanup()
