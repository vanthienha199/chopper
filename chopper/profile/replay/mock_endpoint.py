"""Mocked model endpoint for reproducible agent replay.

Replays a recorded run's model responses over an OpenAI-compatible AND
Anthropic-compatible HTTP API, in order, with the recorded timing
(time-to-first-token and total duration). Point a harness at this instead of
vLLM and the CPU side of an agent run can be replayed deterministically: no
GPU, no model variance, no network variance. This is the "mocked or recorded
model endpoints" piece of the benchmark (hardware comparisons must not be
confounded by model randomness), and the partner of the CPU-command replay
log.

Recording format: JSONL, one object per model call, in call order. Matches
the proxy's model_calls.jsonl (schema_version 8); minimal fields:

    {"turn": 0, "response": "...text...", "ttft_s": 0.42, "duration_s": 3.1,
     "tool_calls": [{"name": "bash", "arguments": "{\"command\": \"ls\"}",
                     "id": "chatcmpl-tool-..."}],
     "finish_reasons": ["tool_calls"],
     "prompt_sha256": "ab12..." (optional), "model": "gpt-oss-120b" (optional)}

Most agent turns have EMPTY response text plus structured tool_calls
(claude-code and codex especially); the mock returns those tool calls in the
protocol's native shape, or the harness would stall waiting for a tool call
that never comes. `arguments` may be a JSON string or an object. Legacy
OpenAI-nested {"function": {"name", "arguments"}} entries are accepted too.

Serving semantics (v2):
  - Responses are served strictly FIFO. If the recording had N calls, call
    N+1 returns HTTP 410 (recording exhausted).
  - POST /v1/chat/completions answers in OpenAI shape; POST /v1/messages
    answers in Anthropic shape (claude-code speaks this). Both support
    stream=true; the protocol of the recorded call does not need to match
    the endpoint being asked, the shape follows the REQUEST.
  - If prompt_sha256 is present, the incoming prompt is hashed and compared;
    a mismatch is logged loudly (the replayed harness diverged from the
    recording) but the response is still served, so a run never wedges.
  - Timing: first byte is delayed by ttft_s, the remaining content is paced
    across duration_s. Non-streaming requests sleep ttft_s + duration_s.
  - --speed N divides all recorded delays by N (10 = fast replay, ~0 waits).

    python -m chopper.profile.replay.mock_endpoint --recording run1.jsonl \
        --port 8123 --speed 1
"""

import argparse
import functools
import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

# unbuffered progress lines: the log must survive a hard kill of the replay
print = functools.partial(print, flush=True)


def _norm_tool_calls(raw: Any) -> list[dict[str, Any]]:
    """Normalize recorded tool calls to [{id, name, arguments:str}]."""
    out: list[dict[str, Any]] = []
    for i, tc in enumerate(raw or []):
        if "function" in tc:  # OpenAI-nested shape
            name = tc["function"].get("name", "")
            args = tc["function"].get("arguments", "{}")
        else:  # proxy's flat shape
            name = tc.get("name", "")
            args = tc.get("arguments", "{}")
        if not isinstance(args, str):
            args = json.dumps(args)
        out.append({"id": tc.get("id") or f"replay-tool-{i}",
                    "name": name, "arguments": args})
    return out


class _Recording:
    def __init__(self, path: str, speed: float, task: str | None = None,
                 journal: str | None = None):
        self.calls: list[dict[str, Any]] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.calls.append(json.loads(line))
        if task:
            # replaying a single task: drop pipeline warmup calls and other
            # tasks' calls, the harness never made those requests
            self.calls = [c for c in self.calls if c.get("task_id") == task]
        self.speed = max(speed, 1e-9)
        self.i = 0
        self.lock = threading.Lock()
        self.mismatches = 0
        # the replay's own model_calls journal: REPLAY-time arrival stamps,
        # so turn windows can be cut on the machine being measured (the
        # recording's timestamps belong to the original machine)
        self.journal = open(journal, "w") if journal else None

    def log_served(self, entry: dict[str, Any]) -> None:
        if self.journal:
            self.journal.write(json.dumps(entry) + "\n")
            self.journal.flush()

    def next_call(self) -> dict[str, Any] | None:
        with self.lock:
            if self.i >= len(self.calls):
                return None
            call = self.calls[self.i]
            self.i += 1
            return call


REC: _Recording | None = None


def _prompt_hash(body: dict[str, Any]) -> str:
    msgs = body.get("messages", body.get("prompt", ""))
    return hashlib.sha256(
        json.dumps(msgs, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _finish_reason(call: dict[str, Any], tools: list[dict[str, Any]]) -> str:
    fr = call.get("finish_reasons") or []
    if isinstance(fr, str):
        fr = [fr]
    if fr:
        return str(fr[0])
    return "tool_calls" if tools else "stop"


_ANTHROPIC_STOP = {"tool_calls": "tool_use", "stop": "end_turn",
                   "length": "max_tokens"}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        pass  # quiet; we do our own logging

    def _json(self, code: int, obj: dict[str, Any]) -> None:
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path.startswith("/v1/models"):
            self._json(200, {"object": "list", "data": [
                {"id": "chopper-replay", "object": "model", "owned_by": "chopper"}]})
        else:
            self._json(404, {"error": "not found"})

    def _sse(self, obj: dict[str, Any], event: str | None = None) -> None:
        if event:
            self.wfile.write(f"event: {event}\n".encode())
        self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
        self.wfile.flush()

    def do_POST(self) -> None:
        assert REC is not None
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        call = REC.next_call()
        if call is None:
            self._json(410, {"error": "recording exhausted"})
            print("[mock-endpoint] recording exhausted, returned 410")
            return

        want = call.get("prompt_sha256")
        if want:
            got = _prompt_hash(body)
            if got != want:
                REC.mismatches += 1
                print(f"[mock-endpoint] WARNING call {REC.i - 1}: prompt hash "
                      f"mismatch (replay diverged from recording), serving anyway")

        text = call.get("response") or ""
        tools = _norm_tool_calls(call.get("tool_calls"))
        finish = _finish_reason(call, tools)
        ttft = float(call.get("ttft_s", 0.0)) / REC.speed
        dur = float(call.get("duration_s", 0.0)) / REC.speed
        REC.log_served({
            "turn": call.get("turn", REC.i - 1),
            "task_id": call.get("task_id"),
            "ts_epoch_s": time.time(),
            "ttft_s": ttft, "duration_s": dur,
            "prompt_tokens": call.get("prompt_tokens", 0),
            "completion_tokens": call.get("completion_tokens", 0),
        })
        model = call.get("model", "chopper-replay")
        usage_in = int(call.get("prompt_tokens", 0))
        usage_out = int(call.get("completion_tokens", 0))
        rid = f"replay-{REC.i - 1}"
        anthropic = self.path.startswith("/v1/messages")

        if body.get("stream"):
            if anthropic:
                self._stream_anthropic(rid, model, text, tools, finish,
                                       ttft, dur, usage_in, usage_out)
            else:
                self._stream_openai(rid, model, text, tools, finish, ttft, dur)
        else:
            time.sleep(ttft + dur)
            if anthropic:
                content: list[dict[str, Any]] = []
                if text:
                    content.append({"type": "text", "text": text})
                for t in tools:
                    content.append({"type": "tool_use", "id": t["id"],
                                    "name": t["name"],
                                    "input": json.loads(t["arguments"] or "{}")})
                self._json(200, {
                    "id": rid, "type": "message", "role": "assistant",
                    "model": model, "content": content,
                    "stop_reason": _ANTHROPIC_STOP.get(finish, finish),
                    "stop_sequence": None,
                    "usage": {"input_tokens": usage_in,
                              "output_tokens": usage_out}})
            else:
                msg: dict[str, Any] = {"role": "assistant",
                                       "content": text if text else None}
                if tools:
                    msg["tool_calls"] = [
                        {"id": t["id"], "type": "function",
                         "function": {"name": t["name"],
                                      "arguments": t["arguments"]}}
                        for t in tools]
                self._json(200, {
                    "id": rid, "object": "chat.completion",
                    "created": int(time.time()), "model": model,
                    "choices": [{"index": 0, "message": msg,
                                 "finish_reason": finish}],
                    "usage": {"prompt_tokens": usage_in,
                              "completion_tokens": usage_out,
                              "total_tokens": usage_in + usage_out}})
        print(f"[mock-endpoint] served call {REC.i - 1}/{len(REC.calls)} "
              f"({'anthropic' if anthropic else 'openai'}, "
              f"{len(tools)} tool_calls, finish={finish}, "
              f"stream={bool(body.get('stream'))})")

    def _paced_chunks(self, text: str) -> list[str]:
        words = text.split(" ") or [""]
        nchunks = min(len(words), 40)
        per = max(1, len(words) // nchunks)
        return [" ".join(words[j:j + per]) + (" " if j + per < len(words) else "")
                for j in range(0, len(words), per)]

    def _stream_openai(self, rid: str, model: str, text: str,
                       tools: list[dict[str, Any]], finish: str,
                       ttft: float, dur: float) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        time.sleep(ttft)
        now = int(time.time())

        def chunk(delta: dict[str, Any], fr: str | None = None) -> None:
            self._sse({"id": rid, "object": "chat.completion.chunk",
                       "created": now, "model": model,
                       "choices": [{"index": 0, "delta": delta,
                                    "finish_reason": fr}]})

        parts = self._paced_chunks(text) if text else []
        nevents = max(1, len(parts) + len(tools))
        delay = dur / nevents
        first = True
        for p in parts:
            d: dict[str, Any] = {"content": p}
            if first:
                d["role"] = "assistant"
                first = False
            chunk(d)
            time.sleep(delay)
        for ti, t in enumerate(tools):
            d = {"tool_calls": [{"index": ti, "id": t["id"], "type": "function",
                                 "function": {"name": t["name"],
                                              "arguments": t["arguments"]}}]}
            if first:
                d["role"] = "assistant"
                first = False
            chunk(d)
            time.sleep(delay)
        chunk({}, finish)
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _stream_anthropic(self, rid: str, model: str, text: str,
                          tools: list[dict[str, Any]], finish: str,
                          ttft: float, dur: float,
                          usage_in: int, usage_out: int) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        time.sleep(ttft)
        self._sse({"type": "message_start", "message": {
            "id": rid, "type": "message", "role": "assistant", "model": model,
            "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": usage_in, "output_tokens": 0}}},
            "message_start")

        parts = self._paced_chunks(text) if text else []
        nevents = max(1, len(parts) + len(tools))
        delay = dur / nevents
        idx = 0
        if parts:
            self._sse({"type": "content_block_start", "index": idx,
                       "content_block": {"type": "text", "text": ""}},
                      "content_block_start")
            for p in parts:
                self._sse({"type": "content_block_delta", "index": idx,
                           "delta": {"type": "text_delta", "text": p}},
                          "content_block_delta")
                time.sleep(delay)
            self._sse({"type": "content_block_stop", "index": idx},
                      "content_block_stop")
            idx += 1
        for t in tools:
            self._sse({"type": "content_block_start", "index": idx,
                       "content_block": {"type": "tool_use", "id": t["id"],
                                         "name": t["name"], "input": {}}},
                      "content_block_start")
            self._sse({"type": "content_block_delta", "index": idx,
                       "delta": {"type": "input_json_delta",
                                 "partial_json": t["arguments"]}},
                      "content_block_delta")
            self._sse({"type": "content_block_stop", "index": idx},
                      "content_block_stop")
            time.sleep(delay)
            idx += 1
        self._sse({"type": "message_delta",
                   "delta": {"stop_reason": _ANTHROPIC_STOP.get(finish, finish),
                             "stop_sequence": None},
                   "usage": {"output_tokens": usage_out}}, "message_delta")
        self._sse({"type": "message_stop"}, "message_stop")


def main() -> None:
    global REC
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recording", required=True, help="JSONL recording of model calls")
    p.add_argument("--port", type=int, default=8123)
    p.add_argument("--speed", type=float, default=1.0,
                   help="divide recorded delays by this factor (10 = fast replay)")
    p.add_argument("--task", default=None,
                   help="serve only this task_id's calls (drops __warmup__ etc.)")
    p.add_argument("--journal", default=None,
                   help="write the replay's own model_calls jsonl here "
                        "(replay-time arrival stamps for turn windows)")
    a = p.parse_args()
    REC = _Recording(a.recording, a.speed, a.task, a.journal)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"[mock-endpoint] {len(REC.calls)} recorded calls, speed {a.speed}x, "
          f"serving http://127.0.0.1:{a.port}/v1 (openai + anthropic)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    print(f"[mock-endpoint] served {REC.i}/{len(REC.calls)} calls, "
          f"{REC.mismatches} prompt mismatches")


if __name__ == "__main__":
    main()
