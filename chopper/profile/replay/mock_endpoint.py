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
the proxy's model_calls.jsonl (schema_version 8 or 9); minimal fields:

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

Serving semantics (v3):
  - Responses are served FIFO WITHIN A SESSION. Each session gets its own
    cursor over the recording, so N concurrent sessions replay N
    independent turn streams from the same file. A session that runs past
    the end of the recording gets HTTP 410 (recording exhausted); the
    other sessions keep going.
  - The session key is the X-Chopper-Session request header when present
    (preferred, and required when several sessions replay the SAME
    recorded task, because the hash fallback would then collide), else the
    SHA-256 of the first message of the request plus the Anthropic
    `system` field. This is the same rule record_proxy.py uses, and the
    proxy forwards the header, so proxy and mock agree on session identity.
  - POST /v1/chat/completions answers in OpenAI shape; POST /v1/messages
    answers in Anthropic shape (claude-code speaks this). Both support
    stream=true; the protocol of the recorded call does not need to match
    the endpoint being asked, the shape follows the REQUEST.
  - If prompt_sha256 is present, the incoming prompt is hashed and compared;
    a mismatch is logged loudly (the replayed harness diverged from the
    recording) but the response is still served, so a run never wedges.
    --no-prompt-check turns the comparison off, which is what a synthetic
    load driver that does not reproduce the original prompts needs.
  - Timing: first byte is delayed by ttft_s, the remaining content is paced
    across duration_s. Non-streaming requests sleep ttft_s + duration_s.
  - --speed N divides all recorded delays by N (10 = fast replay, near-zero
    waits). It does not divide queue wait, which is produced in real time.

Finite-capacity mode (--max-concurrency K):
  With K set, at most K requests are in service at once. A request that
  arrives while K are in service waits in a FIFO queue, and its recorded
  service time starts only after admission. The imposed wait is written to
  the journal as queue_wait_s, which is ground truth for validating the
  client-side queue-wait attribution in multi_request.py.

  This is an EMULATION for exercising the measurement path without a GPU.
  It is not a model of real continuous batching. A real serving engine
  admits a request into a running batch and interleaves its decode steps
  with the other requests in that batch, so the effect of load shows up as
  a slower per-token rate across all in-flight requests. This mock instead
  holds whole requests in a queue and then serves them at their recorded
  rate. Numbers taken from it describe the plumbing, not a real engine.

  Also recorded per call: inflight_at_arrival, the number of requests
  already accepted and not yet finished when this one arrived.

--host defaults to 127.0.0.1. Binding to a routable address lets a harness
on another node reach the mock, and it also exposes the port on this node.

    python -m chopper.profile.replay.mock_endpoint --recording run1.jsonl \
        --port 8123 --speed 1 [--max-concurrency 4] [--host 0.0.0.0]
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

SESSION_HEADER = "X-Chopper-Session"


class _Gate:
    """FIFO admission gate with a fixed number of service slots.

    arrive() stamps the request with a ticket and reports how many requests
    were already in flight. acquire() blocks until this ticket is the oldest
    waiting one and a slot is free, and returns the wait it imposed.
    release() frees the slot. With capacity None the gate never blocks and
    only counts in-flight requests.
    """

    def __init__(self, capacity: int | None):
        self.capacity = capacity
        self.cv = threading.Condition()
        self.next_ticket = 0
        self.now_serving = 0
        self.active = 0
        self.inflight = 0

    def arrive(self) -> tuple[int, int]:
        with self.cv:
            already = self.inflight
            self.inflight += 1
            ticket = self.next_ticket
            self.next_ticket += 1
            return ticket, already

    def acquire(self, ticket: int) -> float:
        t0 = time.time()
        with self.cv:
            while (ticket != self.now_serving
                   or (self.capacity is not None and self.active >= self.capacity)):
                self.cv.wait()
            self.now_serving += 1
            self.active += 1
            self.cv.notify_all()
        return time.time() - t0

    def release(self) -> None:
        with self.cv:
            self.active -= 1
            self.inflight -= 1
            self.cv.notify_all()

    def abandon(self, ticket: int) -> None:
        """Drop a request that never got admitted, without stranding the
        tickets queued behind it."""
        with self.cv:
            if ticket == self.now_serving:
                self.now_serving += 1
            self.inflight -= 1
            self.cv.notify_all()


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
        self.served = 0
        self.cursors: dict[str, int] = {}
        self.lock = threading.Lock()
        self.journal_lock = threading.Lock()
        self.mismatches = 0
        # the replay's own model_calls journal: REPLAY-time arrival stamps,
        # so turn windows can be cut on the machine being measured (the
        # recording's timestamps belong to the original machine)
        self.journal = open(journal, "w") if journal else None

    def log_served(self, entry: dict[str, Any]) -> None:
        if self.journal:
            with self.journal_lock:
                self.journal.write(json.dumps(entry) + "\n")
                self.journal.flush()

    def next_call(self, session: str) -> tuple[dict[str, Any] | None, int]:
        """Next call for one session, plus its 1-based turn in that session."""
        with self.lock:
            i = self.cursors.get(session, 0)
            self.cursors[session] = i + 1
            if i >= len(self.calls):
                return None, i + 1
            self.served += 1
            return self.calls[i], i + 1

    def note_mismatch(self) -> None:
        with self.lock:
            self.mismatches += 1


REC: _Recording | None = None
GATE: _Gate | None = None
ARGS: argparse.Namespace | None = None


def _prompt_hash(body: dict[str, Any]) -> str:
    msgs = body.get("messages", body.get("prompt", ""))
    return hashlib.sha256(
        json.dumps(msgs, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _session_key(headers: Any, body: dict[str, Any]) -> str:
    """Stable per-conversation key. Header first, first-message hash after."""
    hdr = headers.get(SESSION_HEADER)
    if hdr:
        return str(hdr)
    msgs = body.get("messages")
    first = msgs[0] if isinstance(msgs, list) and msgs else body.get("prompt", "")
    seed = {"system": body.get("system"), "first": first}
    digest = hashlib.sha256(
        json.dumps(seed, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return f"sha-{digest[:16]}"


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
        assert REC is not None and GATE is not None
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        session = _session_key(self.headers, body)
        t_arrival = time.time()
        ticket, inflight_at_arrival = GATE.arrive()
        try:
            queue_wait = GATE.acquire(ticket)
        except BaseException:
            GATE.abandon(ticket)
            raise
        try:
            self._serve(body, session, t_arrival, queue_wait,
                        inflight_at_arrival)
        finally:
            GATE.release()

    def _serve(self, body: dict[str, Any], session: str, t_arrival: float,
               queue_wait: float, inflight_at_arrival: int) -> None:
        assert REC is not None and ARGS is not None
        call, turn_in_session = REC.next_call(session)
        if call is None:
            self._json(410, {"error": "recording exhausted"})
            print(f"[mock-endpoint] session {session[:12]} turn "
                  f"{turn_in_session}: recording exhausted, returned 410")
            return

        want = call.get("prompt_sha256")
        if want and not ARGS.no_prompt_check:
            got = _prompt_hash(body)
            if got != want:
                REC.note_mismatch()
                print(f"[mock-endpoint] WARNING session {session[:12]} turn "
                      f"{turn_in_session}: prompt hash mismatch (replay "
                      f"diverged from recording), serving anyway")

        text = call.get("response") or ""
        tools = _norm_tool_calls(call.get("tool_calls"))
        finish = _finish_reason(call, tools)
        ttft = float(call.get("ttft_s", 0.0)) / REC.speed
        dur = float(call.get("duration_s", 0.0)) / REC.speed
        REC.log_served({
            "turn": call.get("turn", turn_in_session),
            "task_id": call.get("task_id"),
            "session_id": session, "turn_in_session": turn_in_session,
            "ts_arrival_epoch_s": t_arrival,
            "ts_epoch_s": time.time(),
            "inflight_at_arrival": inflight_at_arrival,
            "queue_wait_s": round(queue_wait, 6),
            "ttft_s": ttft, "duration_s": dur,
            "prompt_tokens": call.get("prompt_tokens", 0),
            "completion_tokens": call.get("completion_tokens", 0),
        })
        model = call.get("model", "chopper-replay")
        usage_in = int(call.get("prompt_tokens", 0))
        usage_out = int(call.get("completion_tokens", 0))
        rid = f"replay-{session[:8]}-{turn_in_session}"
        anthropic = self.path.startswith("/v1/messages")

        if body.get("stream"):
            if anthropic:
                self._stream_anthropic(rid, model, text, tools, finish,
                                       ttft, dur, usage_in, usage_out)
            else:
                opts = body.get("stream_options") or {}
                self._stream_openai(rid, model, text, tools, finish, ttft, dur,
                                    usage_in, usage_out,
                                    bool(opts.get("include_usage")))
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
        print(f"[mock-endpoint] served session {session[:12]} turn "
              f"{turn_in_session}/{len(REC.calls)} "
              f"({'anthropic' if anthropic else 'openai'}, "
              f"{len(tools)} tool_calls, finish={finish}, "
              f"stream={bool(body.get('stream'))}, "
              f"queue_wait={queue_wait:.3f}s, "
              f"inflight_at_arrival={inflight_at_arrival})")

    def _paced_chunks(self, text: str) -> list[str]:
        words = text.split(" ") or [""]
        nchunks = min(len(words), 40)
        per = max(1, len(words) // nchunks)
        return [" ".join(words[j:j + per]) + (" " if j + per < len(words) else "")
                for j in range(0, len(words), per)]

    def _stream_openai(self, rid: str, model: str, text: str,
                       tools: list[dict[str, Any]], finish: str,
                       ttft: float, dur: float,
                       usage_in: int = 0, usage_out: int = 0,
                       include_usage: bool = False) -> None:
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
        if include_usage:
            # OpenAI stream_options.include_usage: a final choices-empty
            # chunk carrying usage. A recording proxy in front of the mock
            # needs this to recover token counts from a streamed leg.
            self._sse({"id": rid, "object": "chat.completion.chunk",
                       "created": now, "model": model, "choices": [],
                       "usage": {"prompt_tokens": usage_in,
                                 "completion_tokens": usage_out,
                                 "total_tokens": usage_in + usage_out}})
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
    global REC, GATE, ARGS
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recording", required=True, help="JSONL recording of model calls")
    p.add_argument("--host", default="127.0.0.1",
                   help="bind address; 0.0.0.0 or a node IP makes the mock "
                        "reachable from another node and also exposes the "
                        "port on this node")
    p.add_argument("--port", type=int, default=8123)
    p.add_argument("--max-concurrency", type=int, default=None,
                   help="serve at most this many requests at once; the rest "
                        "wait in a FIFO queue and the imposed wait is "
                        "journalled as queue_wait_s. Emulation of a finite "
                        "batch slot count, not real continuous batching")
    p.add_argument("--no-prompt-check", action="store_true",
                   help="skip the prompt_sha256 divergence check (for load "
                        "drivers that do not reproduce the original prompts)")
    p.add_argument("--speed", type=float, default=1.0,
                   help="divide recorded delays by this factor (10 = fast replay)")
    p.add_argument("--task", default=None,
                   help="serve only this task_id's calls (drops __warmup__ etc.)")
    p.add_argument("--journal", default=None,
                   help="write the replay's own model_calls jsonl here "
                        "(replay-time arrival stamps for turn windows)")
    a = p.parse_args()
    ARGS = a
    REC = _Recording(a.recording, a.speed, a.task, a.journal)
    GATE = _Gate(a.max_concurrency)
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    cap = "unlimited" if a.max_concurrency is None else str(a.max_concurrency)
    print(f"[mock-endpoint] {len(REC.calls)} recorded calls, speed {a.speed}x, "
          f"max concurrency {cap}, serving http://{a.host}:{a.port}/v1 "
          f"(openai + anthropic)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    print(f"[mock-endpoint] served {REC.served} calls over "
          f"{len(REC.cursors)} sessions, {REC.mismatches} prompt mismatches")


if __name__ == "__main__":
    main()
