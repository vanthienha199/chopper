"""Mocked model endpoint for reproducible agent replay.

Replays a recorded run's model responses over an OpenAI-compatible HTTP API,
in order, with the recorded timing (time-to-first-token and total duration).
Point a harness at this instead of vLLM and the CPU side of an agent run can
be replayed deterministically: no GPU, no model variance, no network variance.
This is the "mocked or recorded model endpoints" piece of the benchmark
(hardware comparisons must not be confounded by model randomness), and the
partner of the CPU-command replay log.

Recording format: JSONL, one object per model call, in call order:

    {"turn": 0, "response": "...text...", "ttft_s": 0.42, "duration_s": 3.1,
     "prompt_sha256": "ab12..." (optional), "model": "gpt-oss-120b" (optional)}

Serving semantics (v1, single-agent replay):
  - Responses are served strictly FIFO. If the recording had N calls, call
    N+1 returns HTTP 410 (recording exhausted).
  - If prompt_sha256 is present, the incoming prompt is hashed and compared;
    a mismatch is logged loudly (the replayed harness diverged from the
    recording) but the response is still served, so a run never wedges.
  - stream=true is supported: the first SSE chunk is delayed by ttft_s and
    the remaining chunks are paced evenly across duration_s. Non-streaming
    requests sleep ttft_s + duration_s, then return the whole body.
  - --speed N divides all recorded delays by N (10 = fast replay, ~0 waits).

    python -m chopper.profile.replay.mock_endpoint --recording run1.jsonl \
        --port 8123 --speed 1
"""

import argparse
import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class _Recording:
    def __init__(self, path: str, speed: float):
        self.calls: list[dict[str, Any]] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.calls.append(json.loads(line))
        self.speed = max(speed, 1e-9)
        self.i = 0
        self.lock = threading.Lock()
        self.mismatches = 0

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

        text = call.get("response", "")
        ttft = float(call.get("ttft_s", 0.0)) / REC.speed
        dur = float(call.get("duration_s", 0.0)) / REC.speed
        model = call.get("model", "chopper-replay")
        now = int(time.time())
        rid = f"chatcmpl-replay-{REC.i - 1}"

        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            time.sleep(ttft)
            # pace the content evenly across the recorded duration
            words = text.split(" ") or [""]
            nchunks = min(len(words), 40)
            per = max(1, len(words) // nchunks)
            chunks = [" ".join(words[j:j + per]) + (" " if j + per < len(words) else "")
                      for j in range(0, len(words), per)]
            delay = dur / max(1, len(chunks))
            for j, chunk in enumerate(chunks):
                ev = {"id": rid, "object": "chat.completion.chunk", "created": now,
                      "model": model, "choices": [{"index": 0, "delta":
                          ({"role": "assistant", "content": chunk} if j == 0
                           else {"content": chunk}), "finish_reason": None}]}
                self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                self.wfile.flush()
                time.sleep(delay)
            done = {"id": rid, "object": "chat.completion.chunk", "created": now,
                    "model": model, "choices": [{"index": 0, "delta": {},
                                                 "finish_reason": "stop"}]}
            self.wfile.write(f"data: {json.dumps(done)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
            time.sleep(ttft + dur)
            self._json(200, {
                "id": rid, "object": "chat.completion", "created": now,
                "model": model,
                "choices": [{"index": 0, "message":
                             {"role": "assistant", "content": text},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": call.get("prompt_tokens", 0),
                          "completion_tokens": call.get("completion_tokens", 0),
                          "total_tokens": call.get("prompt_tokens", 0)
                          + call.get("completion_tokens", 0)}})
        print(f"[mock-endpoint] served call {REC.i - 1}/{len(REC.calls)} "
              f"(ttft {ttft:.2f}s, dur {dur:.2f}s, stream={bool(body.get('stream'))})")


def main() -> None:
    global REC
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recording", required=True, help="JSONL recording of model calls")
    p.add_argument("--port", type=int, default=8123)
    p.add_argument("--speed", type=float, default=1.0,
                   help="divide recorded delays by this factor (10 = fast replay)")
    a = p.parse_args()
    REC = _Recording(a.recording, a.speed)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"[mock-endpoint] {len(REC.calls)} recorded calls, speed {a.speed}x, "
          f"serving http://127.0.0.1:{a.port}/v1")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    print(f"[mock-endpoint] served {REC.i}/{len(REC.calls)} calls, "
          f"{REC.mismatches} prompt mismatches")


if __name__ == "__main__":
    main()
