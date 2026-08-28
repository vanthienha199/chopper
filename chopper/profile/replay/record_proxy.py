"""Minimal recording proxy: sits between a harness and a live model server,
forwards requests unchanged, and writes a model_calls.jsonl compatible with
decompose.py and the mock endpoint. This makes chopper self-sufficient for
record -> replay: no external proxy needed to produce a recording.

Fields per call: turn, ts_epoch_s (arrival), ttft_s (first byte of a
streaming response; 0.0 for non-streaming, where first token time is not
observable at the proxy), duration_s, prompt_sha256, prompt/completion
token counts when the upstream reports usage, response text and tool_calls
(OpenAI shape), finish_reasons, stream flag.

    python record_proxy.py --upstream http://127.0.0.1:8000 \
        --port 8124 --out model_calls.jsonl [--task TASK_ID]
"""

import argparse
import functools
import hashlib
import json
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

print = functools.partial(print, flush=True)

ARGS = None
LOCK = threading.Lock()
TURN = 0


def _prompt_hash(body: dict) -> str:
    msgs = body.get("messages", body.get("prompt", ""))
    return hashlib.sha256(
        json.dumps(msgs, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        upstream = f"{ARGS.upstream}{self.path}"
        try:
            with urllib.request.urlopen(upstream, timeout=120) as r:
                data = r.read()
                self.send_response(r.status)
                self.send_header("Content-Type",
                                 r.headers.get("Content-Type", "application/json"))
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        except Exception as e:
            self.send_response(502)
            self.end_headers()
            print(f"[record-proxy] GET {self.path} failed: {e}")

    def do_POST(self):
        global TURN
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n)
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            body = {}
        stream = bool(body.get("stream"))
        t0 = time.time()

        req = urllib.request.Request(
            f"{ARGS.upstream}{self.path}", data=raw,
            headers={"Content-Type": "application/json"}, method="POST")
        ttft = 0.0
        chunks = []
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                if stream:
                    self.send_response(r.status)
                    self.send_header("Content-Type",
                                     r.headers.get("Content-Type",
                                                   "text/event-stream"))
                    self.send_header("Connection", "close")
                    self.end_headers()
                    first = True
                    while True:
                        chunk = r.read(8192)
                        if not chunk:
                            break
                        if first:
                            ttft = time.time() - t0
                            first = False
                        chunks.append(chunk)
                        self.wfile.write(chunk)
                        self.wfile.flush()
                else:
                    data = r.read()
                    chunks.append(data)
                    self.send_response(r.status)
                    self.send_header("Content-Type",
                                     r.headers.get("Content-Type",
                                                   "application/json"))
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
        except Exception as e:
            self.send_response(502)
            self.end_headers()
            print(f"[record-proxy] POST {self.path} failed: {e}")
            return
        dur = time.time() - t0

        text, tool_calls, finish, p_tok, c_tok = "", [], [], 0, 0
        try:
            payload = b"".join(chunks).decode()
            if stream:
                for line in payload.splitlines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        ev = json.loads(line[6:])
                        for ch in ev.get("choices", []):
                            d = ch.get("delta", {})
                            text += d.get("content") or ""
                            for tc in d.get("tool_calls") or []:
                                f = tc.get("function", {})
                                tool_calls.append({
                                    "name": f.get("name", ""),
                                    "arguments": f.get("arguments", ""),
                                    "id": tc.get("id", "")})
                            if ch.get("finish_reason"):
                                finish.append(ch["finish_reason"])
                        u = ev.get("usage") or {}
                        p_tok = u.get("prompt_tokens", p_tok)
                        c_tok = u.get("completion_tokens", c_tok)
            else:
                resp = json.loads(payload)
                for ch in resp.get("choices", []):
                    msg = ch.get("message", {})
                    text += msg.get("content") or ""
                    for tc in msg.get("tool_calls") or []:
                        f = tc.get("function", {})
                        tool_calls.append({"name": f.get("name", ""),
                                           "arguments": f.get("arguments", ""),
                                           "id": tc.get("id", "")})
                    if ch.get("finish_reason"):
                        finish.append(ch["finish_reason"])
                u = resp.get("usage") or {}
                p_tok = u.get("prompt_tokens", 0)
                c_tok = u.get("completion_tokens", 0)
        except Exception as e:
            print(f"[record-proxy] parse warning: {e}")

        with LOCK:
            TURN += 1
            entry = {
                "schema_version": 8, "record": "model_call",
                "task_id": ARGS.task, "turn": TURN,
                "ts_epoch_s": t0, "ttft_s": round(ttft, 4),
                "duration_s": round(dur, 4),
                "prompt_tokens": p_tok, "completion_tokens": c_tok,
                "prompt_sha256": _prompt_hash(body),
                "response": text, "tool_calls": tool_calls,
                "finish_reasons": finish, "stream": stream,
                "protocol": "openai", "endpoint": self.path,
            }
            with open(ARGS.out, "a") as f:
                f.write(json.dumps(entry) + "\n")
        print(f"[record-proxy] call {TURN}: {dur:.2f}s, "
              f"{len(tool_calls)} tool_calls, finish={finish}")


def main():
    global ARGS
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream", required=True)
    p.add_argument("--port", type=int, default=8124)
    p.add_argument("--out", required=True)
    p.add_argument("--task", default=None)
    ARGS = p.parse_args()
    open(ARGS.out, "w").close()
    srv = ThreadingHTTPServer(("127.0.0.1", ARGS.port), Handler)
    print(f"[record-proxy] forwarding :{ARGS.port} -> {ARGS.upstream}, "
          f"recording to {ARGS.out}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    print(f"[record-proxy] recorded {TURN} calls")


if __name__ == "__main__":
    main()
