"""Minimal recording proxy: sits between a harness and a live model server,
forwards requests unchanged, and writes a model_calls.jsonl compatible with
decompose.py and the mock endpoint. This makes chopper self-sufficient for
record -> replay: no external proxy needed to produce a recording.

Fields per call: seq (global arrival order), session_id, turn_in_session,
turn (alias of seq, kept so older readers still work), ts_epoch_s
(arrival), inflight_at_arrival, ttft_s (first byte of a streaming response;
0.0 for non-streaming when --no-ttft-upgrade is set, where first token time
is not observable at the proxy), duration_s, prompt_sha256, prompt and
completion token counts when the upstream reports usage, response text and
tool_calls (OpenAI shape), finish_reasons, stream flag.

Multi-request recording: several agent sessions may share one backend and
one proxy. A single global turn counter would interleave their turns and
make the recording unusable, so turns are numbered per session. The session
key is the X-Chopper-Session request header when the harness sets one
(preferred, because it is exact), otherwise the SHA-256 of the first
message of the request (plus the Anthropic `system` field when present),
which stays constant inside one conversation and differs across tasks. The
hash fallback collides when two concurrent sessions start from an identical
first message, for example the same task replayed N times; set the header
in that case. The header is forwarded upstream so a downstream mock
endpoint can key on the same value.

The proxy detects that collision at runtime rather than leaving it to be
found in the data later. A request carrying no assistant message is the
opening turn of a conversation, so when one arrives for a session key that
already has turns recorded, the proxy prints a warning naming the likely
cause, once per session key. It is a warning and not an error, because a
sequential rerun of the same task looks identical and is harmless.

inflight_at_arrival is the number of requests the proxy had already
accepted and not yet finished when this one arrived. It is the proxy-side
view of offered concurrency, which is what multi_request.py groups on.

--host defaults to 127.0.0.1. Binding to a routable address (0.0.0.0 or a
node IP) lets a harness on another node reach the proxy, and it also
exposes the port to anything else that can reach that node, so use it only
on a trusted network.

    python record_proxy.py --upstream http://127.0.0.1:8000 \
        --port 8124 --out model_calls.jsonl [--task TASK_ID] [--host 0.0.0.0]
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

SESSION_HEADER = "X-Chopper-Session"

ARGS = None
LOCK = threading.Lock()
SEQ = 0
TURNS: dict[str, int] = {}
INFLIGHT = 0
WARNED: set[str] = set()


def _assemble_completion(sse_payload: str) -> dict:
    """Fold an OpenAI SSE stream back into one chat.completion object."""
    content = ""
    tool_calls: dict[int, dict] = {}
    finish = None
    usage = {}
    rid, model = "proxy-upgraded", ""
    for line in sse_payload.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        try:
            ev = json.loads(line[6:])
        except json.JSONDecodeError:
            continue
        rid = ev.get("id", rid)
        model = ev.get("model", model)
        if ev.get("usage"):
            usage = ev["usage"]
        for ch in ev.get("choices", []):
            d = ch.get("delta", {})
            content += d.get("content") or ""
            for tc in d.get("tool_calls") or []:
                i = tc.get("index", 0)
                slot = tool_calls.setdefault(
                    i, {"id": "", "type": "function",
                        "function": {"name": "", "arguments": ""}})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                f = tc.get("function", {})
                if f.get("name"):
                    slot["function"]["name"] = f["name"]
                slot["function"]["arguments"] += f.get("arguments") or ""
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
    message = {"role": "assistant", "content": content or None}
    if tool_calls:
        message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
    return {"id": rid, "object": "chat.completion",
            "created": int(time.time()), "model": model,
            "choices": [{"index": 0, "message": message,
                         "finish_reason": finish or "stop"}],
            "usage": usage}


def _prompt_hash(body: dict) -> str:
    msgs = body.get("messages", body.get("prompt", ""))
    return hashlib.sha256(
        json.dumps(msgs, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _session_key(headers, body: dict) -> str:
    """Stable per-conversation key. Header first, first-message hash after."""
    hdr = headers.get(SESSION_HEADER)
    if hdr:
        return hdr
    msgs = body.get("messages")
    first = msgs[0] if isinstance(msgs, list) and msgs else body.get("prompt", "")
    seed = {"system": body.get("system"), "first": first}
    digest = hashlib.sha256(
        json.dumps(seed, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return f"sha-{digest[:16]}"


def _looks_like_first_turn(body: dict) -> bool:
    """True when the request carries no assistant reply yet.

    An agent turn after the first always replays the assistant messages it
    has already received, so a request with none of them is the opening
    turn of a conversation.
    """
    msgs = body.get("messages")
    if not isinstance(msgs, list):
        return False
    return not any(isinstance(m, dict) and m.get("role") == "assistant"
                   for m in msgs)


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
        global SEQ, INFLIGHT
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n)
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            body = {}
        session = _session_key(self.headers, body)
        had_header = bool(self.headers.get(SESSION_HEADER))
        with LOCK:
            SEQ += 1
            seq = SEQ
            turn_in_session = TURNS.get(session, 0) + 1
            TURNS[session] = turn_in_session
            inflight_at_arrival = INFLIGHT
            INFLIGHT += 1
            # merging two conversations into one session produces a
            # plausible-looking but wrong recording, which is worse than a
            # crash, so say so loudly (once per key) and keep going
            collision = (turn_in_session > 1 and _looks_like_first_turn(body)
                         and session not in WARNED)
            if collision:
                WARNED.add(session)
        if collision:
            cause = ("the client reused the X-Chopper-Session value across "
                     "conversations" if had_header else
                     "concurrent clients on the same task with no "
                     f"{SESSION_HEADER} header, whose first messages hash to "
                     "the same key")
            print(f"[record-proxy] WARNING session {session}: an opening-turn "
                  f"request arrived for a session that already has "
                  f"{turn_in_session - 1} turn(s) recorded. Turns from "
                  f"different conversations are being merged into one "
                  f"session and this recording will be wrong. Likely cause: "
                  f"{cause}. Set a distinct {SESSION_HEADER} per "
                  f"conversation. A sequential rerun of the same task looks "
                  f"identical here and is harmless.")
        try:
            self._forward(raw, body, session, seq, turn_in_session,
                          inflight_at_arrival)
        finally:
            with LOCK:
                INFLIGHT -= 1

    def _forward(self, raw, body, session, seq, turn_in_session,
                 inflight_at_arrival):
        stream = bool(body.get("stream"))
        # ttft observation for non-streaming harnesses (e.g. mini-swe-agent,
        # which cannot stream: its model layer reads response.choices[0]
        # directly): upgrade the UPSTREAM leg to streaming so the first
        # chunk timestamps queue+prefill, then reassemble a normal
        # non-streaming response for the harness. Transparent to the client.
        upgraded = False
        if (not stream and not ARGS.no_ttft_upgrade
                and self.path.rstrip("/").endswith("/chat/completions")):
            body = dict(body, stream=True,
                        stream_options={"include_usage": True})
            raw = json.dumps(body).encode()
            upgraded = True
        t0 = time.time()

        fwd = {"Content-Type": "application/json"}
        if self.headers.get(SESSION_HEADER):
            # keep the session identity visible to the upstream (a mock
            # endpoint keys its per-session response stream on this)
            fwd[SESSION_HEADER] = self.headers[SESSION_HEADER]
        req = urllib.request.Request(
            f"{ARGS.upstream}{self.path}", data=raw,
            headers=fwd, method="POST")
        ttft = 0.0
        chunks = []
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                if upgraded:
                    # consume the upstream SSE ourselves, time the first
                    # chunk, rebuild one chat.completion for the client.
                    # read1 returns what has arrived; read(n) would block
                    # for n bytes and report ttft at the END of any
                    # response shorter than n, which is most agent turns
                    first = True
                    while True:
                        chunk = r.read1(8192)
                        if not chunk:
                            break
                        if first:
                            ttft = time.time() - t0
                            first = False
                        chunks.append(chunk)
                    resp = _assemble_completion(b"".join(chunks).decode())
                    data = json.dumps(resp).encode()
                    chunks = [data]
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                elif stream:
                    self.send_response(r.status)
                    self.send_header("Content-Type",
                                     r.headers.get("Content-Type",
                                                   "text/event-stream"))
                    self.send_header("Connection", "close")
                    self.end_headers()
                    first = True
                    while True:
                        chunk = r.read1(8192)
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

        entry = {
            "schema_version": 9, "record": "model_call",
            "task_id": ARGS.task, "seq": seq, "turn": seq,
            "session_id": session, "turn_in_session": turn_in_session,
            "inflight_at_arrival": inflight_at_arrival,
            "ts_epoch_s": t0, "ttft_s": round(ttft, 4),
            "duration_s": round(dur, 4),
            "prompt_tokens": p_tok, "completion_tokens": c_tok,
            "prompt_sha256": _prompt_hash(body),
            "response": text, "tool_calls": tool_calls,
            "finish_reasons": finish, "stream": stream,
            "protocol": "openai", "endpoint": self.path,
        }
        with LOCK:
            with open(ARGS.out, "a") as f:
                f.write(json.dumps(entry) + "\n")
        print(f"[record-proxy] call {seq} (session {session[:12]} turn "
              f"{turn_in_session}, {inflight_at_arrival} in flight on "
              f"arrival): {dur:.2f}s, {len(tool_calls)} tool_calls, "
              f"finish={finish}")


def main():
    global ARGS
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream", required=True)
    p.add_argument("--host", default="127.0.0.1",
                   help="bind address; 0.0.0.0 or a node IP makes the proxy "
                        "reachable from another node and also exposes the "
                        "port on this node")
    p.add_argument("--port", type=int, default=8124)
    p.add_argument("--out", required=True)
    p.add_argument("--task", default=None)
    p.add_argument("--no-ttft-upgrade", action="store_true",
                   help="pass non-streaming requests through unchanged "
                        "instead of upgrading the upstream leg to streaming "
                        "for first-token timing")
    ARGS = p.parse_args()
    open(ARGS.out, "w").close()
    srv = ThreadingHTTPServer((ARGS.host, ARGS.port), Handler)
    print(f"[record-proxy] forwarding {ARGS.host}:{ARGS.port} -> "
          f"{ARGS.upstream}, recording to {ARGS.out}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    print(f"[record-proxy] recorded {SEQ} calls over {len(TURNS)} sessions")


if __name__ == "__main__":
    main()
