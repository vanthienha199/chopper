# Replay record schema (proposal, to agree with the harness side)

Goal: one recording of a live run = enough to replay the host-side work
reproducibly, with the model behind a mocked endpoint. Three files per run,
each produced by the layer that already sees that data.

## 1. model_calls.jsonl  (produced by the proxy)

One line per model call, in call order (proxy schema_version 8):

    {
      "turn": 3,
      "ts_epoch_s": 1754321000.123,        // when the request arrived
      "ttft_s": 0.42,                      // first token latency observed
      "duration_s": 3.10,                  // full response duration
      "prompt_tokens": 8123,
      "completion_tokens": 350,
      "prompt_sha256": "ab12...",          // hash of messages, for divergence check
      "response": "...full text...",       // text part, OFTEN EMPTY on tool turns
      "tool_calls": [                      // structured tool calls, flat shape
        {"name": "bash",
         "arguments": "{\"command\": \"ls -R\"}",   // JSON string (or object)
         "id": "chatcmpl-tool-..."}
      ],
      "finish_reasons": ["tool_calls"],    // list; first entry is used
      "protocol": "openai",                // or "anthropic" (claude-code)
      "stream": false,
      "model": "gpt-oss-120b"
    }

The mocked endpoint (chopper.profile.replay.mock_endpoint) serves exactly
this file back, FIFO, with the recorded timing. Most agent turns carry
empty text plus tool_calls (claude-code/codex especially); the mock returns
them structured, in the protocol of the endpoint being asked:
POST /v1/chat/completions -> OpenAI shape, POST /v1/messages -> Anthropic
shape (text + tool_use content blocks), streaming and non-streaming both.
Validated round-trip against real GH200 recordings (mini-swe-agent openai,
claude-code anthropic).

## 2. tool_calls.jsonl  (produced by the harness adapter; the proxy ships
## this as a richer typed `actions.jsonl`, same idea)

One line per tool execution, in execution order:

    {
      "turn": 3,
      "agent": "swebench-cell-2",          // matches CHOPPER_AGENT_ID
      "ts_epoch_s": 1754321003.4,
      "duration_s": 1.9,
      "command": ["python", "-m", "pytest", "tests/x.py"],
      "cwd": "/repo",
      "exit_code": 0
    }

Plus one header line describing the initial state:

    {"initial_state": {"task_id": "swe-bench-123", "repo_sha": "...",
                       "container": "sha256:..."}}

## 3. profile.json clock anchors  (already produced by chopper_driver)

epoch_s + monotonic_ns + GPU-clock read back-to-back at start and stop.
Everything above is epoch-stamped; the anchors convert epoch to the
telemetry/kernel clock domain, which is how turn boundaries land on the
chopper timeline (and how energy.py integrates power per turn).

## Replay procedure

1. Restore initial_state (same task container/repo sha).
2. Start mock_endpoint with model_calls.jsonl, point the harness at it.
3. Run the harness normally; it re-executes real tool commands, the model
   side is served from the recording.
4. Profile with chopper as usual. Heavier collectors are now affordable
   because the run is deterministic and repeatable.

Divergence policy: replay is best-effort deterministic. The endpoint checks
prompt hashes and counts mismatches; a diverged replay is reported, never
silently trusted.
