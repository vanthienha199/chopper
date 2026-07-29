"""Streaming sinks: where derived records go.

A sink receives one record (a plain dict) at a time via ``emit`` and is
responsible for delivering it somewhere. The important sink is
``JsonLinesSocketSink``, which ships records as newline-delimited JSON over
TCP to the dashboard.

Non-blocking guarantee: ``emit`` never blocks the caller and never raises.
The socket sink hands records to a bounded background queue and a worker
thread does the connecting and sending. If the consumer is absent or slow the
queue fills and the OLDEST records are dropped, so a profiled workload is
never stalled by the dashboard. Dropped-record counts are logged, never
hidden, so the stream is honestly lossy rather than silently truncated.
"""

import json
import logging
import queue
import socket
import threading
import time
from typing import Any, Callable, Protocol

log = logging.getLogger("chopper.stream")

Record = dict[str, Any]


class StreamSink(Protocol):
    """A destination for stream records."""

    def emit(self, record: Record) -> None:
        ...

    def close(self) -> None:
        ...


class NullSink:
    """Discards everything. Useful as a default / for benchmarking overhead."""

    def emit(self, record: Record) -> None:
        pass

    def close(self) -> None:
        pass


class CallbackSink:
    """Calls a Python function per record. Used by tests and in-process demos."""

    def __init__(self, fn: Callable[[Record], None]):
        self._fn = fn

    def emit(self, record: Record) -> None:
        self._fn(record)

    def close(self) -> None:
        pass


class JsonLinesSocketSink:
    """Non-blocking newline-delimited-JSON-over-TCP sink.

    Records are enqueued and sent by a background worker that (re)connects to
    ``host:port`` as needed. Emit is O(1) and lossy under backpressure.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8900,
        max_queue: int = 4096,
        reconnect_s: float = 1.0,
    ):
        self._addr = (host, port)
        self._q: "queue.Queue[Record | None]" = queue.Queue(maxsize=max_queue)
        self._reconnect_s = reconnect_s
        self._dropped = 0
        self._sent = 0
        self._stop = threading.Event()
        self._sock: socket.socket | None = None
        self._worker = threading.Thread(
            target=self._run, name="chopper-stream-sink", daemon=True
        )
        self._worker.start()

    def emit(self, record: Record) -> None:
        try:
            self._q.put_nowait(record)
        except queue.Full:
            # Drop the oldest to make room; keep the stream current, not complete.
            try:
                self._q.get_nowait()
                self._dropped += 1
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(record)
            except queue.Full:
                self._dropped += 1

    def _connect(self) -> None:
        if self._sock is not None:
            return
        try:
            s = socket.create_connection(self._addr, timeout=2.0)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._sock = s
            log.info("stream sink connected to %s:%s", *self._addr)
        except OSError:
            self._sock = None

    def _run(self) -> None:
        while not self._stop.is_set():
            self._connect()
            if self._sock is None:
                time.sleep(self._reconnect_s)
                continue
            try:
                item = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                break
            line = (json.dumps(item, separators=(",", ":")) + "\n").encode("utf-8")
            try:
                self._sock.sendall(line)
                self._sent += 1
            except OSError:
                log.warning("stream sink lost connection, will reconnect")
                try:
                    self._sock.close()
                finally:
                    self._sock = None

    @property
    def stats(self) -> dict[str, int]:
        """Sent / dropped / queued counts, for a status line or end-of-run log."""
        return {"sent": self._sent, "dropped": self._dropped, "queued": self._q.qsize()}

    def close(self) -> None:
        self._stop.set()
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass
        self._worker.join(timeout=2.0)
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None
        if self._dropped:
            log.warning("stream sink dropped %d records under backpressure", self._dropped)
