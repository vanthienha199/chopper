"""Per-node clock anchors and the reader-side conversion to a common timeline.

Chopper's rule is that every collector writes its own native clock and the
conversion happens at read time (TRACE_FORMAT.md, "Clock anchors, not clock
conversion at write time"). A single anchor set is enough on one node. Under
multi-node it is not, for two separate reasons:

1. Each node's ``monotonic_ns`` counts from that node's own boot, so two
   nodes' monotonic values are not comparable at all.
2. The nodes' wall clocks differ from each other by milliseconds even under
   NTP, and each node's monotonic-to-epoch offset moves over a long run as
   the NTP daemon steers the wall clock.

So the anchor becomes a per-node record, taken twice: once when collection
starts and once when it stops. Two anchors per node make the drift over the
run a measured quantity instead of an assumption, and let the reader
interpolate the offset rather than pin it to the value it had at t=0.

Write side (collect.py):

    write_anchor(outdir, "start")   # before the collectors start
    write_anchor(outdir, "stop")    # after they join

Both calls append to ``<outdir>/clock_anchors.json``. Under multi-node SLURM,
``collect.py`` has already namespaced ``outdir`` by hostname, so each node
gets its own anchor file.

Read side:

    anchors = load_anchors("outputs/run")          # globs every node's file
    df = anchors.attach_epoch(gpu_df)              # adds epoch_ns
    print(anchors.drift_table())                   # how big was the fix

    df = anchors.attach_reference_epoch(gpu_df, reference="node0")
    print(anchors.peer_offset_table())             # the cross-node offset

``attach_epoch`` writes the size of the correction into its own column, so a
reader can see how much interpolation moved each row rather than trusting a
silent adjustment.

Why the anchors also carry a peer measurement
---------------------------------------------

Per-node anchors put each node on that node's own wall clock. They cannot
remove a disagreement between two nodes' wall clocks, because neither node can
see the other. That residual was measured on the AMD HPC Fund ``mi2508x``
partition on 2026-09-11, nodes ``k004-004`` and ``k004-005``, SLURM jobs
414426, 414427 and 414431, using an NTP-style socket exchange with 200 rounds
per measurement:

* The two nodes' wall clocks differed by about 400 microseconds. Each single
  measurement was precise: the standard deviation across the 200 rounds was
  about 2 microseconds and the minimum round trip was about 60 microseconds.
* The offset drifts. Within job 414431 it read -364.7, -377.4, -396.7 and
  -419.6 microseconds at t+0, t+60, t+120 and t+180 seconds. That is 54.9
  microseconds over 180 seconds, a relative skew of about 0.30 ppm between the
  two nodes' wall clocks, which is roughly 1.1 milliseconds of extra
  divergence over an hour.
* chrony is not a usable substitute. Its "System time" field is the residual
  against that daemon's own filtered estimate, not a bound on absolute
  accuracy, and both nodes reported a root delay near 34 ms with a root
  dispersion near 1.5 ms. Differencing the two nodes' chrony values on the same
  node pair gave +225.2, +15.2 and +175.5 microseconds across the three jobs,
  while the direct method was stable and drifted smoothly. The chrony estimate
  is noise at this scale.

So the offset has to be measured directly, and because it drifts it has to be
measured at least twice per run, exactly as the monotonic anchors already are.
``measure_peer_offset`` takes one such measurement and ``write_anchor`` records
it on the anchor. The reader interpolates between the start and stop
measurements in ``to_reference_epoch_ns``, which places every node on one
chosen reference node's wall clock and reports the size of that cross-node
correction in its own column.
"""

import json
import socket
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import numpy as np
import pandas as pd
from loguru import logger

ANCHOR_FILENAME = "clock_anchors.json"
ANCHOR_SCHEMA = "chopper-clock-anchors/1"

#: Where an anchor was taken in the run. The pair bounds the collection window.
TAKEN_START = "start"
TAKEN_STOP = "stop"

#: Default TCP port for the peer offset exchange. It is unprivileged and not
#: registered to anything, and the exchange only runs between compute nodes of
#: one job.
PEER_PORT = 48213

#: Default number of request/response rounds in one peer measurement. One
#: 200 round measurement took 14 ms of wall time on the HPC Fund fabric, where
#: the round trip floor is about 60 microseconds, and 200 rounds are what make
#: the lowest-delay decile large enough to average.
PEER_ROUNDS = 200


class CrossNodeOffsetUnavailable(Exception):
    """No measured wall clock offset links a node to the chosen reference node.

    The cross-node path raises this instead of returning uncorrected
    timestamps. See ``AnchorSet.reference_correction_ns`` for why that is an
    exception and not a flagged column.
    """


@dataclass
class PeerOffset:
    """One measured wall clock offset between this node and one peer node.

    Attributes:
        peer_node: Name of the node the offset was measured against.
        offset_ns: The peer's wall clock minus this node's wall clock, in
            nanoseconds, at the instant of the measurement. A positive value
            means the peer's clock reads later than this node's clock.
        uncertainty_ns: Standard deviation of the per-round offset estimates
            within this one measurement. It is the spread of the raw rounds,
            not the standard error of the mean, because the raw spread is the
            conservative number and the rounds are not independent of the
            network path.
        rtt_min_ns: Smallest round trip seen in the measurement. The offset
            error from path asymmetry is bounded by about half of it, so a
            reader needs it to judge whether the offset is worth applying.
    """

    peer_node: str
    offset_ns: int
    uncertainty_ns: float
    rtt_min_ns: int


@dataclass
class Anchor:
    """One simultaneous reading of every clock chopper can see on one node.

    ``epoch_ns`` is the authoritative wall clock field and ``epoch_s`` is the
    same instant in seconds, kept because TRACE_FORMAT.md names that field. A
    float64 holding present-day seconds since the epoch resolves to about
    240 ns, which is coarser than the drift these anchors exist to measure, so
    the reader prefers the integer and only falls back to the float for
    anchors written before the integer existed.

    Attributes:
        node: Hostname of the node the reading was taken on.
        taken: Either "start" or "stop".
        epoch_ns: Wall clock nanoseconds since the Unix epoch (``time.time_ns``).
        monotonic_ns: The node's monotonic clock in nanoseconds.
        epoch_s: The same wall clock reading in seconds.
        gpu_clock_ns: The GPU trace clock in nanoseconds, or None when no GPU
            clock is reachable from this process.
        gpu_clock: Which GPU clock was read ("rocprofiler", "cupti" or None).
        peer_node: Name of the peer node a wall clock offset was measured
            against, or None. All four peer fields are None together on a
            single-node run, which is the common case and costs nothing.
        peer_offset_ns: The peer's wall clock minus this node's wall clock in
            nanoseconds, at this anchor.
        peer_offset_uncertainty_ns: Spread of the per-round estimates inside
            that one measurement.
        peer_rtt_min_ns: Smallest round trip seen during that measurement.
    """

    node: str
    taken: str
    epoch_ns: int
    monotonic_ns: int
    epoch_s: float = 0.0
    gpu_clock_ns: Optional[int] = None
    gpu_clock: Optional[str] = None
    peer_node: Optional[str] = None
    peer_offset_ns: Optional[int] = None
    peer_offset_uncertainty_ns: Optional[float] = None
    peer_rtt_min_ns: Optional[int] = None

    @property
    def peer_offset(self) -> Optional[PeerOffset]:
        """The peer measurement as a PeerOffset, or None when this anchor has none."""
        if self.peer_node is None or self.peer_offset_ns is None:
            return None
        return PeerOffset(
            peer_node=self.peer_node,
            offset_ns=int(self.peer_offset_ns),
            uncertainty_ns=float(self.peer_offset_uncertainty_ns or 0.0),
            rtt_min_ns=int(self.peer_rtt_min_ns or 0),
        )

    @property
    def offset_ns(self) -> int:
        """Epoch minus monotonic at this instant, in nanoseconds.

        Adding this to a monotonic timestamp taken at the same instant gives
        the epoch timestamp. The whole point of taking two anchors is that
        this number is not constant over a run.

        This stays an integer on purpose. Present-day epoch nanoseconds are
        around 1.8e18, which a float64 can only represent to 256 ns, and the
        drift being measured here is smaller than that.
        """
        return int(self.epoch_ns - self.monotonic_ns)

    @property
    def gpu_offset_ns(self) -> Optional[int]:
        """Epoch minus GPU clock at this instant, or None without a GPU clock."""
        if self.gpu_clock_ns is None:
            return None
        return int(self.epoch_ns - self.gpu_clock_ns)


def _read_gpu_clock(which: str = "auto") -> tuple[Optional[int], Optional[str]]:
    """Read the GPU trace clock. Returns (value_ns, clock_name), both None if absent.

    The GPU clock is the one the kernel traces and device counters are stamped
    with, so anchoring it here is what lets a kernel timestamp reach the common
    epoch timeline. Neither backend is present on a laptop, and that is a
    supported case: the anchor then carries only epoch and monotonic.
    """
    if which in ("auto", "rocprofiler"):
        try:
            from chopper.profile.telemetry import rocprofiler_clock
            if rocprofiler_clock.is_available():
                ts = rocprofiler_clock.get_timestamp()
                if ts is not None:
                    return int(ts), "rocprofiler"
        except Exception as e:  # a missing backend must never break collection
            logger.debug(f"rocprofiler clock unavailable for anchor: {e}")
    if which in ("auto", "cupti"):
        try:
            from chopper.profile.telemetry import cupti_clock
            if cupti_clock.is_available():
                ts = cupti_clock.get_timestamp()
                if ts is not None:
                    return int(ts), "cupti"
        except Exception as e:
            logger.debug(f"cupti clock unavailable for anchor: {e}")
    return None, None


def serve_peer_offset(bind_host: str = "0.0.0.0", port: int = PEER_PORT,
                      rounds: int = PEER_ROUNDS,
                      timeout_s: float = 120.0) -> int:
    """Reflect this node's wall clock so one peer can measure the offset to it.

    This is the passive half of the exchange. It accepts one connection, then
    answers ``rounds`` requests with ``time.time_ns()`` and closes. The node
    running this learns nothing; the node running ``measure_peer_offset``
    computes the offset and records it on its own anchor.

    ``TCP_NODELAY`` is set because Nagle's algorithm would hold a 64 byte reply
    back behind the previous acknowledgement and inflate the round trip well
    past the 60 microseconds the fabric actually costs.

    Args:
        bind_host: Address to listen on. The default accepts from any
            interface, which is what a compute node needs.
        port: TCP port, matched by the peer.
        rounds: How many requests to answer before closing.
        timeout_s: How long to wait for the peer to connect.

    Returns:
        The number of rounds actually answered, which is less than ``rounds``
        if the peer disconnected early.
    """
    served = 0
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((bind_host, port))
        srv.listen(1)
        srv.settimeout(timeout_s)
        conn, _addr = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        with conn:
            for _ in range(rounds):
                if not conn.recv(64):
                    break
                conn.sendall(str(time.time_ns()).encode().ljust(64))
                served += 1
    finally:
        srv.close()
    return served


def measure_peer_offset(peer_node: str, port: int = PEER_PORT,
                        rounds: int = PEER_ROUNDS,
                        connect_timeout_s: float = 120.0,
                        peer_host: Optional[str] = None) -> PeerOffset:
    """Measure this node's wall clock offset to one peer over a TCP socket.

    The peer must be running ``serve_peer_offset``. Each round sends a request
    at local time ``t0``, the peer stamps ``t1`` when it answers, and the reply
    lands at local time ``t3``. The NTP estimate is

        offset = ((t1 - t0) + (t2 - t3)) / 2

    with ``t2`` the peer's send time. This exchange takes a single timestamp on
    the peer, so ``t1`` equals ``t2`` in that formula. That biases the result
    by half of the peer's processing time between receiving and sending, in the
    direction of the peer's clock appearing later than it is. That gap is a
    ``recv`` return plus a ``time_ns`` call plus a ``sendall``, which is well
    under a microsecond, and the offsets this exists to measure are hundreds of
    microseconds, so the bias is small next to them. It is not zero, and it is
    the reason ``rtt_min_ns`` is reported alongside the offset.

    Only the lowest-delay decile of rounds is averaged. A round trip longer
    than the floor spent its extra time on one side of the path or the other,
    and the NTP formula assumes the two directions are symmetric, so the slow
    rounds are exactly the ones carrying asymmetry error.

    Args:
        peer_node: Name recorded on the anchor for the peer.
        port: TCP port, matched by the peer.
        rounds: How many exchanges to run. More rounds shrink the spread and
            cost about one round trip each.
        connect_timeout_s: How long to keep retrying the connection while the
            peer gets to its own call.
        peer_host: Address to connect to, when it differs from ``peer_node``.

    Returns:
        A PeerOffset carrying the offset, its uncertainty and the minimum round
        trip. An offset without its uncertainty is not usable, so the three
        always travel together.

    Raises:
        OSError: The peer was unreachable for the whole connect timeout.
        RuntimeError: The connection opened but produced no complete round.
    """
    host = peer_host or peer_node
    deadline = time.monotonic() + connect_timeout_s
    sock: Optional[socket.socket] = None
    last_error: Optional[OSError] = None
    while True:
        try:
            sock = socket.create_connection((host, port), timeout=10.0)
            break
        except OSError as e:
            last_error = e
            if time.monotonic() >= deadline:
                raise OSError(
                    f"could not reach peer {host}:{port} within "
                    f"{connect_timeout_s} s: {e}") from last_error
            time.sleep(0.5)

    offsets: list[float] = []
    delays: list[int] = []
    with sock:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        for _ in range(rounds):
            t0 = time.time_ns()
            sock.sendall(b"t".ljust(64))
            payload = sock.recv(64)
            t3 = time.time_ns()
            if not payload:
                break
            t1 = int(payload.strip())
            offsets.append(((t1 - t0) + (t1 - t3)) / 2.0)
            delays.append(t3 - t0)

    if not offsets:
        raise RuntimeError(
            f"peer {host}:{port} accepted the connection but returned no "
            f"complete round; the peer may have exited early")

    paired = sorted(zip(delays, offsets))
    best = [o for _d, o in paired[: max(1, len(paired) // 10)]]
    result = PeerOffset(
        peer_node=peer_node,
        offset_ns=int(round(statistics.fmean(best))),
        uncertainty_ns=float(statistics.pstdev(offsets)) if len(offsets) > 1 else 0.0,
        rtt_min_ns=int(min(delays)),
    )
    logger.info(
        f"peer clock offset to {peer_node}: {result.offset_ns} ns "
        f"(stdev {result.uncertainty_ns:.0f} ns, min rtt {result.rtt_min_ns} ns, "
        f"{len(offsets)} rounds)")
    return result


def take_anchor(taken: str, node: Optional[str] = None,
                gpu_clock: str = "auto", peer: Optional[PeerOffset] = None) -> Anchor:
    """Read every available clock as close together as the interpreter allows.

    Args:
        taken: "start" or "stop".
        node: Hostname override. Defaults to this machine's hostname.
        gpu_clock: "auto", "rocprofiler", "cupti", or "none".
        peer: A peer offset measured just before this call, or None. The
            measurement is taken by the caller rather than here because it
            costs a round trip per round and needs the peer to be serving at
            the same moment, neither of which belongs inside a clock read.

    Returns:
        One Anchor holding the simultaneous readings.
    """
    assert taken in (TAKEN_START, TAKEN_STOP), f"bad anchor kind: {taken!r}"
    gpu_ns, gpu_name = (None, None) if gpu_clock == "none" else _read_gpu_clock(gpu_clock)
    # Read the two software clocks last and adjacent, so the pair that carries
    # the offset is as close to simultaneous as possible.
    mono = time.monotonic_ns()
    epoch_ns = time.time_ns()
    return Anchor(
        node=node or socket.gethostname(),
        taken=taken,
        epoch_ns=epoch_ns,
        monotonic_ns=mono,
        epoch_s=epoch_ns / 1e9,
        gpu_clock_ns=gpu_ns,
        gpu_clock=gpu_name,
        peer_node=None if peer is None else peer.peer_node,
        peer_offset_ns=None if peer is None else peer.offset_ns,
        peer_offset_uncertainty_ns=None if peer is None else peer.uncertainty_ns,
        peer_rtt_min_ns=None if peer is None else peer.rtt_min_ns,
    )


def write_anchor(outdir: Union[str, Path], taken: str,
                 node: Optional[str] = None, gpu_clock: str = "auto",
                 peer: Optional[PeerOffset] = None) -> Path:
    """Take an anchor and append it to this node's anchor file.

    The file lives in the per-node output directory that collect.py creates
    under multi-node SLURM, so one file per node with two records in it.

    Args:
        outdir: Per-node output directory.
        taken: "start" or "stop".
        node: Hostname override.
        gpu_clock: "auto", "rocprofiler", "cupti", or "none".
        peer: A peer offset from ``measure_peer_offset``, or None. Leaving it
            None writes exactly the anchor this function wrote before peer
            measurement existed.

    Returns:
        Path of the anchor file that was written.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / ANCHOR_FILENAME

    anchor = take_anchor(taken, node=node, gpu_clock=gpu_clock, peer=peer)

    doc: dict[str, Any] = {"schema": ANCHOR_SCHEMA, "node": anchor.node, "anchors": []}
    if path.is_file():
        try:
            with open(path) as f:
                existing = json.load(f)
            if isinstance(existing, dict) and "anchors" in existing:
                doc = existing
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"could not read {path}, starting a fresh anchor file: {e}")

    record = asdict(anchor)
    if anchor.peer_offset is None:
        # A run with no peer measurement writes exactly the record it wrote
        # before peer measurement existed, rather than four null keys. The
        # reader treats a missing key and a null key the same way, so this is
        # only about keeping single-node anchor files unchanged.
        for key in ("peer_node", "peer_offset_ns",
                    "peer_offset_uncertainty_ns", "peer_rtt_min_ns"):
            record.pop(key, None)

    doc["anchors"] = [a for a in doc.get("anchors", []) if a.get("taken") != taken]
    doc["anchors"].append(record)
    doc["node"] = anchor.node
    doc["schema"] = ANCHOR_SCHEMA

    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    logger.info(f"clock anchor ({taken}) for {anchor.node} -> {path}")
    return path


def _anchor_from_dict(d: dict[str, Any], node: str) -> Anchor:
    # epoch_ns is authoritative. Anchors written before it existed carry only
    # the float seconds, which costs about 240 ns of resolution.
    if d.get("epoch_ns") is not None:
        epoch_ns = int(d["epoch_ns"])
    else:
        epoch_ns = int(round(float(d["epoch_s"]) * 1e9))
    return Anchor(
        node=str(d.get("node", node)),
        taken=str(d["taken"]),
        epoch_ns=epoch_ns,
        monotonic_ns=int(d["monotonic_ns"]),
        epoch_s=float(d.get("epoch_s", epoch_ns / 1e9)),
        gpu_clock_ns=None if d.get("gpu_clock_ns") is None else int(d["gpu_clock_ns"]),
        gpu_clock=d.get("gpu_clock"),
        # An anchor file written before peer measurement existed has none of
        # these keys, so every one of them defaults to None rather than
        # raising. That is what keeps old runs loadable.
        peer_node=None if d.get("peer_node") is None else str(d["peer_node"]),
        peer_offset_ns=(None if d.get("peer_offset_ns") is None
                        else int(d["peer_offset_ns"])),
        peer_offset_uncertainty_ns=(None if d.get("peer_offset_uncertainty_ns") is None
                                    else float(d["peer_offset_uncertainty_ns"])),
        peer_rtt_min_ns=(None if d.get("peer_rtt_min_ns") is None
                         else int(d["peer_rtt_min_ns"])),
    )


def _same_node(a: str, b: str) -> bool:
    """True when two node names refer to the same node.

    SLURM hands out short names in ``SLURM_JOB_NODELIST`` while
    ``socket.gethostname`` often returns the fully qualified name, so a run on
    the HPC Fund records the peer as ``k004-004`` and the node itself as
    ``k004-004.hpcfund``. Comparing the part before the first dot as well as
    the whole string keeps those two from looking like different nodes.
    """
    if a == b:
        return True
    return a.split(".", 1)[0] == b.split(".", 1)[0]


class AnchorSet:
    """Every node's anchor pair, and the conversion onto one epoch timeline.

    For node ``n`` with a start anchor at monotonic ``m0`` and a stop anchor at
    ``m1``, the epoch-minus-monotonic offset is ``o0`` and ``o1``. The offset at
    an arbitrary monotonic time ``m`` is interpolated linearly:

        offset(m) = o0 + (o1 - o0) * (m - m0) / (m1 - m0)

    and the epoch timestamp is ``m + offset(m)``. Extrapolation outside the
    anchor window uses the same line, which is the honest thing to do for a
    sample taken a few milliseconds before the start anchor.

    With a single anchor on a node the offset is constant and the drift over
    the run is unknown, not zero. ``drift_table`` marks that case.
    """

    def __init__(self, anchors: Iterable[Anchor]):
        by_node: dict[str, dict[str, Anchor]] = {}
        for a in anchors:
            by_node.setdefault(a.node, {})[a.taken] = a
        assert by_node, "AnchorSet needs at least one anchor"
        self._by_node = by_node

    # -- construction ----------------------------------------------------

    @property
    def nodes(self) -> list[str]:
        """Node names present, sorted."""
        return sorted(self._by_node)

    def pair(self, node: str) -> tuple[Anchor, Optional[Anchor]]:
        """Return (start_anchor, stop_anchor) for a node. Stop may be None."""
        assert node in self._by_node, (
            f"no clock anchor for node {node!r}; have {self.nodes}")
        got = self._by_node[node]
        start = got.get(TAKEN_START) or got.get(TAKEN_STOP)
        assert start is not None, f"node {node!r} has an empty anchor record"
        stop = got.get(TAKEN_STOP) if got.get(TAKEN_START) is not None else None
        return start, stop

    # -- monotonic conversion --------------------------------------------

    def _line(self, node: str) -> tuple[int, int, float]:
        """Return (m0, o0, slope) for the node's offset line, slope in ns per ns.

        m0 and o0 stay integers so the arithmetic never rounds at the 1.8e18
        magnitude of real timestamps. Only the slope, which multiplies the
        small elapsed-time difference, is a float.
        """
        start, stop = self.pair(node)
        m0 = int(start.monotonic_ns)
        o0 = start.offset_ns
        if stop is None or stop.monotonic_ns == start.monotonic_ns:
            return m0, o0, 0.0
        slope = (stop.offset_ns - o0) / float(int(stop.monotonic_ns) - m0)
        return m0, o0, slope

    @staticmethod
    def _elapsed(monotonic_ns: Any, m0: int) -> Any:
        """Nanoseconds since the start anchor, as int64, without losing bits."""
        m = np.asarray(monotonic_ns)
        if m.dtype.kind == "f":
            # A float64 carrying 1.8e18 already lost the low bits; rounding
            # here at least keeps the subtraction exact from that point on.
            m = np.rint(m).astype("int64")
        else:
            m = m.astype("int64")
        return m - np.int64(m0)

    def offset_ns(self, node: str, monotonic_ns: Any) -> Any:
        """Epoch-minus-monotonic offset for this node at these monotonic times."""
        m0, o0, slope = self._line(node)
        d = self._elapsed(monotonic_ns, m0)
        out = np.int64(o0) + np.rint(slope * d).astype("int64")
        return out if out.ndim else int(out)

    def drift_correction_ns(self, node: str, monotonic_ns: Any) -> Any:
        """How far interpolation moved a timestamp away from the start-anchor offset.

        This is the number a reader should look at before trusting a cross-node
        comparison. It is zero at the start anchor and equals the node's total
        measured drift at the stop anchor.
        """
        m0, _o0, slope = self._line(node)
        d = self._elapsed(monotonic_ns, m0)
        out = np.rint(slope * d).astype("int64")
        return out if out.ndim else int(out)

    def to_epoch_ns(self, node: str, monotonic_ns: Any) -> Any:
        """Convert this node's monotonic timestamps to the common epoch timeline.

        Returns int64 nanoseconds. A float64 cannot hold a present-day epoch
        nanosecond value to better than 256 ns, which would be coarser than
        the drift this conversion is correcting.
        """
        m0, o0, slope = self._line(node)
        d = self._elapsed(monotonic_ns, m0)
        out = np.int64(m0) + d + np.int64(o0) + np.rint(slope * d).astype("int64")
        return out if out.ndim else int(out)

    # -- GPU clock conversion --------------------------------------------

    def has_gpu_clock(self, node: str) -> bool:
        """True when this node anchored a GPU clock reading."""
        start, stop = self.pair(node)
        return start.gpu_clock_ns is not None

    def _gpu_line(self, node: str) -> tuple[int, int, float]:
        """Return (g0, o0, slope) for the node's GPU-clock offset line."""
        start, stop = self.pair(node)
        assert start.gpu_clock_ns is not None, (
            f"node {node!r} has no GPU clock anchor; "
            f"collect on a machine where rocprofiler or CUPTI is reachable")
        g0 = int(start.gpu_clock_ns)
        o0 = start.gpu_offset_ns
        assert o0 is not None
        slope = 0.0
        if stop is not None and stop.gpu_clock_ns is not None:
            g1 = int(stop.gpu_clock_ns)
            o1 = stop.gpu_offset_ns
            if o1 is not None and g1 != g0:
                slope = (o1 - o0) / float(g1 - g0)
        return g0, o0, slope

    def gpu_to_epoch_ns(self, node: str, gpu_clock_ns: Any) -> Any:
        """Convert this node's GPU trace clock timestamps to the epoch timeline.

        Kernel traces and device counters are stamped with the GPU clock, so
        this is the path that puts a kernel on the same axis as a power sample
        from another node.
        """
        g0, o0, slope = self._gpu_line(node)
        d = self._elapsed(gpu_clock_ns, g0)
        out = np.int64(g0) + d + np.int64(o0) + np.rint(slope * d).astype("int64")
        return out if out.ndim else int(out)

    def gpu_drift_correction_ns(self, node: str, gpu_clock_ns: Any) -> Any:
        """Size of the interpolated GPU-clock correction, zero at the start anchor."""
        g0, _o0, slope = self._gpu_line(node)
        d = self._elapsed(gpu_clock_ns, g0)
        out = np.rint(slope * d).astype("int64")
        return out if out.ndim else int(out)

    # -- cross-node alignment ---------------------------------------------

    def has_peer_offset(self, node: str) -> bool:
        """True when this node measured a wall clock offset to some peer."""
        start, _stop = self.pair(node)
        return start.peer_offset is not None

    def peer_node(self, node: str) -> Optional[str]:
        """Name of the peer this node measured against, or None."""
        start, _stop = self.pair(node)
        return start.peer_node

    def _peer_line(self, node: str) -> tuple[int, int, float]:
        """Return (e0, p0, slope) for the node's peer-offset line.

        The line is parameterised by this node's own epoch timestamp rather
        than its monotonic clock, so that a correction measured on one node can
        be evaluated at a timestamp that has already been put on the epoch
        timeline. The two parameterisations differ by the node's own
        monotonic-to-epoch drift, which the measured runs put below 1 ppm, so
        evaluating the line in the epoch domain changes the interpolated
        correction by well under a nanosecond over a run of hours.
        """
        start, stop = self.pair(node)
        p_start = start.peer_offset
        if p_start is None:
            raise CrossNodeOffsetUnavailable(
                f"node {node!r} has no peer clock offset on its start anchor")
        e0 = int(start.epoch_ns)
        p0 = int(p_start.offset_ns)
        if stop is None:
            return e0, p0, 0.0
        p_stop = stop.peer_offset
        if p_stop is None or int(stop.epoch_ns) == e0:
            return e0, p0, 0.0
        if not _same_node(p_stop.peer_node, p_start.peer_node):
            raise CrossNodeOffsetUnavailable(
                f"node {node!r} measured {p_start.peer_node!r} at start but "
                f"{p_stop.peer_node!r} at stop; the two cannot be interpolated")
        slope = (int(p_stop.offset_ns) - p0) / float(int(stop.epoch_ns) - e0)
        return e0, p0, slope

    def peer_offset_at_ns(self, node: str, epoch_ns: Any) -> Any:
        """This node's measured offset to its peer, interpolated to these instants.

        The value is the peer's wall clock minus this node's wall clock, so
        adding it to one of this node's epoch timestamps expresses that
        timestamp on the peer's wall clock.
        """
        e0, p0, slope = self._peer_line(node)
        d = self._elapsed(epoch_ns, e0)
        out = np.int64(p0) + np.rint(slope * d).astype("int64")
        return out if out.ndim else int(out)

    def reference_correction_ns(self, node: str, epoch_ns: Any,
                                reference: str) -> Any:
        """Nanoseconds to add to this node's epoch timestamps to reach the reference node.

        The correction comes from whichever of the two nodes measured the
        other. If ``node`` measured ``reference``, the interpolated offset is
        added as measured. If ``reference`` measured ``node``, the same line is
        used with the sign flipped, evaluated at the same instant, which is
        valid because the two nodes' epoch timelines differ by the few hundred
        microseconds this correction is about to remove and the line changes by
        picoseconds over that gap.

        Only a direct measurement between the two nodes is used. A chain
        through a third node is not followed, because each hop adds its own
        asymmetry bias and this code has no measurement of how large that
        becomes.

        Raises:
            CrossNodeOffsetUnavailable: No measurement links the two nodes.
                This is an exception rather than a zero-filled column on
                purpose. A zero correction and a missing correction produce
                identical numbers in the output, so a caller that ignored a
                flag column would ship a cross-node comparison carrying the
                full unmeasured offset, which the HPC Fund measurement puts at
                about 400 microseconds and growing at 0.30 ppm. Refusing to
                answer is the only form of the answer a caller cannot mistake
                for a corrected one. Call ``has_peer_offset`` or read
                ``peer_offset_table`` first when a run may not have the
                measurement.
        """
        shape = np.asarray(epoch_ns)
        if _same_node(node, reference):
            out = np.zeros(shape.shape, dtype="int64")
            return out if out.ndim else int(out)

        if self.has_peer_offset(node):
            peer = self.peer_node(node)
            if peer is not None and _same_node(peer, reference):
                return self.peer_offset_at_ns(node, epoch_ns)

        if reference in self._by_node and self.has_peer_offset(reference):
            peer = self.peer_node(reference)
            if peer is not None and _same_node(peer, node):
                out = -np.asarray(self.peer_offset_at_ns(reference, epoch_ns))
                out = out.astype("int64")
                return out if out.ndim else int(out)

        raise CrossNodeOffsetUnavailable(
            f"no measured wall clock offset links node {node!r} to reference "
            f"{reference!r}; this run has peer measurements "
            f"{ {n: self.peer_node(n) for n in self.nodes if self.has_peer_offset(n)} }. "
            f"Re-collect with measure_peer_offset on one node of each pair, or "
            f"compare within a single node.")

    def to_reference_epoch_ns(self, node: str, ts: Any, reference: str,
                              domain: str = "monotonic") -> Any:
        """Put this node's timestamps on one reference node's wall clock.

        ``to_epoch_ns`` already puts a node on that node's own wall clock. This
        goes one step further and removes the disagreement between that wall
        clock and the reference node's, which is the part per-node anchors
        cannot see.

        Args:
            node: Node the timestamps were collected on.
            ts: Native timestamps, monotonic or GPU clock.
            reference: Node whose wall clock becomes the common timeline.
            domain: "monotonic" or "gpu".

        Returns:
            int64 nanoseconds on the reference node's wall clock.

        Raises:
            CrossNodeOffsetUnavailable: See ``reference_correction_ns``.
        """
        assert domain in ("monotonic", "gpu"), f"unknown clock domain {domain!r}"
        epoch = (self.gpu_to_epoch_ns(node, ts) if domain == "gpu"
                 else self.to_epoch_ns(node, ts))
        out = np.asarray(epoch).astype("int64") + np.asarray(
            self.reference_correction_ns(node, epoch, reference)).astype("int64")
        return out if out.ndim else int(out)

    # -- reporting --------------------------------------------------------

    def drift_table(self) -> pd.DataFrame:
        """Per-node measured drift over the collection window.

        Columns:
            node: Node name.
            span_s: Seconds between the node's start and stop anchor.
            start_offset_ns: Epoch minus monotonic at the start anchor.
            end_offset_ns: The same at the stop anchor.
            drift_ns: End offset minus start offset. This is the amount the
                node's wall clock moved relative to its monotonic clock during
                the run, and therefore the size of the correction interpolation
                applies by the end of the run.
            drift_ppm: Drift expressed as parts per million of the span.
            gpu_drift_ns: The same quantity measured on the GPU clock, or NaN
                when no GPU clock was anchored.
            paired: False when the node has only one anchor, in which case the
                drift is unknown rather than zero.
        """
        rows = []
        for node in self.nodes:
            start, stop = self.pair(node)
            span_ns = 0
            drift: Optional[int] = None
            gpu_drift: Optional[int] = None
            end_offset: Optional[int] = None
            if stop is not None:
                span_ns = int(stop.monotonic_ns - start.monotonic_ns)
                end_offset = stop.offset_ns
                drift = int(stop.offset_ns - start.offset_ns)
                if start.gpu_offset_ns is not None and stop.gpu_offset_ns is not None:
                    gpu_drift = int(stop.gpu_offset_ns - start.gpu_offset_ns)
            rows.append({
                "node": node,
                "span_s": span_ns / 1e9 if stop is not None else float("nan"),
                "start_offset_ns": start.offset_ns,
                "end_offset_ns": end_offset,
                "drift_ns": drift,
                "drift_ppm": (drift / span_ns * 1e6
                              if drift is not None and span_ns else float("nan")),
                "gpu_drift_ns": gpu_drift,
                "paired": stop is not None,
            })
        return pd.DataFrame(rows)

    def peer_offset_table(self) -> pd.DataFrame:
        """Per-node measured wall clock offset to its peer, and how that offset drifted.

        This is the cross-node counterpart of ``drift_table``. ``drift_table``
        reports how far one node's wall clock moved against its own monotonic
        clock; this reports how far one node's wall clock sits from another
        node's, which is a separate error and the larger one on the hardware
        measured so far.

        Columns:
            node: Node that took the measurement.
            peer_node: Node it measured against.
            span_s: Seconds between the two measurements.
            start_offset_ns: Peer wall clock minus this node's wall clock at
                the start anchor.
            end_offset_ns: The same at the stop anchor.
            drift_ns: End minus start. This is how much the cross-node offset
                moved during the run, and therefore the error a reader would
                carry by treating a single measurement as valid throughout.
            drift_ppm: That drift as parts per million of the span, which is
                the relative skew between the two nodes' wall clocks.
            start_uncertainty_ns: Spread within the start measurement.
            end_uncertainty_ns: Spread within the stop measurement.
            rtt_min_ns: Smallest round trip across both measurements.
            paired: False when only one measurement exists, in which case the
                drift is unknown rather than zero.

        An empty frame means the run has no peer measurement at all, and a
        cross-node comparison in that run carries an unmeasured offset.
        """
        rows = []
        for node in self.nodes:
            start, stop = self.pair(node)
            p_start = start.peer_offset
            if p_start is None:
                continue
            p_stop = stop.peer_offset if stop is not None else None
            if p_stop is not None and not _same_node(p_stop.peer_node, p_start.peer_node):
                logger.warning(
                    f"node {node} measured {p_start.peer_node} at start and "
                    f"{p_stop.peer_node} at stop; reporting the start only")
                p_stop = None
            span_ns = (int(stop.epoch_ns - start.epoch_ns)
                       if stop is not None and p_stop is not None else 0)
            drift = (int(p_stop.offset_ns - p_start.offset_ns)
                     if p_stop is not None else None)
            rows.append({
                "node": node,
                "peer_node": p_start.peer_node,
                "span_s": span_ns / 1e9 if p_stop is not None else float("nan"),
                "start_offset_ns": p_start.offset_ns,
                "end_offset_ns": None if p_stop is None else p_stop.offset_ns,
                "drift_ns": drift,
                "drift_ppm": (drift / span_ns * 1e6
                              if drift is not None and span_ns else float("nan")),
                "start_uncertainty_ns": p_start.uncertainty_ns,
                "end_uncertainty_ns": None if p_stop is None else p_stop.uncertainty_ns,
                "rtt_min_ns": (p_start.rtt_min_ns if p_stop is None
                               else min(p_start.rtt_min_ns, p_stop.rtt_min_ns)),
                "paired": p_stop is not None,
            })
        return pd.DataFrame(rows)

    def pairwise_offsets_ns(self) -> pd.DataFrame:
        """Wall clock disagreement between every pair of nodes at the start anchor.

        Two nodes under NTP typically disagree by some milliseconds. This is
        the residual a cross-node comparison carries even after the monotonic
        origins are removed, because epoch conversion can only be as good as
        each node's own wall clock.
        """
        rows = []
        nodes = self.nodes
        for i, a in enumerate(nodes):
            for b in nodes[i + 1:]:
                sa, _ = self.pair(a)
                sb, _ = self.pair(b)
                rows.append({
                    "node_a": a,
                    "node_b": b,
                    "start_epoch_delta_ns": int(sa.epoch_ns - sb.epoch_ns),
                })
        return pd.DataFrame(rows)

    # -- DataFrame helper -------------------------------------------------

    def attach_epoch(self, df: pd.DataFrame, ts_col: str = "ts",
                     node_col: str = "node", out_col: str = "epoch_ns",
                     domain: str = "monotonic") -> pd.DataFrame:
        """Add a common-timeline column to a collector frame, plus the correction size.

        Args:
            df: Any collector frame carrying a timestamp and a node column.
            ts_col: Name of the native timestamp column.
            node_col: Name of the node column written by the collectors.
            out_col: Name of the epoch column to add.
            domain: "monotonic" for telemetry frames, "gpu" for kernel traces
                and device counter samples.

        Returns:
            A copy of the frame with ``out_col`` and ``<out_col>_drift_ns``
            added. The drift column is the magnitude of the interpolated
            correction for that row, exposed rather than folded in silently.
        """
        assert ts_col in df.columns, f"frame has no {ts_col!r} column"
        out = df.copy()
        if node_col not in out.columns:
            assert len(self.nodes) == 1, (
                f"frame has no {node_col!r} column but the anchor set covers "
                f"{len(self.nodes)} nodes {self.nodes}; a timestamp cannot be "
                f"placed on the common timeline without knowing its node")
            out[node_col] = self.nodes[0]

        assert domain in ("monotonic", "gpu"), f"unknown clock domain {domain!r}"
        convert = self.gpu_to_epoch_ns if domain == "gpu" else self.to_epoch_ns
        correct = (self.gpu_drift_correction_ns if domain == "gpu"
                   else self.drift_correction_ns)

        epoch = np.zeros(len(out), dtype="int64")
        corr = np.zeros(len(out), dtype="int64")
        for node, idx in out.groupby(node_col).groups.items():
            pos = out.index.get_indexer(idx)
            ts = out.loc[idx, ts_col].to_numpy()
            epoch[pos] = convert(str(node), ts)
            corr[pos] = correct(str(node), ts)
        out[out_col] = epoch
        out[f"{out_col}_drift_ns"] = corr
        return out

    def attach_reference_epoch(self, df: pd.DataFrame, reference: str,
                               ts_col: str = "ts", node_col: str = "node",
                               out_col: str = "ref_epoch_ns",
                               domain: str = "monotonic") -> pd.DataFrame:
        """Place every node's rows on one reference node's wall clock.

        ``attach_epoch`` puts each row on its own node's wall clock, which is
        as far as per-node anchors can go. This applies the measured cross-node
        offset on top, so rows from different nodes become directly
        comparable.

        Args:
            df: Any collector frame carrying a timestamp and a node column.
            reference: Node whose wall clock becomes the common timeline. Its
                own rows are returned unchanged, so choosing the node that
                carries the events of interest keeps those exact.
            ts_col: Name of the native timestamp column.
            node_col: Name of the node column written by the collectors.
            out_col: Name of the aligned timestamp column to add.
            domain: "monotonic" for telemetry frames, "gpu" for kernel traces
                and device counter samples.

        Returns:
            A copy of the frame with three columns added: ``out_col`` holding
            the aligned timestamp, ``<out_col>_drift_ns`` holding the node's
            own monotonic-to-epoch interpolation as ``attach_epoch`` reports
            it, and ``<out_col>_crossnode_ns`` holding the cross-node
            correction on its own. The cross-node term is separate because it
            is a different measurement with a different uncertainty, and a
            reader comparing two nodes needs to see how much of the answer came
            from it.

        Raises:
            CrossNodeOffsetUnavailable: Some node in the frame has no measured
                offset to the reference node. The whole call fails rather than
                returning a frame where some rows are aligned and others only
                look aligned.
        """
        assert ts_col in df.columns, f"frame has no {ts_col!r} column"
        assert domain in ("monotonic", "gpu"), f"unknown clock domain {domain!r}"
        out = df.copy()
        if node_col not in out.columns:
            assert len(self.nodes) == 1, (
                f"frame has no {node_col!r} column but the anchor set covers "
                f"{len(self.nodes)} nodes {self.nodes}; a timestamp cannot be "
                f"placed on the common timeline without knowing its node")
            out[node_col] = self.nodes[0]

        assert reference in self._by_node, (
            f"reference node {reference!r} has no anchor; have {self.nodes}")

        convert = self.gpu_to_epoch_ns if domain == "gpu" else self.to_epoch_ns
        correct = (self.gpu_drift_correction_ns if domain == "gpu"
                   else self.drift_correction_ns)

        aligned = np.zeros(len(out), dtype="int64")
        drift = np.zeros(len(out), dtype="int64")
        cross = np.zeros(len(out), dtype="int64")
        for node, idx in out.groupby(node_col).groups.items():
            pos = out.index.get_indexer(idx)
            ts = out.loc[idx, ts_col].to_numpy()
            epoch = np.asarray(convert(str(node), ts)).astype("int64")
            shift = np.asarray(
                self.reference_correction_ns(str(node), epoch, reference)
            ).astype("int64")
            aligned[pos] = epoch + shift
            drift[pos] = correct(str(node), ts)
            cross[pos] = shift
        out[out_col] = aligned
        out[f"{out_col}_drift_ns"] = drift
        out[f"{out_col}_crossnode_ns"] = cross
        return out


def load_anchors(paths: Union[str, Path, Iterable[Union[str, Path]]]) -> AnchorSet:
    """Load every node's anchor file under the given path or paths.

    Accepts a run root directory (every ``clock_anchors.json`` beneath it is
    read, which is exactly the per-node subdirectory layout collect.py writes),
    a single anchor file, or an iterable of either.

    Returns:
        An AnchorSet covering every node found.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]

    files: list[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            direct = p / ANCHOR_FILENAME
            if direct.is_file():
                files.append(direct)
            files.extend(sorted(p.glob(f"*/{ANCHOR_FILENAME}")))
            files.extend(sorted(p.glob(f"*/*/{ANCHOR_FILENAME}")))
        elif p.is_file():
            files.append(p)
        else:
            raise FileNotFoundError(f"no clock anchors at {p}")

    seen: set[str] = set()
    anchors: list[Anchor] = []
    for f in files:
        key = str(f.resolve())
        if key in seen:
            continue
        seen.add(key)
        with open(f) as fh:
            doc = json.load(fh)
        node = str(doc.get("node", f.parent.name))
        for d in doc.get("anchors", []):
            anchors.append(_anchor_from_dict(d, node))

    if not anchors:
        raise FileNotFoundError(
            f"found no {ANCHOR_FILENAME} under {[str(p) for p in paths]}; "
            f"re-collect with a chopper that writes clock anchors")
    logger.info(f"loaded {len(anchors)} clock anchors from {len(files)} file(s)")
    return AnchorSet(anchors)


def _cli() -> int:
    from argparse import ArgumentParser
    parser = ArgumentParser(description=(
        "Take a clock anchor, or report measured drift across a run's nodes."))
    parser.add_argument("path", help="output directory (write) or run root (report)")
    parser.add_argument("--take", choices=[TAKEN_START, TAKEN_STOP],
                        help="take an anchor and append it to path/clock_anchors.json")
    parser.add_argument("--gpu-clock", choices=["auto", "rocprofiler", "cupti", "none"],
                        default="auto")
    parser.add_argument("--peer", help=(
        "measure the wall clock offset to this peer node before taking the "
        "anchor; the peer must be running --serve-peer at the same time"))
    parser.add_argument("--serve-peer", action="store_true", help=(
        "answer one peer's offset measurement and exit, taking no anchor"))
    parser.add_argument("--peer-port", type=int, default=PEER_PORT)
    parser.add_argument("--peer-rounds", type=int, default=PEER_ROUNDS)
    args = parser.parse_args()

    if args.serve_peer:
        served = serve_peer_offset(port=args.peer_port, rounds=args.peer_rounds)
        print(f"served {served} peer offset rounds")
        return 0

    if args.take:
        peer = None
        if args.peer:
            peer = measure_peer_offset(args.peer, port=args.peer_port,
                                       rounds=args.peer_rounds)
        write_anchor(args.path, args.take, gpu_clock=args.gpu_clock, peer=peer)
        return 0

    anchors = load_anchors(args.path)
    pd.set_option("display.width", 200)
    print(anchors.drift_table().to_string(index=False))
    if len(anchors.nodes) > 1:
        print()
        print(anchors.pairwise_offsets_ns().to_string(index=False))
        peers = anchors.peer_offset_table()
        print()
        if peers.empty:
            print("no measured cross-node clock offset in this run; a "
                  "comparison across these nodes carries an unmeasured "
                  "wall clock difference")
        else:
            print(peers.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
