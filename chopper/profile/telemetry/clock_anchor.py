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

``attach_epoch`` writes the size of the correction into its own column, so a
reader can see how much interpolation moved each row rather than trusting a
silent adjustment.
"""

import json
import socket
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
    """

    node: str
    taken: str
    epoch_ns: int
    monotonic_ns: int
    epoch_s: float = 0.0
    gpu_clock_ns: Optional[int] = None
    gpu_clock: Optional[str] = None

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


def take_anchor(taken: str, node: Optional[str] = None,
                gpu_clock: str = "auto") -> Anchor:
    """Read every available clock as close together as the interpreter allows.

    Args:
        taken: "start" or "stop".
        node: Hostname override. Defaults to this machine's hostname.
        gpu_clock: "auto", "rocprofiler", "cupti", or "none".

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
    )


def write_anchor(outdir: Union[str, Path], taken: str,
                 node: Optional[str] = None, gpu_clock: str = "auto") -> Path:
    """Take an anchor and append it to this node's anchor file.

    The file lives in the per-node output directory that collect.py creates
    under multi-node SLURM, so one file per node with two records in it.

    Returns:
        Path of the anchor file that was written.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / ANCHOR_FILENAME

    anchor = take_anchor(taken, node=node, gpu_clock=gpu_clock)

    doc: dict[str, Any] = {"schema": ANCHOR_SCHEMA, "node": anchor.node, "anchors": []}
    if path.is_file():
        try:
            with open(path) as f:
                existing = json.load(f)
            if isinstance(existing, dict) and "anchors" in existing:
                doc = existing
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"could not read {path}, starting a fresh anchor file: {e}")

    doc["anchors"] = [a for a in doc.get("anchors", []) if a.get("taken") != taken]
    doc["anchors"].append(asdict(anchor))
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
    )


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
    args = parser.parse_args()

    if args.take:
        write_anchor(args.path, args.take, gpu_clock=args.gpu_clock)
        return 0

    anchors = load_anchors(args.path)
    pd.set_option("display.width", 200)
    print(anchors.drift_table().to_string(index=False))
    if len(anchors.nodes) > 1:
        print()
        print(anchors.pairwise_offsets_ns().to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
