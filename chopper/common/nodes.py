"""The node axis: shared helpers for keeping two nodes' rows apart.

Every collector stamps a ``node`` column (commit c42e95c). This module holds
the small amount of logic that decides when that column has to take part in a
join or a groupby, and the guard that turns a silently wrong cross-node join
into an exception.

Why a guard and not a warning: the counter join keys on kernel name plus an
instance index within a device. On one node that pair is unique. Across nodes
the same kernel name reaches the same instance index on every node, so the
join has several equally good right-hand rows, pandas takes one of them, and
the result is a full frame of plausible numbers attached to the wrong host.
A crash is recoverable; that frame is not.

Why the node key is conditional: a run that touched one node keys and groups
exactly as it did before this module existed, so old single-node outputs stay
comparable and old analysis scripts keep working.
"""

from typing import Iterable, Optional, Sequence

import pandas as pd

#: Column name every collector writes its hostname into.
NODE_COL = "node"


def nodes_in(df: pd.DataFrame) -> list[str]:
    """Sorted distinct node labels in a frame, or [] when it carries no node column."""
    if NODE_COL not in df.columns:
        return []
    return sorted(str(n) for n in df[NODE_COL].dropna().unique())


def is_multi_node(*frames: pd.DataFrame) -> bool:
    """True when the frames together name more than one node."""
    seen: set[str] = set()
    for df in frames:
        seen.update(nodes_in(df))
    return len(seen) > 1


def device_keys(*frames: pd.DataFrame, base: Sequence[str] = ("gpu",)) -> list[str]:
    """Keys that identify one physical device across the given frames.

    Returns ``['node', 'gpu']`` when the data spans more than one node and
    ``['gpu']`` otherwise, so single-node behaviour is unchanged.
    """
    if is_multi_node(*frames):
        return [NODE_COL] + list(base)
    return list(base)


def require_node_axis(df: pd.DataFrame, what: str,
                      other_nodes: Optional[Iterable[str]] = None) -> None:
    """Raise when a frame that spans several nodes cannot say which node a row is from.

    Args:
        df: The frame about to be joined or grouped.
        what: Human-readable name of the frame, used in the message.
        other_nodes: Node labels known from the other side of the join. When
            that side names more than one node, this frame needs a node column
            even if its own rows happen to look uniform.

    Raises:
        ValueError: When the node axis is missing and more than one node's
            data is in play.
    """
    mine = nodes_in(df)
    theirs = sorted(set(str(n) for n in (other_nodes or [])))
    if NODE_COL in df.columns and df[NODE_COL].isna().any() and (len(mine) > 1 or len(theirs) > 1):
        raise ValueError(
            f"{what} has {int(df[NODE_COL].isna().sum())} rows with no node label "
            f"while {len(set(mine) | set(theirs))} nodes are present; "
            f"a join on these rows would attach counters to the wrong host")
    if NODE_COL not in df.columns and len(theirs) > 1:
        raise ValueError(
            f"{what} has no {NODE_COL!r} column but the run spans {len(theirs)} "
            f"nodes {theirs}; re-collect with a chopper that stamps the node "
            f"column, or pass the node explicitly, because keying only on "
            f"kernel name and instance index would mix the nodes' rows")


def resolve_node_from_path(path: str, known_nodes: Iterable[str]) -> Optional[str]:
    """Infer which node a file came from by matching a known hostname in its path.

    ``collect.py`` puts each node's output under a subdirectory named by that
    node's hostname when SLURM reports more than one node, so a known node name
    appearing as a path component identifies the file. Short hostname and FQDN
    are treated as the same node.

    Returns:
        The node label, or None when nothing matched or several matched.
    """
    from pathlib import Path

    parts = [p for p in Path(path).resolve().parts]
    short = {p.split(".")[0]: p for p in parts}
    hits = []
    for node in known_nodes:
        node = str(node)
        if node in parts or node.split(".")[0] in short:
            hits.append(node)
    hits = sorted(set(hits))
    return hits[0] if len(hits) == 1 else None
