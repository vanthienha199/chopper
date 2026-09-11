"""Data loading and preprocessing utilities for trace analysis."""

import pandas as pd
from typing import Optional, List, Dict

from chopper.common.annotations import (
    no_overlap_mask,
    assign_operator_type,
    assign_chunks as do_assign_chunks,
    fix_names as do_fix_names,
)
from chopper.common.cache import load_pickle
from chopper.common.nodes import NODE_COL, device_keys, is_multi_node


def select_iters(df: pd.DataFrame, iters: List) -> pd.DataFrame:
    """Select specific iterations from trace data.

    Args:
        df: DataFrame containing trace data with 'iteration' column
        iters: List of iteration indices to select

    Returns:
        Filtered DataFrame containing only selected iterations
    """
    u_iters = df.loc[~df['iteration'].isna(), 'iteration'].unique()
    iters = [u_iters[i] for i in iters]
    return df[df['iteration'].isin(iters)]


def get_df(
    fn: str,
    iter_idxs: Optional[List] = None,
    assign_chunks: bool = False,
    assign_optype: bool = False,
    remove_nan_chunks: bool = False,
    remove_overlap: bool = False,
    fix_names: bool = False,
    group_arr: Optional[List] = None,
    group_map: Optional[Dict[str, List[str]]] = None,
    sort_value: Optional[str] = None,
) -> pd.DataFrame:
    """Load and preprocess trace data with optional transformations.

    Main entry point for loading trace files with flexible preprocessing options
    including filtering, grouping, and aggregation.

    Args:
        fn: Path to trace pickle file
        iter_idxs: Optional list of iteration indices to select
        assign_chunks: If True, assign training phase chunks (fwd/bwd/opt)
        assign_optype: If True, categorize operators by type (GEMM/FA/Vec)
        remove_nan_chunks: If True, remove rows with unassigned chunks
        remove_overlap: If True, filter out communication-overlapped kernels
        fix_names: If True, normalize operator names
        group_arr: Optional list of columns to group by for aggregation
        group_map: Optional dict mapping columns to aggregation functions
        sort_value: Optional column name to sort by after grouping

    Returns:
        Processed DataFrame with applied transformations
    """
    df = load_pickle(fn)
    df['layer'] = df['layer'].fillna(-1)
    df = df[df['name'] != 'Memcpy HtoD (Host -> Device)']

    if iter_idxs:
        df = select_iters(df, iter_idxs)

    if remove_overlap:
        df = df[no_overlap_mask(df)]
    if assign_optype:
        df = assign_operator_type(df)

    if assign_chunks:
        df = do_assign_chunks(df)
        if remove_nan_chunks:
            df = df[~df['chunk'].isna()]

    if fix_names:
        df = do_fix_names(df)

    if group_arr:
        assert group_map, f"Null group_map is invalid with non-null group_arr: {group_arr}"
        assert all(col in df.columns.tolist() for col in group_map.keys())

        missing_cols = tuple(col for col in group_map.keys()
                             if col not in df.columns.tolist())
        assert len(missing_cols) == 0, f"Missing: {missing_cols}"

        weight_metrics: dict[str, list[str]] = {}
        new_group_map = {}
        for metric, aggs in group_map.items():
            for agg in aggs:
                if agg in df.columns.tolist():
                    weight_metrics.setdefault(metric, []).append(agg)
                    new_group_map[agg] = (agg, 'sum')
                    new_group_map[metric] = (metric, 'sum')
                else:
                    new_group_map[metric if agg ==
                                  'sum' else f"{metric}_{agg}"] = (metric, agg)

        for metric, weights in weight_metrics.items():
            assert len(weights) == 1, "cannot weigh by multiple metrics"
            df[metric] *= df[weights[0]]

        df = df.groupby(group_arr, dropna=False).agg(**new_group_map)

        for metric, weights in weight_metrics.items():
            df[metric] /= df[weights[0]]

        if sort_value:
            df = df.sort_values(sort_value).reset_index()
        else:
            df = df.reset_index()

    return df


def get_straggler_df(
    fn: str,
    iter_idxs: Optional[List] = None,
    agg_meth: str = 'max',
    kernel_name: bool = False,
    scope: str = 'auto',
) -> pd.DataFrame:
    """Load and compute straggler metrics from trace data.

    Processes trace data to identify performance stragglers by computing
    how much each GPU lags behind the slowest GPU for each operation.

    Straggler scope under multi-node:
        The reference timestamp is a max (or min, or mean) taken across
        devices. Whether that max should run across all nodes or within each
        node is a real choice, not a detail, so it is a parameter rather than
        an accident of which columns happen to be present.

        'global' compares every device in the job against one reference. That
        is the right question for a single data-parallel group whose devices
        all synchronise together, and it REQUIRES that the timestamps were
        already converted to a common timeline, because each node's kernel
        trace clock has its own origin. Comparing raw per-node timestamps
        globally produces an s-value dominated by the clock offset, which can
        be seconds, not the microseconds straggling actually costs.

        'per_node' takes the reference within each node, so the result answers
        "which GPU lags inside its own host" and needs no clock conversion.

        'auto' (the default) picks 'per_node' when the frame names more than
        one node and 'global' otherwise, so a single-node run behaves exactly
        as it did before and a multi-node run defaults to the answer that is
        correct on unconverted timestamps.

    Args:
        fn: Path to trace pickle file
        iter_idxs: Optional list of iteration indices to select
        agg_meth: Aggregation method ('max', 'min', 'mean') for straggler reference
        kernel_name: If True, include kernel names in grouping
        scope: 'auto', 'per_node', or 'global'. See above.

    Returns:
        DataFrame with straggler metrics including 's-value' (lag time) and
        's-delta' (change in lag between operations). On a multi-node frame the
        result carries a 'straggler-scope' column recording which comparison
        produced the numbers.
    """
    assert scope in ('auto', 'per_node', 'global'), f"bad scope: {scope!r}"

    group_arr = ['iteration', 'layer', 'operator-name',
                 'name'] if kernel_name else ['iteration', 'layer', 'operator-name']

    probe = load_pickle(fn)
    multi_node = is_multi_node(probe)
    dev_keys = [NODE_COL, 'gpu'] if multi_node else ['gpu']
    if scope == 'auto':
        scope = 'per_node' if multi_node else 'global'
    assert not (scope == 'per_node' and not multi_node), (
        "scope='per_node' needs a frame with a 'node' column; this trace has none")

    # The reference is taken across devices; per_node keeps the node fixed.
    ref_arr = ([NODE_COL] + group_arr) if scope == 'per_node' else group_arr

    df = get_df(
        fn,
        iter_idxs=iter_idxs,
        assign_chunks=True,
        remove_nan_chunks=True,
        remove_overlap=True,
        fix_names=True,
        group_arr=dev_keys + group_arr,
        group_map={
            'ts': ['first', 'last'],
            'dur': ['sum', 'last'],
        },
        sort_value='ts_first',
    )
    agg_df = df.groupby(
        ref_arr,
        dropna=False
    ).agg(
        **{f'ts_first_{agg_meth}': ('ts_first', agg_meth)}
    ).sort_values(f'ts_first_{agg_meth}').reset_index()

    df = df.merge(
        agg_df,
        on=ref_arr,
        how='left'
    )

    df['s-value'] = (
        df[f'ts_first_{agg_meth}'] - df['ts_first'])

    df['s-delta'] = df.groupby(
        dev_keys
    )['s-value'].transform(lambda x: x.shift(-1) - x)

    last_op_of_iter_mask = df.groupby(
        dev_keys + ['iteration']).cumcount(ascending=False) == 0
    df.loc[last_op_of_iter_mask, 's-delta'] = 0

    if multi_node:
        df['straggler-scope'] = scope

    return df


def get_straggler_contributors(
    df: pd.DataFrame,
    group_arr: Optional[List[str]] = None,
    delta: bool = False,
    agg_cols: List[str] = ['min', 'max', 'median', 'sum'],
):
    """Aggregate straggler contributions by operator or GPU.

    Args:
        df: DataFrame from get_straggler_df() containing straggler metrics
        group_arr: List of columns to group by (e.g., ['gpu'], ['operator-name']).
            Defaults to ['gpu', 'operator-name'] on a single-node frame and
            ['node', 'gpu', 'operator-name'] on a multi-node one, because
            grouping by gpu alone would pool every node's GPU 3 into one row.
        delta: If True, analyze s-delta instead of s-value
        agg_cols: List of aggregation functions to apply

    Returns:
        DataFrame with aggregated straggler contributions
    """
    if group_arr is None:
        group_arr = device_keys(df) + ['operator-name']
    return df.groupby(group_arr)[
        's-delta' if delta else 's-value'
    ].agg(list(agg_cols)).reset_index()


def get_overlap_df(
    fn: str,
    iter_idxs: Optional[List] = None,
    kernel_name: bool = False,
    include_comm_df: bool = False,
):
    """Compute communication-computation overlap ratios.

    Analyzes how much computation overlaps with communication operations
    to assess pipeline efficiency.

    Args:
        fn: Path to trace pickle file
        iter_idxs: Optional list of iteration indices to select
        kernel_name: If True, include kernel names in grouping
        include_comm_df: If True, return both overlap and communication DataFrames

    Returns:
        If include_comm_df is False: DataFrame with overlap_ratio column
        If include_comm_df is True: Tuple of (overlap_df, comm_df)

    Overlap is computed within one device. On a multi-node frame the device is
    (node, gpu), so a compute kernel is never matched against communication
    from another host's GPU of the same index.
    """
    comm_df = get_df(
        fn,
        iter_idxs=iter_idxs,
        sort_value='ts',
    )
    comm_df = comm_df[~no_overlap_mask(comm_df)]
    comm_df['end_ts'] = comm_df['ts'] + comm_df['dur']

    dev_keys = device_keys(comm_df)
    tail = ['iteration', 'layer', 'operator-name',
            'name'] if kernel_name else ['iteration', 'layer', 'operator-name']

    comp_df = get_df(
        fn,
        iter_idxs=iter_idxs,
        assign_chunks=True,
        remove_nan_chunks=True,
        remove_overlap=True,
        fix_names=True,
        group_arr=dev_keys + tail,
        group_map={
            'ts': ['first', 'last'],
            'dur': ['sum', 'last'],
        },
        sort_value='ts_first',
    )
    comp_df['end_ts'] = comp_df['ts_last'] + comp_df['dur_last']
    comp_df['elapsed'] = comp_df['end_ts'] - comp_df['ts_first']

    def add_overlap(group):
        key = group.name
        gpu_comm_df = comm_df
        for col, val in zip(dev_keys, key if isinstance(key, tuple) else (key,)):
            gpu_comm_df = gpu_comm_df[gpu_comm_df[col] == val]

        for op_idx, operation in group.iterrows():
            start = operation['ts_first']
            end = operation['end_ts']
            elapsed = operation['elapsed']
            overlapped_comm = gpu_comm_df[
                (gpu_comm_df['ts'] <= end) &
                (gpu_comm_df['end_ts'] >= start)
            ]

            # FIXME maybe double counts some overlap
            total_ovr = 0
            for _, comm_kern in overlapped_comm.iterrows():
                ovr_start = max(start, comm_kern['ts'])
                ovr_end = min(end, comm_kern['end_ts'])
                total_ovr += max(0, ovr_end - ovr_start)

            total_ovr = min(elapsed, total_ovr)
            ratio = 100 * total_ovr / elapsed
            group.loc[op_idx, "overlap_ratio"] = ratio
        return group

    by = dev_keys[0] if len(dev_keys) == 1 else dev_keys
    ovr_df = comp_df.groupby(by).apply(add_overlap).reset_index(
        level=list(range(len(dev_keys))))
    if include_comm_df:
        return ovr_df, comm_df
    else:
        return ovr_df


def get_slack_adv_df(
    fn: str,
    iter_idxs: Optional[List] = None,
    kernel_name: bool = False,
    agg_meth: str = 'max',
    scope: str = 'auto',
):
    """Compute slack advantage metrics for communication operations.

    Analyzes the slack advantage of communication operations -- how much
    idle time (slack) is available relative to computation.

    Args:
        fn: Path to trace pickle file
        iter_idxs: Optional list of iteration indices to select
        kernel_name: If True, include kernel names in grouping
        agg_meth: Aggregation method ('max', 'min', 'mean') for slack reference
        scope: 'auto', 'per_node', or 'global'. Same meaning as in
            get_straggler_df: the reference timestamp is taken across devices,
            and 'global' only makes sense once the timestamps sit on a common
            timeline.

    Returns:
        Tuple of (comm_df, comp_df) with timing and straggler information
    """
    assert scope in ('auto', 'per_node', 'global'), f"bad scope: {scope!r}"

    group_arr = ['iteration', 'layer', 'operator-name',
                 'name'] if kernel_name else ['iteration', 'layer', 'operator-name']

    multi_node = is_multi_node(load_pickle(fn))
    dev_keys = [NODE_COL, 'gpu'] if multi_node else ['gpu']
    if scope == 'auto':
        scope = 'per_node' if multi_node else 'global'
    assert not (scope == 'per_node' and not multi_node), (
        "scope='per_node' needs a frame with a 'node' column; this trace has none")
    ref_arr = ([NODE_COL] + group_arr) if scope == 'per_node' else group_arr

    comm_df = get_df(
        fn,
        iter_idxs=iter_idxs,
        group_arr=dev_keys + group_arr,
        group_map={
            'ts': ['first', 'last'],
            'dur': ['sum', 'last'],
        },
        sort_value='ts_first',
    )
    comm_df = comm_df[~no_overlap_mask(comm_df)]
    comm_df = comm_df[comm_df['name'] != 'Memcpy HtoD (Host -> Device)']
    comm_df['end_ts'] = comm_df['ts_last'] + comm_df['dur']
    comm_df['elapsed'] = comm_df['end_ts'] - comm_df['ts_first']

    comp_df = get_df(
        fn,
        iter_idxs=iter_idxs,
        assign_chunks=True,
        remove_nan_chunks=True,
        remove_overlap=True,
        fix_names=True,
        group_arr=dev_keys + group_arr,
        group_map={
            'ts': ['first', 'last'],
            'dur': ['sum', 'last'],
        },
        sort_value='ts_first',
    )
    comp_df['end_ts'] = comp_df['ts_last'] + comp_df['dur_last']
    comp_df['elapsed'] = comp_df['end_ts'] - comp_df['ts_first']

    agg_df = comm_df.groupby(
        ref_arr,
        dropna=False
    ).agg(
        **{f'ts_{agg_meth}': ('ts_first', agg_meth)}
    ).sort_values(f'ts_{agg_meth}').reset_index()

    comm_df = comm_df.merge(
        agg_df,
        on=ref_arr,
        how='left'
    )

    comm_df['s-value'] = (
        comm_df[f'ts_{agg_meth}'] - comm_df['ts_first'])

    if multi_node:
        comm_df['straggler-scope'] = scope

    return comm_df, comp_df
