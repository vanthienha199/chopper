"""Full trace merge: parse PyTorch Chrome traces into a kernel DataFrame."""
import json
import time
import re
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from loguru import logger

from chopper.common.nodes import (
    NODE_COL,
    nodes_in,
    require_node_axis,
    resolve_node_from_path,
)


def assign_ranges(timestamps, ranges):
    """Assign labels from (start, end, label) ranges to timestamps.
    Ranges must be sorted longest-first so innermost overwrites."""
    ts_arr = np.array(timestamps)
    sort_idx = ts_arr.argsort()
    sorted_ts = ts_arr[sort_idx]
    result = np.empty(len(sorted_ts), dtype=object)
    for start, end, label in ranges:
        lo = np.searchsorted(sorted_ts, start, side='left')
        hi = np.searchsorted(sorted_ts, end, side='right')
        result[lo:hi] = label
    final = np.empty_like(result)
    final[sort_idx] = result
    return final


def parse(filename):
    with open(filename) as f:
        trace = json.load(f)

    cpu_ops = {}
    annotations = []
    fwdbwd = {}
    kernels = []
    runtime = {}

    for e in trace['traceEvents']:
        cat = e.get('cat', '')
        if cat == 'cpu_op':
            cpu_ops[e['args']['External id']] = {
                'name': e['name'],
                'ts': int(e['ts'] * 1000),
                'dur': int(e['dur'] * 1000),
                'seq': e['args'].get('Sequence number'),
            }
        elif cat == 'user_annotation':
            ts = int(e['ts'] * 1000)
            dur = int(e['dur'] * 1000)
            annotations.append({'name': e['name'], 'ts': ts, 'end_ts': ts + dur})
        elif cat == 'fwdbwd':
            fid = e['id']
            if fid not in fwdbwd:
                fwdbwd[fid] = {}
            fwdbwd[fid]['bwd' if 'bp' in e else 'fwd'] = int(e['ts'] * 1000)
        elif cat == 'kernel':
            kernels.append({
                'name': e['name'],
                'ts': int(e['ts'] * 1000),
                'dur': int(e['dur'] * 1000),
                'correlation': e['args']['correlation'],
            })
        elif cat == 'cuda_runtime':
            runtime[e['args']['correlation']] = {
                'ext_id': e['args'].get('External id'),
                'ts': int(e['ts'] * 1000),
            }

    return cpu_ops, annotations, fwdbwd, kernels, runtime


def classify_annotations(annotations):
    """Split annotations into layers, iterations, and operator annotations."""
    layers = []
    iterations = []
    ops = []
    for a in annotations:
        if (m := re.fullmatch(r'Layer(\d+)', a['name'])):
            layers.append((a['ts'], a['end_ts'], int(m.group(1))))
            continue
        if (m := re.fullmatch(r'Iteration(\d+)', a['name'])):
            iterations.append((a['ts'], a['end_ts'], int(m.group(1))))
            continue
        ops.append((a['ts'], a['end_ts'], a['name']))

    # Sort longest-first so innermost overwrites
    layers.sort(key=lambda r: r[1] - r[0], reverse=True)
    iterations.sort(key=lambda r: r[1] - r[0], reverse=True)
    ops.sort(key=lambda r: r[1] - r[0], reverse=True)
    return layers, iterations, ops


def link_fwdbwd(cpu_ops, op_ranges, fwdbwd, layer_ranges):
    """Link fwd<->bwd cpu_ops.
    Returns (ext_id -> f_/b_ annotation name, ext_id -> layer)."""
    ts_to_ext = {op['ts']: eid for eid, op in cpu_ops.items()}
    fwd_data = []
    fwd_timestamps = []

    for fid, ep in fwdbwd.items():
        assert 'fwd' in ep and 'bwd' in ep, f"fwdbwd {fid} missing endpoint: {ep}"
        fwd_ext = ts_to_ext.get(ep['fwd'])
        bwd_ext = ts_to_ext.get(ep['bwd'])
        assert fwd_ext is not None, f"fwdbwd {fid} fwd ts {ep['fwd']} not in cpu_ops"
        assert bwd_ext is not None, f"fwdbwd {fid} bwd ts {ep['bwd']} not in cpu_ops"
        fwd_data.append((fwd_ext, bwd_ext))
        fwd_timestamps.append(cpu_ops[fwd_ext]['ts'])

    anns = assign_ranges(fwd_timestamps, op_ranges)
    fwd_layers = assign_ranges(fwd_timestamps, layer_ranges)

    results = {}
    fwdbwd_layers = {}
    for i, (fwd_ext, bwd_ext) in enumerate(fwd_data):
        ann = anns[i]
        assert ann is not None, f"cpu_op {fwd_ext} ({cpu_ops[fwd_ext]['name']}) has no annotation"
        results[fwd_ext] = 'f_' + ann
        results[bwd_ext] = 'b_' + ann
        fwdbwd_layers[fwd_ext] = fwd_layers[i]
        fwdbwd_layers[bwd_ext] = fwd_layers[i]

    return results, fwdbwd_layers


def promote(cpu_ops, labeled):
    """Promote fwdbwd labels up to parent cpu_ops with the same sequence number.
    This ensures siblings of the linked op also get covered when we propagate down."""
    seq_to_label = {}
    for eid, label in labeled.items():
        seq = cpu_ops[eid].get('seq')
        if seq is not None:
            seq_to_label[seq] = label

    result = dict(labeled)
    for eid, op in cpu_ops.items():
        if eid in result:
            continue
        seq = op.get('seq')
        if seq is not None and seq in seq_to_label:
            result[eid] = seq_to_label[seq]
    return result


def propagate(cpu_ops, labeled):
    """Propagate labels to child cpu_ops. Returns ext_id -> label (expanded)."""
    parents = [(cpu_ops[eid]['ts'], cpu_ops[eid]['ts'] + cpu_ops[eid]['dur'], label)
               for eid, label in labeled.items()]
    parents.sort(key=lambda p: p[1] - p[0], reverse=True)

    unlabeled_eids = [eid for eid in cpu_ops if eid not in labeled]
    unlabeled_ts = [cpu_ops[eid]['ts'] for eid in unlabeled_eids]
    assigned = assign_ranges(unlabeled_ts, parents)

    result = dict(labeled)
    for i, eid in enumerate(unlabeled_eids):
        # None is expected: some autograd backward ops (e.g. DivBackward0) aren't
        # contained by any fwdbwd-labeled parent. These get picked up by assign_unlabeled.
        if assigned[i] is not None:
            result[eid] = assigned[i]
    return result


def assign_unlabeled(cpu_ops, labeled, op_ranges):
    """Assign annotations to cpu_ops not covered by fwdbwd (opt, FSDP, etc)."""
    unlabeled_eids = [eid for eid in cpu_ops if eid not in labeled]
    unlabeled_ts = [cpu_ops[eid]['ts'] for eid in unlabeled_eids]
    assigned = assign_ranges(unlabeled_ts, op_ranges)

    result = dict(labeled)
    for i, eid in enumerate(unlabeled_eids):
        # None is expected: some cpu_ops occur outside any user_annotation range
        # (e.g. during profiler init before training starts). These remain unlabeled.
        if assigned[i] is not None:
            result[eid] = assigned[i]
    return result


def build_kernel_df(cpu_ops, kernels, runtime, labels, layer_ranges, iter_ranges, fwdbwd_layers):
    """Build final kernel DataFrame with all context."""
    # Assign layer and iteration to all cpu_ops via timestamp ranges
    all_eids = list(cpu_ops.keys())
    all_ts = [cpu_ops[eid]['ts'] for eid in all_eids]
    layers = assign_ranges(all_ts, layer_ranges)
    iterations = assign_ranges(all_ts, iter_ranges)
    eid_to_layer = {eid: layers[i] for i, eid in enumerate(all_eids)}
    eid_to_iter = {eid: iterations[i] for i, eid in enumerate(all_eids)}

    # fwdbwd_layers covers both fwd and bwd ops (bwd gets fwd's layer).
    # Overwrite timestamp-based layers with these authoritative values.
    eid_to_layer.update(fwdbwd_layers)

    rows = []
    for k in kernels:
        rt = runtime.get(k['correlation'])
        assert rt is not None, f"kernel {k['correlation']} has no runtime match"
        ext_id = rt['ext_id']
        cpu_op = cpu_ops.get(ext_id)
        assert cpu_op is not None, f"ext_id {ext_id} not in cpu_ops"

        rows.append({
            'name': k['name'],
            'ts': k['ts'],
            'dur': k['dur'],
            'ts_cuda_runtime': rt['ts'],
            'name_cpu_op': cpu_op['name'],
            'operator-name': labels.get(ext_id),
            'layer': eid_to_layer.get(ext_id),
            'iteration': eid_to_iter.get(ext_id),
        })

    return pd.DataFrame(rows)


def parse_trace(filename):
    """Full pipeline: parse trace -> kernel DataFrame."""
    cpu_ops, annotations, fwdbwd, kernels, runtime = parse(filename)
    layer_ranges, iter_ranges, op_ranges = classify_annotations(annotations)

    # Step 1: fwd/bwd linking (also assigns layers to both fwd and bwd ops)
    labeled, fwdbwd_layers = link_fwdbwd(cpu_ops, op_ranges, fwdbwd, layer_ranges)

    # Step 2: promote labels up to parent cpu_ops with same sequence number
    labeled = promote(cpu_ops, labeled)
    fwdbwd_layers = promote(cpu_ops, fwdbwd_layers)

    # HACK
    # some autograd ops don't have sequence numbers (e.g., bwd all gathers)
    # they are spawned by other operations (e.g., residual add launches bwd all gathers)
    # Need to use timestamp based promotion of sibling fwdbwd labels to parents, instead of seq number
    # Then, parent propogate layers to all gather in the bwd pass
    autograd_eids = [
        eid for eid, op in cpu_ops.items()
        if eid not in fwdbwd_layers
        and op['name'].startswith('autograd::engine::evaluate_function:')
        and op.get('seq') is None
    ]
    if autograd_eids:
        labeled_ts = [cpu_ops[eid]['ts'] for eid in fwdbwd_layers]
        labeled_layers = list(fwdbwd_layers.values())
        autograd_ranges = [(cpu_ops[eid]['ts'], cpu_ops[eid]['ts'] + cpu_ops[eid]['dur'], eid)
                           for eid in autograd_eids]
        autograd_ranges.sort(key=lambda r: r[1] - r[0], reverse=True)
        parent_eids = assign_ranges(labeled_ts, autograd_ranges)
        for i, parent_eid in enumerate(parent_eids):
            if parent_eid is not None and parent_eid not in fwdbwd_layers:
                fwdbwd_layers[parent_eid] = labeled_layers[i]
    # end of HACK

    # Step 3: propagate labels and layers to children
    labeled = propagate(cpu_ops, labeled)
    fwdbwd_layers = propagate(cpu_ops, fwdbwd_layers)

    # Step 4: assign annotations to remaining unlabeled ops
    labeled = assign_unlabeled(cpu_ops, labeled, op_ranges)

    # Step 5: build kernel DataFrame
    return build_kernel_df(cpu_ops, kernels, runtime, labeled, layer_ranges, iter_ranges, fwdbwd_layers)


def get_pivoted(csv_filename):
    """Pivot rocprofv3 CSV from long format to wide format.

    rocprofv3 --pmc outputs one row per (kernel, counter). This pivots so
    each kernel dispatch is one row with counter names as columns.
    """
    df = pd.read_csv(csv_filename)
    meta_cols = [c for c in df.columns if c not in ("Counter_Name", "Counter_Value")]
    pivoted = df.pivot_table(
        index=meta_cols, columns="Counter_Name", values="Counter_Value",
    ).sort_values("Start_Timestamp", ascending=True).reset_index()
    return pivoted


def get_combined_counters(csv_list):
    """Merge multiple counter CSV batches for one GPU.

    Each batch collected different counters from the same workload re-run.
    Joins on [Kernel_Name, _mi] where _mi is the instance index within
    each kernel name.
    """
    kname = "Kernel_Name"
    df_combined = None
    for cur_csv in csv_list:
        df_cur = get_pivoted(cur_csv)
        df_cur["_mi"] = df_cur.groupby(kname).cumcount()
        if df_combined is None:
            df_combined = df_cur
        else:
            df_combined = df_combined.merge(
                df_cur, on=[kname, "_mi"], how="left", suffixes=("", "_new"),
            )
            df_combined = df_combined.loc[
                :, ~df_combined.columns.str.endswith("_new")]
    assert df_combined is not None, "no counter CSV files provided"
    df_combined = df_combined.drop(columns=["_mi"])
    return df_combined


def _assign_counter_nodes(per_device, ts_nodes, counter_nodes):
    """Work out which node each counter device's CSV files came from.

    Args:
        per_device: Per-device lists of CSV paths (the transposed batches).
        ts_nodes: Node labels present in the trace frame.
        counter_nodes: Explicit override. Either one label for every device or
            one label per device, in the same order as per_device.

    Returns:
        List of node labels (or None where the node could not be determined),
        one per device.
    """
    n = len(per_device)
    if counter_nodes:
        counter_nodes = [str(x) for x in counter_nodes]
        if len(counter_nodes) == 1:
            return counter_nodes * n
        assert len(counter_nodes) == n, (
            f"got {len(counter_nodes)} counter nodes for {n} counter devices; "
            f"pass one label for all devices or one label per device")
        return counter_nodes

    inferred = [resolve_node_from_path(fns[0], ts_nodes) for fns in per_device]
    if all(x is None for x in inferred) and len(ts_nodes) == 1:
        # Single-node run: every counter file belongs to the only node there is.
        return [ts_nodes[0]] * n
    return inferred


def _gpu_index_within_node(device_nodes):
    """Number the devices 0..N-1 within each node, in file-sorted order.

    ``sorted()`` groups a node's files together because collect.py writes them
    under a directory named by the hostname, and within a node the existing
    convention is that sorted file order is GPU order.
    """
    seen: dict = {}
    out = []
    for node in device_nodes:
        idx = seen.get(node, 0)
        out.append(idx)
        seen[node] = idx + 1
    return out


def merge_counters(df_ts, counter_batches, counter_nodes=None):
    """Join hardware counter data with trace pickle.

    The join keys on kernel name plus an instance index within one device. On
    a multi-node run the device is (node, gpu), not gpu alone, because GPU 3
    exists on every node and reaches the same instance index there. Keying on
    gpu alone would pick an arbitrary node's row.

    Args:
        df_ts: Trace DataFrame (from ts.pkl)
        counter_batches: List of lists -- each inner list is one batch of
            CSV files (sorted = GPU order). E.g.:
            [["batch0/gpu0.csv", "batch0/gpu1.csv"],
             ["batch1/gpu0.csv", "batch1/gpu1.csv"]]
        counter_nodes: Optional node label per counter device (or one label for
            all of them). Defaults to inferring the node from each file's path,
            which works with the per-node output directories collect.py creates.

    Raises:
        ValueError: When more than one node's data is present but the node of
            some rows or some counter files cannot be determined.
    """
    # Transpose from per-batch to per-device
    per_gpu = [list(fns) for fns in zip(*[sorted(b) for b in counter_batches])]
    n_counter_gpus = len(per_gpu)

    ts_nodes = nodes_in(df_ts)
    device_nodes = _assign_counter_nodes(per_gpu, ts_nodes, counter_nodes)
    known_nodes = sorted(set(ts_nodes) | set(n for n in device_nodes if n is not None))
    use_node = len(known_nodes) > 1

    gpus = sorted(df_ts["gpu"].unique())
    if use_node:
        require_node_axis(df_ts, "trace frame", other_nodes=known_nodes)
        missing = [fns[0] for fns, node in zip(per_gpu, device_nodes) if node is None]
        if missing:
            raise ValueError(
                f"{len(missing)} counter device(s) could not be attributed to a node "
                f"while {len(known_nodes)} nodes {known_nodes} are present "
                f"(first: {missing[0]}); pass --counter-nodes, because joining "
                f"them on kernel name and instance index alone would mix nodes")
        n_trace_devices = len(df_ts[[NODE_COL, "gpu"]].drop_duplicates())
    else:
        n_trace_devices = len(gpus)
        if n_counter_gpus > n_trace_devices and not known_nodes:
            raise ValueError(
                f"{n_counter_gpus} counter devices but only {n_trace_devices} "
                f"device(s) in the trace, and no {NODE_COL!r} column on either "
                f"side; this is what a multi-node run looks like to a merge that "
                f"cannot tell the nodes apart. Re-collect with node stamping or "
                f"pass --counter-nodes")

    if n_trace_devices != n_counter_gpus:
        logger.warning(
            f"{n_trace_devices} devices in trace but {n_counter_gpus} counter files")

    # Load and combine counters per device
    device_gpus = _gpu_index_within_node(device_nodes) if use_node else list(range(n_counter_gpus))
    frames = []
    for i, fns in enumerate(per_gpu):
        d = get_combined_counters(fns).assign(gpu=device_gpus[i])
        if use_node:
            d[NODE_COL] = device_nodes[i]
        frames.append(d)
    df_cntr = pd.concat(frames, ignore_index=True)
    if use_node:
        logger.info(f"Loaded {len(df_cntr)} counter rows across {n_counter_gpus} "
                    f"devices on {len(known_nodes)} nodes {known_nodes}")
    else:
        logger.info(f"Loaded {len(df_cntr)} counter rows across {n_counter_gpus} GPUs")

    kname = "Kernel_Name"

    # Select first iteration from trace for matching
    first_iter = df_ts["iteration"].dropna().unique().min()
    logger.info(f"Matching counters against iteration {first_iter}")
    match_mask = df_ts["iteration"] == first_iter
    df_match = df_ts[match_mask].copy()
    df_rest = df_ts[~match_mask].copy()

    # Warn about kernel name mismatches
    ts_names = set(df_match["name"])
    cntr_names = set(df_cntr[kname])
    cntr_only = cntr_names - ts_names
    ts_only = ts_names - cntr_names
    if cntr_only:
        logger.warning(f"{len(cntr_only)} kernels in counters but not trace")
    if ts_only:
        logger.warning(f"{len(ts_only)} kernels in trace but not counters")

    # Remove mismatched kernels from counter side
    df_cntr = df_cntr[~df_cntr[kname].isin(cntr_only)]

    # Assign reverse-cumcount _mi (count from end so late dispatches align).
    # The instance index only means anything within one device, so on a
    # multi-node run the device key carries the node as well as the GPU.
    dev_keys = [NODE_COL, "gpu"] if use_node else ["gpu"]
    left_keys = ["name"] + dev_keys
    right_keys = [kname] + dev_keys

    df_match["_mi"] = (
        df_match.iloc[::-1].groupby(left_keys).cumcount().iloc[::-1]
    )
    df_cntr["_mi"] = (
        df_cntr.iloc[::-1].groupby(right_keys).cumcount().iloc[::-1]
    )

    # Merge
    df_merged = df_match.merge(
        df_cntr,
        left_on=left_keys + ["_mi"],
        right_on=right_keys + ["_mi"],
        how="left",
        suffixes=("", "_y"),
    )

    # Drop temp and duplicate columns
    drop_cols = ["_mi", kname]
    drop_cols += [c for c in df_merged.columns if c.endswith("_y")]
    df_merged = df_merged.drop(columns=[c for c in drop_cols if c in df_merged.columns])

    # Rejoin with non-matched iterations
    df_result = pd.concat([df_merged, df_rest], ignore_index=True)
    df_result = df_result.sort_values("ts").reset_index(drop=True)

    # Report
    counter_cols = [c for c in df_merged.columns if c not in df_ts.columns and c != "_mi"]
    n_matched = df_merged[counter_cols[0]].notna().sum() if counter_cols else 0
    logger.info(f"Merged {n_matched}/{len(df_match)} kernels with counters")
    logger.info(f"Counter columns: {counter_cols}")

    return df_result


def _prepare_device_samples(counter_df):
    """Sum across HW dims, pivot wide, diff to get per-interval deltas.

    Returns (pivoted DataFrame with delta columns + timestamp_ns, half_dt_ns).
    """
    totals = counter_df.groupby(
        ["timestamp_ns", "counter_name"]
    )["counter_value"].sum().reset_index()

    piv = totals.pivot(
        index="timestamp_ns", columns="counter_name", values="counter_value"
    ).reset_index()
    piv.columns.name = None
    piv = piv.sort_values("timestamp_ns").reset_index(drop=True)

    # Diff: cumulative -> per-interval
    for col in [c for c in piv.columns if c != "timestamp_ns"]:
        piv[col] = piv[col].diff()

    dt_ns = piv["timestamp_ns"].diff()
    half_dt_ns = int(dt_ns.median() / 2)
    piv = piv.iloc[1:].reset_index(drop=True)

    return piv, half_dt_ns


def scan_rank_files(group_dir):
    """Discover the per-rank device files in one counter group directory.

    The device tool suffixes its output with the GLOBAL rank (RANK, else
    SLURM_PROCID, else LOCAL_RANK) and writes a rank_info sidecar naming the
    node, the global rank and the local rank. The local rank is the index of
    the GPU on its own node, which is the number that lines up with the trace
    frame's 'gpu' column.

    Runs collected before the sidecar existed have no node information. There
    the suffix is treated as both the global and the local rank, which is
    correct for a single-node run, and the caller's node guard is what has to
    catch a multi-node one.

    Args:
        group_dir: One chopper_device_counters* directory.

    Returns:
        List of dicts sorted by (node, local_rank), each with keys
        global_rank, local_rank, node, counter_file, trace_file.
    """
    from pathlib import Path

    group_dir = Path(group_dir)
    records = []
    for cf in sorted(group_dir.glob("counter_samples_rank*.csv")):
        suffix = cf.stem.split("_rank")[1]
        tf = group_dir / f"kernel_traces_rank{suffix}.csv"
        info_path = group_dir / f"rank_info_rank{suffix}.csv"
        node = None
        local_rank = int(suffix)
        global_rank = int(suffix)
        if info_path.is_file():
            info = pd.read_csv(info_path)
            assert len(info) == 1, f"{info_path} should hold exactly one row"
            row = info.iloc[0]
            node = str(row["node"])
            local_rank = int(row["local_rank"])
            global_rank = int(row["global_rank"])
        records.append({
            "global_rank": global_rank,
            "local_rank": local_rank,
            "node": node,
            "counter_file": cf,
            "trace_file": tf,
        })
    records.sort(key=lambda r: (r["node"] or "", r["local_rank"]))
    return records


def merge_device_with_traces(device_dir, trace_pkl, output):
    """Merge device counter samples with PyTorch trace annotations.

    Each counter group keeps its own runtime kernel trace and samples.
    Annotations (operator-name, layer, iteration) are stolen from ts.pkl
    via reverse-cumcount matching, keyed per device. On a multi-node run the
    device is (node, gpu): the same kernel name reaches the same instance
    index on every node, so a gpu-only key would steal another host's
    annotations.

    Output: {
        "groups": {gi: {"kernels": df, "samples": df, "counters": [...]}},
        "counter_to_group": {counter_name: group_index},
    }
    """
    import pickle
    from pathlib import Path

    t0 = time.time()

    # Load ts.pkl for annotations
    df_ts = pd.read_pickle(trace_pkl)
    last_iter = df_ts["iteration"].dropna().unique().max()
    logger.info(f"Stealing annotations from iteration {last_iter} of {trace_pkl}")
    df_last = df_ts[df_ts["iteration"] == last_iter].copy()
    df_last = df_last.sort_values("ts").reset_index(drop=True)
    gpus = sorted(df_last["gpu"].unique())

    # Load device sampling CSVs
    device_dir = Path(device_dir)
    group_dirs = sorted(device_dir.glob("chopper_device_counters*"))
    assert group_dirs, f"No chopper_device_counters* dirs in {device_dir}"

    ann_cols = ["operator-name", "layer", "iteration"]
    for c in ann_cols:
        assert c in df_last.columns, f"ts.pkl missing required column: {c}"

    # Decide once whether the node takes part in the device key.
    probe = scan_rank_files(group_dirs[0])
    assert probe, f"No counter_samples_rank*.csv in {group_dirs[0]}"
    file_nodes = sorted(set(r["node"] for r in probe if r["node"] is not None))
    ts_nodes = nodes_in(df_last)
    known_nodes = sorted(set(file_nodes) | set(ts_nodes))
    use_node = len(known_nodes) > 1
    if use_node:
        require_node_axis(df_last, f"trace frame {trace_pkl}", other_nodes=known_nodes)
        unlabelled = [r for r in probe if r["node"] is None]
        if unlabelled:
            raise ValueError(
                f"{len(unlabelled)} device file(s) in {group_dirs[0]} carry no "
                f"rank_info sidecar while {len(known_nodes)} nodes {known_nodes} "
                f"are present; those files cannot be attributed to a host and "
                f"annotating them by gpu index alone would take another node's rows")

    def device_key(node, gpu):
        return (node, gpu) if use_node else (gpu,)

    # Prepare ts.pkl reverse cumcount per device (done once)
    ts_by_device = {}
    if use_node:
        device_pairs = [(str(n), g) for n, g in
                        df_last[[NODE_COL, "gpu"]].drop_duplicates().itertuples(index=False)]
    else:
        device_pairs = [(None, g) for g in gpus]
    for node, gpu in device_pairs:
        sel = df_last["gpu"] == gpu
        if use_node:
            sel = sel & (df_last[NODE_COL].astype(str) == node)
        ts_dev = df_last[sel].copy()
        ts_dev["_mi"] = (
            ts_dev.iloc[::-1].groupby("name").cumcount().iloc[::-1]
        )
        ts_by_device[device_key(node, gpu)] = ts_dev

    output_groups = {}
    counter_to_group = {}

    for gi, group_dir in enumerate(group_dirs):
        rank_records = scan_rank_files(group_dir)
        assert rank_records, f"No counter_samples_rank*.csv in {group_dir}"
        found = {device_key(r["node"], r["local_rank"]) for r in rank_records}
        for key in ts_by_device:
            assert key in found, f"Missing device files for {key} in {group_dir}"

        all_kernels = []
        all_samples = []
        group_counter_names = None

        for rec in rank_records:
            node = rec["node"]
            gpu = rec["local_rank"]
            counter_file = rec["counter_file"]
            trace_file = rec["trace_file"]
            assert counter_file.is_file(), f"Missing {counter_file}"
            assert trace_file.is_file(), f"Missing {trace_file}"

            runtime_df = pd.read_csv(trace_file)
            assert len(runtime_df) > 0, f"Empty kernel trace: {trace_file}"
            runtime_df = runtime_df.sort_values("start_ns").reset_index(drop=True)

            counter_df = pd.read_csv(counter_file)
            assert len(counter_df) > 0, f"Empty counter file: {counter_file}"
            samples_piv, half_dt_ns = _prepare_device_samples(counter_df)

            if group_counter_names is None:
                group_counter_names = [c for c in samples_piv.columns if c != "timestamp_ns"]

            # Apply midpoint correction
            samples_piv = samples_piv.copy()
            samples_piv["timestamp_ns"] = samples_piv["timestamp_ns"] - half_dt_ns
            samples_piv["gpu"] = gpu
            if use_node:
                samples_piv[NODE_COL] = node

            where = f"GPU {gpu}" if not use_node else f"{node} GPU {gpu}"
            logger.info(f"Group {gi}, {where}: {len(samples_piv)} samples, "
                        f"{len(runtime_df)} runtime kernels")

            # Annotate runtime kernels from ts.pkl
            kernel_rows = runtime_df.rename(columns={
                "kernel_name": "name",
                "start_ns": "ts",
                "duration_ns": "dur",
            }).copy()
            kernel_rows["gpu"] = gpu
            if use_node:
                kernel_rows[NODE_COL] = node

            # Reverse cumcount match to steal annotations
            key = device_key(node, gpu)
            assert key in ts_by_device, (
                f"device {key} has counter files but no rows in {trace_pkl}; "
                f"trace has {sorted(ts_by_device)}")
            ts_gpu = ts_by_device[key]
            ts_names = set(ts_gpu["name"])
            rt_names = set(kernel_rows["name"])
            rt_only = rt_names - ts_names
            if rt_only:
                logger.warning(f"  Group {gi}, {where}: {len(rt_only)} kernels not in ts.pkl")

            kernel_rows["_mi"] = (
                kernel_rows.iloc[::-1].groupby("name").cumcount().iloc[::-1]
            )
            kernel_rows = kernel_rows.merge(
                ts_gpu[["name", "_mi"] + ann_cols],
                on=["name", "_mi"],
                how="left",
            )
            kernel_rows = kernel_rows.drop(
                columns=["_mi", "end_ns", "agent_id", "queue_id", "correlation_id"],
                errors="ignore",
            )

            all_kernels.append(kernel_rows)
            all_samples.append(samples_piv)

        df_kernels = pd.concat(all_kernels, ignore_index=True)
        df_samples = pd.concat(all_samples, ignore_index=True)
        sample_sort = ([NODE_COL, "gpu", "timestamp_ns"] if use_node
                       else ["gpu", "timestamp_ns"])
        df_samples = df_samples.sort_values(sample_sort).reset_index(drop=True)

        n_annotated = df_kernels["operator-name"].notna().sum()
        logger.info(f"Group {gi}: {len(df_kernels)} kernels ({n_annotated} annotated), "
                    f"{len(df_samples)} samples, counters: {group_counter_names}")

        output_groups[gi] = {
            "kernels": df_kernels,
            "samples": df_samples,
            "counters": group_counter_names,
        }
        for cname in group_counter_names:
            counter_to_group[cname] = gi

    result = {
        "groups": output_groups,
        "counter_to_group": counter_to_group,
    }
    with open(output, "wb") as f:
        pickle.dump(result, f)

    t1 = time.time()
    logger.info(f"Wrote {output} in {t1 - t0:.2f}s")
    if use_node:
        logger.info(f"  {len(output_groups)} groups, {len(ts_by_device)} devices "
                    f"on {len(known_nodes)} nodes {known_nodes}")
    else:
        logger.info(f"  {len(output_groups)} groups, {len(gpus)} GPUs")
    logger.info(f"  Counter -> group: {counter_to_group}")


def merge_device_counters(device_dir, output):
    """Store device sampling CSVs into a pickle. No transformations.

    Reads counter_samples_rank*.csv and kernel_traces_rank*.csv from
    chopper_device_counters*/ subdirs.

    Dict keys are (group_index, global_rank). The rank in the filename is the
    GLOBAL rank, so the keys stay unique across nodes. When more than one node
    contributed, the result carries an extra 'rank_index' frame mapping each
    global rank to its node and local GPU index, because the global rank on
    its own does not say which physical GPU produced the rows.
    """
    from pathlib import Path

    device_dir = Path(device_dir)
    groups = sorted(device_dir.glob("chopper_device_counters*"))
    assert groups, f"No chopper_device_counters* dirs in {device_dir}"

    counter_samples = {}
    kernel_traces = {}
    counter_to_group = {}  # {counter_name: group_index}
    rank_rows = []

    for gi, group_dir in enumerate(groups):
        counter_files = sorted(group_dir.glob("counter_samples_rank*.csv"))
        trace_files = sorted(group_dir.glob("kernel_traces_rank*.csv"))
        assert counter_files, f"No counter_samples_rank*.csv in {group_dir}"
        assert trace_files, f"No kernel_traces_rank*.csv in {group_dir}"

        for rec in scan_rank_files(group_dir):
            rank_rows.append({
                "group": gi,
                "global_rank": rec["global_rank"],
                "local_rank": rec["local_rank"],
                NODE_COL: rec["node"],
            })

        for cf in counter_files:
            rank = int(cf.stem.split("_rank")[1])
            df = pd.read_csv(cf)
            assert len(df) > 0, f"Empty counter file: {cf}"
            assert (gi, rank) not in counter_samples, (
                f"two counter files claim group {gi} rank {rank} ({cf}); the "
                f"device tool names files by global rank, so this means two "
                f"nodes wrote into one directory with local ranks")
            counter_samples[(gi, rank)] = df
            for name in df["counter_name"].unique():
                counter_to_group[name] = gi

        for tf in trace_files:
            rank = int(tf.stem.split("_rank")[1])
            df = pd.read_csv(tf)
            if len(df) > 0:
                kernel_traces[(gi, rank)] = df

    result = {
        "counter_samples": counter_samples,
        "kernel_traces": kernel_traces,
        "counter_to_group": counter_to_group,
    }

    # Only present on a multi-node run, so a single-node pickle is unchanged.
    seen_nodes = sorted(set(r[NODE_COL] for r in rank_rows if r[NODE_COL] is not None))
    if len(seen_nodes) > 1:
        result["rank_index"] = pd.DataFrame(rank_rows)
        logger.info(f"  Nodes: {seen_nodes}")

    import pickle
    with open(output, "wb") as f:
        pickle.dump(result, f)

    ranks = sorted(set(r for _, r in counter_samples.keys()))
    total_samples = sum(len(df) for df in counter_samples.values())
    total_kernels = sum(len(df) for df in kernel_traces.values())
    logger.info(f"Wrote {output}: {total_samples} counter samples, {total_kernels} kernel records")
    logger.info(f"  Groups: {len(groups)}, Ranks: {ranks}")
    logger.info(f"  Counter -> group: {counter_to_group}")


def _gpu_busy_per_bin(starts, ends, t0, bin_ns, n_bins):
    """Fraction of each time bin covered by kernel execution (union of intervals).

    Overlapping kernels (concurrent compute/comm streams) are merged so a bin is
    never counted past 100%. starts/ends are ns in the rocprofiler clock domain.
    """
    order = starts.argsort()
    s = (starts[order] - t0)
    e = (ends[order] - t0)
    # merge overlapping intervals
    merged = []
    cs, ce = int(s[0]), int(e[0])
    for i in range(1, len(s)):
        si, ei = int(s[i]), int(e[i])
        if si <= ce:
            ce = max(ce, ei)
        else:
            merged.append((cs, ce))
            cs, ce = si, ei
    merged.append((cs, ce))

    busy = np.zeros(n_bins)
    for cs, ce in merged:
        lo_bin = max(0, cs // bin_ns)
        hi_bin = min(n_bins - 1, ce // bin_ns)
        for b in range(lo_bin, hi_bin + 1):
            lo = b * bin_ns
            hi = lo + bin_ns
            busy[b] += max(0, min(ce, hi) - max(cs, lo))
    return 100.0 * busy / bin_ns


def merge_cpu_gpu_timeline(cpu_pkl, kernel_csv, output, bin_ms=10):
    """Join CPU telemetry and the GPU kernel timeline on the shared clock.

    When cpu.pkl is collected with --cpu-clock rocprofiler, its timestamps are in
    the same domain as the device sampler's kernel_traces.csv (both from
    rocprofiler_get_timestamp). This bins both into fixed windows and reports, per
    bin: mean CPU utilization (averaged over cores) and GPU busy% (fraction of the
    bin covered by kernel execution). The result is the CPU-and-GPU-on-one-timeline
    view: when GPU is busy, CPU is typically idle and vice versa.
    """
    import pickle

    cpu = pd.read_pickle(cpu_pkl)
    cpu_nodes = nodes_in(cpu)
    if len(cpu_nodes) > 1:
        raise ValueError(
            f"{cpu_pkl} holds CPU samples from {len(cpu_nodes)} nodes {cpu_nodes} "
            f"but {kernel_csv} is one node's kernel trace; binning them onto one "
            f"axis would average another host's cores against these kernels. "
            f"Filter the frame to one node, or bin each node separately")
    domain = cpu.attrs.get("clock_domain")
    if domain != "rocprofiler":
        logger.warning(
            f"cpu.pkl clock_domain={domain!r} (expected 'rocprofiler'). "
            "CPU and GPU axes may not align; re-collect with --cpu-clock rocprofiler."
        )

    k = pd.read_csv(kernel_csv)
    bin_ns = bin_ms * 1_000_000

    cpu_ts = cpu["ts"].to_numpy().astype("int64")
    starts = k["start_ns"].to_numpy().astype("int64")
    ends = k["end_ns"].to_numpy().astype("int64")

    t0 = int(min(cpu_ts.min(), starts.min()))
    t_end = int(max(cpu_ts.max(), ends.max()))
    n_bins = (t_end - t0) // bin_ns + 1

    # CPU: mean percent across cores per bin
    cpu = cpu.copy()
    cpu["bin"] = (cpu_ts - t0) // bin_ns
    cpu_busy = cpu.groupby("bin")["percent"].mean()

    gpu_busy = _gpu_busy_per_bin(starts, ends, t0, bin_ns, n_bins)

    rows = []
    for b in range(n_bins):
        rows.append({
            "t_ms": b * bin_ms,
            "cpu_busy_pct": float(cpu_busy.get(b, 0.0)),
            "gpu_busy_pct": float(gpu_busy[b]),
        })
    df = pd.DataFrame(rows)

    result = {"timeline": df, "clock_domain": domain, "bin_ms": bin_ms}
    with open(output, "wb") as f:
        pickle.dump(result, f)

    corr = df["cpu_busy_pct"].corr(df["gpu_busy_pct"])
    logger.info(f"Wrote {output}: {n_bins} bins x {bin_ms}ms, clock={domain}")
    logger.info(f"  mean CPU busy={df['cpu_busy_pct'].mean():.1f}% "
                f"mean GPU busy={df['gpu_busy_pct'].mean():.1f}% "
                f"CPU-GPU corr={corr:.2f}")


def _trace_node_labels(traces, trace_nodes):
    """Decide the node label for each trace file, or None for a single-node run.

    Args:
        traces: Trace file paths, in the order they will be parsed.
        trace_nodes: What the caller passed for --trace-nodes. Either None,
            the literal "auto" (use each trace's parent directory name, which
            is the hostname under the per-node output layout), one label for
            every trace, or one label per trace.

    Returns:
        List of labels the same length as traces, or None when the run is
        single-node and the frame should stay exactly as it was before.

    Raises:
        ValueError: When the traces come from several directories and the
            caller did not say which node each belongs to.
    """
    from pathlib import Path

    if trace_nodes:
        labels = [str(x) for x in trace_nodes]
        if labels == ["auto"]:
            labels = [Path(t).resolve().parent.name for t in traces]
        elif len(labels) == 1:
            labels = labels * len(traces)
        else:
            assert len(labels) == len(traces), (
                f"got {len(labels)} trace nodes for {len(traces)} traces; pass "
                f"one label for all of them, one per trace, or 'auto'")
        return labels if len(set(labels)) > 1 else None

    parents = sorted(set(str(Path(t).resolve().parent) for t in traces))
    if len(parents) > 1:
        raise ValueError(
            f"traces come from {len(parents)} directories {parents} and carry "
            f"no node label; on a multi-node run that means several hosts' "
            f"GPU 0 would all be numbered gpu=0 and later joins would mix "
            f"them. Pass --trace-nodes auto to label each trace by its parent "
            f"directory, an explicit label per trace, or a single label if "
            f"this really is one node")
    return None


def main(traces, pickles, counters, device_dir, output,
         cpu_pkl=None, kernel_csv=None, bin_ms=10,
         counter_nodes=None, trace_nodes=None):
    if cpu_pkl and kernel_csv:
        merge_cpu_gpu_timeline(cpu_pkl, kernel_csv, output, bin_ms)
        return

    if device_dir and pickles:
        assert len(pickles) == 1, "pass exactly one pickle with --device-dir"
        merge_device_with_traces(device_dir, pickles[0], output)
        return

    if device_dir:
        merge_device_counters(device_dir, output)
        return

    assert not (traces and pickles), "pass either -t or -p, not both"
    assert traces or pickles, "pass -t (traces) or -p (pickles)"
    assert not (traces and counters), "cannot use -c with -t; merge traces first, then add counters with -p"

    t0 = time.time()

    if pickles and counters:
        assert len(pickles) == 1, "pass exactly one pickle with -c"
        df = pd.read_pickle(pickles[0])
        df = merge_counters(df, counters, counter_nodes=counter_nodes)
        df.to_pickle(output)
        t1 = time.time()
        logger.info(f"Merged counters into {len(df)} kernels -> {output} in {t1-t0:.2f}s")
        return

    if pickles:
        dfs = [pd.read_pickle(p) for p in pickles]
        labelled = [NODE_COL in d.columns for d in dfs]
        if any(labelled) and not all(labelled):
            raise ValueError(
                f"{sum(labelled)} of {len(dfs)} pickles carry a {NODE_COL!r} "
                f"column; concatenating them would leave the unlabelled rows "
                f"unattributable, and every later join keys on the node. "
                f"Re-merge the unlabelled inputs with --trace-nodes")
        df = pd.concat(dfs, ignore_index=True)
        df = df.sort_values('ts').reset_index(drop=True)
        df.to_pickle(output)
        t1 = time.time()
        nodes = nodes_in(df)
        if len(nodes) > 1:
            logger.info(f"  {len(nodes)} nodes: {nodes}")
        logger.info(f"Merged {len(pickles)} pickles, {len(df)} kernels -> {output} in {t1-t0:.2f}s")
        return

    node_labels = _trace_node_labels(traces, trace_nodes)

    if len(traces) == 1:
        df = parse_trace(traces[0])
        df['gpu'] = 0
        if node_labels:
            df[NODE_COL] = node_labels[0]
    else:
        with ProcessPoolExecutor(max_workers=len(traces)) as ex:
            dfs = list(ex.map(parse_trace, traces))
        if node_labels:
            # gpu is the index of the GPU on its own node, so it restarts at 0
            # for each node. The (node, gpu) pair is what identifies a device.
            per_node: dict = {}
            for d, node in zip(dfs, node_labels):
                idx = per_node.get(node, 0)
                d['gpu'] = idx
                d[NODE_COL] = node
                per_node[node] = idx + 1
        else:
            for i, d in enumerate(dfs):
                d['gpu'] = i
        df = pd.concat(dfs, ignore_index=True)

    df = df.sort_values('ts').reset_index(drop=True)
    df.to_pickle(output)
    t1 = time.time()

    logger.info(f"Wrote {len(df)} kernels to {output} in {t1-t0:.2f}s")
    if node_labels:
        logger.info(f"Nodes: {nodes_in(df)}")
        logger.warning(
            "kernel timestamps from different nodes are in different clock "
            "domains; convert them with chopper.profile.telemetry.clock_anchor "
            "before comparing across nodes")
    logger.info(f"Columns: {list(df.columns)}")


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description=(
        "Merge PyTorch traces into a kernel pickle, combine pickles, "
        "join hardware counters, or merge device sampling data.\n"
        "  1) -t trace*.json -o ts.pkl                       (parse traces)\n"
        "  2) -p iter*.pkl -o ts.pkl                          (combine pickles)\n"
        "  3) -p ts.pkl -c batch0/*.csv -o out.pkl            (add counters)\n"
        "  4) --device-dir outputs/run -o device.pkl           (raw device CSVs)\n"
        "  5) --device-dir outputs/run -p ts.pkl -o merged.pkl (device + traces)\n"
        "  6) --cpu-pkl cpu.pkl --kernel-csv kt.csv -o tl.pkl  (CPU+GPU timeline)\n"
    ))
    parser.add_argument('-t', '--traces', nargs='+')
    parser.add_argument('-p', '--pickles', nargs='+')
    parser.add_argument('-c', '--counters', action='append', nargs='+',
                        help='Counter CSV files (one -c per batch, sorted = GPU order)')
    parser.add_argument('--device-dir',
                        help='Device sampling output directory (chopper --device)')
    parser.add_argument('--cpu-pkl',
                        help='cpu.pkl (collect with --cpu-clock rocprofiler) for CPU+GPU timeline')
    parser.add_argument('--kernel-csv',
                        help='Device sampler kernel_traces.csv for CPU+GPU timeline')
    parser.add_argument('--bin-ms', type=int, default=10,
                        help='Time bin size for CPU+GPU timeline (default 10ms)')
    parser.add_argument('--counter-nodes', nargs='+',
                        help='Node (hostname) each counter device belongs to, in '
                             'sorted-file order. Pass one label for all devices or '
                             'one per device. Defaults to reading the node from the '
                             'per-node output directory in each file path.')
    parser.add_argument('--trace-nodes', nargs='+',
                        help="Node each trace file belongs to. Pass 'auto' to use "
                             "each trace's parent directory name, one label for all "
                             "traces, or one label per trace. Required when the "
                             "traces come from more than one directory.")
    parser.add_argument('-o', '--output', required=True)
    args = parser.parse_args()
    main(sorted(args.traces) if args.traces else None,
         sorted(args.pickles) if args.pickles else None,
         args.counters,
         args.device_dir,
         args.output,
         args.cpu_pkl,
         args.kernel_csv,
         args.bin_ms,
         args.counter_nodes,
         args.trace_nodes)
