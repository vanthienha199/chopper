"""Full trace merge: parse PyTorch Chrome traces into a kernel DataFrame."""
import json
import time
import re
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from loguru import logger


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


def merge_counters(df_ts, counter_batches):
    """Join hardware counter data with trace pickle.

    Args:
        df_ts: Trace DataFrame (from ts.pkl)
        counter_batches: List of lists -- each inner list is one batch of
            CSV files (sorted = GPU order). E.g.:
            [["batch0/gpu0.csv", "batch0/gpu1.csv"],
             ["batch1/gpu0.csv", "batch1/gpu1.csv"]]
    """
    # Transpose from per-batch to per-GPU
    per_gpu = [list(fns) for fns in zip(*[sorted(b) for b in counter_batches])]
    n_counter_gpus = len(per_gpu)

    gpus = sorted(df_ts["gpu"].unique())
    if len(gpus) != n_counter_gpus:
        logger.warning(f"{len(gpus)} GPUs in trace but {n_counter_gpus} counter files")

    # Load and combine counters per GPU
    df_cntr = pd.concat(
        [df.assign(gpu=i) for i, df in enumerate(
            map(get_combined_counters, per_gpu))],
        ignore_index=True,
    )
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

    # Assign reverse-cumcount _mi (count from end so late dispatches align)
    left_keys = ["name", "gpu"]
    right_keys = [kname, "gpu"]

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


def merge_device_with_traces(device_dir, trace_pkl, output):
    """Merge device counter samples with PyTorch trace annotations.

    Each counter group keeps its own runtime kernel trace and samples.
    Annotations (operator-name, layer, iteration) are stolen from ts.pkl
    via reverse-cumcount matching.

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

    # Prepare ts.pkl reverse cumcount per GPU (done once)
    ts_by_gpu = {}
    for gpu in gpus:
        ts_gpu = df_last[df_last["gpu"] == gpu].copy()
        ts_gpu["_mi"] = (
            ts_gpu.iloc[::-1].groupby("name").cumcount().iloc[::-1]
        )
        ts_by_gpu[gpu] = ts_gpu

    output_groups = {}
    counter_to_group = {}

    for gi, group_dir in enumerate(group_dirs):
        counter_files = sorted(group_dir.glob("counter_samples_rank*.csv"))
        trace_files = sorted(group_dir.glob("kernel_traces_rank*.csv"))
        assert counter_files, f"No counter_samples_rank*.csv in {group_dir}"
        assert trace_files, f"No kernel_traces_rank*.csv in {group_dir}"

        all_kernels = []
        all_samples = []
        group_counter_names = None

        for gpu in gpus:
            counter_file = group_dir / f"counter_samples_rank{gpu}.csv"
            trace_file = group_dir / f"kernel_traces_rank{gpu}.csv"
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

            logger.info(f"Group {gi}, GPU {gpu}: {len(samples_piv)} samples, "
                        f"{len(runtime_df)} runtime kernels")

            # Annotate runtime kernels from ts.pkl
            kernel_rows = runtime_df.rename(columns={
                "kernel_name": "name",
                "start_ns": "ts",
                "duration_ns": "dur",
            }).copy()
            kernel_rows["gpu"] = gpu

            # Reverse cumcount match to steal annotations
            ts_gpu = ts_by_gpu[gpu]
            ts_names = set(ts_gpu["name"])
            rt_names = set(kernel_rows["name"])
            ts_names & rt_names
            rt_only = rt_names - ts_names
            if rt_only:
                logger.warning(f"  Group {gi}, GPU {gpu}: {len(rt_only)} kernels not in ts.pkl")

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
        df_samples = df_samples.sort_values(["gpu", "timestamp_ns"]).reset_index(drop=True)

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
    logger.info(f"  {len(output_groups)} groups, {len(gpus)} GPUs")
    logger.info(f"  Counter -> group: {counter_to_group}")


def merge_device_counters(device_dir, output):
    """Store device sampling CSVs into a pickle. No transformations.

    Reads counter_samples_rank*.csv and kernel_traces_rank*.csv from
    chopper_device_counters*/ subdirs.
    """
    from pathlib import Path

    device_dir = Path(device_dir)
    groups = sorted(device_dir.glob("chopper_device_counters*"))
    assert groups, f"No chopper_device_counters* dirs in {device_dir}"

    counter_samples = {}
    kernel_traces = {}
    counter_to_group = {}  # {counter_name: group_index}

    for gi, group_dir in enumerate(groups):
        counter_files = sorted(group_dir.glob("counter_samples_rank*.csv"))
        trace_files = sorted(group_dir.glob("kernel_traces_rank*.csv"))
        assert counter_files, f"No counter_samples_rank*.csv in {group_dir}"
        assert trace_files, f"No kernel_traces_rank*.csv in {group_dir}"

        for cf in counter_files:
            rank = int(cf.stem.split("_rank")[1])
            df = pd.read_csv(cf)
            assert len(df) > 0, f"Empty counter file: {cf}"
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


def main(traces, pickles, counters, device_dir, output,
         cpu_pkl=None, kernel_csv=None, bin_ms=10):
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
        df = merge_counters(df, counters)
        df.to_pickle(output)
        t1 = time.time()
        logger.info(f"Merged counters into {len(df)} kernels -> {output} in {t1-t0:.2f}s")
        return

    if pickles:
        dfs = [pd.read_pickle(p) for p in pickles]
        df = pd.concat(dfs, ignore_index=True)
        df = df.sort_values('ts').reset_index(drop=True)
        df.to_pickle(output)
        t1 = time.time()
        logger.info(f"Merged {len(pickles)} pickles, {len(df)} kernels -> {output} in {t1-t0:.2f}s")
        return

    if len(traces) == 1:
        df = parse_trace(traces[0])
        df['gpu'] = 0
    else:
        with ProcessPoolExecutor(max_workers=len(traces)) as ex:
            dfs = list(ex.map(parse_trace, traces))
        for i, d in enumerate(dfs):
            d['gpu'] = i
        df = pd.concat(dfs, ignore_index=True)

    df = df.sort_values('ts').reset_index(drop=True)
    df.to_pickle(output)
    t1 = time.time()

    logger.info(f"Wrote {len(df)} kernels to {output} in {t1-t0:.2f}s")
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
    parser.add_argument('-o', '--output', required=True)
    args = parser.parse_args()
    main(sorted(args.traces) if args.traces else None,
         sorted(args.pickles) if args.pickles else None,
         args.counters,
         args.device_dir,
         args.output,
         args.cpu_pkl,
         args.kernel_csv,
         args.bin_ms)
