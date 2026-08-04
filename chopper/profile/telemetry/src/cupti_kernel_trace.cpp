// CUPTI Activity kernel tracer for Chopper (NVIDIA side).
//
// The NVIDIA analog of device_counters_tool.cpp's kernel-dispatch tracing:
// records each GPU kernel's name + start/end timestamp (in CUPTI's clock, the
// same domain as cuptiGetTimestamp) to a CSV. Loaded via CUDA_INJECTION64_PATH
// so no app changes are needed. Output columns match the AMD kernel_traces.csv
// (kernel_name,start_ns,end_ns,duration_ns) so merge_cpu_gpu_timeline works
// unchanged.
//
// Safety rules (an injected library shares the app's address space, so it
// must fail SAFE, never fatal):
//   - The kernel record is cast to the CUpti_ActivityKernel struct of the
//     headers this file was COMPILED against. If the CUPTI loaded at runtime
//     is a different major version, that cast reads garbage (a bad `name`
//     pointer crashed vLLM's EngineCore on its first inference). So if the
//     runtime CUPTI version differs from the compile-time one, tracing is
//     disabled with a warning instead of enabled with a landmine.
//   - Every CUPTI call result is checked; failure disables tracing, never
//     aborts the app.
//   - Multi-process servers (e.g. vLLM tensor parallel) inject this into
//     every worker; with CHOPPER_NV_TRACE_PER_PID=1 each process writes
//     <output>.<pid>.csv instead of all truncating one file.
#include <cupti.h>
#include <cupti_version.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cxxabi.h>
#include <fstream>
#include <mutex>
#include <string>
#include <unistd.h>

namespace {
std::ofstream* g_out = nullptr;
std::mutex g_mtx;
std::atomic<uint64_t> g_count{0};
bool g_active = false;

std::string demangle(const char* name) {
    if (!name) return "??";
    int status = 0;
    char* d = abi::__cxa_demangle(name, nullptr, nullptr, &status);
    std::string r = (status == 0 && d) ? std::string(d) : std::string(name);
    free(d);
    return r;
}

void CUPTIAPI bufferRequested(uint8_t** buffer, size_t* size, size_t* maxNumRecords) {
    // CUPTI requires 8-byte alignment; malloc already guarantees >= 16 on
    // glibc, so hand the pointer over as-is (an aligned-offset copy here made
    // the free() in bufferCompleted undefined behavior).
    //
    // Small buffer on purpose: a record's kernel-name pointer references the
    // owning module's name table, and JIT engines (Triton) load and replace
    // modules mid-run. The longer records sit undelivered, the wider the
    // window in which a module unload leaves pending name pointers dangling
    // (native EngineCore death on DeltaAI GH200, first seen right as Triton
    // JIT-compiled new kernels mid-inference). Small buffers + a periodic
    // flush keep that window near zero.
    size_t s = 1 * 1024 * 1024;
    *buffer = (uint8_t*)malloc(s);
    *size = s;
    *maxNumRecords = 0;
}

std::string csv_escape(const std::string& s) {
    // Triton emits Python-derived kernel names (dots, <locals>, and
    // potentially quotes); embedded quotes would break the CSV quoting.
    std::string r;
    r.reserve(s.size());
    for (char c : s) {
        if (c == '"') r += "\"\"";
        else r += c;
    }
    return r;
}

void CUPTIAPI bufferCompleted(CUcontext, uint32_t, uint8_t* buffer,
                              size_t, size_t validSize) {
    std::lock_guard<std::mutex> lk(g_mtx);
    CUpti_Activity* record = nullptr;
    while (true) {
        CUptiResult status = cuptiActivityGetNextRecord(buffer, validSize, &record);
        if (status != CUPTI_SUCCESS) break;  // includes MAX_LIMIT_REACHED
        if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL ||
            record->kind == CUPTI_ACTIVITY_KIND_KERNEL) {
            // Struct version matching guaranteed by the version gate in
            // InitializeInjection.
            auto* k = (CUpti_ActivityKernel9*)record;
            if (g_out) {
                (*g_out) << "\"" << csv_escape(demangle(k->name)) << "\","
                         << k->start << "," << k->end << ","
                         << (k->end - k->start) << "\n";
                g_count++;
            }
        }
#if CUPTI_API_VERSION >= 16
        else if (record->kind == CUPTI_ACTIVITY_KIND_GRAPH_TRACE) {
            // One record per CUDA graph launch instead of per kernel node.
            // Graph-level start/end is what the GPU-busy timeline needs, and
            // it avoids per-node instrumentation of graphs entirely (a driver
            // segfault in cuGraphLaunch on DeltaAI GH200 when vLLM captured
            // new graphs mid-inference while per-kernel tracing was active).
            auto* g = (CUpti_ActivityGraphTrace*)record;
            if (g_out) {
                (*g_out) << "\"[cuda_graph " << g->graphId << "]\","
                         << g->start << "," << g->end << ","
                         << (g->end - g->start) << "\n";
                g_count++;
            }
        }
#endif
    }
    free(buffer);
}

void finalize() {
    if (!g_active) return;
    cuptiActivityFlushAll(1);
    if (g_out) {
        g_out->flush();
        fprintf(stderr, "[cupti-trace] pid %d wrote %llu kernel records\n",
                (int)getpid(), (unsigned long long)g_count.load());
    }
}
}  // namespace

extern "C" int InitializeInjection(void) {
    // Version gate: refuse to trace when the runtime CUPTI is not the one we
    // compiled against. The activity record layout is version-specific and a
    // mismatched cast crashes the host application.
    uint32_t rt_version = 0;
    if (cuptiGetVersion(&rt_version) != CUPTI_SUCCESS) {
        fprintf(stderr, "[cupti-trace] cannot read CUPTI version; tracing disabled\n");
        return 1;
    }
    if (rt_version != CUPTI_API_VERSION) {
        fprintf(stderr,
                "[cupti-trace] runtime CUPTI version %u != compiled-against %u; "
                "tracing DISABLED to avoid corrupting the host app. Rebuild the "
                "tracer against this toolkit's cupti headers.\n",
                rt_version, (unsigned)CUPTI_API_VERSION);
        return 1;
    }

    const char* path = getenv("CHOPPER_NV_TRACE_OUTPUT");
    std::string out = path ? path : "kernel_traces.csv";
    const char* per_pid = getenv("CHOPPER_NV_TRACE_PER_PID");
    if (per_pid && per_pid[0] == '1') {
        // vLLM-style multi-process servers inject this into every worker;
        // give each process its own file instead of truncating one another's.
        size_t dot = out.rfind(".csv");
        std::string stem = (dot == std::string::npos) ? out : out.substr(0, dot);
        out = stem + "." + std::to_string((int)getpid()) + ".csv";
    }
    g_out = new std::ofstream(out);
    if (!g_out->is_open()) {
        fprintf(stderr, "[cupti-trace] cannot open %s; tracing disabled\n", out.c_str());
        return 1;
    }
    (*g_out) << "kernel_name,start_ns,end_ns,duration_ns\n";

    if (cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted) != CUPTI_SUCCESS) {
        fprintf(stderr, "[cupti-trace] callback registration failed; tracing disabled\n");
        return 1;
    }
    if (cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) != CUPTI_SUCCESS) {
        fprintf(stderr, "[cupti-trace] activity enable failed; tracing disabled\n");
        return 1;
    }
#if CUPTI_API_VERSION >= 16
    // Graph-launched kernels are traced as ONE record per graph launch by
    // default, not per kernel node. Per-node instrumentation of graphs
    // segfaulted the CUDA driver (cuGraphLaunch, DeltaAI GH200, CUPTI 28)
    // when vLLM captured new graphs mid-inference after a Triton JIT compile.
    // Eager (non-graph) kernels are still traced individually. Set
    // CHOPPER_NV_PER_KERNEL_GRAPHS=1 to opt back into per-node tracing.
    const char* pk = getenv("CHOPPER_NV_PER_KERNEL_GRAPHS");
    if (!(pk && pk[0] == '1')) {
        if (cuptiActivityEnable(CUPTI_ACTIVITY_KIND_GRAPH_TRACE) != CUPTI_SUCCESS) {
            fprintf(stderr, "[cupti-trace] note: graph-trace unavailable, "
                            "graph kernels traced per node\n");
        } else {
            fprintf(stderr, "[cupti-trace] graph launches traced as single records "
                            "(CHOPPER_NV_PER_KERNEL_GRAPHS=1 to override)\n");
        }
    }
#endif
    // Deliver records promptly (default is only-when-full). Keeps pending
    // records from outliving JIT module reloads; see bufferRequested.
    if (cuptiActivityFlushPeriod(500) != CUPTI_SUCCESS) {
        fprintf(stderr, "[cupti-trace] note: periodic flush unavailable, using buffer-full delivery\n");
    }
    g_active = true;
    atexit(finalize);
    fprintf(stderr, "[cupti-trace] pid %d injection initialized (CUPTI %u) -> %s\n",
            (int)getpid(), rt_version, out.c_str());
    return 1;
}
