// CUPTI Activity kernel tracer for Chopper (NVIDIA side).
//
// The NVIDIA analog of device_counters_tool.cpp's kernel-dispatch tracing:
// records each GPU kernel's name + start/end timestamp (in CUPTI's clock, the
// same domain as cuptiGetTimestamp) to a CSV. Loaded via CUDA_INJECTION64_PATH
// so no app changes are needed. Output columns match the AMD kernel_traces.csv
// (kernel_name,start_ns,end_ns,duration_ns) so merge_cpu_gpu_timeline works
// unchanged.
#include <cupti.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cxxabi.h>
#include <fstream>
#include <mutex>
#include <string>

namespace {
std::ofstream* g_out = nullptr;
std::mutex g_mtx;
std::atomic<uint64_t> g_count{0};

#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buf, align) \
    (((uintptr_t)(buf) & ((align) - 1)) \
         ? ((buf) + (align) - ((uintptr_t)(buf) & ((align) - 1))) \
         : (buf))

std::string demangle(const char* name) {
    if (!name) return "??";
    int status = 0;
    char* d = abi::__cxa_demangle(name, nullptr, nullptr, &status);
    std::string r = (status == 0 && d) ? std::string(d) : std::string(name);
    free(d);
    return r;
}

void CUPTIAPI bufferRequested(uint8_t** buffer, size_t* size, size_t* maxNumRecords) {
    size_t s = 8 * 1024 * 1024;
    uint8_t* b = (uint8_t*)malloc(s + ALIGN_SIZE);
    *buffer = ALIGN_BUFFER(b, ALIGN_SIZE);
    *size = s;
    *maxNumRecords = 0;
}

void CUPTIAPI bufferCompleted(CUcontext, uint32_t, uint8_t* buffer,
                              size_t, size_t validSize) {
    std::lock_guard<std::mutex> lk(g_mtx);
    CUpti_Activity* record = nullptr;
    while (true) {
        CUptiResult status = cuptiActivityGetNextRecord(buffer, validSize, &record);
        if (status == CUPTI_SUCCESS) {
            if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL ||
                record->kind == CUPTI_ACTIVITY_KIND_KERNEL) {
                auto* k = (CUpti_ActivityKernel9*)record;
                if (g_out) {
                    (*g_out) << "\"" << demangle(k->name) << "\","
                             << k->start << "," << k->end << ","
                             << (k->end - k->start) << "\n";
                    g_count++;
                }
            }
        } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
            break;
        } else {
            break;
        }
    }
    free(buffer);
}

void finalize() {
    cuptiActivityFlushAll(1);
    if (g_out) {
        g_out->flush();
        fprintf(stderr, "[cupti-trace] wrote %llu kernel records\n",
                (unsigned long long)g_count.load());
    }
}
}  // namespace

extern "C" int InitializeInjection(void) {
    const char* path = getenv("CHOPPER_NV_TRACE_OUTPUT");
    g_out = new std::ofstream(path ? path : "kernel_traces.csv");
    (*g_out) << "kernel_name,start_ns,end_ns,duration_ns\n";
    cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    atexit(finalize);
    fprintf(stderr, "[cupti-trace] injection initialized\n");
    return 1;
}
