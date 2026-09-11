// Device counter sampling + kernel dispatch tracing tool.
//
// Uses two rocprofiler-sdk contexts to avoid GPU hangs during init:
//
//   1. Tracing context (g_ctx): started immediately. Handles kernel dispatch
//      tracing and code object callbacks. This is safe during HSA/NCCL init
//      because it only installs callbacks -- no PMC hardware access.
//
//   2. Counter context (g_counter_ctx): started ONLY after the first kernel
//      dispatch is observed via the tracing context. Starting the device
//      counting service configures PMC hardware, which hangs the GPU if done
//      during HSA or NCCL initialization. Waiting for the first dispatch
//      guarantees that init is complete.
//
// IMPORTANT: counter samples will not include the very first kernel dispatches
// (those that occur before the counter context starts). Kernel dispatch traces
// are captured from the start, but PMC counter data begins ~100ms after the
// first kernel is seen.
//
// Loaded via LD_PRELOAD. Outputs kernel_traces.csv and counter_samples.csv.
//
// Env var configuration:
//   CHOPPER_COUNTERS       - comma-separated counter names (default: SQ_WAVES)
//   CHOPPER_SAMPLE_MS      - sampling interval in ms (default: 1)
//   CHOPPER_COUNTER_OUTPUT - counter CSV path (default: counter_samples.csv)
//   CHOPPER_TRACE_OUTPUT   - trace CSV path (default: kernel_traces.csv)
//   CHOPPER_TRACE_ONLY     - if set, collect dispatch traces only (no device counters)
//
// Rank semantics (multi-node):
//   LOCAL_RANK  - selects which GPU on THIS node to profile. Repeats on every
//                 node, so it must never appear in an output filename.
//   RANK        - global rank across the job (torchrun). Used for the output
//                 filename suffix so two nodes cannot clobber each other.
//   SLURM_PROCID- fallback global rank when torchrun is not in use.
// Each process also writes rank_info_rank<GLOBAL>.csv naming its node, global
// rank and local rank, so the reader can map a file back to a physical GPU.

#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cxxabi.h>
#include <fstream>
#include <iostream>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <unistd.h>  // gethostname, for the rank_info sidecar

#define ROCPROFILER_CALL(result, msg)                                                    \
    {                                                                                    \
        rocprofiler_status_t CHECKSTATUS = result;                                       \
        if(CHECKSTATUS != ROCPROFILER_STATUS_SUCCESS)                                    \
        {                                                                                \
            const char* status_msg = rocprofiler_get_status_string(CHECKSTATUS);         \
            std::cerr << "[tool] " << msg << " failed with status " << CHECKSTATUS       \
                      << " (" << (status_msg ? status_msg : "unknown") << ")\n";         \
            std::abort();                                                                \
        }                                                                                \
    }

namespace
{

// Demangle a kernel name, stripping the .kd suffix first.
// Same approach as rocprofiler-sdk's code_object_tracing sample.
std::string
demangle_kernel_name(const char* name)
{
    auto mangled = std::string{name};
    // Strip .kd suffix
    if(mangled.size() > 3 && mangled.substr(mangled.size() - 3) == ".kd")
        mangled = mangled.substr(0, mangled.size() - 3);

    int   status     = 0;
    char* demangled   = abi::__cxa_demangle(mangled.c_str(), nullptr, nullptr, &status);
    auto  result      = (status == 0 && demangled) ? std::string{demangled} : mangled;
    ::free(demangled);
    return result;
}

//
// Kernel symbol tracking (from api_buffered_tracing example)
//
using kernel_symbol_data_t = rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t;
using kernel_symbol_map_t  = std::unordered_map<rocprofiler_kernel_id_t, kernel_symbol_data_t>;

kernel_symbol_map_t client_kernels   = {};
std::mutex          client_kernels_mutex;

//
// Kernel trace records collected in buffer callback
//
struct kernel_record_t
{
    std::string kernel_name;
    uint64_t    start_ns;
    uint64_t    end_ns;
    uint64_t    agent_id;
    uint64_t    queue_id;
    uint64_t    correlation_id;
};
std::vector<kernel_record_t> g_kernel_records;
std::mutex                   g_kernel_records_mutex;

//
// Counter dimension metadata (per counter_id)
//
struct dim_info_t
{
    std::string                        name;
    rocprofiler_counter_dimension_id_t id;
    size_t                             size;
};

struct counter_dim_info_t
{
    std::string             name;
    std::vector<dim_info_t> dims;
};
std::unordered_map<uint64_t, counter_dim_info_t> g_counter_dims;
std::vector<std::string>                         g_all_dim_names;

//
// Counter sample records
//
struct counter_sample_t
{
    uint64_t                                timestamp_ns;
    std::string                             counter_name;
    double                                  value;
    uint64_t                                agent_id;
    std::unordered_map<std::string, size_t> dim_positions;
};
std::vector<counter_sample_t> g_counter_samples;
std::mutex                    g_counter_samples_mutex;

//
// Shared state
//
rocprofiler_context_id_t g_ctx{0};         // dispatch tracing (started immediately)
rocprofiler_context_id_t g_counter_ctx{0}; // device counting (deferred until first dispatch)
rocprofiler_buffer_id_t  g_trace_buffer{};
rocprofiler_buffer_id_t  g_counter_buffer{};
std::atomic<bool>        g_exit{false};
std::atomic<bool>        g_thread_started{false};
std::atomic<bool>        g_first_dispatch{false}; // set by buffer callback on first kernel
int                      g_sample_interval_ms = 1;
bool                     g_trace_only = false;    // skip device counting, keep dispatch traces

std::unordered_map<uint64_t, rocprofiler_counter_config_id_t> g_profile_cache;

//
// Code object callback -- tracks kernel symbols for name resolution.
// Flushes the trace buffer before code objects unload so kernel name
// pointers remain valid.
//
void tool_code_object_callback(rocprofiler_callback_tracing_record_t record,
                               rocprofiler_user_data_t*              user_data,
                               void*                                 callback_data)
{
    if(record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT
       && record.operation == ROCPROFILER_CODE_OBJECT_LOAD)
    {
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_UNLOAD)
        {
            auto flush_status = rocprofiler_flush_buffer(g_trace_buffer);
            if(flush_status != ROCPROFILER_STATUS_ERROR_BUFFER_BUSY)
            {
                (void)flush_status;
            }
        }
    }
    else if(record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT
            && record.operation == ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER)
    {
        auto* data = static_cast<kernel_symbol_data_t*>(record.payload);
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_LOAD)
        {
            std::lock_guard<std::mutex> lk(client_kernels_mutex);
            client_kernels.emplace(data->kernel_id, *data);
        }
    }

    (void)user_data;
    (void)callback_data;
}

//
// Buffer callback -- handles both kernel dispatch records and counter records.
// Sets g_first_dispatch on the first kernel dispatch, which unblocks the
// sampling thread to start the counter context.
//
void tool_buffer_callback(rocprofiler_context_id_t,
                          rocprofiler_buffer_id_t,
                          rocprofiler_record_header_t** headers,
                          size_t                        num_headers,
                          void*,
                          uint64_t)
{
    for(size_t i = 0; i < num_headers; ++i)
    {
        auto* header = headers[i];

        if(header->category == ROCPROFILER_BUFFER_CATEGORY_TRACING
           && header->kind == ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH)
        {
            auto* record = static_cast<rocprofiler_buffer_tracing_kernel_dispatch_record_t*>(
                header->payload);

            auto kernel_id   = record->dispatch_info.kernel_id;
            std::string kernel_name = "??";
            {
                std::lock_guard<std::mutex> lk(client_kernels_mutex);
                auto it = client_kernels.find(kernel_id);
                if(it != client_kernels.end())
                    kernel_name = demangle_kernel_name(it->second.kernel_name);
            }

            // Signal that GPU init is complete -- safe to start PMC sampling
            g_first_dispatch.store(true);

            std::lock_guard<std::mutex> lk(g_kernel_records_mutex);
            g_kernel_records.push_back({std::move(kernel_name),
                                        record->start_timestamp,
                                        record->end_timestamp,
                                        record->dispatch_info.agent_id.handle,
                                        record->dispatch_info.queue_id.handle,
                                        record->correlation_id.internal});
        }
        else if(header->category == ROCPROFILER_BUFFER_CATEGORY_COUNTERS
                && header->kind == ROCPROFILER_COUNTER_RECORD_PROFILE_COUNTING_DISPATCH_HEADER)
        {
            // skip dispatch headers
        }
        else if(header->category == ROCPROFILER_BUFFER_CATEGORY_COUNTERS
                && header->kind == ROCPROFILER_COUNTER_RECORD_VALUE)
        {
            auto* record = static_cast<rocprofiler_counter_record_t*>(header->payload);

            rocprofiler_counter_id_t counter_id = {.handle = 0};
            rocprofiler_query_record_counter_id(record->id, &counter_id);

            std::string counter_name = "??";
            std::unordered_map<std::string, size_t> positions;

            auto it = g_counter_dims.find(counter_id.handle);
            if(it != g_counter_dims.end())
            {
                counter_name = it->second.name;
                for(const auto& dim : it->second.dims)
                {
                    size_t pos = 0;
                    rocprofiler_query_record_dimension_position(record->id, dim.id, &pos);
                    positions[dim.name] = pos;
                }
            }

            std::lock_guard<std::mutex> lk(g_counter_samples_mutex);
            g_counter_samples.push_back({record->user_data.value,
                                         counter_name,
                                         record->counter_value,
                                         record->agent_id.handle,
                                         std::move(positions)});
        }
    }
}

//
// Device counting: set_profile callback
//
void set_profile(rocprofiler_context_id_t               context_id,
                 rocprofiler_agent_id_t                 agent,
                 rocprofiler_device_counting_agent_cb_t set_config,
                 void*)
{
    auto pos = g_profile_cache.find(agent.handle);
    assert(pos != g_profile_cache.end());
    set_config(context_id, pos->second);
}

//
// Build counter profile for an agent. Reads CHOPPER_COUNTERS env var
// and discovers counter dimensions for CSV output.
//
rocprofiler_counter_config_id_t build_profile_for_agent(rocprofiler_agent_id_t agent)
{
    std::set<std::string> counters_to_collect = {"SQ_WAVES"};

    if(auto* env = std::getenv("CHOPPER_COUNTERS"); env)
    {
        std::istringstream ss(env);
        std::string        token;
        counters_to_collect.clear();
        while(std::getline(ss, token, ','))
        {
            if(!token.empty()) counters_to_collect.insert(token);
        }
    }

    std::vector<rocprofiler_counter_id_t> gpu_counters;
    ROCPROFILER_CALL(rocprofiler_iterate_agent_supported_counters(
                         agent,
                         [](rocprofiler_agent_id_t,
                            rocprofiler_counter_id_t* counters,
                            size_t                    num_counters,
                            void*                     user_data)
                         {
                             auto* vec = static_cast<std::vector<rocprofiler_counter_id_t>*>(user_data);
                             for(size_t i = 0; i < num_counters; i++)
                                 vec->push_back(counters[i]);
                             return ROCPROFILER_STATUS_SUCCESS;
                         },
                         static_cast<void*>(&gpu_counters)),
                     "Could not fetch supported counters");

    std::vector<rocprofiler_counter_id_t> collect_counters;
    for(auto& counter : gpu_counters)
    {
        rocprofiler_counter_info_v0_t info;
        ROCPROFILER_CALL(rocprofiler_query_counter_info(counter,
                                                        ROCPROFILER_COUNTER_INFO_VERSION_0,
                                                        static_cast<void*>(&info)),
                         "Could not query info for counter");
        if(counters_to_collect.count(std::string(info.name)) > 0)
        {
            std::clog << "[tool] Collecting counter: " << info.name << "\n";
            collect_counters.push_back(counter);
        }
    }

    for(auto& counter : collect_counters)
    {
        rocprofiler_counter_info_v1_t info_v1;
        info_v1.size = sizeof(rocprofiler_counter_info_v1_t);
        auto status = rocprofiler_query_counter_info(
            counter, ROCPROFILER_COUNTER_INFO_VERSION_1, static_cast<void*>(&info_v1));
        if(status == ROCPROFILER_STATUS_SUCCESS)
        {
            counter_dim_info_t cdi;
            cdi.name = info_v1.name;
            for(uint64_t d = 0; d < info_v1.dimensions_count; d++)
            {
                std::string dname = info_v1.dimensions[d]->name;
                cdi.dims.push_back({dname,
                                    info_v1.dimensions[d]->id,
                                    info_v1.dimensions[d]->instance_size});
                bool found = false;
                for(const auto& existing : g_all_dim_names)
                    if(existing == dname) { found = true; break; }
                if(!found)
                    g_all_dim_names.push_back(dname);
            }
            g_counter_dims[counter.handle] = std::move(cdi);
        }
    }

    rocprofiler_counter_config_id_t profile = {.handle = 0};
    ROCPROFILER_CALL(rocprofiler_create_counter_config(agent,
                                                       collect_counters.data(),
                                                       collect_counters.size(),
                                                       &profile),
                     "Could not construct profile cfg");

    return profile;
}

//
// tool_init -- sets up two contexts:
//   g_ctx:         dispatch tracing, started immediately
//   g_counter_ctx: device counting, started after first kernel dispatch
//
int tool_init(rocprofiler_client_finalize_t, void*)
{
    if(auto* env = std::getenv("CHOPPER_SAMPLE_MS"); env)
        g_sample_interval_ms = std::atoi(env);

    g_trace_only = (std::getenv("CHOPPER_TRACE_ONLY") != nullptr);

    std::clog << "[tool] Initializing with sample interval " << g_sample_interval_ms << " ms"
              << (g_trace_only ? " (trace-only mode, no device counters)" : "") << "\n";

    // --- Context 1: dispatch tracing (safe to start immediately) ---
    ROCPROFILER_CALL(rocprofiler_create_context(&g_ctx), "tracing context creation");

    auto code_object_ops = std::vector<rocprofiler_tracing_operation_t>{
        ROCPROFILER_CODE_OBJECT_LOAD,
        ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER};

    ROCPROFILER_CALL(
        rocprofiler_configure_callback_tracing_service(g_ctx,
                                                       ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
                                                       code_object_ops.data(),
                                                       code_object_ops.size(),
                                                       tool_code_object_callback,
                                                       nullptr),
        "code object tracing service configure");

    ROCPROFILER_CALL(rocprofiler_create_buffer(g_ctx,
                                               16384,
                                               16384 - 2048,
                                               ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                               tool_buffer_callback,
                                               nullptr,
                                               &g_trace_buffer),
                     "trace buffer creation");

    ROCPROFILER_CALL(
        rocprofiler_configure_buffer_tracing_service(g_ctx,
                                                     ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
                                                     nullptr,
                                                     0,
                                                     g_trace_buffer),
        "buffer tracing service for kernel dispatch configure");

    ROCPROFILER_CALL(rocprofiler_start_context(g_ctx), "start tracing context");

    auto client_thread = rocprofiler_callback_thread_t{};
    ROCPROFILER_CALL(rocprofiler_create_callback_thread(&client_thread),
                     "failure creating callback thread");
    ROCPROFILER_CALL(rocprofiler_assign_callback_thread(g_trace_buffer, client_thread),
                     "failed to assign thread for trace buffer");

    if(!g_trace_only)
    {
        // --- Context 2: device counting (deferred start) ---
        // Starting the device counting context configures PMC hardware on the GPU.
        // If done during HSA or NCCL initialization, this causes a GPU hang.
        // We defer this until the first kernel dispatch is observed, which
        // guarantees that GPU init is complete.
        ROCPROFILER_CALL(rocprofiler_create_context(&g_counter_ctx), "counter context creation");

        ROCPROFILER_CALL(rocprofiler_create_buffer(g_counter_ctx,
                                                   4096,
                                                   2048,
                                                   ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                                   tool_buffer_callback,
                                                   nullptr,
                                                   &g_counter_buffer),
                         "counter buffer creation");

        // Discover GPU agents and select the one matching LOCAL_RANK
        std::vector<rocprofiler_agent_v0_t>     agents;
        rocprofiler_query_available_agents_cb_t iterate_cb =
            [](rocprofiler_agent_version_t agents_ver,
               const void**                agents_arr,
               size_t                      num_agents,
               void*                       udata)
        {
            if(agents_ver != ROCPROFILER_AGENT_INFO_VERSION_0)
                throw std::runtime_error{"unexpected rocprofiler agent version"};
            auto* agents_v = static_cast<std::vector<rocprofiler_agent_v0_t>*>(udata);
            for(size_t i = 0; i < num_agents; ++i)
                agents_v->emplace_back(
                    *static_cast<const rocprofiler_agent_v0_t*>(agents_arr[i]));
            return ROCPROFILER_STATUS_SUCCESS;
        };

        ROCPROFILER_CALL(rocprofiler_query_available_agents(
                             ROCPROFILER_AGENT_INFO_VERSION_0,
                             iterate_cb,
                             sizeof(rocprofiler_agent_t),
                             const_cast<void*>(static_cast<const void*>(&agents))),
                         "query available agents");

        // LOCAL_RANK is the right variable HERE: it indexes the GPUs visible
        // on this node, which is exactly what the agent list enumerates. It
        // is the wrong variable for output filenames (see rank_suffix below),
        // because it restarts at 0 on every node of a multi-node job.
        int target_gpu_index = 0;
        if(auto* lr = std::getenv("LOCAL_RANK"); lr)
            target_gpu_index = std::atoi(lr);

        rocprofiler_agent_id_t target_agent = {.handle = 0};
        int gpu_index = 0;
        for(const auto& agent : agents)
        {
            if(agent.type == ROCPROFILER_AGENT_TYPE_GPU)
            {
                if(gpu_index == target_gpu_index)
                {
                    g_profile_cache.emplace(agent.id.handle,
                                            build_profile_for_agent(agent.id));
                    target_agent = agent.id;
                    std::clog << "[tool] rank " << target_gpu_index
                              << " using GPU agent: " << agent.id.handle << "\n";
                    break;
                }
                gpu_index++;
            }
        }

        if(target_agent.handle == 0)
        {
            std::cerr << "[tool] No GPU agents found\n";
            return 1;
        }

        ROCPROFILER_CALL(rocprofiler_configure_device_counting_service(g_counter_ctx,
                                                                       g_counter_buffer,
                                                                       target_agent,
                                                                       set_profile,
                                                                       nullptr),
                         "Could not setup device counting service");

        ROCPROFILER_CALL(rocprofiler_assign_callback_thread(g_counter_buffer, client_thread),
                         "failed to assign thread for counter buffer");
    }

    // --- Sampling / flush thread ---
    // In full mode: waits for first dispatch, starts counter context, samples.
    // In trace-only mode: just keeps flushing the trace buffer so records
    // are delivered promptly.
    g_thread_started.store(true);
    std::thread(
        [=]()
        {
            if(g_trace_only)
            {
                while(!g_exit.load())
                {
                    rocprofiler_flush_buffer(g_trace_buffer);
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
                g_exit.store(false);
                return;
            }

            std::clog << "[tool] Waiting for first kernel dispatch...\n";
            while(!g_exit.load() && !g_first_dispatch.load())
            {
                rocprofiler_flush_buffer(g_trace_buffer);
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            if(g_exit.load()) { g_exit.store(false); return; }

            ROCPROFILER_CALL(rocprofiler_start_context(g_counter_ctx),
                             "start counter context");
            std::clog << "[tool] First dispatch seen, counter context started\n";

            uint64_t count = 0;
            while(!g_exit.load())
            {
                rocprofiler_timestamp_t ts = 0;
                rocprofiler_get_timestamp(&ts);

                auto status = rocprofiler_sample_device_counting_service(
                    g_counter_ctx,
                    {.value = ts},
                    ROCPROFILER_COUNTER_FLAG_NONE,
                    nullptr,
                    nullptr);
                if(status != ROCPROFILER_STATUS_SUCCESS)
                    std::cerr << "[tool] WARNING: sample failed at ts=" << ts
                              << " status=" << status << "\n";
                count++;
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(g_sample_interval_ms));
            }
            g_exit.store(false);
            std::clog << "[tool] Sampling thread exiting after " << count << " samples\n";
        })
        .detach();

    std::clog << "[tool] Initialization complete\n";
    return 0;
}

//
// tool_fini -- stop sampling, flush buffers, write CSVs.
//
void tool_fini(void*)
{
    std::clog << "[tool] Finalizing...\n";

    if(!g_thread_started.load())
    {
        std::clog << "[tool] No sampling thread was started, skipping.\n";
        return;
    }

    g_exit.store(true);
    while(g_exit.load() == true)
    {};

    rocprofiler_stop_context(g_ctx);
    ROCPROFILER_CALL(rocprofiler_flush_buffer(g_trace_buffer), "trace buffer flush");

    if(!g_trace_only)
    {
        rocprofiler_stop_context(g_counter_ctx);
        ROCPROFILER_CALL(rocprofiler_flush_buffer(g_counter_buffer), "counter buffer flush");
    }

    // Determine the rank suffix for multi-process runs.
    //
    // LOCAL_RANK restarts at 0 on every node, so two nodes writing into one
    // directory both produce counter_samples_rank0.csv and the second one
    // wins. The suffix therefore uses the GLOBAL rank (torchrun sets RANK,
    // srun sets SLURM_PROCID), which is unique across the whole job. On a
    // single-node run RANK equals LOCAL_RANK, so the filenames are the same
    // as before and old outputs stay readable.
    const char* global_rank_env = std::getenv("RANK");
    const char* global_rank_src = "RANK";
    if(!global_rank_env)
    {
        global_rank_env = std::getenv("SLURM_PROCID");
        global_rank_src = "SLURM_PROCID";
    }
    const char* local_rank_env = std::getenv("LOCAL_RANK");
    if(!global_rank_env)
    {
        global_rank_env = local_rank_env;
        global_rank_src = "LOCAL_RANK";
    }

    std::string rank_suffix;
    if(global_rank_env)
        rank_suffix = std::string("_rank") + global_rank_env;

    // Sidecar that says what the suffix means, so merge.py can map a global
    // rank back to (node, local GPU index) instead of assuming rank == GPU.
    if(global_rank_env)
    {
        char hostname[256] = {0};
        if(gethostname(hostname, sizeof(hostname) - 1) != 0)
            std::snprintf(hostname, sizeof(hostname), "unknown");

        std::string info_path = "rank_info" + rank_suffix + ".csv";
        if(auto* env = std::getenv("CHOPPER_COUNTER_OUTPUT"); env)
        {
            std::string base = env;
            auto slash = base.rfind('/');
            std::string dir = (slash == std::string::npos) ? std::string(".")
                                                            : base.substr(0, slash);
            info_path = dir + "/rank_info" + rank_suffix + ".csv";
        }
        std::ofstream info(info_path);
        info << "node,global_rank,local_rank,global_rank_source\n";
        info << hostname << "," << global_rank_env << ","
             << (local_rank_env ? local_rank_env : global_rank_env) << ","
             << global_rank_src << "\n";
        std::clog << "[tool] Wrote rank info to " << info_path << "\n";
    }

    // Write kernel traces CSV
    {
        std::string path = "kernel_traces" + rank_suffix + ".csv";
        if(auto* env = std::getenv("CHOPPER_TRACE_OUTPUT"); env)
        {
            path = env;
            if(!rank_suffix.empty())
            {
                auto dot = path.rfind('.');
                if(dot != std::string::npos)
                    path = path.substr(0, dot) + rank_suffix + path.substr(dot);
                else
                    path += rank_suffix;
            }
        }

        std::ofstream out(path);
        out << "kernel_name,start_ns,end_ns,duration_ns,agent_id,queue_id,correlation_id\n";
        std::lock_guard<std::mutex> lk(g_kernel_records_mutex);
        for(const auto& r : g_kernel_records)
        {
            out << "\"" << r.kernel_name << "\","
                << r.start_ns << ","
                << r.end_ns << ","
                << (r.end_ns - r.start_ns) << ","
                << r.agent_id << ","
                << r.queue_id << ","
                << r.correlation_id << "\n";
        }
        std::clog << "[tool] Wrote " << g_kernel_records.size()
                  << " kernel records to " << path << "\n";
    }

    // Write counter samples CSV (skip in trace-only mode)
    if(!g_trace_only)
    {
        std::string path = "counter_samples" + rank_suffix + ".csv";
        if(auto* env = std::getenv("CHOPPER_COUNTER_OUTPUT"); env)
        {
            path = env;
            if(!rank_suffix.empty())
            {
                auto dot = path.rfind('.');
                if(dot != std::string::npos)
                    path = path.substr(0, dot) + rank_suffix + path.substr(dot);
                else
                    path += rank_suffix;
            }
        }

        std::ofstream out(path);
        out << "timestamp_ns,counter_name,counter_value,agent_id";
        for(const auto& d : g_all_dim_names)
            out << "," << d;
        out << "\n";
        std::lock_guard<std::mutex> lk(g_counter_samples_mutex);
        for(const auto& s : g_counter_samples)
        {
            out << s.timestamp_ns << ","
                << s.counter_name << ","
                << s.value << ","
                << s.agent_id;
            for(const auto& d : g_all_dim_names)
            {
                auto it = s.dim_positions.find(d);
                out << "," << (it != s.dim_positions.end() ? it->second : 0);
            }
            out << "\n";
        }
        std::clog << "[tool] Wrote " << g_counter_samples.size()
                  << " counter samples to " << path << "\n";
    }

    std::clog << "[tool] Done.\n";
}

} // namespace

//
// rocprofiler-sdk entry point
//
extern "C" rocprofiler_tool_configure_result_t* rocprofiler_configure(uint32_t    version,
                                                                      const char* runtime_version,
                                                                      uint32_t    priority,
                                                                      rocprofiler_client_id_t* id)
{
    id->name = "chopper_device_profiler";

    uint32_t major = version / 10000;
    uint32_t minor = (version % 10000) / 100;
    uint32_t patch = version % 100;

    std::clog << id->name << " (priority=" << priority << ") is using rocprofiler-sdk v"
              << major << "." << minor << "." << patch << " (" << runtime_version << ")\n";

    static auto cfg
        = rocprofiler_tool_configure_result_t{sizeof(rocprofiler_tool_configure_result_t),
                                              &tool_init,
                                              &tool_fini,
                                              nullptr};

    return &cfg;
}
