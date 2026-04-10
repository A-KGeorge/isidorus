#include "session.h"
#include "graph.h"
#include <uv.h>
#include <cstring>
#include <thread>
#include <cstdio>

// ---------------------------------------------------------------------------
// ConfigProto — minimal binary encoding for TF thread configuration.
//
// Proto3 wire format:
//   field 2 (intra_op_parallelism_threads) : varint  tag=0x10
//   field 5 (inter_op_parallelism_threads) : varint  tag=0x28
// ---------------------------------------------------------------------------

static constexpr int MAX_VARINT1 = 127;

static void make_config_proto(
    uint8_t *buf,
    size_t &len,
    int intra_op,
    int inter_op)
{
    if (intra_op > MAX_VARINT1)
        intra_op = MAX_VARINT1;
    if (inter_op > MAX_VARINT1)
        inter_op = MAX_VARINT1;
    if (intra_op < 1)
        intra_op = 1;
    if (inter_op < 1)
        inter_op = 1;
    len = 0;
    buf[len++] = 0x10;
    buf[len++] = static_cast<uint8_t>(intra_op);
    buf[len++] = 0x28;
    buf[len++] = static_cast<uint8_t>(inter_op);
}

// ---------------------------------------------------------------------------
// CPU affinity helpers
//
// Design:
//   affinity_mask_all()        — bitmask of all online cores
//   affinity_mask_range(f, n)  — bits [f, f+n) set
//   affinity_get()             — current thread's affinity
//   affinity_set(mask)         — set current thread's affinity
//
// These are called from OnRunWork (libuv thread pool thread) immediately
// before and after TF_SessionRun. TF's eigen threadpool inherits the
// calling thread's affinity when it spawns new threads, so restricting
// the libuv worker's affinity before TF_SessionRun pins TF's threads
// to the designated cores.
//
// After TF_SessionRun returns, the libuv worker's affinity is restored
// to full_affinity_mask so it can service unrelated work (I/O callbacks,
// other native addons) on any core.
// ---------------------------------------------------------------------------

#ifdef _WIN32

// Windows specific includes/types are in session.h if needed, but we use Windows.h if required.
AffinityMask affinity_mask_all()
{
    DWORD_PTR proc_mask = 0, sys_mask = 0;
    if (GetProcessAffinityMask(GetCurrentProcess(), &proc_mask, &sys_mask))
        return static_cast<AffinityMask>(proc_mask);
    // Fallback: all 64 bits set.
    return static_cast<AffinityMask>(~0ULL);
}

AffinityMask affinity_mask_range(int first_core, int num_cores)
{
    AffinityMask mask = 0;
    for (int i = first_core; i < first_core + num_cores; ++i)
        mask |= (AffinityMask(1) << i);
    return mask;
}

AffinityMask affinity_mask_numa_node(int numa_node)
{
    ULONGLONG mask = 0;
    if (GetNumaNodeProcessorMask(static_cast<UCHAR>(numa_node), &mask))
        return static_cast<AffinityMask>(mask);
    return affinity_mask_all();
}

AffinityMask affinity_get()
{
    // Windows has no GetThreadAffinityMask. We set to full and get back
    // the previous value as the return of SetThreadAffinityMask.
    AffinityMask full = affinity_mask_all();
    DWORD_PTR prev = SetThreadAffinityMask(GetCurrentThread(),
                                           static_cast<DWORD_PTR>(full));
    if (prev)
    {
        // Restore immediately — we only wanted to read.
        SetThreadAffinityMask(GetCurrentThread(), prev);
        return static_cast<AffinityMask>(prev);
    }
    return full;
}

bool affinity_set(AffinityMask mask)
{
    return SetThreadAffinityMask(
               GetCurrentThread(),
               static_cast<DWORD_PTR>(mask)) != 0;
}

#elif defined(__APPLE__)

#include <mach/thread_act.h>
#include <mach/mach_init.h>
#include <mach/thread_policy.h>
#include <pthread.h>
#if defined(__arm64__)
#include <pthread/qos.h>
#endif

// macOS does not support core pinning via the public API.
// thread_affinity_policy is a scheduler hint only — the kernel may ignore it.
// On Apple Silicon (arm64), QoS classes are more effective than affinity tags
// for protecting the event loop from TF compute threads.

// Arbitrary nonzero tag used to group TF eigen threads together.
// The event loop never calls affinity_set, so it is implicitly in a
// different group (tag 0 / THREAD_AFFINITY_TAG_NULL).
static constexpr int TF_AFFINITY_TAG = 42;

AffinityMask affinity_mask_all()
{
    return static_cast<AffinityMask>(~0ULL);
}

// Not meaningful on macOS — retained for API compatibility.
AffinityMask affinity_mask_range(int first_core, int num_cores)
{
    AffinityMask mask = 0;
    for (int i = first_core; i < first_core + num_cores; ++i)
        mask |= (AffinityMask(1) << i);
    return mask;
}
AffinityMask affinity_mask_numa_node(int numa_node)
{
    // macOS is UMA, return all cores fallback
    return affinity_mask_all();
}
AffinityMask affinity_get()
{
    // No public API to read core affinity on macOS.
    return static_cast<AffinityMask>(~0ULL);
}

bool affinity_set(AffinityMask mask)
{
    bool ok = true;

#if defined(__arm64__)
    // Apple Silicon: use QoS to deprioritise TF threads onto E-cores.
    // UTILITY = background batch work, yields P-cores to higher-priority tasks.
    // This is more effective than affinity tags on M-series chips.
    if (mask != static_cast<AffinityMask>(~0ULL) && mask != 0)
    {
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    }
    else
    {
        // Restoring full mask — reset to USER_INTERACTIVE so libuv workers
        // can run at full priority for I/O and other native work.
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    }
#endif

    // Affinity tag hint — weak on all macOS, near-useless on Apple Silicon,
    // but costs nothing to set and may help on Intel Macs with discrete L2s.
    thread_affinity_policy_data_t policy;
    policy.affinity_tag = (mask == static_cast<AffinityMask>(~0ULL) || mask == 0)
                              ? THREAD_AFFINITY_TAG_NULL
                              : TF_AFFINITY_TAG;

    mach_port_t mach_thread = pthread_mach_thread_np(pthread_self());
    kern_return_t kr = thread_policy_set(
        mach_thread,
        THREAD_AFFINITY_POLICY,
        reinterpret_cast<thread_policy_t>(&policy),
        THREAD_AFFINITY_POLICY_COUNT);

    return ok && (kr == KERN_SUCCESS);
}

#else // POSIX (Linux)

#include <sched.h>
#include <pthread.h>
#include <fstream>
#include <sstream>

AffinityMask affinity_mask_all()
{
    cpu_set_t cs;
    CPU_ZERO(&cs);
    if (sched_getaffinity(0, sizeof(cs), &cs) != 0)
    {
        // Fallback: mark all 64 bits.
        return static_cast<AffinityMask>(~0ULL);
    }
    AffinityMask mask = 0;
    int count = static_cast<int>(sizeof(AffinityMask) * 8);
    for (int i = 0; i < count; ++i)
        if (CPU_ISSET(i, &cs))
            mask |= (AffinityMask(1) << i);
    return mask;
}

AffinityMask affinity_mask_range(int first_core, int num_cores)
{
    AffinityMask mask = 0;
    for (int i = first_core; i < first_core + num_cores; ++i)
        mask |= (AffinityMask(1) << i);
    return mask;
}

AffinityMask affinity_mask_numa_node(int numa_node)
{
    AffinityMask mask = 0;
    std::ostringstream path;
    path << "/sys/devices/system/node/node" << numa_node << "/cpulist";
    std::ifstream file(path.str());
    if (file)
    {
        std::string line;
        if (std::getline(file, line))
        {
            std::stringstream ss(line);
            std::string token;
            while (std::getline(ss, token, ','))
            {
                size_t dash = token.find('-');
                if (dash != std::string::npos)
                {
                    int start = std::stoi(token.substr(0, dash));
                    int end = std::stoi(token.substr(dash + 1));
                    for (int i = start; i <= end; i++)
                        mask |= (AffinityMask(1) << i);
                }
                else
                {
                    int core = std::stoi(token);
                    mask |= (AffinityMask(1) << core);
                }
            }
        }
    }
    if (mask == 0)
        return affinity_mask_all();
    return mask;
}

AffinityMask affinity_get()
{
    cpu_set_t cs;
    CPU_ZERO(&cs);
    sched_getaffinity(0, sizeof(cs), &cs);
    AffinityMask mask = 0;
    int count = static_cast<int>(sizeof(AffinityMask) * 8);
    for (int i = 0; i < count; ++i)
        if (CPU_ISSET(i, &cs))
            mask |= (AffinityMask(1) << i);
    return mask;
}

bool affinity_set(AffinityMask mask)
{
    cpu_set_t cs;
    CPU_ZERO(&cs);
    int count = static_cast<int>(sizeof(AffinityMask) * 8);
    for (int i = 0; i < count; ++i)
        if (mask & (AffinityMask(1) << i))
            CPU_SET(i, &cs);
    return pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs) == 0;
}

#endif // _WIN32

static void SessionCompletionCallJs(
    napi_env env, napi_value js_cb, void *context, void *data);

// ---------------------------------------------------------------------------
// SessionWrap
// ---------------------------------------------------------------------------

Napi::Object SessionWrap::Init(Napi::Env env, Napi::Object exports)
{
    Napi::Function func = DefineClass(env, "Session", {
                                                          InstanceMethod<&SessionWrap::Run>("run"),
                                                          InstanceMethod<&SessionWrap::RunAsync>("runAsync"),
                                                          InstanceMethod<&SessionWrap::Destroy>("destroy"),
                                                          InstanceAccessor<&SessionWrap::IntraOpThreads>("intraOpThreads"),
                                                          InstanceAccessor<&SessionWrap::InterOpThreads>("interOpThreads"),
                                                          InstanceAccessor<&SessionWrap::TfAffinityMask>("tfAffinityMask"),
                                                          InstanceAccessor<&SessionWrap::FullAffinityMask>("fullAffinityMask"),
                                                      });
    auto *ctor = new Napi::FunctionReference(Napi::Persistent(func));
    env.SetInstanceData<Napi::FunctionReference>(ctor);
    exports.Set("Session", func);
    return exports;
}

// ---------------------------------------------------------------------------
// Constructor
//
// Options:
//   strategy?:        "worker-pool" | "tf-parallel"
//   intraOpThreads?:  number  (overrides strategy default)
//   interOpThreads?:  number  (overrides strategy default)
//   reserveCores?:    number  (reserve first N cores for event loop / other libs)
//
// CPU affinity model:
//
//   reserveCores = R means:
//     Cores 0..(R-1)  → reserved for event loop, libuv I/O, opencv, etc.
//     Cores R..(N-1)  → given to TF_SessionRun via thread affinity
//
//   Before each TF_SessionRun:
//     libuv worker's affinity → tf_affinity_mask_  (cores R..N-1)
//     TF's eigen threads inherit this affinity when spawned
//
//   After each TF_SessionRun:
//     libuv worker's affinity → full_affinity_mask_ (all cores)
//     Worker returns to normal scheduling
//
//   reserveCores = 0 (default): no affinity restriction, TF may use any core.
// ---------------------------------------------------------------------------
SessionWrap::SessionWrap(const Napi::CallbackInfo &info)
    : Napi::ObjectWrap<SessionWrap>(info)
{
    Napi::Env env = info.Env();

    if (info.Length() < 1 || !info[0].IsObject())
    {
        Napi::TypeError::New(env, "Session(graph: Graph, options?)")
            .ThrowAsJavaScriptException();
        return;
    }

    GraphWrap *gw = Napi::ObjectWrap<GraphWrap>::Unwrap(
        info[0].As<Napi::Object>());
    if (!gw || !gw->GetGraph())
    {
        Napi::Error::New(env, "Invalid or destroyed Graph")
            .ThrowAsJavaScriptException();
        return;
    }

    graph_ = gw->GetGraph();
    graph_ref_ = Napi::ObjectReference::New(info[0].As<Napi::Object>(), 1);

    // ── Thread count defaults ───────────────────────────────────────────────
    int intra_op = 1;
    int inter_op = 1;
    int reserve_cores = 0;
    int numa_node = -1;

    if (info.Length() >= 2 && info[1].IsObject())
    {
        auto opts = info[1].As<Napi::Object>();

        // Strategy sets thread count defaults.
        if (opts.Has("strategy"))
        {
            std::string strat = opts.Get("strategy")
                                    .As<Napi::String>()
                                    .Utf8Value();
            if (strat == "tf-parallel")
            {
                unsigned hw = std::thread::hardware_concurrency();
                intra_op = hw > 0 ? static_cast<int>(hw) : 4;
                inter_op = 1;
            }
        }

        // Explicit values always override strategy defaults.
        if (opts.Has("intraOpThreads"))
            intra_op = opts.Get("intraOpThreads").As<Napi::Number>().Int32Value();
        if (opts.Has("interOpThreads"))
            inter_op = opts.Get("interOpThreads").As<Napi::Number>().Int32Value();
        if (opts.Has("reserveCores"))
            reserve_cores = opts.Get("reserveCores")
                                .As<Napi::Number>()
                                .Int32Value();
        if (opts.Has("numaNode"))
            numa_node = opts.Get("numaNode")
                            .As<Napi::Number>()
                            .Int32Value();
    }

    intra_op_threads_ = intra_op;
    inter_op_threads_ = inter_op;

    // ── Affinity masks ──────────────────────────────────────────────────────
    full_affinity_mask_ = affinity_mask_all();

    if (numa_node >= 0)
    {
        tf_affinity_mask_ = affinity_mask_numa_node(numa_node);
        fprintf(stderr,
                "[isidorus] affinity: binding to NUMA node %d, "
                "TF mask=0x%llx full mask=0x%llx\n",
                numa_node,
                static_cast<unsigned long long>(tf_affinity_mask_),
                static_cast<unsigned long long>(full_affinity_mask_));
    }
    else if (reserve_cores > 0)
    {
        int total_cores = static_cast<int>(
            sizeof(AffinityMask) * 8);

        // Count actual online cores from the full mask.
        int online = 0;
        for (int i = 0; i < total_cores; ++i)
            if (full_affinity_mask_ & (AffinityMask(1) << i))
                ++online;

        int tf_cores = online - reserve_cores;
        if (tf_cores < 1)
            tf_cores = 1;

        // TF gets the LAST tf_cores online cores (highest indices).
        // Reserved cores are the FIRST reserve_cores online cores.
        // This keeps core 0 (event loop affinity default) reserved.
        AffinityMask tf_mask = 0;
        int assigned = 0;
        for (int i = total_cores - 1; i >= 0 && assigned < tf_cores; --i)
        {
            if (full_affinity_mask_ & (AffinityMask(1) << i))
            {
                tf_mask |= (AffinityMask(1) << i);
                ++assigned;
            }
        }
        tf_affinity_mask_ = tf_mask;

        fprintf(stderr,
                "[isidorus] affinity: reserving %d core(s), "
                "TF mask=0x%llx full mask=0x%llx\n",
                reserve_cores,
                static_cast<unsigned long long>(tf_affinity_mask_),
                static_cast<unsigned long long>(full_affinity_mask_));
    }
    // tf_affinity_mask_ = 0 means "no restriction" — checked in OnRunWork.

    // ── ConfigProto and session creation ────────────────────────────────────
    uint8_t config_buf[4];
    size_t config_len = 0;
    make_config_proto(config_buf, config_len, intra_op, inter_op);

    TF_SessionOptions *opts = TF_NewSessionOptions();
    StatusGuard config_status;
    TF_SetConfig(opts, config_buf, config_len, config_status.s);
    if (!config_status.ok())
    {
        TF_DeleteSessionOptions(opts);
        Napi::Error::New(env,
                         "TF_SetConfig failed: " + config_status.message())
            .ThrowAsJavaScriptException();
        return;
    }

    StatusGuard status;
    session_ = TF_NewSession(graph_, opts, status.s);
    TF_DeleteSessionOptions(opts);

    if (!status.ok() || !session_)
    {
        Napi::Error::New(env,
                         "TF_NewSession failed: " + status.message())
            .ThrowAsJavaScriptException();
        return;
    }

    // ── Create TSFN for batched completion notifications ─────────────────────
    // max_queue_size=0 → unlimited. Non-blocking calls from OnRunWork never
    // block the libuv thread. SessionCompletionCallJs drains the entire
    // completion_queue_ on each invocation, coalescing N concurrent
    // completions into 1 V8 callback.
    napi_value async_name;
    napi_create_string_utf8(env, "isidorus:session:completions",
                            NAPI_AUTO_LENGTH, &async_name);
    napi_create_threadsafe_function(
        env,
        nullptr, // no JS function to wrap
        nullptr, // no async resource
        async_name,
        0,       // max_queue_size = unlimited
        1,       // initial_thread_count
        nullptr, // thread_finalize_data
        nullptr, // thread_finalize_cb
        this,    // context passed to call_js_cb
        SessionCompletionCallJs,
        &completion_tsfn_);

    // Unref so the TSFN doesn't prevent the event loop from exiting when
    // there are no in-flight requests.
    if (completion_tsfn_)
        napi_unref_threadsafe_function(env, completion_tsfn_);
}

SessionWrap::~SessionWrap() { cleanup(); }

void SessionWrap::cleanup()
{
    if (session_)
    {
        StatusGuard s;
        TF_CloseSession(session_, s.s);
        TF_DeleteSession(session_, s.s);
        session_ = nullptr;
    }
    graph_ = nullptr;

    if (!graph_ref_.IsEmpty())
        graph_ref_.Reset();

    // Release the TSFN. This decrements the thread count to 0, which causes
    // the TSFN to drain any queued call_js_cb invocations and then release
    // itself. The finalizer runs on the event loop thread after all pending
    // completions have been processed.
    if (completion_tsfn_)
    {
        napi_release_threadsafe_function(completion_tsfn_, napi_tsfn_release);
        completion_tsfn_ = nullptr;
    }
}

Napi::Value SessionWrap::IntraOpThreads(const Napi::CallbackInfo &info)
{
    return Napi::Number::New(info.Env(), intra_op_threads_);
}
Napi::Value SessionWrap::InterOpThreads(const Napi::CallbackInfo &info)
{
    return Napi::Number::New(info.Env(), inter_op_threads_);
}
Napi::Value SessionWrap::TfAffinityMask(const Napi::CallbackInfo &info)
{
    return Napi::Number::New(info.Env(),
                             static_cast<double>(tf_affinity_mask_));
}
Napi::Value SessionWrap::FullAffinityMask(const Napi::CallbackInfo &info)
{
    return Napi::Number::New(info.Env(),
                             static_cast<double>(full_affinity_mask_));
}

// ---------------------------------------------------------------------------
// Feed / fetch / output helpers
// ---------------------------------------------------------------------------

static bool parse_feeds(
    TF_Graph *graph,
    Napi::Array feeds_arr,
    std::vector<TF_Output> &inputs,
    std::vector<TF_Tensor *> &input_tensors,
    std::string &error)
{
    for (uint32_t i = 0; i < feeds_arr.Length(); ++i)
    {
        auto feed = feeds_arr.Get(i).As<Napi::Object>();
        std::string n = feed.Get("opName").As<Napi::String>().Utf8Value();
        int idx = feed.Get("index").As<Napi::Number>().Int32Value();
        TF_Operation *op = TF_GraphOperationByName(graph, n.c_str());
        if (!op)
        {
            error = "Feed op not found: " + n;
            return false;
        }
        inputs.push_back({op, idx});

        auto t = feed.Get("tensor").As<Napi::Object>();
        auto dtype = static_cast<TF_DataType>(
            t.Get("dtype").As<Napi::Number>().Int32Value());
        auto data = t.Get("data").As<Napi::Buffer<uint8_t>>();
        auto darr = t.Get("shape").As<Napi::Array>();
        std::vector<int64_t> dims(darr.Length());
        for (uint32_t j = 0; j < darr.Length(); ++j)
            dims[j] = darr.Get(j).As<Napi::Number>().Int64Value();

        TF_Tensor *tensor = TF_AllocateTensor(
            dtype, dims.data(), static_cast<int>(dims.size()),
            data.ByteLength());
        if (!tensor)
        {
            error = "TF_AllocateTensor failed";
            return false;
        }
        std::memcpy(TF_TensorData(tensor), data.Data(), data.ByteLength());
        input_tensors.push_back(tensor);
    }
    return true;
}

static bool parse_fetches(
    TF_Graph *graph,
    Napi::Array fetches_arr,
    std::vector<TF_Output> &outputs,
    std::string &error)
{
    for (uint32_t i = 0; i < fetches_arr.Length(); ++i)
    {
        auto fetch = fetches_arr.Get(i).As<Napi::Object>();
        std::string n = fetch.Get("opName").As<Napi::String>().Utf8Value();
        int idx = fetch.Get("index").As<Napi::Number>().Int32Value();
        TF_Operation *op = TF_GraphOperationByName(graph, n.c_str());
        if (!op)
        {
            error = "Fetch op not found: " + n;
            return false;
        }
        outputs.push_back({op, idx});
    }
    return true;
}

static Napi::Array pack_outputs(
    Napi::Env env,
    std::vector<TF_Tensor *> &output_tensors)
{
    Napi::Array result = Napi::Array::New(env, output_tensors.size());
    for (size_t i = 0; i < output_tensors.size(); ++i)
    {
        TF_Tensor *t = output_tensors[i];
        if (!t)
        {
            result.Set(i, env.Null());
            continue;
        }

        Napi::Object obj = Napi::Object::New(env);
        TF_DataType dtype = TF_TensorType(t);
        int ndims = TF_NumDims(t);
        size_t nb = TF_TensorByteSize(t);

        obj.Set("dtype", Napi::Number::New(env, static_cast<double>(dtype)));
        Napi::Array shape = Napi::Array::New(env, ndims);
        for (int j = 0; j < ndims; ++j)
            shape.Set(j, Napi::Number::New(env,
                                           static_cast<double>(TF_Dim(t, j))));
        obj.Set("shape", shape);

        auto buf = Napi::Buffer<uint8_t>::Copy(
            env,
            reinterpret_cast<const uint8_t *>(TF_TensorData(t)),
            nb);
        obj.Set("data", buf);
        TF_DeleteTensor(t);
        result.Set(i, obj);
    }
    return result;
}

// ---------------------------------------------------------------------------
// run — synchronous (no affinity — runs on event loop thread)
// ---------------------------------------------------------------------------
Napi::Value SessionWrap::Run(const Napi::CallbackInfo &info)
{
    Napi::Env env = info.Env();

    if (!session_)
    {
        Napi::Error::New(env, "Session destroyed").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    std::vector<TF_Output> tf_inputs, tf_outputs;
    std::vector<TF_Tensor *> tf_input_tensors;
    std::string error;

    if (!parse_feeds(graph_, info[0].As<Napi::Array>(),
                     tf_inputs, tf_input_tensors, error) ||
        !parse_fetches(graph_, info[1].As<Napi::Array>(),
                       tf_outputs, error))
    {
        for (auto *t : tf_input_tensors)
            TF_DeleteTensor(t);
        Napi::Error::New(env, error).ThrowAsJavaScriptException();
        return env.Undefined();
    }

    std::vector<TF_Operation *> target_ops;
    if (info.Length() >= 3 && info[2].IsArray())
    {
        auto targets = info[2].As<Napi::Array>();
        for (uint32_t i = 0; i < targets.Length(); ++i)
        {
            std::string n = targets.Get(i).As<Napi::String>().Utf8Value();
            TF_Operation *op = TF_GraphOperationByName(graph_, n.c_str());
            if (op)
                target_ops.push_back(op);
        }
    }

    std::vector<TF_Tensor *> output_tensors(tf_outputs.size(), nullptr);
    StatusGuard status;
    TF_SessionRun(
        session_, nullptr,
        tf_inputs.data(), tf_input_tensors.data(),
        static_cast<int>(tf_inputs.size()),
        tf_outputs.data(), output_tensors.data(),
        static_cast<int>(tf_outputs.size()),
        target_ops.data(), static_cast<int>(target_ops.size()),
        nullptr, status.s);

    for (auto *t : tf_input_tensors)
        TF_DeleteTensor(t);

    if (!status.ok())
    {
        for (auto *t : output_tensors)
            if (t)
                TF_DeleteTensor(t);
        Napi::Error::New(env, "TF_SessionRun failed: " + status.message()).ThrowAsJavaScriptException();
        return env.Undefined();
    }

    return pack_outputs(env, output_tensors);
}

// ---------------------------------------------------------------------------
// runAsync — TF_SessionRun on libuv thread pool with affinity fencing
//
// Affinity fencing in OnRunWork:
//
//   1. Save the libuv worker's current affinity (full_affinity_mask).
//   2. Set the worker's affinity to tf_affinity_mask (TF cores only).
//   3. Call TF_SessionRun.
//      TF's eigen threadpool spawns threads that inherit this affinity.
//      All TF compute stays on the designated cores.
//   4. Restore the worker's affinity to full_affinity_mask.
//      The libuv worker returns to unrestricted scheduling.
//
// If tf_affinity_mask == 0, affinity fencing is skipped entirely.
// ---------------------------------------------------------------------------
// Pre-packed output from a single TF output tensor.
// Populated in OnRunWork (libuv thread) — no memcpy on the event loop.
struct PackedOutput
{
    std::vector<uint8_t> data; // raw tensor bytes, copied off TF heap
    int32_t dtype;
    std::vector<int64_t> shape;
};

// ── CompletionData ────────────────────────────────────────────────────────────
// One inference completion, queued from the libuv thread and drained in bulk
// by the TSFN call_js_cb on the event loop thread.
//
// Uses raw napi handles (napi_deferred, napi_ref) so it can live in a plain
// C++ struct that crosses thread boundaries. All V8 access happens only in
// call_js_cb which runs on the event loop thread.
struct CompletionData
{
    napi_deferred deferred; // raw handle — JS thread only
    napi_ref self_ref;      // raw ref — JS thread only
    bool ok = true;
    std::string error_message;
    std::vector<PackedOutput> packed_outputs;
};

// Forward declaration for RunCtx → SessionWrap* pointer.
class SessionWrap;

struct RunCtx
{
    uv_work_t req;
    TF_Session *session;
    TF_Graph *graph;
    AffinityMask tf_affinity_mask;
    AffinityMask full_affinity_mask;
    std::vector<TF_Output> tf_inputs;
    std::vector<TF_Tensor *> tf_input_tensors;
    std::vector<TF_Output> tf_outputs;
    std::vector<TF_Operation *> target_ops;
    std::vector<TF_Tensor *> output_tensors;
    std::vector<PackedOutput> packed_outputs;
    bool ok = true;
    std::string error_message;
    // Raw handles set in RunAsync on the JS thread; consumed only in call_js_cb.
    napi_deferred raw_deferred = nullptr;
    napi_ref raw_self_ref = nullptr;
    // Back-pointer so OnRunWork can push to the session's completion queue.
    SessionWrap *session_wrap = nullptr;
};

// ── TSFN completion batching ─────────────────────────────────────────────────
//
// Design: N concurrent runAsync() completions may arrive between two event
// loop ticks. Each pushes a CompletionData to the session's completion_queue_
// and calls napi_call_threadsafe_function (non-blocking, from libuv thread).
// The TSFN enqueues N wake signals but the event loop may process them in
// one batch. SessionCompletionCallJs drains the ENTIRE queue each invocation,
// so N concurrent completions → 1 V8 callback (instead of N).
//
// For maxConcurrent=1 (large models), there is at most one item per drain —
// the benefit is eliminating uv_queue_work's after_work_cb overhead and
// the per-completion event loop context-switch cost.
static void SessionCompletionCallJs(
    napi_env env, napi_value /*js_cb*/, void *context, void * /*data*/)
{
    if (!env)
        return; // env is null during TSFN teardown

    auto *sw = static_cast<SessionWrap *>(context);

    // Drain the entire completion queue in one lock acquisition.
    std::vector<CompletionData *> batch;
    {
        std::lock_guard<std::mutex> lg(sw->completion_mu_);
        batch = std::move(sw->completion_queue_);
        // completion_queue_ is now empty; subsequent call_js_cb invocations
        // from the same burst will see an empty queue and return immediately.
    }

    Napi::Env n_env(env);
    Napi::HandleScope scope(n_env);

    for (auto *cd : batch)
    {
        if (!cd->ok)
        {
            napi_value err_msg, error;
            napi_create_string_utf8(env, cd->error_message.c_str(),
                                    NAPI_AUTO_LENGTH, &err_msg);
            napi_create_error(env, nullptr, err_msg, &error);
            napi_reject_deferred(env, cd->deferred, error);
        }
        else
        {
            // Build the JS result array from pre-packed data.
            // Napi::Buffer::New with external pointer — zero extra memcpy.
            napi_value result;
            napi_create_array_with_length(env, cd->packed_outputs.size(), &result);

            for (size_t i = 0; i < cd->packed_outputs.size(); ++i)
            {
                auto &po = cd->packed_outputs[i];

                napi_value obj;
                napi_create_object(env, &obj);

                napi_value dtype_val;
                napi_create_double(env, static_cast<double>(po.dtype), &dtype_val);
                napi_set_named_property(env, obj, "dtype", dtype_val);

                napi_value shape_arr;
                napi_create_array_with_length(env, po.shape.size(), &shape_arr);
                for (size_t j = 0; j < po.shape.size(); ++j)
                {
                    napi_value dim;
                    napi_create_double(env, static_cast<double>(po.shape[j]), &dim);
                    napi_set_element(env, shape_arr, static_cast<uint32_t>(j), dim);
                }
                napi_set_named_property(env, obj, "shape", shape_arr);

                // Transfer ownership of po.data to the Buffer finalizer.
                auto *heap_vec = new std::vector<uint8_t>(std::move(po.data));
                napi_value buf;
                napi_create_external_buffer(
                    env, heap_vec->size(), heap_vec->data(),
                    [](napi_env, void *, void *hint)
                    {
                        delete static_cast<std::vector<uint8_t> *>(hint);
                    },
                    heap_vec, &buf);
                napi_set_named_property(env, obj, "data", buf);
                napi_set_element(env, result, static_cast<uint32_t>(i), obj);
            }

            napi_resolve_deferred(env, cd->deferred, result);
        }

        napi_delete_reference(env, cd->self_ref);
        delete cd;
    }
}

static void OnRunWork(uv_work_t *req)
{
    auto *ctx = reinterpret_cast<RunCtx *>(req);
    ctx->output_tensors.assign(ctx->tf_outputs.size(), nullptr);

    // ── Affinity fence in ────────────────────────────────────────────────────
    // Pin this libuv worker to TF cores so TF's eigen threads inherit
    // the restricted affinity when they are spawned by TF_SessionRun.
    bool affinity_applied = false;
    if (ctx->tf_affinity_mask != 0)
    {
        affinity_applied = affinity_set(ctx->tf_affinity_mask);
    }

    StatusGuard status;
    TF_SessionRun(
        ctx->session, nullptr,
        ctx->tf_inputs.data(), ctx->tf_input_tensors.data(),
        static_cast<int>(ctx->tf_inputs.size()),
        ctx->tf_outputs.data(), ctx->output_tensors.data(),
        static_cast<int>(ctx->tf_outputs.size()),
        ctx->target_ops.data(), static_cast<int>(ctx->target_ops.size()),
        nullptr, status.s);

    // ── Affinity fence out ───────────────────────────────────────────────────
    // Restore unrestricted scheduling so this libuv worker can service
    // other work (I/O callbacks, opencv, etc.) on any core.
    if (affinity_applied)
    {
        affinity_set(ctx->full_affinity_mask);
    }

    if (!status.ok())
    {
        ctx->ok = false;
        ctx->error_message = status.message();
        // Free output tensors on this thread to avoid any work on event loop.
        for (auto *t : ctx->output_tensors)
            if (t)
                TF_DeleteTensor(t);
        ctx->output_tensors.clear();
    }
    else
    {
        // Pre-pack outputs on the libuv thread (no V8 access needed).
        // This moves the expensive memcpy + TF_DeleteTensor off the event
        // loop thread so OnRunAfter only does cheap V8 object construction.
        ctx->packed_outputs.reserve(ctx->output_tensors.size());
        for (auto *t : ctx->output_tensors)
        {
            PackedOutput po;
            po.dtype = static_cast<int32_t>(TF_TensorType(t));
            int ndims = TF_NumDims(t);
            po.shape.reserve(ndims);
            for (int j = 0; j < ndims; ++j)
                po.shape.push_back(TF_Dim(t, j));
            size_t nb = TF_TensorByteSize(t);
            po.data.resize(nb);
            std::memcpy(po.data.data(),
                        reinterpret_cast<const uint8_t *>(TF_TensorData(t)),
                        nb);
            TF_DeleteTensor(t); // free TF allocation here, not on event loop
            ctx->packed_outputs.push_back(std::move(po));
        }
        ctx->output_tensors.clear();
    }

    // Delete input tensors here too — we're done with them after TF_SessionRun.
    for (auto *t : ctx->tf_input_tensors)
        TF_DeleteTensor(t);
    ctx->tf_input_tensors.clear();

    // ── Push to session completion queue + signal event loop ─────────────────
    // Build CompletionData from ctx fields (raw napi handles are safe to pass
    // cross-thread; they are only dereferenced in SessionCompletionCallJs on
    // the event loop thread).
    auto *cd = new CompletionData();
    cd->deferred = ctx->raw_deferred;
    cd->self_ref = ctx->raw_self_ref;
    cd->ok = ctx->ok;
    cd->error_message = std::move(ctx->error_message);
    cd->packed_outputs = std::move(ctx->packed_outputs);

    SessionWrap *sw = ctx->session_wrap;
    {
        std::lock_guard<std::mutex> lg(sw->completion_mu_);
        sw->completion_queue_.push_back(cd);
    }

    // Signal the event loop. Non-blocking — if the TSFN queue is full
    // (shouldn't happen with max_queue_size=0) the item is dropped, so we
    // keep our own queue above as the authoritative store.
    napi_call_threadsafe_function(sw->completion_tsfn_,
                                  nullptr,
                                  napi_tsfn_nonblocking);
}

// OnRunAfter — trivial cleanup only.
// All completion work (V8 object construction, promise resolution) has been
// moved to SessionCompletionCallJs via the TSFN. This callback just frees
// the RunCtx allocation so uv_queue_work's bookkeeping is satisfied.
static void OnRunAfter(uv_work_t *req, int /*status*/)
{
    delete reinterpret_cast<RunCtx *>(req);
}

Napi::Value SessionWrap::RunAsync(const Napi::CallbackInfo &info)
{
    Napi::Env env = info.Env();
    if (!session_)
    {
        auto d = Napi::Promise::Deferred::New(env);
        d.Reject(Napi::Error::New(env, "Session destroyed").Value());
        return d.Promise();
    }

    // Create promise + raw deferred handle.
    // We store raw napi_deferred (not Napi::Promise::Deferred) because the
    // completion travels to SessionCompletionCallJs via the completion queue
    // (a C++ struct), which cannot hold Napi++ RAII types safely.
    napi_deferred raw_deferred = nullptr;
    napi_value promise_val = nullptr;
    if (napi_create_promise(env, &raw_deferred, &promise_val) != napi_ok)
    {
        Napi::Error::New(env, "Failed to create promise").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    auto *ctx = new RunCtx();
    ctx->session = session_;
    ctx->graph = graph_;
    ctx->tf_affinity_mask = tf_affinity_mask_;
    ctx->full_affinity_mask = full_affinity_mask_;
    ctx->raw_deferred = raw_deferred;
    ctx->session_wrap = this;

    // Create a raw napi_ref to keep 'this' alive until the completion fires.
    napi_ref raw_self_ref = nullptr;
    napi_create_reference(env, info.This(), 1, &raw_self_ref);
    ctx->raw_self_ref = raw_self_ref;

    std::string error;
    if (!parse_feeds(graph_, info[0].As<Napi::Array>(),
                     ctx->tf_inputs, ctx->tf_input_tensors, error) ||
        !parse_fetches(graph_, info[1].As<Napi::Array>(),
                       ctx->tf_outputs, error))
    {
        napi_value err_msg, err_obj;
        napi_create_string_utf8(env, error.c_str(), NAPI_AUTO_LENGTH, &err_msg);
        napi_create_error(env, nullptr, err_msg, &err_obj);
        napi_reject_deferred(env, raw_deferred, err_obj);
        napi_delete_reference(env, raw_self_ref);
        delete ctx;
        return Napi::Value(env, promise_val);
    }

    if (info.Length() >= 3 && info[2].IsArray())
    {
        auto targets = info[2].As<Napi::Array>();
        for (uint32_t i = 0; i < targets.Length(); ++i)
        {
            std::string n = targets.Get(i).As<Napi::String>().Utf8Value();
            TF_Operation *op = TF_GraphOperationByName(graph_, n.c_str());
            if (op)
                ctx->target_ops.push_back(op);
        }
    }

    uv_loop_t *loop = nullptr;
    if (napi_get_uv_event_loop(env, &loop) != napi_ok || !loop ||
        uv_queue_work(loop, &ctx->req, OnRunWork, OnRunAfter) != 0)
    {
        for (auto *t : ctx->tf_input_tensors)
            TF_DeleteTensor(t);
        napi_value err_msg, err_obj;
        napi_create_string_utf8(env, "Failed to queue runAsync work",
                                NAPI_AUTO_LENGTH, &err_msg);
        napi_create_error(env, nullptr, err_msg, &err_obj);
        napi_reject_deferred(env, raw_deferred, err_obj);
        napi_delete_reference(env, raw_self_ref);
        delete ctx;
        return Napi::Value(env, promise_val);
    }

    return Napi::Value(env, promise_val);
}

Napi::Value SessionWrap::Destroy(const Napi::CallbackInfo &info)
{
    cleanup();
    return info.Env().Undefined();
}