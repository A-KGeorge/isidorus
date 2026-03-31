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
        ok = pthread_set_qos_class_self_np(
                 pthread_self(), QOS_CLASS_UTILITY, 0) == 0;
    }
    else
    {
        // Restoring full mask — reset to USER_INTERACTIVE so libuv workers
        // can run at full priority for I/O and other native work.
        pthread_set_qos_class_self_np(
            pthread_self(), QOS_CLASS_USER_INTERACTIVE, 0);
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
    }

    intra_op_threads_ = intra_op;
    inter_op_threads_ = inter_op;

    // ── Affinity masks ──────────────────────────────────────────────────────
    full_affinity_mask_ = affinity_mask_all();

    if (reserve_cores > 0)
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
    {
        graph_ref_.Reset();
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
    auto deferred = Napi::Promise::Deferred::New(env);

    if (!session_)
    {
        deferred.Reject(Napi::Error::New(env, "Session destroyed").Value());
        return deferred.Promise();
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
        deferred.Reject(Napi::Error::New(env, error).Value());
        return deferred.Promise();
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
        deferred.Reject(Napi::Error::New(env,
                                         "TF_SessionRun failed: " + status.message())
                            .Value());
        return deferred.Promise();
    }

    deferred.Resolve(pack_outputs(env, output_tensors));
    return deferred.Promise();
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
    bool ok = true;
    std::string error_message;
    Napi::Promise::Deferred deferred;
    Napi::ObjectReference self_ref;
    explicit RunCtx(Napi::Env env)
        : req{}, session(nullptr), graph(nullptr),
          tf_affinity_mask(0), full_affinity_mask(0),
          deferred(Napi::Promise::Deferred::New(env)) {}
};

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
    }
}

static void OnRunAfter(uv_work_t *req, int)
{
    auto *ctx = reinterpret_cast<RunCtx *>(req);
    Napi::Env env = ctx->deferred.Env();
    Napi::HandleScope scope(env);
    for (auto *t : ctx->tf_input_tensors)
        TF_DeleteTensor(t);
    if (!ctx->ok)
    {
        for (auto *t : ctx->output_tensors)
            if (t)
                TF_DeleteTensor(t);
        ctx->deferred.Reject(Napi::Error::New(env,
                                              "TF_SessionRun failed: " + ctx->error_message)
                                 .Value());
    }
    else
    {
        ctx->deferred.Resolve(pack_outputs(env, ctx->output_tensors));
    }
    ctx->self_ref.Unref();
    delete ctx;
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

    auto *ctx = new RunCtx(env);
    ctx->session = session_;
    ctx->graph = graph_;

    // Pass affinity masks so OnRunWork can fence without touching the JS heap.
    ctx->tf_affinity_mask = tf_affinity_mask_;
    ctx->full_affinity_mask = full_affinity_mask_;

    ctx->self_ref = Napi::ObjectReference::New(
        info.This().As<Napi::Object>(), 1);

    std::string error;
    if (!parse_feeds(graph_, info[0].As<Napi::Array>(),
                     ctx->tf_inputs, ctx->tf_input_tensors, error) ||
        !parse_fetches(graph_, info[1].As<Napi::Array>(),
                       ctx->tf_outputs, error))
    {
        auto promise = ctx->deferred.Promise();
        ctx->deferred.Reject(Napi::Error::New(env, error).Value());
        ctx->self_ref.Unref();
        delete ctx;
        return promise;
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
        auto promise = ctx->deferred.Promise();
        ctx->deferred.Reject(Napi::Error::New(env,
                                              "Failed to queue runAsync work")
                                 .Value());
        ctx->self_ref.Unref();
        delete ctx;
        return promise;
    }

    return ctx->deferred.Promise();
}

Napi::Value SessionWrap::Destroy(const Napi::CallbackInfo &info)
{
    cleanup();
    return info.Env().Undefined();
}