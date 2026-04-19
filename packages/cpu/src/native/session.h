#pragma once
#include <napi.h>
#include "platform_tf.h"
#include <string>
#include <vector>
#include <mutex>
#include <atomic>
#include <cstdint>

// ----------------------------------------------------------------------------
// Platform CPU affinity abstractions
// ----------------------------------------------------------------------------
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
using AffinityMask = DWORD_PTR;
#else
#include <sched.h>
#include <pthread.h>
using AffinityMask = uint64_t;
#endif

#ifndef ISIDORUS_STATUS_GUARD_DEFINED
#define ISIDORUS_STATUS_GUARD_DEFINED
struct StatusGuard
{
    TF_Status *s;
    StatusGuard() : s(TF_NewStatus()) {}
    ~StatusGuard()
    {
        if (s)
            TF_DeleteStatus(s);
    }
    bool ok() const { return TF_GetCode(s) == TF_OK; }
    std::string message() const { return TF_Message(s); }
};
#endif

// Affinity helpers — declared here, defined in session.cc
AffinityMask affinity_get();
bool affinity_set(AffinityMask mask);
AffinityMask affinity_mask_range(int first_core, int num_cores);
AffinityMask affinity_mask_numa_node(int numa_node);
AffinityMask affinity_mask_all();

// Maximum items in the native completion queue before rejecting new requests.
static constexpr size_t MAX_NATIVE_COMPLETION_QUEUE = 512;

// Timeout (seconds) for the cleanup() spin-wait.  If in-flight workers have
// not drained within this window a warning is printed and cleanup proceeds.
static constexpr int CLEANUP_SPIN_TIMEOUT_SEC = 30;

// ----------------------------------------------------------------------------
// SessionWrap
// ----------------------------------------------------------------------------
class SessionWrap : public Napi::ObjectWrap<SessionWrap>
{
public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);

    explicit SessionWrap(const Napi::CallbackInfo &info);
    ~SessionWrap() override;

    AffinityMask tf_affinity_mask_ = 0;
    AffinityMask full_affinity_mask_ = 0;

public:
    // Public for SessionCompletionCallJs (static fn, no friend access).
    std::atomic<bool> destroyed_{false};
    std::atomic<int> in_flight_count_{0};
    napi_threadsafe_function completion_tsfn_ = nullptr;
    std::mutex completion_mu_;
    std::vector<struct CompletionData *> completion_queue_;

private:
    TF_Graph *graph_ = nullptr;
    TF_Session *session_ = nullptr;
    Napi::ObjectReference graph_ref_;
    int intra_op_threads_ = 1;
    int inter_op_threads_ = 1;

    napi_env env_ = nullptr;

    void cleanup();

    Napi::Value Run(const Napi::CallbackInfo &info);
    Napi::Value RunAsync(const Napi::CallbackInfo &info);
    Napi::Value Destroy(const Napi::CallbackInfo &info);
    Napi::Value IntraOpThreads(const Napi::CallbackInfo &info);
    Napi::Value InterOpThreads(const Napi::CallbackInfo &info);
    Napi::Value TfAffinityMask(const Napi::CallbackInfo &info);
    Napi::Value FullAffinityMask(const Napi::CallbackInfo &info);
};