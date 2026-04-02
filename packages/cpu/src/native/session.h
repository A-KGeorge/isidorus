#pragma once
#include <napi.h>
#include "platform_tf.h"
#include <string>
#include <vector>
#include <cstdint>

// ----------------------------------------------------------------------------
// Platform CPU affinity abstractions
//
// AffinityMask: bitmask where bit N = 1 means "this thread may run on core N"
// Supports up to 64 cores. Sytems with > 64 cores need processor groups
// (Windows) or cpu_set_t extension (Linux) - not handled here
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
#endif // _WIN32

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

// ----------------------------------------------------------------------------
// Affinity helpers - declared here, defined in session.cc
// ----------------------------------------------------------------------------

// Get the affinity mask of the calling thread.
AffinityMask affinity_get();

// Set the affinity mask of the calling thread.
// Returns true on success.
bool affinity_set(AffinityMask mask);

// Build a mask covering cores [first_core, num_cores).
// e.g. affinity_mask_range(2, 6) = cores 2,3,4,5 (bits 2..5 set)
AffinityMask affinity_mask_range(int first_core, int num_cores);

// Build a mask covering all cores on a given NUMA node.
AffinityMask affinity_mask_numa_node(int numa_node);

// Build  a mask covering all online cores.
AffinityMask affinity_mask_all();

// ----------------------------------------------------------------------------
// SessionWrap
// ----------------------------------------------------------------------------
class SessionWrap : public Napi::ObjectWrap<SessionWrap>
{
public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);

    explicit SessionWrap(const Napi::CallbackInfo &info);
    ~SessionWrap() override;

    // Affinity masks used by RunCtx - set in constructor, read in OnRunWork.
    AffinityMask tf_affinity_mask_ = 0; // 0 = no restriction
    AffinityMask full_affinity_mask_ = 0;

private:
    TF_Graph *graph_ = nullptr;
    TF_Session *session_ = nullptr;
    Napi::ObjectReference graph_ref_;
    int intra_op_threads_ = 1;
    int inter_op_threads_ = 1;

    void cleanup();

    Napi::Value Run(const Napi::CallbackInfo &info);
    Napi::Value RunAsync(const Napi::CallbackInfo &info);
    Napi::Value Destroy(const Napi::CallbackInfo &info);
    Napi::Value IntraOpThreads(const Napi::CallbackInfo &info);
    Napi::Value InterOpThreads(const Napi::CallbackInfo &info);
    Napi::Value TfAffinityMask(const Napi::CallbackInfo &info);
    Napi::Value FullAffinityMask(const Napi::CallbackInfo &info);
};