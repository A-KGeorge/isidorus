#pragma once
#include <napi.h>
#include <string>
#include <vector>
#include "platform_tf.h"

#ifndef ISIDORUS_STATUS_GUARD_DEFINED
#define ISIDORUS_STATUS_GUARD_DEFINED
struct StatusGuard
{
    TF_Status *s;
    StatusGuard() : s(TF_NewStatus()) {}
    ~StatusGuard() { TF_DeleteStatus(s); }
    bool ok() const { return TF_GetCode(s) == TF_OK; }
    std::string message() const { return TF_Message(s); }
};
#endif

class GraphWrap : public Napi::ObjectWrap<GraphWrap>
{
public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);

    explicit GraphWrap(const Napi::CallbackInfo &info);
    ~GraphWrap();

    // Expose the raw graph pointer to Session and op helpers.
    TF_Graph *GetGraph() const { return graph_; }

private:
    TF_Graph *graph_ = nullptr;
    int op_counter_ = 0;

    Napi::Value AddOp(const Napi::CallbackInfo &info);
    Napi::Value HasOp(const Napi::CallbackInfo &info);
    Napi::Value OpOutputType(const Napi::CallbackInfo &info);
    Napi::Value OpOutputShape(const Napi::CallbackInfo &info);
    Napi::Value ToGraphDef(const Napi::CallbackInfo &info);
    Napi::Value NumOps(const Napi::CallbackInfo &info);
    Napi::Value ImportGraphDef(const Napi::CallbackInfo &info);
    Napi::Value AddGradients(const Napi::CallbackInfo &info);
};