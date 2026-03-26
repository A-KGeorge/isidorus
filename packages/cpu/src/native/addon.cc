#include <napi.h>
#include "graph.h"
#include "session.h"

Napi::Object InitAll(Napi::Env env, Napi::Object exports)
{
    GraphWrap::Init(env, exports);
    SessionWrap::Init(env, exports);
    return exports;
}

NODE_API_MODULE(isidorus_cpu, InitAll)