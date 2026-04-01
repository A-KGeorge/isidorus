#include "graph.h"
#include <cstring>
#include <unordered_map>
#include <unordered_set>

Napi::Object GraphWrap::Init(Napi::Env env, Napi::Object exports)
{
    Napi::Function func = DefineClass(env, "Graph", {
                                                        InstanceMethod<&GraphWrap::AddOp>("addOp"),
                                                        InstanceMethod<&GraphWrap::HasOp>("hasOp"),
                                                        InstanceMethod<&GraphWrap::OpOutputType>("opOutputType"),
                                                        InstanceMethod<&GraphWrap::OpOutputShape>("opOutputShape"),
                                                        InstanceMethod<&GraphWrap::ToGraphDef>("toGraphDef"),
                                                        InstanceMethod<&GraphWrap::NumOps>("numOps"),
                                                        InstanceMethod<&GraphWrap::ImportGraphDef>("importGraphDef"),
                                                        InstanceMethod<&GraphWrap::AddGradients>("addGradients"),
                                                        InstanceMethod<&GraphWrap::ListOpsOfType>("listOpsOfType"),
                                                        InstanceMethod<&GraphWrap::ListSinkOps>("listSinkOps"),
                                                    });
    auto *ctor = new Napi::FunctionReference(Napi::Persistent(func));
    env.SetInstanceData<Napi::FunctionReference>(ctor);
    exports.Set("Graph", func);
    return exports;
}

GraphWrap::GraphWrap(const Napi::CallbackInfo &info)
    : Napi::ObjectWrap<GraphWrap>(info)
{
    graph_ = TF_NewGraph();
    if (!graph_)
        Napi::Error::New(info.Env(), "Failed to create TF_Graph")
            .ThrowAsJavaScriptException();
}

GraphWrap::~GraphWrap()
{
    if (graph_)
    {
        TF_DeleteGraph(graph_);
        graph_ = nullptr;
    }
}

// ---------------------------------------------------------------------------
// addOp — JS signature:
//   addOp(
//     type:          string,
//     inputs:        { opName: string; index: number }[],
//     attrs?:        Record<string, AttrValue>,
//     name?:         string,
//     controlInputs?: string[],   // ← NEW: op names that must run first
//   ) -> { opName: string; numOutputs: number }
//
// Control inputs are wired via TF_AddControlInput.  TF guarantees the
// described op will not execute until every control-input op has completed.
// This is the mechanism behind globalVariablesInitializer: a NoOp with
// control edges to all init AssignVariableOps — running the NoOp forces all
// assignments to execute first.
// ---------------------------------------------------------------------------
Napi::Value GraphWrap::AddOp(const Napi::CallbackInfo &info)
{
    Napi::Env env = info.Env();

    if (!graph_)
    {
        Napi::Error::New(env, "Graph has been destroyed")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    if (info.Length() < 2 || !info[0].IsString() || !info[1].IsArray())
    {
        Napi::TypeError::New(env,
                             "addOp(type, inputs, attrs?, name?, controlInputs?)")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    std::string op_type = info[0].As<Napi::String>().Utf8Value();
    std::string op_name = op_type + "_" + std::to_string(op_counter_++);
    if (info.Length() >= 4 && info[3].IsString())
        op_name = info[3].As<Napi::String>().Utf8Value();

    // ── Resolve data inputs ─────────────────────────────────────────────────
    std::vector<TF_Output> resolved_inputs;
    auto inputs_arr = info[1].As<Napi::Array>();
    resolved_inputs.reserve(inputs_arr.Length());
    for (uint32_t i = 0; i < inputs_arr.Length(); i++)
    {
        auto obj = inputs_arr.Get(i).As<Napi::Object>();
        std::string dep_name = obj.Get("opName").As<Napi::String>().Utf8Value();
        int dep_idx = obj.Get("index").As<Napi::Number>().Int32Value();
        TF_Operation *dep_op = TF_GraphOperationByName(graph_, dep_name.c_str());
        if (!dep_op)
        {
            Napi::Error::New(env, "Input op not found: " + dep_name)
                .ThrowAsJavaScriptException();
            return env.Undefined();
        }
        resolved_inputs.push_back({dep_op, dep_idx});
    }

    // ── Resolve control inputs (arg index 4) ───────────────────────────────
    std::vector<TF_Operation *> ctrl_ops;
    if (info.Length() >= 5 && info[4].IsArray())
    {
        auto ctrl_arr = info[4].As<Napi::Array>();
        ctrl_ops.reserve(ctrl_arr.Length());
        for (uint32_t i = 0; i < ctrl_arr.Length(); i++)
        {
            std::string ctrl_name =
                ctrl_arr.Get(i).As<Napi::String>().Utf8Value();
            TF_Operation *ctrl_op =
                TF_GraphOperationByName(graph_, ctrl_name.c_str());
            if (!ctrl_op)
            {
                Napi::Error::New(env,
                                 "Control input op not found: " + ctrl_name)
                    .ThrowAsJavaScriptException();
                return env.Undefined();
            }
            ctrl_ops.push_back(ctrl_op);
        }
    }

    // ── Attr dispatch ───────────────────────────────────────────────────────
    enum class AttrKind
    {
        Int,
        Float,
        Bool,
        Type,
        Shape,
        ListType,
        ListInt,
        Tensor,
        String,
        Unknown
    };
    static const std::unordered_map<std::string, AttrKind> kind_map = {
        {"int", AttrKind::Int},
        {"float", AttrKind::Float},
        {"bool", AttrKind::Bool},
        {"type", AttrKind::Type},
        {"shape", AttrKind::Shape},
        {"list_type", AttrKind::ListType},
        {"list_int", AttrKind::ListInt},
        {"tensor", AttrKind::Tensor},
        {"string", AttrKind::String},
    };

    auto apply_attrs = [&](TF_OperationDescription *desc) -> bool
    {
        if (!(info.Length() >= 3 && info[2].IsObject()))
            return true;

        auto attrs = info[2].As<Napi::Object>();
        auto attrs_keys = attrs.GetPropertyNames();
        for (uint32_t i = 0; i < attrs_keys.Length(); i++)
        {
            std::string attr_name =
                attrs_keys.Get(i).As<Napi::String>().Utf8Value();
            auto attr_val = attrs.Get(attr_name).As<Napi::Object>();
            std::string kind_str =
                attr_val.Get("kind").As<Napi::String>().Utf8Value();

            auto it = kind_map.find(kind_str);
            AttrKind kind = (it != kind_map.end()) ? it->second
                                                   : AttrKind::Unknown;
            switch (kind)
            {
            case AttrKind::Int:
            {
                int64_t v = static_cast<int64_t>(
                    attr_val.Get("value").As<Napi::Number>().Int64Value());
                TF_SetAttrInt(desc, attr_name.c_str(), v);
                break;
            }
            case AttrKind::Float:
            {
                float f = attr_val.Get("value").As<Napi::Number>().FloatValue();
                TF_SetAttrFloat(desc, attr_name.c_str(), f);
                break;
            }
            case AttrKind::Bool:
            {
                unsigned char b =
                    attr_val.Get("value").As<Napi::Boolean>().Value() ? 1 : 0;
                TF_SetAttrBool(desc, attr_name.c_str(), b);
                break;
            }
            case AttrKind::Type:
            {
                TF_DataType v = static_cast<TF_DataType>(
                    attr_val.Get("value").As<Napi::Number>().Int32Value());
                TF_SetAttrType(desc, attr_name.c_str(), v);
                break;
            }
            case AttrKind::Shape:
            {
                auto dims_arr = attr_val.Get("value").As<Napi::Array>();
                std::vector<int64_t> dims(dims_arr.Length());
                for (uint32_t j = 0; j < dims_arr.Length(); ++j)
                    dims[j] = static_cast<int64_t>(
                        dims_arr.Get(j).As<Napi::Number>().Int64Value());
                TF_SetAttrShape(desc, attr_name.c_str(),
                                dims.data(), static_cast<int>(dims.size()));
                break;
            }
            case AttrKind::ListType:
            {
                auto vals = attr_val.Get("value").As<Napi::Array>();
                std::vector<TF_DataType> types(vals.Length());
                for (uint32_t j = 0; j < vals.Length(); ++j)
                    types[j] = static_cast<TF_DataType>(
                        vals.Get(j).As<Napi::Number>().Int32Value());
                TF_SetAttrTypeList(desc, attr_name.c_str(),
                                   types.data(),
                                   static_cast<int>(types.size()));
                break;
            }
            case AttrKind::ListInt:
            {
                auto vals_int = attr_val.Get("value").As<Napi::Array>();
                std::vector<int64_t> ints(vals_int.Length());
                for (uint32_t j = 0; j < vals_int.Length(); ++j)
                    ints[j] = static_cast<int64_t>(
                        vals_int.Get(j).As<Napi::Number>().Int64Value());
                TF_SetAttrIntList(desc, attr_name.c_str(),
                                  ints.data(),
                                  static_cast<int>(ints.size()));
                break;
            }
            case AttrKind::Tensor:
            {
                auto tv = attr_val.Get("value").As<Napi::Object>();
                TF_DataType dtype = static_cast<TF_DataType>(
                    tv.Get("dtype").As<Napi::Number>().Int32Value());
                auto data_buf = tv.Get("data").As<Napi::Buffer<uint8_t>>();
                auto dims_arr = tv.Get("shape").As<Napi::Array>();
                std::vector<int64_t> dims(dims_arr.Length());
                for (uint32_t j = 0; j < dims_arr.Length(); ++j)
                    dims[j] = static_cast<int64_t>(
                        dims_arr.Get(j).As<Napi::Number>().Int64Value());
                StatusGuard ts;
                TF_Tensor *tensor = TF_AllocateTensor(
                    dtype, dims.data(), static_cast<int>(dims.size()),
                    data_buf.Length());
                if (tensor)
                {
                    std::memcpy(TF_TensorData(tensor),
                                data_buf.Data(), data_buf.ByteLength());
                    TF_SetAttrTensor(desc, attr_name.c_str(), tensor, ts.s);
                    TF_DeleteTensor(tensor);
                }
                break;
            }
            case AttrKind::String:
            {
                // TF_SetAttrString expects raw bytes + length.
                // The JS value is a JS string — convert via Utf8Value().
                std::string sv;
                auto vnode = attr_val.Get("value");
                if (vnode.IsString())
                    sv = vnode.As<Napi::String>().Utf8Value();
                TF_SetAttrString(desc, attr_name.c_str(),
                                 sv.data(), sv.size());
                break;
            }
            default:
                Napi::Error::New(env, "Unsupported attr kind: " + kind_str)
                    .ThrowAsJavaScriptException();
                return false;
            }
        }
        return true;
    };

    // ── Build the op description ────────────────────────────────────────────
    auto finish_op = [&](bool list_input,
                         std::string &err) -> TF_Operation *
    {
        StatusGuard status;
        TF_OperationDescription *desc =
            TF_NewOperation(graph_, op_type.c_str(), op_name.c_str());
        if (!desc)
        {
            err = "TF_NewOperation failed for " + op_type;
            return nullptr;
        }

        if (list_input)
        {
            if (!resolved_inputs.empty())
                TF_AddInputList(desc, resolved_inputs.data(),
                                static_cast<int>(resolved_inputs.size()));
        }
        else
        {
            for (const auto &inp : resolved_inputs)
                TF_AddInput(desc, inp);
        }

        // Wire control dependencies — these guarantee ordering without
        // passing tensor data. The described op will not run until all
        // control-input ops have finished executing.
        for (TF_Operation *ctrl : ctrl_ops)
            TF_AddControlInput(desc, ctrl);

        if (!apply_attrs(desc))
            return nullptr;

        TF_Operation *op = TF_FinishOperation(desc, status.s);
        if (!status.ok() || !op)
        {
            err = status.message();
            return nullptr;
        }
        return op;
    };

    std::string first_error;
    TF_Operation *op = finish_op(false, first_error);

    // Some ops have a single list-valued input arg — retry with AddInputList.
    if (!op &&
        first_error.find("expected list") != std::string::npos &&
        !resolved_inputs.empty())
    {
        std::string retry_error;
        op = finish_op(true, retry_error);
        if (!op)
            first_error = retry_error;
    }

    if (!op)
    {
        if (!env.IsExceptionPending())
            Napi::Error::New(env,
                             "TF_FinishOperation failed for " + op_type +
                                 ": " + first_error)
                .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    Napi::Object result = Napi::Object::New(env);
    result.Set("opName", Napi::String::New(env, op_name));
    result.Set("numOutputs", Napi::Number::New(env,
                                               static_cast<double>(TF_OperationNumOutputs(op))));
    return result;
}

// ---------------------------------------------------------------------------
// addGradients — JS signature:
//   addGradients(
//     y:   { opName: string; index: number }[],   // loss outputs
//     x:   { opName: string; index: number }[],   // inputs to diff w.r.t.
//     dx?: { opName: string; index: number }[],   // initial upstream grads
//   ) -> { opName: string; index: number }[]      // gradient tensors, len = |x|
//
// Wraps TF_AddGradients.  TF injects gradient ops directly into the graph.
// The returned outputs are the dL/dx_i tensors, one per x_i.
//
// dx defaults to ones (i.e. dL/dy = 1 for scalar loss, standard convention).
// Pass explicit dx when you need to chain gradients or scale the loss.
//
// Throws if any op in the path is non-differentiable (e.g. ArgMax).
// ---------------------------------------------------------------------------
Napi::Value GraphWrap::AddGradients(const Napi::CallbackInfo &info)
{
    Napi::Env env = info.Env();

    if (!graph_)
    {
        Napi::Error::New(env, "Graph destroyed").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    if (info.Length() < 2 || !info[0].IsArray() || !info[1].IsArray())
    {
        Napi::TypeError::New(env, "addGradients(y, x, dx?)")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    auto resolve_outputs = [&](Napi::Array arr,
                               std::vector<TF_Output> &out,
                               const std::string &label) -> bool
    {
        for (uint32_t i = 0; i < arr.Length(); ++i)
        {
            auto obj = arr.Get(i).As<Napi::Object>();
            std::string name = obj.Get("opName").As<Napi::String>().Utf8Value();
            int idx = obj.Get("index").As<Napi::Number>().Int32Value();
            TF_Operation *op = TF_GraphOperationByName(graph_, name.c_str());
            if (!op)
            {
                Napi::Error::New(env, label + " op not found: " + name)
                    .ThrowAsJavaScriptException();
                return false;
            }
            out.push_back({op, idx});
        }
        return true;
    };

    std::vector<TF_Output> y_vec, x_vec, dx_vec;

    if (!resolve_outputs(info[0].As<Napi::Array>(), y_vec, "y"))
        return env.Undefined();
    if (!resolve_outputs(info[1].As<Napi::Array>(), x_vec, "x"))
        return env.Undefined();

    // Optional initial gradients.
    TF_Output *dx_ptr = nullptr;
    if (info.Length() >= 3 && info[2].IsArray())
    {
        if (!resolve_outputs(info[2].As<Napi::Array>(), dx_vec, "dx"))
            return env.Undefined();
        if (dx_vec.size() != y_vec.size())
        {
            Napi::Error::New(env,
                             "addGradients: dx length must equal y length")
                .ThrowAsJavaScriptException();
            return env.Undefined();
        }
        dx_ptr = dx_vec.data();
    }

    // Allocate output array — TF_AddGradients writes one gradient per x.
    std::vector<TF_Output> dy(x_vec.size());

    StatusGuard status;
    TF_AddGradients(
        graph_,
        y_vec.data(), static_cast<int>(y_vec.size()),
        x_vec.data(), static_cast<int>(x_vec.size()),
        dx_ptr,
        status.s,
        dy.data());

    if (!status.ok())
    {
        Napi::Error::New(env,
                         "TF_AddGradients failed: " + status.message())
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    // Return array of { opName, index } — same shape as x.
    Napi::Array result = Napi::Array::New(env, dy.size());
    for (size_t i = 0; i < dy.size(); ++i)
    {
        const char *raw_name = TF_OperationName(dy[i].oper);
        Napi::Object obj = Napi::Object::New(env);
        obj.Set("opName", Napi::String::New(env, raw_name ? raw_name : ""));
        obj.Set("index", Napi::Number::New(env, dy[i].index));
        result.Set(static_cast<uint32_t>(i), obj);
    }
    return result;
}

// ---------------------------------------------------------------------------
// Remaining methods — unchanged from previous iteration
// ---------------------------------------------------------------------------

Napi::Value GraphWrap::HasOp(const Napi::CallbackInfo &info)
{
    if (!info[0].IsString())
        return Napi::Boolean::New(info.Env(), false);
    std::string name = info[0].As<Napi::String>().Utf8Value();
    return Napi::Boolean::New(info.Env(),
                              graph_ && TF_GraphOperationByName(graph_, name.c_str()) != nullptr);
}

Napi::Value GraphWrap::OpOutputType(const Napi::CallbackInfo &info)
{
    Napi::Env env = info.Env();
    if (!graph_ || !info[0].IsString())
        return env.Null();
    std::string name = info[0].As<Napi::String>().Utf8Value();
    int idx = info.Length() >= 2 ? info[1].As<Napi::Number>().Int32Value() : 0;
    TF_Operation *op = TF_GraphOperationByName(graph_, name.c_str());
    if (!op)
        return env.Null();
    return Napi::Number::New(env,
                             static_cast<double>(TF_OperationOutputType({op, idx})));
}

Napi::Value GraphWrap::OpOutputShape(const Napi::CallbackInfo &info)
{
    Napi::Env env = info.Env();
    if (!graph_ || !info[0].IsString())
        return env.Null();
    std::string name = info[0].As<Napi::String>().Utf8Value();
    int idx = info.Length() >= 2 ? info[1].As<Napi::Number>().Int32Value() : 0;
    TF_Operation *op = TF_GraphOperationByName(graph_, name.c_str());
    if (!op)
        return env.Null();

    TF_Output out{op, idx};
    StatusGuard s1;
    int ndims = TF_GraphGetTensorNumDims(graph_, out, s1.s);
    if (!s1.ok() || ndims < 0)
        return env.Null();

    std::vector<int64_t> dims(ndims, -1);
    StatusGuard s2;
    TF_GraphGetTensorShape(graph_, out, dims.data(), ndims, s2.s);

    Napi::Array arr = Napi::Array::New(env, ndims);
    for (int i = 0; i < ndims; ++i)
        arr.Set(i, Napi::Number::New(env, static_cast<double>(dims[i])));
    return arr;
}

Napi::Value GraphWrap::ToGraphDef(const Napi::CallbackInfo &info)
{
    Napi::Env env = info.Env();
    if (!graph_)
        return env.Null();

    StatusGuard status;
    TF_Buffer *buf = TF_NewBuffer();
    TF_GraphToGraphDef(graph_, buf, status.s);
    if (!status.ok())
    {
        TF_DeleteBuffer(buf);
        Napi::Error::New(env,
                         "TF_GraphToGraphDef failed: " + status.message())
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }
    auto node_buf = Napi::Buffer<uint8_t>::Copy(
        env,
        reinterpret_cast<const uint8_t *>(buf->data), buf->length);
    TF_DeleteBuffer(buf);
    return node_buf;
}

Napi::Value GraphWrap::NumOps(const Napi::CallbackInfo &info)
{
    if (!graph_)
        return Napi::Number::New(info.Env(), 0);
    size_t pos = 0;
    int count = 0;
    while (TF_GraphNextOperation(graph_, &pos))
        ++count;
    return Napi::Number::New(info.Env(), count);
}

Napi::Value GraphWrap::ImportGraphDef(const Napi::CallbackInfo &info)
{
    Napi::Env env = info.Env();
    if (!graph_)
    {
        Napi::Error::New(env, "Graph destroyed").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    if (info.Length() < 1 || !info[0].IsBuffer())
    {
        Napi::TypeError::New(env, "importGraphDef(buffer: Buffer)")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    auto buf = info[0].As<Napi::Buffer<uint8_t>>();
    TF_Buffer *graphdef = TF_NewBufferFromString(buf.Data(), buf.ByteLength());
    TF_ImportGraphDefOptions *opts = TF_NewImportGraphDefOptions();
    TF_ImportGraphDefOptionsSetPrefix(opts, "");

    StatusGuard status;
    TF_GraphImportGraphDef(graph_, graphdef, opts, status.s);
    TF_DeleteImportGraphDefOptions(opts);
    TF_DeleteBuffer(graphdef);

    if (!status.ok())
        Napi::Error::New(env,
                         "TF_GraphImportGraphDef failed: " + status.message())
            .ThrowAsJavaScriptException();
    return env.Undefined();
}

Napi::Value GraphWrap::ListOpsOfType(const Napi::CallbackInfo &info)
{
    Napi::Env env = info.Env();
    if (!graph_ || !info[0].IsString())
    {
        return Napi::Array::New(env, 0);
    }
    std::string target_type = info[0].As<Napi::String>().Utf8Value();

    std::vector<std::string> matches;
    size_t pos = 0;
    TF_Operation *op = nullptr;
    while ((op = TF_GraphNextOperation(graph_, &pos)) != nullptr)
    {
        const char *op_type = TF_OperationOpType(op);
        if (op_type && target_type == op_type)
        {
            matches.push_back(TF_OperationName(op));
        }
    }

    Napi::Array result = Napi::Array::New(env, matches.size());
    for (size_t i = 0; i < matches.size(); ++i)
        result.Set(static_cast<uint32_t>(i), Napi::String::New(env, matches[i]));
    return result;
}

Napi::Value GraphWrap::ListSinkOps(const Napi::CallbackInfo &info)
{
    Napi::Env env = info.Env();
    if (!graph_)
        return Napi::Array::New(env, 0);

    // Op types that must never be returned as output ops.
    // These are control/infrastructure nodes that appear in frozen SavedModel
    // graphs but have zero outputs or are not tensor-producing ops.
    //   NoOp          — zero outputs, used as a control dependency fence
    //   VarHandleOp   — resource handle for a variable, not a model output
    //   Placeholder   — graph inputs, explicitly not outputs
    //   Const         — constant initializer values, not inference outputs
    //   AssignVariableOp / ReadVariableOp — variable lifecycle ops
    static const std::unordered_set<std::string> excluded_types = {
        "NoOp",
        "VarHandleOp",
        "Placeholder",
        "Const",
        "AssignVariableOp",
        "ReadVariableOp",
        "SaveV2",
        "RestoreV2",
        "MergeV2Checkpoints",
        "StringJoin",
        "ShardedFilename",
        "_Arg",
        "_Retval",
    };

    // Collect every op that is consumed as a data input by some other op.
    std::unordered_set<TF_Operation *> consumed;

    size_t pos = 0;
    TF_Operation *op = nullptr;
    while ((op = TF_GraphNextOperation(graph_, &pos)) != nullptr)
    {
        int num_inputs = TF_OperationNumInputs(op);
        for (int i = 0; i < num_inputs; ++i)
        {
            TF_Output src = TF_OperationInput({op, i});
            consumed.insert(src.oper);
        }
    }

    // A sink op is one that:
    //   1. Is not consumed as an input by any other op
    //   2. Has at least one output (excludes NoOp and other zero-output nodes)
    //   3. Is not an excluded infrastructure op type
    std::vector<std::string> sinks;
    pos = 0;
    while ((op = TF_GraphNextOperation(graph_, &pos)) != nullptr)
    {
        if (consumed.find(op) != consumed.end())
            continue;
        if (TF_OperationNumOutputs(op) < 1)
            continue;
        const char *op_type = TF_OperationOpType(op);
        if (op_type && excluded_types.count(op_type))
            continue;
        sinks.push_back(TF_OperationName(op));
    }

    Napi::Array result = Napi::Array::New(env, sinks.size());
    for (size_t i = 0; i < sinks.size(); ++i)
        result.Set(static_cast<uint32_t>(i), Napi::String::New(env, sinks[i]));
    return result;
}