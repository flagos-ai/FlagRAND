#include <acl/acl.h>
#include <acl/acl_op_compiler.h>
#include <aclnnop/aclnn_exp.h>
#include <aclnnop/aclnn_normal.h>
#include <aclnnop/aclnn_uniform.h>

#include <cstdint>

// api: 0=raw32, 1=uniform, 2=normal, 3=lognormal, 5=Poisson.
// dtype: 0=int32, 1=float32, 2=float64. The caller owns output and stream.
struct CannRng {
    int api;
    int dtype;
    int64_t n;
    int64_t seed;
    int64_t offset;
    int64_t offset_step;
    size_t bytes;
    aclrtStream stream = nullptr;
    void* output = nullptr;
    aclTensor* tensor = nullptr;
    void* workspace = nullptr;
    uint64_t workspace_bytes = 0;
    void* exp_workspace = nullptr;
    uint64_t exp_workspace_bytes = 0;
    const char* graph_operation = nullptr;
    int input_count = 0;
    int32_t alg = 1;
    uint64_t key_host = 0;
    void* key = nullptr;
    void* rate = nullptr;
    void* state = nullptr;
    uint64_t* uploads = nullptr;
    size_t upload_slot = 0;
    aclTensorDesc* inputs[4] = {};
    aclDataBuffer* input_buffers[4] = {};
    aclTensorDesc* output_desc = nullptr;
    aclDataBuffer* output_buffer = nullptr;
    aclopAttr* attributes = nullptr;
};

static const char* last_operation = "none";
#define CALL(expression) do { last_operation = #expression; const int status = (expression); if (status != 0) return status; } while (0)

static bool uses_graph(const CannRng* h) {
    return h->api == 0 || h->api == 5 || h->dtype == 2;
}

static int query_rng(CannRng* h, uint64_t* bytes, aclOpExecutor** executor) {
    if (h->api == 1) {
        CALL(aclnnInplaceUniformGetWorkspaceSize(h->tensor, 0.0, 1.0, static_cast<uint64_t>(h->seed),
                                               static_cast<uint64_t>(h->offset), bytes, executor));
    } else {
        CALL(aclnnInplaceNormalGetWorkspaceSize(h->tensor, 0.0f, 1.0f, h->seed, h->offset, bytes, executor));
    }
    return 0;
}

static int run_rng(CannRng* h, uint64_t bytes, aclOpExecutor* executor) {
    if (h->api == 1) {
        CALL(aclnnInplaceUniform(h->workspace, bytes, executor, h->stream));
    } else {
        CALL(aclnnInplaceNormal(h->workspace, bytes, executor, h->stream));
    }
    return 0;
}

extern "C" const char* cann_rng_last_operation() { return last_operation; }
extern "C" const char* cann_rng_recent_error() { return aclGetRecentErrMsg(); }

// One immutable pinned seed/counter per queued call; reuse after synchronization.
extern "C" int cann_rng_reserve_calls(CannRng* h, int64_t calls) {
    if (!uses_graph(h)) return 0;
    CALL(aclrtSynchronizeStream(h->stream));
    if (h->uploads != nullptr) CALL(aclrtFreeHost(h->uploads));
    CALL(aclrtMallocHost(reinterpret_cast<void**>(&h->uploads), calls * 2 * sizeof(uint64_t)));
    h->upload_slot = 0;
    return 0;
}

static int run_graph(CannRng* h) {
    uint64_t* state = h->uploads + 2 * h->upload_slot++;
    state[0] = static_cast<uint64_t>(h->api == 5 ? h->seed : h->offset);
    state[1] = h->api == 5 ? static_cast<uint64_t>(h->offset) : 0;
    CALL(aclrtMemcpyAsync(h->state, 2 * sizeof(uint64_t), state, 2 * sizeof(uint64_t),
                         ACL_MEMCPY_HOST_TO_DEVICE, h->stream));
    const aclTensorDesc* inputs[] = {h->inputs[0], h->inputs[1], h->inputs[2], h->inputs[3]};
    const aclDataBuffer* buffers[] = {
        h->input_buffers[0], h->input_buffers[1], h->input_buffers[2], h->input_buffers[3]};
    const aclTensorDesc* outputs[] = {h->output_desc};
    aclDataBuffer* output_buffers[] = {h->output_buffer};
    CALL(aclopCompileAndExecute(h->graph_operation, h->input_count, inputs, buffers, 1, outputs,
                               output_buffers, h->attributes, ACL_ENGINE_SYS, ACL_COMPILE_SYS, nullptr, h->stream));
    return 0;
}

extern "C" int cann_rng_generate(CannRng* h) {
    if (uses_graph(h)) {
        CALL(run_graph(h));
    } else {
        uint64_t bytes = 0;
        aclOpExecutor* executor = nullptr;
        CALL(query_rng(h, &bytes, &executor));
        CALL(run_rng(h, bytes, executor));
    }
    if (h->api == 3) {
        uint64_t bytes = 0;
        aclOpExecutor* executor = nullptr;
        CALL(aclnnInplaceExpGetWorkspaceSize(h->tensor, &bytes, &executor));
        CALL(aclnnInplaceExp(h->exp_workspace, bytes, executor, h->stream));
    }
    h->offset += h->offset_step;
    return 0;
}

static int setup_graph(CannRng* h, double lam) {
    CALL(aclSetCompileopt(ACL_OP_JIT_COMPILE, "enable"));
    const int64_t one[] = {1}, two[] = {2}, dims[] = {h->n};
    h->inputs[0] = aclCreateTensorDesc(ACL_INT64, 1, one, ACL_FORMAT_ND);
    CALL(aclSetTensorPlaceMent(h->inputs[0], ACL_MEMTYPE_HOST));
    CALL(aclSetTensorConst(h->inputs[0], &h->n, sizeof(int64_t)));
    h->input_buffers[0] = aclCreateDataBuffer(&h->n, sizeof(int64_t));
    CALL(aclrtMalloc(&h->state, 2 * sizeof(uint64_t), ACL_MEM_MALLOC_HUGE_FIRST));
    if (h->api == 5) {
        h->graph_operation = "StatelessRandomPoisson";
        h->input_count = 3;
        const float rate = static_cast<float>(lam);
        CALL(aclrtMalloc(&h->rate, sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
        CALL(aclrtMemcpy(h->rate, sizeof(float), &rate, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
        h->inputs[1] = aclCreateTensorDesc(ACL_INT64, 1, two, ACL_FORMAT_ND);
        h->inputs[2] = aclCreateTensorDesc(ACL_FLOAT, 0, nullptr, ACL_FORMAT_ND);
        h->input_buffers[1] = aclCreateDataBuffer(h->state, 2 * sizeof(int64_t));
        h->input_buffers[2] = aclCreateDataBuffer(h->rate, sizeof(float));
    } else {
        h->graph_operation = h->api == 0 ? "StatelessRandomUniformFullIntV2"
                             : h->api == 1 ? "StatelessRandomUniformV2" : "StatelessRandomNormalV2";
        h->input_count = 4;
        h->key_host = static_cast<uint64_t>(h->seed);
        CALL(aclrtMalloc(&h->key, sizeof(uint64_t), ACL_MEM_MALLOC_HUGE_FIRST));
        CALL(aclrtMemcpy(h->key, sizeof(uint64_t), &h->key_host, sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE));
        h->inputs[1] = aclCreateTensorDesc(ACL_UINT64, 1, one, ACL_FORMAT_ND);
        h->inputs[2] = aclCreateTensorDesc(ACL_UINT64, 1, two, ACL_FORMAT_ND);
        h->inputs[3] = aclCreateTensorDesc(ACL_INT32, 0, nullptr, ACL_FORMAT_ND);
        CALL(aclSetTensorPlaceMent(h->inputs[3], ACL_MEMTYPE_HOST));
        CALL(aclSetTensorConst(h->inputs[3], &h->alg, sizeof(int32_t)));
        h->input_buffers[1] = aclCreateDataBuffer(h->key, sizeof(uint64_t));
        h->input_buffers[2] = aclCreateDataBuffer(h->state, 2 * sizeof(uint64_t));
        h->input_buffers[3] = aclCreateDataBuffer(&h->alg, sizeof(int32_t));
    }
    const aclDataType dtype = h->dtype == 2 ? ACL_DOUBLE : ACL_INT32;
    h->output_desc = aclCreateTensorDesc(dtype, 1, dims, ACL_FORMAT_ND);
    h->output_buffer = aclCreateDataBuffer(h->output, h->bytes);
    h->attributes = aclopCreateAttr();
    CALL(aclopSetAttrDataType(h->attributes, "dtype", dtype));
    CALL(cann_rng_reserve_calls(h, 1));
    return 0;
}

static int setup(CannRng* h, double lam) {
    if (!uses_graph(h) || h->api == 3) {
        const int64_t dims[] = {h->n}, strides[] = {1};
        h->tensor = aclCreateTensor(dims, 1, h->dtype == 2 ? ACL_DOUBLE : ACL_FLOAT,
                                    strides, 0, ACL_FORMAT_ND, dims, 1, h->output);
    }
    uint64_t bytes = 0;
    aclOpExecutor* executor = nullptr;
    if (uses_graph(h)) {
        CALL(setup_graph(h, lam));
        CALL(run_graph(h));
    } else {
        CALL(query_rng(h, &bytes, &executor));
        h->workspace_bytes = bytes;
        if (bytes != 0) CALL(aclrtMalloc(&h->workspace, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
        CALL(run_rng(h, bytes, executor));
    }
    if (h->api == 3) {
        aclOpExecutor* exp_executor = nullptr;
        CALL(aclnnInplaceExpGetWorkspaceSize(h->tensor, &h->exp_workspace_bytes, &exp_executor));
        if (h->exp_workspace_bytes != 0) {
            CALL(aclrtMalloc(&h->exp_workspace, h->exp_workspace_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
        }
        CALL(aclnnInplaceExp(h->exp_workspace, h->exp_workspace_bytes, exp_executor, h->stream));
    }
    CALL(aclrtSynchronizeStream(h->stream));
    h->offset = 0;
    h->upload_slot = 0;
    return 0;
}

// The caller has initialized ACL, selected the device, and owns output/stream.
extern "C" int cann_rng_bind(int api, int dtype, int64_t n, int64_t seed, int64_t offset_step,
                              double lam, void* output, void* stream, CannRng** result) {
    auto* h = new CannRng{api, dtype, n, seed, 0, offset_step, static_cast<size_t>(n) * (dtype == 2 ? 8 : 4)};
    h->output = output;
    h->stream = static_cast<aclrtStream>(stream);
    *result = h;
    CALL(setup(h, lam));
    return 0;
}

extern "C" int cann_rng_set_state(CannRng* h, int64_t seed, int64_t offset) {
    if (uses_graph(h) && h->api != 5) {
        // A key change must not alter an earlier queued graph call.
        CALL(aclrtSynchronizeStream(h->stream));
        h->upload_slot = 0;
        h->key_host = static_cast<uint64_t>(seed);
        CALL(aclrtMemcpy(h->key, sizeof(uint64_t), &h->key_host, sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE));
    }
    h->seed = seed;
    h->offset = offset;
    return 0;
}

extern "C" int cann_rng_synchronize(CannRng* h) {
    CALL(aclrtSynchronizeStream(h->stream));
    h->upload_slot = 0;
    return 0;
}

extern "C" uint64_t cann_rng_workspace_bytes(CannRng* h) {
    return h->workspace_bytes + h->exp_workspace_bytes;
}

extern "C" int cann_rng_destroy(CannRng* h) {
    CALL(aclrtSynchronizeStream(h->stream));
    if (uses_graph(h)) {
        for (int i = 0; i < h->input_count; ++i) {
            aclDestroyTensorDesc(h->inputs[i]);
            aclDestroyDataBuffer(h->input_buffers[i]);
        }
        aclDestroyTensorDesc(h->output_desc);
        aclDestroyDataBuffer(h->output_buffer);
        aclopDestroyAttr(h->attributes);
        CALL(aclrtFree(h->state));
        CALL(aclrtFreeHost(h->uploads));
        if (h->api == 5) CALL(aclrtFree(h->rate));
        else CALL(aclrtFree(h->key));
    }
    if (h->tensor != nullptr) CALL(aclDestroyTensor(h->tensor));
    if (h->workspace_bytes != 0) CALL(aclrtFree(h->workspace));
    if (h->exp_workspace_bytes != 0) CALL(aclrtFree(h->exp_workspace));
    delete h;
    return 0;
}
