#include <acl/acl.h>
#include <aclnnop/aclnn_exp.h>

#include <cstdint>
#include <cstring>

struct RuntimeRng {
    int api;
    aclrtRandomNumTaskInfo task{};
    aclrtStream stream = nullptr;
    void* counter = nullptr;
    void* parameters = nullptr;
    aclTensor* tensor = nullptr;
    void* exp_workspace = nullptr;
    uint64_t exp_workspace_bytes = 0;
    int task_type_before = -1;
    int task_type_after = -1;
};

static thread_local const char* last_operation = "none";
#define CALL(expression) do { last_operation = #expression; const int code = (expression); if (code) return code; } while (0)

template <typename T>
static void immediate(aclrtRandomParaInfo& parameter, T value) {
    parameter = {};
    std::memcpy(parameter.valueOrAddr, &value, sizeof(value));
    parameter.size = sizeof(value);
}

extern "C" const char* runtime_rng_last_operation() { return last_operation; }
extern "C" const char* runtime_rng_recent_error() { return aclGetRecentErrMsg(); }

static int submit_random(RuntimeRng* h) {
    // CANN 9.1 converts dataType from the ACL enum to the RT enum in place.
    // Submit a fresh local copy so the next call retains the ACL template.
    aclrtRandomNumTaskInfo task = h->task;
    h->task_type_before = static_cast<int>(task.dataType);
    last_operation = "aclrtRandomNumAsync(&task, h->stream, nullptr)";
    const int status = aclrtRandomNumAsync(&task, h->stream, nullptr);
    h->task_type_after = static_cast<int>(task.dataType);
    return status;
}

// api: 0=raw32, 1=uniform FP32, 2=normal FP32, 3=lognormal FP32.
// Torch owns device initialization, the output allocation and the stream.
extern "C" int runtime_rng_bind(int api, uint64_t n, uint64_t seed,
                                void* output, void* stream, RuntimeRng** result) {
    auto* h = new RuntimeRng;
    *result = h;
    h->api = api;
    h->stream = static_cast<aclrtStream>(stream);
    CALL(aclrtMalloc(&h->counter, 16, ACL_MEM_MALLOC_HUGE_FIRST));
    // Runtime 9.1 SetRandomSqeCommonInfo writes two 8-byte parameters here.
    CALL(aclrtMalloc(&h->parameters, 16, ACL_MEM_MALLOC_HUGE_FIRST));
    CALL(aclrtMemset(h->counter, 16, 0, 16));
    h->task.dataType = api == 0 ? ACL_UINT32 : ACL_FLOAT;
    h->task.randomNumFuncParaInfo.funcType = api >= 2
        ? ACL_RT_RANDOM_NUM_FUNC_TYPE_NORMAL_DIS : ACL_RT_RANDOM_NUM_FUNC_TYPE_UNIFORM_DIS;
    h->task.randomParaAddr = h->parameters;
    h->task.randomCounterAddr = h->counter;
    h->task.randomResultAddr = output;
    immediate(h->task.randomSeed, seed);
    immediate(h->task.randomNum, n);
    if (api == 0) {
        immediate(h->task.randomNumFuncParaInfo.paramInfo.uniformDisInfo.min, uint32_t{0});
        immediate(h->task.randomNumFuncParaInfo.paramInfo.uniformDisInfo.max, UINT32_MAX);
    } else if (api == 1) {
        immediate(h->task.randomNumFuncParaInfo.paramInfo.uniformDisInfo.min, 0.0f);
        immediate(h->task.randomNumFuncParaInfo.paramInfo.uniformDisInfo.max, 1.0f);
    } else {
        immediate(h->task.randomNumFuncParaInfo.paramInfo.normalDisInfo.mean, 0.0f);
        immediate(h->task.randomNumFuncParaInfo.paramInfo.normalDisInfo.stddev, 1.0f);
    }
    CALL(submit_random(h));
    if (api == 3) {
        const int64_t dims[] = {static_cast<int64_t>(n)}, strides[] = {1};
        h->tensor = aclCreateTensor(dims, 1, ACL_FLOAT, strides, 0, ACL_FORMAT_ND, dims, 1, output);
        aclOpExecutor* executor = nullptr;
        CALL(aclnnInplaceExpGetWorkspaceSize(h->tensor, &h->exp_workspace_bytes, &executor));
        if (h->exp_workspace_bytes != 0) {
            CALL(aclrtMalloc(&h->exp_workspace, h->exp_workspace_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
        }
        CALL(aclnnInplaceExp(h->exp_workspace, h->exp_workspace_bytes, executor, h->stream));
    }
    CALL(aclrtSynchronizeStream(h->stream));
    CALL(aclrtMemset(h->counter, 16, 0, 16));
    return 0;
}

extern "C" int runtime_rng_generate(RuntimeRng* h, void* stream) {
    h->stream = static_cast<aclrtStream>(stream);
    CALL(submit_random(h));
    if (h->api == 3) {
        uint64_t bytes = 0;
        aclOpExecutor* executor = nullptr;
        // Two-phase executors are single-use; query and execute every call.
        CALL(aclnnInplaceExpGetWorkspaceSize(h->tensor, &bytes, &executor));
        CALL(aclnnInplaceExp(h->exp_workspace, bytes, executor, h->stream));
    }
    return 0;
}

extern "C" int runtime_rng_synchronize(RuntimeRng* h) {
    CALL(aclrtSynchronizeStream(h->stream));
    return 0;
}

extern "C" int runtime_rng_reset(RuntimeRng* h, uint64_t seed, uint64_t low, uint64_t high) {
    CALL(runtime_rng_synchronize(h));
    immediate(h->task.randomSeed, seed);
    const uint64_t counter[] = {low, high};
    CALL(aclrtMemcpy(h->counter, 16, counter, 16, ACL_MEMCPY_HOST_TO_DEVICE));
    return 0;
}

extern "C" void runtime_rng_task_types(RuntimeRng* h, int* output) {
    output[0] = static_cast<int>(h->task.dataType);
    output[1] = h->task_type_before;
    output[2] = h->task_type_after;
}

extern "C" int runtime_rng_read_counter(RuntimeRng* h, uint64_t* output) {
    CALL(runtime_rng_synchronize(h));
    CALL(aclrtMemcpy(output, 16, h->counter, 16, ACL_MEMCPY_DEVICE_TO_HOST));
    return 0;
}

extern "C" uint64_t runtime_rng_workspace_bytes(RuntimeRng* h) {
    return h->exp_workspace_bytes;
}

extern "C" int runtime_rng_destroy(RuntimeRng* h) {
    CALL(runtime_rng_synchronize(h));
    if (h->tensor != nullptr) CALL(aclDestroyTensor(h->tensor));
    if (h->exp_workspace_bytes != 0) CALL(aclrtFree(h->exp_workspace));
    CALL(aclrtFree(h->parameters));
    CALL(aclrtFree(h->counter));
    delete h;
    return 0;
}
