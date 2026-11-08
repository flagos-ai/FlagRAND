// CoreX SDK raw RNG plus CPU Poisson table / integer alias reference.
// This is a corrected native reference, not the original SDK Poisson API.
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand_mtgp32_host.h>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cfloat>
#include "integer_mrg.cuh"
extern "C" int probe_host_init(uint64_t seed, unsigned n, uint32_t *out);

constexpr unsigned kStreams = 32768;
constexpr unsigned kThreads = 256;
constexpr unsigned kMtgpStates = 200;
constexpr unsigned kMtgpCycle = kMtgpStates * kThreads;
using MtgpState = curandStateMtgp32_t;

struct Context {
    int engine;
    int api;
    float mean;
    float stddev;
    double lambda;
    void *states;
    mtgp32_kernel_params_t *params;
    uint32_t *tail;
    unsigned cursor;
    uint32_t *thresholds;
    uint32_t *aliases;
    unsigned table_size;
    int first;
    int last;
    double retained_mass;
    double chernoff_tail_bound;
};

static const char *last_operation = "none";
#define CUDA_RETURN(call) do { last_operation = #call; auto status = (call); if (status != cudaSuccess) return int(status); } while (0)
#define RNG_RETURN(call) do { last_operation = #call; auto status = (call); if (status != CURAND_STATUS_SUCCESS) return int(status); } while (0)

// Poisson probabilities are prepared on the CPU. Device sampling uses only
// original SDK raw words and integer table lookups; it does not call the
// SDK Poisson functions whose device double math failed the fixed-input probe.
static int build_alias(Context *context, double lambda) {
    const long double lam = static_cast<long double>(lambda);
    const int first = std::max(0, int(std::floor(lam - 14 * std::sqrt(lam) - 64)));
    const int last = int(std::ceil(lam + 14 * std::sqrt(lam) + 64));
    const unsigned support = unsigned(last - first + 1);
    unsigned size = 1;
    while (size < support) size <<= 1;
    std::vector<long double> probabilities(size, 0), scaled(size, 0);
    long double mass = 0;
    for (unsigned i = 0; i < support; ++i) {
        const long double k = first + i;
        probabilities[i] = std::exp(-lam + k * std::log(lam) - std::lgamma(k + 1));
        mass += probabilities[i];
    }
    std::vector<unsigned> low, high;
    for (unsigned i = 0; i < size; ++i) {
        scaled[i] = probabilities[i] * size / mass;
        (scaled[i] < 1 ? low : high).push_back(i);
    }
    std::vector<uint32_t> thresholds(size, UINT32_MAX), aliases(size);
    for (unsigned i = 0; i < size; ++i) aliases[i] = i;
    while (!low.empty() && !high.empty()) {
        const unsigned small = low.back(); low.pop_back();
        const unsigned large = high.back(); high.pop_back();
        const long double fixed = std::floor(std::max(0.L, scaled[small]) * 4294967296.L);
        thresholds[small] = uint32_t(std::min(fixed, 4294967295.L));
        aliases[small] = large;
        scaled[large] -= 1 - scaled[small];
        (scaled[large] < 1 ? low : high).push_back(large);
    }
    context->first = first;
    context->last = last;
    context->retained_mass = double(mass);
    const long double upper = last + 1;
    long double tail_bound = std::exp(-lam + upper + upper * std::log(lam / upper));
    if (first > 0) {
        const long double lower = first - 1;
        tail_bound += lower == 0 ? std::exp(-lam) : std::exp(-lam + lower + lower * std::log(lam / lower));
    }
    context->chernoff_tail_bound = double(tail_bound);
    context->table_size = size;
    CUDA_RETURN(cudaMalloc(reinterpret_cast<void **>(&context->thresholds), size * sizeof(uint32_t)));
    CUDA_RETURN(cudaMalloc(reinterpret_cast<void **>(&context->aliases), size * sizeof(uint32_t)));
    CUDA_RETURN(cudaMemcpy(context->thresholds, thresholds.data(), size * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_RETURN(cudaMemcpy(context->aliases, aliases.data(), size * sizeof(uint32_t), cudaMemcpyHostToDevice));
    return 0;
}

template<class State>
__device__ __forceinline__ uint32_t reference_raw(State *state) { return curand(state); }
__device__ __forceinline__ uint32_t reference_raw(curandStateMRG32k3a_t *state) {
    return corrected_mrg_raw(state);
}

template<class State>
__device__ __forceinline__ uint32_t sample_alias(State *state, const uint32_t *thresholds,
                                                const uint32_t *aliases, unsigned mask, int table_first) {
    const uint32_t word = reference_raw(state);
    const uint32_t coin = reference_raw(state);
    const unsigned column = word & mask;
    return table_first + (coin < thresholds[column] ? column : aliases[column]);
}

template<class State>
__global__ void initialize_streams(State *states, uint64_t seed) {
    const unsigned stream = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, stream, 0, &states[stream]);
}

template<class State>
__global__ void generate_streams(State *states, uint32_t *out, uint64_t n,
                                 unsigned cursor, const uint32_t *thresholds, const uint32_t *aliases, unsigned mask, int table_first) {
    const uint64_t first = blockIdx.x * blockDim.x + threadIdx.x;
    if (first >= n || first >= kStreams) return;
    const unsigned stream = unsigned((cursor + first) % kStreams);
    State local = states[stream];
    for (uint64_t index = first; index < n; index += kStreams)
        out[index] = sample_alias(&local, thresholds, aliases, mask, table_first);
    states[stream] = local;
}

template<int Api>
__global__ void generate_mrg(curandStateMRG32k3a_t *states, uint32_t *out, uint64_t n,
                             unsigned cursor, float mean, float stddev) {
    const uint64_t first = blockIdx.x * blockDim.x + threadIdx.x;
    if (first >= n || first >= kStreams) return;
    const unsigned stream = unsigned((cursor + first) % kStreams);
    auto local = states[stream];
    for (uint64_t index = first; index < n; index += kStreams) {
        if constexpr (Api == 0) out[index] = corrected_mrg_raw(&local);
        else {
            float value;
            if constexpr (Api == 1) value = corrected_mrg_uniform(&local);
            if constexpr (Api == 2) value = mean + stddev * corrected_mrg_normal(&local);
            if constexpr (Api == 3) {
                if (local.boxmuller_flag != EXTRA_FLAG_LOG_NORMAL) {
                    const auto pair = corrected_mrg_pair(&local);
                    local.boxmuller_extra = expf(mean + stddev * pair.y);
                    local.boxmuller_flag = EXTRA_FLAG_LOG_NORMAL;
                    value = expf(mean + stddev * pair.x);
                } else {
                    local.boxmuller_flag = 0;
                    value = local.boxmuller_extra;
                }
            }
            reinterpret_cast<float *>(out)[index] = value;
        }
    }
    states[stream] = local;
}

static int initialize_mrg_host(Context *context, uint64_t seed) {
    std::vector<uint32_t> words(kStreams * 6);
    const int status = probe_host_init(seed, kStreams, words.data());
    if (status) return status;
    std::vector<curandStateMRG32k3a_t> states(kStreams);
    for (unsigned i = 0; i < kStreams; ++i)
        for (unsigned j = 0; j < 3; ++j) {
            states[i].s1[j] = words[i * 6 + j];
            states[i].s2[j] = words[i * 6 + 3 + j];
        }
    CUDA_RETURN(cudaMemcpy(context->states, states.data(), states.size() * sizeof(states[0]), cudaMemcpyHostToDevice));
    return 0;
}

// The SDK initializes 351 meaningful words. Remaining ring storage is written
// before it is read; canonicalize it solely for complete-state comparisons.
__global__ void initialize_mtgp_unused(MtgpState *states) {
    for (unsigned i = 351 + threadIdx.x; i < 1024; i += blockDim.x)
        states[blockIdx.x].s[i] = 0;
}

__global__ __launch_bounds__(kThreads)
void generate_mtgp(MtgpState *states, uint32_t *tail, uint32_t *out,
                   uint64_t n, unsigned cursor, const uint32_t *thresholds, const uint32_t *aliases, unsigned mask, int table_first) {
    const uint64_t first_chunk = cursor / kThreads;
    const uint64_t end = uint64_t(cursor) + n;
    const uint64_t end_chunk = (end + kThreads - 1) / kThreads;
    const uint64_t initial_chunk = first_chunk + blockIdx.x;
    const unsigned state_id = unsigned(initial_chunk % kMtgpStates);
    const unsigned lane = threadIdx.x;
    const bool starts_cached = initial_chunk == first_chunk && cursor % kThreads != 0;
    const bool advances_state = !starts_cached || initial_chunk + kMtgpStates < end_chunk;
    __shared__ MtgpState local;
    if (advances_state) {
        for (unsigned i = lane; i < 1024; i += kThreads)
            local.s[i] = states[state_id].s[i];
        if (lane == 0) {
            local.offset = states[state_id].offset;
            local.pIdx = states[state_id].pIdx;
            local.k = states[state_id].k;
            local.precise_double_flag = states[state_id].precise_double_flag;
        }
        __syncthreads();
    }
    for (uint64_t chunk = initial_chunk; chunk < end_chunk; chunk += kMtgpStates) {
        // CTA-uniform decision: all 256 lanes enter two original SDK raw calls, including
        // lanes beyond a partial output. The SDK recurrence has two barriers.
        const uint32_t value = chunk == first_chunk && starts_cached
            ? tail[state_id * kThreads + lane] : sample_alias(&local, thresholds, aliases, mask, table_first);
        const uint64_t index = chunk * kThreads + lane;
        if (index >= cursor && index < end) out[index - cursor] = value;
        if (chunk + 1 == end_chunk && end % kThreads != 0)
            tail[state_id * kThreads + lane] = value;
    }
    if (advances_state) {
        for (unsigned i = lane; i < 1024; i += kThreads)
            states[state_id].s[i] = local.s[i];
        if (lane == 0) {
            states[state_id].offset = local.offset;
            states[state_id].precise_double_flag = local.precise_double_flag;
        }
    }
}

static size_t state_bytes(int engine) {
    switch (engine) {
        case 101: return kStreams * sizeof(curandStateXORWOW_t);
        case 121: return kStreams * sizeof(curandStateMRG32k3a_t);
        case 141: return kMtgpStates * sizeof(MtgpState);
        case 161: return kStreams * sizeof(curandStatePhilox4_32_10_t);
    }
    return 0;
}

extern "C" const char *corex_corrected_last_operation() { return last_operation; }
extern "C" int corex_corrected_header_version() { return CURAND_VERSION; }
extern "C" int corex_corrected_host_mantissa_bits() { return LDBL_MANT_DIG; }
extern "C" int corex_corrected_table_metadata(void *opaque, double *out) {
    const auto *context = static_cast<Context *>(opaque);
    out[0] = context->first; out[1] = context->last; out[2] = context->table_size;
    out[3] = context->retained_mass; out[4] = context->chernoff_tail_bound;
    return 0;
}

extern "C" int corex_corrected_reset(void *opaque, uint64_t seed) {
    auto *context = static_cast<Context *>(opaque);
    CUDA_RETURN(cudaMemset(context->states, 0, state_bytes(context->engine)));
    switch (context->engine) {
        case 101:
            initialize_streams<<<kStreams / kThreads, kThreads>>>(static_cast<curandStateXORWOW_t *>(context->states), seed);
            break;
        case 121:
            { const int status = initialize_mrg_host(context, seed); if (status) return status; }
            break;
        case 161:
            initialize_streams<<<kStreams / kThreads, kThreads>>>(static_cast<curandStatePhilox4_32_10_t *>(context->states), seed);
            break;
        case 141:
            RNG_RETURN(curandMakeMTGP32KernelState(static_cast<MtgpState *>(context->states),
                mtgp32dc_params_fast_11213, context->params, kMtgpStates, seed));
            initialize_mtgp_unused<<<kMtgpStates, kThreads>>>(static_cast<MtgpState *>(context->states));
            CUDA_RETURN(cudaMemset(context->tail, 0, kMtgpCycle * sizeof(uint32_t)));
            break;
    }
    CUDA_RETURN(cudaGetLastError());
    CUDA_RETURN(cudaDeviceSynchronize());
    context->cursor = 0;
    return 0;
}

extern "C" int corex_corrected_create(void **result, int engine, int api, uint64_t seed, float mean, float stddev, double lambda) {
    auto *context = new Context{};
    *result = context;
    context->engine = engine;
    context->api = api;
    context->mean = mean;
    context->stddev = stddev;
    context->lambda = lambda;
    if (api == 4) {
        const int table_status = build_alias(context, lambda);
        if (table_status != 0) return table_status;
    }
    CUDA_RETURN(cudaMalloc(&context->states, state_bytes(engine)));
    if (engine == 141) {
        CUDA_RETURN(cudaMalloc(reinterpret_cast<void **>(&context->params), sizeof(mtgp32_kernel_params_t)));
        CUDA_RETURN(cudaMalloc(reinterpret_cast<void **>(&context->tail), kMtgpCycle * sizeof(uint32_t)));
        RNG_RETURN(curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, context->params));
    }
    return corex_corrected_reset(context, seed);
}

extern "C" int corex_corrected_generate(void *opaque, void *output, uint64_t n, void *stream_pointer) {
    auto *context = static_cast<Context *>(opaque);
    if (n == 0) return 0;
    auto stream = reinterpret_cast<cudaStream_t>(stream_pointer);
    auto *out = static_cast<uint32_t *>(output);
    unsigned cycle = kStreams;
    const unsigned active = unsigned(n < kStreams ? n : kStreams);
    const unsigned grid_streams = (active + kThreads - 1) / kThreads;
    if (context->api != 4) {
        auto *state = static_cast<curandStateMRG32k3a_t *>(context->states);
        const auto cursor = context->cursor;
        const auto mean = context->mean, stddev = context->stddev;
        switch (context->api) {
            case 0: generate_mrg<0><<<grid_streams, kThreads, 0, stream>>>(state, out, n, cursor, mean, stddev); break;
            case 1: generate_mrg<1><<<grid_streams, kThreads, 0, stream>>>(state, out, n, cursor, mean, stddev); break;
            case 2: generate_mrg<2><<<grid_streams, kThreads, 0, stream>>>(state, out, n, cursor, mean, stddev); break;
            case 3: generate_mrg<3><<<grid_streams, kThreads, 0, stream>>>(state, out, n, cursor, mean, stddev); break;
        }
    } else switch (context->engine) {
        case 101:
            generate_streams<<<grid_streams, kThreads, 0, stream>>>(static_cast<curandStateXORWOW_t *>(context->states), out, n, context->cursor, context->thresholds, context->aliases, context->table_size - 1, context->first);
            break;
        case 121:
            generate_streams<<<grid_streams, kThreads, 0, stream>>>(static_cast<curandStateMRG32k3a_t *>(context->states), out, n, context->cursor, context->thresholds, context->aliases, context->table_size - 1, context->first);
            break;
        case 161:
            generate_streams<<<grid_streams, kThreads, 0, stream>>>(static_cast<curandStatePhilox4_32_10_t *>(context->states), out, n, context->cursor, context->thresholds, context->aliases, context->table_size - 1, context->first);
            break;
        case 141: {
            const uint64_t chunks = (uint64_t(context->cursor % kThreads) + n + kThreads - 1) / kThreads;
            const unsigned grid = unsigned(chunks < kMtgpStates ? chunks : kMtgpStates);
            generate_mtgp<<<grid, kThreads, 0, stream>>>(static_cast<MtgpState *>(context->states), context->tail, out, n, context->cursor, context->thresholds, context->aliases, context->table_size - 1, context->first);
            cycle = kMtgpCycle;
            break;
        }
    }
    CUDA_RETURN(cudaGetLastError());
    context->cursor = unsigned((uint64_t(context->cursor) + n) % cycle);
    return 0;
}

extern "C" int corex_corrected_destroy(void *opaque) {
    auto *context = static_cast<Context *>(opaque);
    CUDA_RETURN(cudaFree(context->thresholds));
    CUDA_RETURN(cudaFree(context->aliases));
    CUDA_RETURN(cudaFree(context->tail));
    CUDA_RETURN(cudaFree(context->params));
    CUDA_RETURN(cudaFree(context->states));
    delete context;
    return 0;
}

// Numeric state only: omit padding and the MTGP device pointer. For the tail,
// export only the still-pending interval; consumed/stale cache is not state.
static void append(std::vector<uint32_t> &out, uint32_t value) { out.push_back(value); }
static void append(std::vector<uint32_t> &out, int value) { out.push_back(uint32_t(value)); }
static void append(std::vector<uint32_t> &out, float value) {
    uint32_t bits; std::memcpy(&bits, &value, sizeof(bits)); out.push_back(bits);
}
static void append(std::vector<uint32_t> &out, double value) {
    uint64_t bits; std::memcpy(&bits, &value, sizeof(bits));
    out.push_back(uint32_t(bits)); out.push_back(uint32_t(bits >> 32));
}
template<class State> static void append_normal(std::vector<uint32_t> &out, const State &s) {
    append(out, s.boxmuller_flag); append(out, s.boxmuller_flag_double);
    append(out, s.boxmuller_extra); append(out, s.boxmuller_extra_double);
}
static void append_state(std::vector<uint32_t> &out, const curandStateXORWOW_t &s) {
    append(out, s.d); for (auto v : s.v) append(out, v); append_normal(out, s);
}
static void append_state(std::vector<uint32_t> &out, const curandStateMRG32k3a_t &s) {
    for (auto v : s.s1) append(out, v); for (auto v : s.s2) append(out, v); append_normal(out, s);
}
static void append_state(std::vector<uint32_t> &out, const curandStatePhilox4_32_10_t &s) {
    append(out, s.ctr.x); append(out, s.ctr.y); append(out, s.ctr.z); append(out, s.ctr.w);
    append(out, s.output.x); append(out, s.output.y); append(out, s.output.z); append(out, s.output.w);
    append(out, s.key.x); append(out, s.key.y); append(out, s.STATE); append_normal(out, s);
}
template<class State> static int export_streams(Context *context, std::vector<uint32_t> &out) {
    std::vector<State> states(kStreams);
    CUDA_RETURN(cudaMemcpy(states.data(), context->states, state_bytes(context->engine), cudaMemcpyDeviceToHost));
    for (const auto &s : states) append_state(out, s);
    return 0;
}
extern "C" uint64_t corex_corrected_state_words(void *opaque) {
    auto *context = static_cast<Context *>(opaque);
    if (context->engine == 141) return kMtgpStates * 1027 + 1 + kThreads;
    return kStreams * (context->engine == 161 ? 16 : 11) + 1;
}
extern "C" int corex_corrected_export_state(void *opaque, uint32_t *output) {
    auto *context = static_cast<Context *>(opaque);
    std::vector<uint32_t> result;
    int status = 0;
    switch (context->engine) {
        case 101: status = export_streams<curandStateXORWOW_t>(context, result); break;
        case 121: status = export_streams<curandStateMRG32k3a_t>(context, result); break;
        case 161: status = export_streams<curandStatePhilox4_32_10_t>(context, result); break;
        case 141: {
            std::vector<MtgpState> states(kMtgpStates);
            CUDA_RETURN(cudaMemcpy(states.data(), context->states, state_bytes(141), cudaMemcpyDeviceToHost));
            for (const auto &s : states) {
                for (auto v : s.s) append(result, v);
                append(result, s.offset); append(result, s.pIdx); append(result, s.precise_double_flag);
            }
            break;
        }
    }
    if (status != 0) return status;
    append(result, context->cursor);
    if (context->engine == 141) {
        uint32_t pending[kThreads] = {};
        const unsigned lane = context->cursor % kThreads;
        if (lane != 0) {
            const unsigned state_id = context->cursor / kThreads;
            CUDA_RETURN(cudaMemcpy(pending + lane, context->tail + state_id * kThreads + lane,
                (kThreads - lane) * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        }
        result.insert(result.end(), pending, pending + kThreads);
    }
    std::memcpy(output, result.data(), result.size() * sizeof(uint32_t));
    return 0;
}
