// CoreX SDK corrected reference: integer recurrence derived from
// curand_kernel.h:930-958 (__CUDA_ARCH__ > 600 branch).
// The SDK license is retained in SDK-LICENSE.txt in this experiment package.
// This is not the installed stock host or device cuRAND implementation.
#pragma once
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdint>

__device__ __forceinline__ uint32_t corrected_mrg_next(curandStateMRG32k3a_t *state) {
    constexpr uint32_t m1 = 4294967087u, m2 = 4294944443u;
    // Same arithmetic as the SDK integer branch; portable multiply/add replaces
    // NVIDIA-specific inline mad.wide.u32. Do not override __CUDA_ARCH__.
    uint64_t p1 = uint64_t(1403580u) * state->s1[1] + uint64_t(810728u) * (m1 - state->s1[0]);
    p1 = (p1 & 0xffffffffULL) + (p1 >> 32) * 209u;
    if (p1 >= m1) p1 -= m1;
    state->s1[0] = state->s1[1]; state->s1[1] = state->s1[2]; state->s1[2] = uint32_t(p1);
    uint64_t p2 = uint64_t(527612u) * state->s2[2] + uint64_t(1370589u) * (m2 - state->s2[0]);
    p2 = (p2 & 0xffffffffULL) + (p2 >> 32) * 22853u;
    p2 = (p2 & 0xffffffffULL) + (p2 >> 32) * 22853u;
    if (p2 >= m2) p2 -= m2;
    state->s2[0] = state->s2[1]; state->s2[1] = state->s2[2]; state->s2[2] = uint32_t(p2);
    return uint32_t(p1 <= p2 ? p1 + m1 - p2 : p1 - p2);
}

__device__ __forceinline__ float corrected_mrg_uniform(curandStateMRG32k3a_t *state) {
    // FP32 (0,1] mapping, explicitly changed from the SDK's double multiply.
    return float(corrected_mrg_next(state)) * 0x1p-32f;
}

__device__ __forceinline__ uint32_t corrected_mrg_raw(curandStateMRG32k3a_t *state) {
    const uint64_t z = corrected_mrg_next(state);
    // Integer evaluation of the SDK decimal BITS_NORM=1.000000048662.
    // This avoids software double; exact recurrence remains separately exposed.
    return uint32_t(z + z * 24331ULL / 500000000000ULL);
}

__device__ __forceinline__ float2 corrected_mrg_pair(curandStateMRG32k3a_t *state) {
    // SDK curand_normal.h:85-102 FP32 Box-Muller structure.
    const float x = corrected_mrg_uniform(state);
    const float y = corrected_mrg_uniform(state) * CURAND_2PI;
    float2 pair;
    const float scale = sqrtf(-2.0f * logf(x));
    __sincosf(y, &pair.x, &pair.y);
    pair.x *= scale; pair.y *= scale;
    return pair;
}

__device__ __forceinline__ float corrected_mrg_normal(curandStateMRG32k3a_t *state) {
    if (state->boxmuller_flag != EXTRA_FLAG_NORMAL) {
        const float2 pair = corrected_mrg_pair(state);
        state->boxmuller_extra = pair.y;
        state->boxmuller_flag = EXTRA_FLAG_NORMAL;
        return pair.x;
    }
    state->boxmuller_flag = 0;
    return state->boxmuller_extra;
}

__device__ __forceinline__ float corrected_mrg_lognormal(curandStateMRG32k3a_t *state) {
    // Diagnostic standard parameters mean=0, stddev=1. SDK spare stores the
    // transformed second result; normal/lognormal use distinct spare flags.
    if (state->boxmuller_flag != EXTRA_FLAG_LOG_NORMAL) {
        const float2 pair = corrected_mrg_pair(state);
        state->boxmuller_extra = expf(pair.y);
        state->boxmuller_flag = EXTRA_FLAG_LOG_NORMAL;
        return expf(pair.x);
    }
    state->boxmuller_flag = 0;
    return state->boxmuller_extra;
}
