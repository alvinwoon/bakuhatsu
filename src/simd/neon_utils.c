#include "bakuhatsu/simd/neon_utils.h"
#include <math.h>
#include <string.h>

#ifdef HAVE_NEON
#ifdef __linux__
#include <sys/auxv.h>
#include <asm/hwcap.h>
#endif
#endif

bool simd_is_neon_available(void) {
#ifdef HAVE_NEON
    // Check for NEON support at runtime on Linux
    #ifdef __linux__
    return (getauxval(AT_HWCAP) & HWCAP_NEON) != 0;
    #else
    // On other ARM platforms, assume NEON is available if compiled with it
    return true;
    #endif
#else
    return false;
#endif
}

simd_f32x4_t simd_load_aligned_f32(const float* ptr) {
#ifdef HAVE_NEON
    return vld1q_f32(ptr);
#else
    simd_f32x4_t result;
    memcpy(result.v, ptr, 4 * sizeof(float));
    return result;
#endif
}

void simd_store_aligned_f32(float* ptr, simd_f32x4_t vec) {
#ifdef HAVE_NEON
    vst1q_f32(ptr, vec);
#else
    memcpy(ptr, vec.v, 4 * sizeof(float));
#endif
}

simd_f32x4_t simd_set1_f32(float value) {
#ifdef HAVE_NEON
    return vdupq_n_f32(value);
#else
    simd_f32x4_t result;
    result.v[0] = result.v[1] = result.v[2] = result.v[3] = value;
    return result;
#endif
}

simd_f32x4_t simd_set_f32(float a, float b, float c, float d) {
#ifdef HAVE_NEON
    float temp[4] = {a, b, c, d};
    return vld1q_f32(temp);
#else
    simd_f32x4_t result;
    result.v[0] = a; result.v[1] = b; result.v[2] = c; result.v[3] = d;
    return result;
#endif
}

simd_f32x4_t simd_add_f32(simd_f32x4_t a, simd_f32x4_t b) {
#ifdef HAVE_NEON
    return vaddq_f32(a, b);
#else
    simd_f32x4_t result;
    for (int i = 0; i < 4; i++) {
        result.v[i] = a.v[i] + b.v[i];
    }
    return result;
#endif
}

simd_f32x4_t simd_mul_f32(simd_f32x4_t a, simd_f32x4_t b) {
#ifdef HAVE_NEON
    return vmulq_f32(a, b);
#else
    simd_f32x4_t result;
    for (int i = 0; i < 4; i++) {
        result.v[i] = a.v[i] * b.v[i];
    }
    return result;
#endif
}

simd_f32x4_t simd_fmadd_f32(simd_f32x4_t a, simd_f32x4_t b, simd_f32x4_t c) {
#ifdef HAVE_NEON
    return vfmaq_f32(c, a, b);
#else
    simd_f32x4_t result;
    for (int i = 0; i < 4; i++) {
        result.v[i] = a.v[i] * b.v[i] + c.v[i];
    }
    return result;
#endif
}

simd_f32x4_t simd_sqrt_f32(simd_f32x4_t a) {
#ifdef HAVE_NEON
    // Use Newton-Raphson iteration for higher precision
    simd_f32x4_t rsqrt = vrsqrteq_f32(a);
    rsqrt = vmulq_f32(rsqrt, vrsqrtsq_f32(vmulq_f32(a, rsqrt), rsqrt));
    rsqrt = vmulq_f32(rsqrt, vrsqrtsq_f32(vmulq_f32(a, rsqrt), rsqrt));
    return vmulq_f32(a, rsqrt);
#else
    simd_f32x4_t result;
    for (int i = 0; i < 4; i++) {
        result.v[i] = sqrtf(a.v[i]);
    }
    return result;
#endif
}

simd_f32x4_t simd_rsqrt_f32(simd_f32x4_t a) {
#ifdef HAVE_NEON
    // Fast reciprocal square root with one Newton-Raphson iteration
    simd_f32x4_t rsqrt = vrsqrteq_f32(a);
    return vmulq_f32(rsqrt, vrsqrtsq_f32(vmulq_f32(a, rsqrt), rsqrt));
#else
    simd_f32x4_t result;
    for (int i = 0; i < 4; i++) {
        result.v[i] = 1.0f / sqrtf(a.v[i]);
    }
    return result;
#endif
}

float simd_horizontal_sum_f32(simd_f32x4_t a) {
#ifdef HAVE_NEON
    float32x2_t low = vget_low_f32(a);
    float32x2_t high = vget_high_f32(a);
    float32x2_t sum = vadd_f32(low, high);
    return vget_lane_f32(vpadd_f32(sum, sum), 0);
#else
    return a.v[0] + a.v[1] + a.v[2] + a.v[3];
#endif
}

simd_u32x4_t simd_cmplt_f32(simd_f32x4_t a, simd_f32x4_t b) {
#ifdef HAVE_NEON
    return vcltq_f32(a, b);
#else
    simd_u32x4_t result;
    for (int i = 0; i < 4; i++) {
        result.v[i] = (a.v[i] < b.v[i]) ? 0xFFFFFFFF : 0x00000000;
    }
    return result;
#endif
}

simd_f32x4_t simd_select_f32(simd_u32x4_t mask, simd_f32x4_t a, simd_f32x4_t b) {
#ifdef HAVE_NEON
    return vbslq_f32(mask, a, b);
#else
    simd_f32x4_t result;
    for (int i = 0; i < 4; i++) {
        result.v[i] = mask.v[i] ? a.v[i] : b.v[i];
    }
    return result;
#endif
}

simd_f32x4_t simd_cvt_u32_to_f32(simd_u32x4_t a) {
#ifdef HAVE_NEON
    return vcvtq_f32_u32(a);
#else
    simd_f32x4_t result;
    for (int i = 0; i < 4; i++) {
        result.v[i] = (float)a.v[i];
    }
    return result;
#endif
}

float simd_extract_f32(simd_f32x4_t vec, int index) {
#ifdef HAVE_NEON
    switch (index) {
        case 0: return vgetq_lane_f32(vec, 0);
        case 1: return vgetq_lane_f32(vec, 1);
        case 2: return vgetq_lane_f32(vec, 2);
        case 3: return vgetq_lane_f32(vec, 3);
        default: return 0.0f;
    }
#else
    if (index >= 0 && index < 4) {
        return vec.v[index];
    }
    return 0.0f;
#endif
}

void simd_memory_barrier(void) {
#ifdef HAVE_NEON
    __sync_synchronize();
#endif
}

void simd_prefetch(const void* ptr) {
#ifdef HAVE_NEON
    __builtin_prefetch(ptr, 0, 3);  // Read, high temporal locality
#else
    (void)ptr;  // Suppress unused parameter warning
#endif
}