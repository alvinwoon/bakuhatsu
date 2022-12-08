// SIMD Box-Muller using the parallel MT generators
// This should be much faster than scalar version

#ifdef __ARM_NEON
#include <arm_neon.h>
#include "neon_math.c"  // Our SIMD math functions

static int has_spare_simd = 0;
static float32x4_t spare_values;

// Generate 4 Gaussian random numbers using SIMD Box-Muller
float32x4_t box_muller_simd() {
    if (has_spare_simd) {
        has_spare_simd = 0;
        return spare_values;
    }
    
    // Generate two sets of uniform random numbers
    float32x4_t u1 = mt_simd_uniform();
    float32x4_t u2 = mt_simd_uniform();
    
    // Add small epsilon to avoid log(0)
    float32x4_t epsilon = vdupq_n_f32(1e-10f);
    u1 = vmaxq_f32(u1, epsilon);
    
    // Box-Muller: z0 = sqrt(-2*ln(u1)) * cos(2*pi*u2)  
    //             z1 = sqrt(-2*ln(u1)) * sin(2*pi*u2)
    
    float32x4_t ln_u1 = vlog_f32(u1);
    float32x4_t neg_two_ln = vmulq_f32(vdupq_n_f32(-2.0f), ln_u1);
    float32x4_t magnitude = vsqrtq_f32(neg_two_ln);
    
    float32x4_t two_pi_u2 = vmulq_f32(vdupq_n_f32(6.28318530718f), u2);
    
    float32x4_t cos_val = vcos_f32(two_pi_u2);
    float32x4_t sin_val = vsin_f32(two_pi_u2);
    
    float32x4_t z0 = vmulq_f32(magnitude, cos_val);
    float32x4_t z1 = vmulq_f32(magnitude, sin_val);
    
    // Store z1 for next call
    spare_values = z1;
    has_spare_simd = 1;
    
    return z0;
}

// Generate many Gaussian samples efficiently
void box_muller_simd_array(float *output, int count) {
    int simd_count = (count / 4) * 4;  // Round down to multiple of 4
    
    for (int i = 0; i < simd_count; i += 4) {
        float32x4_t gaussian = box_muller_simd();
        vst1q_f32(&output[i], gaussian);
    }
}

#endif

// This should give massive speedup over scalar version!
// 4x from SIMD + better cache utilization from batching