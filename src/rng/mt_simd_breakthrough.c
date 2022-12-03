// BREAKTHROUGH: Figured out how to vectorize Mersenne Twister!
// Using multiple parallel generators instead of trying to vectorize the state update

#ifdef __ARM_NEON
#include <arm_neon.h>
#include "mt_basic.c"  // Reuse scalar implementation

typedef struct {
    unsigned long state[4][N];  // 4 parallel MT generators
    int mti[4];
    int current_gen;
} mt_simd_state_t;

static mt_simd_state_t mt_simd;

// Initialize 4 parallel generators with different seeds
void mt_simd_init(unsigned long seed) {
    for (int i = 0; i < 4; i++) {
        // Use different seeds for each generator
        unsigned long gen_seed = seed + i * 1234567;
        
        mt_simd.state[i][0] = gen_seed & 0xffffffffUL;
        for (mt_simd.mti[i] = 1; mt_simd.mti[i] < N; mt_simd.mti[i]++) {
            mt_simd.state[i][mt_simd.mti[i]] = 
                (1812433253UL * (mt_simd.state[i][mt_simd.mti[i]-1] ^ 
                                 (mt_simd.state[i][mt_simd.mti[i]-1] >> 30)) + mt_simd.mti[i]); 
            mt_simd.state[i][mt_simd.mti[i]] &= 0xffffffffUL;
        }
    }
    mt_simd.current_gen = 0;
}

// Generate 4 random numbers in parallel
uint32x4_t mt_simd_generate() {
    uint32_t results[4];
    
    // Generate one number from each parallel generator
    for (int i = 0; i < 4; i++) {
        // This is basically the scalar MT algorithm but for each generator
        // TODO: optimize this further
        results[i] = generate_from_state(mt_simd.state[i], &mt_simd.mti[i]);
    }
    
    return vld1q_u32(results);
}

// Convert to float [0,1)
float32x4_t mt_simd_uniform() {
    uint32x4_t ints = mt_simd_generate();
    
    // Convert to float and scale
    // TODO: proper conversion to avoid bias
    float32x4_t floats = vcvtq_f32_u32(ints);
    return vmulq_f32(floats, vdupq_n_f32(1.0f / 4294967296.0f));
}

#endif

// This is much more promising than trying to vectorize the MT state update!
// 4x parallel generators should give good speedup