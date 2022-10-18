#include "bakuhatsu/rng/mersenne_twister_simd.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#ifndef PI_F
#define PI_F 3.14159265358979323846f
#endif

// MT19937 tempering constants
#define TEMPERING_MASK_B 0x9d2c5680UL
#define TEMPERING_MASK_C 0xefc60000UL
#define TEMPERING_SHIFT_U(y) (y >> 11)
#define TEMPERING_SHIFT_S(y) (y << 7)
#define TEMPERING_SHIFT_T(y) (y << 15)
#define TEMPERING_SHIFT_L(y) (y >> 18)

// SIMD-optimized constants
static const uint32_t MT_SIMD_MAGIC[4] = {MT_MATRIX_A, MT_MATRIX_A, MT_MATRIX_A, MT_MATRIX_A};

void mt_init(mersenne_twister_state_t* mt, uint32_t seed) {
    mt->mt[0] = seed;
    mt->seed = seed;
    mt->generated_count = 0;
    
    for (mt->mti = 1; mt->mti < MT_N; mt->mti++) {
        mt->mt[mt->mti] = (1812433253UL * (mt->mt[mt->mti - 1] ^ (mt->mt[mt->mti - 1] >> 30)) + mt->mti);
    }
}

static void mt_twist(mersenne_twister_state_t* mt) {
    uint32_t y;
    static const uint32_t mag01[2] = {0x0UL, MT_MATRIX_A};
    
    int kk;
    for (kk = 0; kk < MT_N - MT_M; kk++) {
        y = (mt->mt[kk] & MT_UPPER_MASK) | (mt->mt[kk + 1] & MT_LOWER_MASK);
        mt->mt[kk] = mt->mt[kk + MT_M] ^ (y >> 1) ^ mag01[y & 0x1UL];
    }
    
    for (; kk < MT_N - 1; kk++) {
        y = (mt->mt[kk] & MT_UPPER_MASK) | (mt->mt[kk + 1] & MT_LOWER_MASK);
        mt->mt[kk] = mt->mt[kk + (MT_M - MT_N)] ^ (y >> 1) ^ mag01[y & 0x1UL];
    }
    
    y = (mt->mt[MT_N - 1] & MT_UPPER_MASK) | (mt->mt[0] & MT_LOWER_MASK);
    mt->mt[MT_N - 1] = mt->mt[MT_M - 1] ^ (y >> 1) ^ mag01[y & 0x1UL];
    
    mt->mti = 0;
}

uint32_t mt_genrand_uint32(mersenne_twister_state_t* mt) {
    uint32_t y;
    
    if (mt->mti >= MT_N) {
        mt_twist(mt);
    }
    
    y = mt->mt[mt->mti++];
    mt->generated_count++;
    
    // Tempering
    y ^= TEMPERING_SHIFT_U(y);
    y ^= TEMPERING_SHIFT_S(y) & TEMPERING_MASK_B;
    y ^= TEMPERING_SHIFT_T(y) & TEMPERING_MASK_C;
    y ^= TEMPERING_SHIFT_L(y);
    
    return y;
}

float mt_genrand_float(mersenne_twister_state_t* mt) {
    // Generate random number in [0, 1) with 32-bit precision
    uint32_t a = mt_genrand_uint32(mt) >> 5;  // Upper 27 bits
    uint32_t b = mt_genrand_uint32(mt) >> 6;  // Upper 26 bits
    return (a * 67108864.0f + b) * (1.0f / 9007199254740992.0f);
}

simd_mt_pool_t* simd_mt_pool_create(uint32_t pool_size, uint64_t seed) {
    if (pool_size == 0 || pool_size > 1024) {
        return NULL;  // Reasonable limits
    }
    
    simd_mt_pool_t* pool = (simd_mt_pool_t*)malloc(sizeof(simd_mt_pool_t));
    if (!pool) return NULL;
    
    // Allocate generators array
    pool->generators = (mersenne_twister_state_t*)malloc(pool_size * sizeof(mersenne_twister_state_t));
    if (!pool->generators) {
        free(pool);
        return NULL;
    }
    
    // Initialize generators with different seeds
    for (uint32_t i = 0; i < pool_size; i++) {
        mt_init(&pool->generators[i], (uint32_t)(seed + i * 1234567891UL));
    }
    
    pool->pool_size = pool_size;
    pool->current_index = 0;
    pool->total_generated = 0;
    
    // Initialize batch buffer for performance
    pool->batch_size = 1024;  // Pre-generate 1024 values
    pool->batch_buffer = (uint32_t*)malloc(pool->batch_size * sizeof(uint32_t));
    pool->batch_position = pool->batch_size;  // Force initial fill
    
    if (!pool->batch_buffer) {
        free(pool->generators);
        free(pool);
        return NULL;
    }
    
    return pool;
}

void simd_mt_pool_destroy(simd_mt_pool_t* pool) {
    if (pool) {
        free(pool->batch_buffer);
        free(pool->generators);
        free(pool);
    }
}

simd_f32x4_t simd_mt_generate_uniform_batch(simd_mt_pool_t* pool) {
    if (!pool) return simd_set1_f32(0.0f);
    
    // Generate 4 uint32 values using round-robin
    uint32_t values[4];
    for (int i = 0; i < 4; i++) {
        values[i] = mt_genrand_uint32(&pool->generators[pool->current_index]);
        pool->current_index = (pool->current_index + 1) % pool->pool_size;
    }
    
    pool->total_generated += 4;
    
    // Convert to [0, 1) using SIMD
    simd_u32x4_t uint_vec = simd_load_aligned_f32((const float*)values);
    
    // Mask to get 24 bits of precision for float conversion
    simd_u32x4_t mask = simd_set1_f32(*(float*)&(uint32_t){0x00FFFFFF});
    uint_vec = simd_cmplt_f32(uint_vec, mask);  // Actually bitwise AND operation
    
    // Convert to float and scale to [0, 1)
    simd_f32x4_t float_vec = simd_cvt_u32_to_f32(uint_vec);
    simd_f32x4_t scale = simd_set1_f32(1.0f / 16777216.0f);  // 1 / 2^24
    
    return simd_mul_f32(float_vec, scale);
}

simd_u32x4_t simd_mt_generate_uint32_batch(simd_mt_pool_t* pool) {
    if (!pool) return simd_set1_f32(*(float*)&(uint32_t){0});
    
    uint32_t values[4] __attribute__((aligned(16)));
    for (int i = 0; i < 4; i++) {
        values[i] = mt_genrand_uint32(&pool->generators[pool->current_index]);
        pool->current_index = (pool->current_index + 1) % pool->pool_size;
    }
    
    pool->total_generated += 4;
    return simd_load_aligned_f32((const float*)values);
}

uint32_t simd_mt_generate_uniform_array(simd_mt_pool_t* pool, float* output, uint32_t count) {
    if (!pool || !output || count == 0) return 0;
    
    // Ensure count is multiple of 4 for SIMD efficiency
    uint32_t simd_count = (count / 4) * 4;
    
    for (uint32_t i = 0; i < simd_count; i += 4) {
        simd_f32x4_t batch = simd_mt_generate_uniform_batch(pool);
        simd_store_aligned_f32(&output[i], batch);
    }
    
    return simd_count;
}

void simd_mt_pool_advance(simd_mt_pool_t* pool, uint64_t steps) {
    if (!pool) return;
    
    // Advance each generator by the specified number of steps
    for (uint32_t i = 0; i < pool->pool_size; i++) {
        for (uint64_t j = 0; j < steps; j++) {
            mt_genrand_uint32(&pool->generators[i]);
        }
    }
}

void simd_mt_pool_reseed(simd_mt_pool_t* pool, uint64_t seed) {
    if (!pool) return;
    
    for (uint32_t i = 0; i < pool->pool_size; i++) {
        mt_init(&pool->generators[i], (uint32_t)(seed + i * 1234567891UL));
    }
    
    pool->current_index = 0;
    pool->total_generated = 0;
    pool->batch_position = pool->batch_size;  // Force batch refill
}

uint64_t simd_mt_pool_get_count(const simd_mt_pool_t* pool) {
    return pool ? pool->total_generated : 0;
}

void simd_mt_pool_prefill_batch(simd_mt_pool_t* pool) {
    if (!pool || !pool->batch_buffer) return;
    
    for (uint32_t i = 0; i < pool->batch_size; i++) {
        pool->batch_buffer[i] = mt_genrand_uint32(&pool->generators[pool->current_index]);
        pool->current_index = (pool->current_index + 1) % pool->pool_size;
    }
    
    pool->batch_position = 0;
}

uint32_t simd_mt_pool_check_twist_needed(const simd_mt_pool_t* pool) {
    if (!pool) return 0;
    
    uint32_t twist_count = 0;
    for (uint32_t i = 0; i < pool->pool_size; i++) {
        if (pool->generators[i].mti >= MT_N) {
            twist_count++;
        }
    }
    
    return twist_count;
}

void simd_mt_pool_force_twist_all(simd_mt_pool_t* pool) {
    if (!pool) return;
    
    for (uint32_t i = 0; i < pool->pool_size; i++) {
        if (pool->generators[i].mti >= MT_N) {
            mt_twist(&pool->generators[i]);
        }
    }
}

void simd_mt_generate_unrolled_batch(simd_mt_pool_t* pool, float* batch_output) {
    if (!pool || !batch_output) return;
    
    // Generate 4 SIMD vectors (16 floats) with manual unrolling
    simd_f32x4_t batch0 = simd_mt_generate_uniform_batch(pool);
    simd_f32x4_t batch1 = simd_mt_generate_uniform_batch(pool);
    simd_f32x4_t batch2 = simd_mt_generate_uniform_batch(pool);
    simd_f32x4_t batch3 = simd_mt_generate_uniform_batch(pool);
    
    simd_store_aligned_f32(&batch_output[0], batch0);
    simd_store_aligned_f32(&batch_output[4], batch1);
    simd_store_aligned_f32(&batch_output[8], batch2);
    simd_store_aligned_f32(&batch_output[12], batch3);
}

bool simd_mt_pool_validate_state(const simd_mt_pool_t* pool) {
    if (!pool) return false;
    if (!pool->generators || !pool->batch_buffer) return false;
    if (pool->pool_size == 0 || pool->pool_size > 1024) return false;
    if (pool->current_index >= pool->pool_size) return false;
    if (pool->batch_position > pool->batch_size) return false;
    
    // Check individual generator states
    for (uint32_t i = 0; i < pool->pool_size; i++) {
        if (pool->generators[i].mti > MT_N) return false;
    }
    
    return true;
}