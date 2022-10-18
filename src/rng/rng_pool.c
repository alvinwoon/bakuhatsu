#include "bakuhatsu/rng/rng_pool.h"
#include "bakuhatsu/rng/mersenne_twister_simd.h"
#include <stdlib.h>
#include <string.h>

/**
 * @brief RNG pool implementation using Mersenne Twister
 */
struct simd_rng_pool {
    rng_algorithm_t algorithm;
    uint32_t pool_size;
    union {
        simd_mt_pool_t* mt_pool;
        // Add other RNG implementations here
    } generator;
    rng_performance_stats_t stats;
    uint64_t creation_time;
};

simd_rng_pool_t* simd_rng_pool_create(uint32_t pool_size, rng_algorithm_t algorithm, uint64_t seed) {
    if (pool_size == 0 || pool_size > 1024) {
        return NULL;
    }
    
    simd_rng_pool_t* pool = (simd_rng_pool_t*)malloc(sizeof(simd_rng_pool_t));
    if (!pool) return NULL;
    
    memset(pool, 0, sizeof(simd_rng_pool_t));
    pool->algorithm = algorithm;
    pool->pool_size = pool_size;
    
    switch (algorithm) {
        case RNG_MERSENNE_TWISTER:
            pool->generator.mt_pool = simd_mt_pool_create(pool_size, seed);
            if (!pool->generator.mt_pool) {
                free(pool);
                return NULL;
            }
            break;
            
        case RNG_XORSHIFT128_PLUS:
        case RNG_PHILOX_4X32:
        default:
            // Not implemented yet - fallback to MT
            pool->generator.mt_pool = simd_mt_pool_create(pool_size, seed);
            if (!pool->generator.mt_pool) {
                free(pool);
                return NULL;
            }
            break;
    }
    
    return pool;
}

void simd_rng_pool_destroy(simd_rng_pool_t* pool) {
    if (!pool) return;
    
    switch (pool->algorithm) {
        case RNG_MERSENNE_TWISTER:
            simd_mt_pool_destroy(pool->generator.mt_pool);
            break;
        default:
            simd_mt_pool_destroy(pool->generator.mt_pool);
            break;
    }
    
    free(pool);
}

simd_f32x4_t simd_generate_uniform_batch(simd_rng_pool_t* pool) {
    if (!pool) return simd_set1_f32(0.0f);
    
    switch (pool->algorithm) {
        case RNG_MERSENNE_TWISTER:
        default:
            return simd_mt_generate_uniform_batch(pool->generator.mt_pool);
    }
}

simd_u32x4_t simd_generate_uint32_batch(simd_rng_pool_t* pool) {
    if (!pool) return simd_set1_f32(*(float*)&(uint32_t){0});
    
    switch (pool->algorithm) {
        case RNG_MERSENNE_TWISTER:
        default:
            return simd_mt_generate_uint32_batch(pool->generator.mt_pool);
    }
}

uint32_t simd_generate_uniform_array(simd_rng_pool_t* pool, float* output, uint32_t count) {
    if (!pool || !output || count == 0) return 0;
    
    switch (pool->algorithm) {
        case RNG_MERSENNE_TWISTER:
        default:
            return simd_mt_generate_uniform_array(pool->generator.mt_pool, output, count);
    }
}

void simd_rng_pool_advance(simd_rng_pool_t* pool, uint64_t steps) {
    if (!pool) return;
    
    switch (pool->algorithm) {
        case RNG_MERSENNE_TWISTER:
        default:
            simd_mt_pool_advance(pool->generator.mt_pool, steps);
            break;
    }
}

void simd_rng_pool_reseed(simd_rng_pool_t* pool, uint64_t seed) {
    if (!pool) return;
    
    switch (pool->algorithm) {
        case RNG_MERSENNE_TWISTER:
        default:
            simd_mt_pool_reseed(pool->generator.mt_pool, seed);
            break;
    }
}

rng_quality_metrics_t simd_rng_get_quality_metrics(rng_algorithm_t algorithm) {
    rng_quality_metrics_t metrics = {0};
    
    switch (algorithm) {
        case RNG_MERSENNE_TWISTER:
            metrics.period_log10 = 19.937;  // 2^19937 - 1
            metrics.equidistribution = 623;
            metrics.entropy_estimate = 0.99;
            metrics.passes_diehard = true;
            metrics.passes_testu01 = true;
            break;
            
        case RNG_XORSHIFT128_PLUS:
            metrics.period_log10 = 38.5;   // 2^128 - 1
            metrics.equidistribution = 64;
            metrics.entropy_estimate = 0.95;
            metrics.passes_diehard = true;
            metrics.passes_testu01 = false;
            break;
            
        case RNG_PHILOX_4X32:
            metrics.period_log10 = 38.5;   // 2^128
            metrics.equidistribution = 32;
            metrics.entropy_estimate = 0.98;
            metrics.passes_diehard = true;
            metrics.passes_testu01 = true;
            break;
    }
    
    return metrics;
}

rng_performance_stats_t simd_rng_pool_get_stats(const simd_rng_pool_t* pool) {
    rng_performance_stats_t stats = {0};
    if (!pool) return stats;
    
    switch (pool->algorithm) {
        case RNG_MERSENNE_TWISTER:
        default:
            stats.numbers_generated = simd_mt_pool_get_count(pool->generator.mt_pool);
            stats.generation_rate_mps = 100.0;  // Placeholder
            stats.simd_efficiency = 0.95;       // Placeholder
            break;
    }
    
    return stats;
}

bool simd_rng_pool_check_period_safety(const simd_rng_pool_t* pool, uint64_t count) {
    if (!pool) return false;
    
    // Very conservative check - in practice, period exhaustion is extremely unlikely
    return count < (1ULL << 32);  // Safe for any reasonable simulation
}

bool simd_rng_pool_get_info(const simd_rng_pool_t* pool, uint32_t* pool_size, rng_algorithm_t* algorithm) {
    if (!pool || !pool_size || !algorithm) return false;
    
    *pool_size = pool->pool_size;
    *algorithm = pool->algorithm;
    return true;
}

void simd_rng_pool_warmup(simd_rng_pool_t* pool, uint32_t warmup_count) {
    if (!pool) return;
    
    // Generate and discard warmup_count values
    for (uint32_t i = 0; i < warmup_count; i += 4) {
        simd_generate_uniform_batch(pool);
    }
}

bool simd_rng_pool_statistical_test(simd_rng_pool_t* pool, uint32_t test_size) {
    if (!pool || test_size < 1000) return false;
    
    // Basic statistical tests
    float* samples = (float*)malloc(test_size * sizeof(float));
    if (!samples) return false;
    
    uint32_t generated = simd_generate_uniform_array(pool, samples, test_size);
    if (generated < test_size) {
        free(samples);
        return false;
    }
    
    // Check mean and variance
    double sum = 0.0, sum_sq = 0.0;
    for (uint32_t i = 0; i < test_size; i++) {
        sum += samples[i];
        sum_sq += samples[i] * samples[i];
    }
    
    double mean = sum / test_size;
    double variance = (sum_sq / test_size) - (mean * mean);
    
    // Expected: mean ≈ 0.5, variance ≈ 1/12 ≈ 0.0833
    bool mean_ok = (mean > 0.45 && mean < 0.55);
    bool var_ok = (variance > 0.07 && variance < 0.095);
    
    free(samples);
    return mean_ok && var_ok;
}