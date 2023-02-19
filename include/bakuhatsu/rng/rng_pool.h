#ifndef BAKUHATSU_RNG_POOL_H
#define BAKUHATSU_RNG_POOL_H

#include "bakuhatsu/simd/neon_utils.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file rng_pool.h
 * @brief High-performance parallel random number generation pool for Monte Carlo simulations
 */

// Forward declarations
typedef struct simd_rng_pool simd_rng_pool_t;

/**
 * @brief RNG algorithm types
 */
typedef enum {
    RNG_MERSENNE_TWISTER,
    RNG_XORSHIFT128_PLUS,
    RNG_PHILOX_4X32
} rng_algorithm_t;

/**
 * @brief RNG quality metrics
 */
typedef struct {
    double period_log10;          // Log10 of period length
    uint32_t equidistribution;    // k-dimensional equidistribution
    double entropy_estimate;      // Estimated entropy per bit
    bool passes_diehard;         // Passes Diehard battery
    bool passes_testu01;         // Passes TestU01 Crush
} rng_quality_metrics_t;

/**
 * @brief RNG performance statistics
 */
typedef struct {
    uint64_t numbers_generated;   // Total numbers generated
    double generation_rate_mps;   // Million numbers per second
    uint64_t cache_misses;       // Cache miss count
    double simd_efficiency;      // SIMD utilization percentage
} rng_performance_stats_t;

/**
 * @brief Create parallel RNG pool
 * @param pool_size Number of parallel generators (should be multiple of 4 for SIMD)
 * @param algorithm RNG algorithm to use
 * @param seed Initial seed value
 * @return Pointer to RNG pool or NULL on failure
 */
simd_rng_pool_t* simd_rng_pool_create(uint32_t pool_size, rng_algorithm_t algorithm, uint64_t seed);

/**
 * @brief Destroy RNG pool and free resources
 * @param pool RNG pool to destroy
 */
void simd_rng_pool_destroy(simd_rng_pool_t* pool);

/**
 * @brief Generate batch of uniform random floats [0, 1)
 * @param pool RNG pool
 * @return SIMD vector with 4 uniform random floats
 */
simd_f32x4_t simd_generate_uniform_batch(simd_rng_pool_t* pool);

/**
 * @brief Generate large array of uniform random floats
 * @param pool RNG pool
 * @param output Output array (must be 16-byte aligned)
 * @param count Number of floats to generate (must be multiple of 4)
 * @return Number of floats actually generated
 */
uint32_t simd_generate_uniform_array(simd_rng_pool_t* pool, float* output, uint32_t count);

/**
 * @brief Generate batch of uniform random integers [0, 2^32)
 * @param pool RNG pool
 * @return SIMD vector with 4 uniform random uint32
 */
simd_u32x4_t simd_generate_uint32_batch(simd_rng_pool_t* pool);

/**
 * @brief Advance RNG state by specified number of steps
 * @param pool RNG pool
 * @param steps Number of steps to advance
 */
void simd_rng_pool_advance(simd_rng_pool_t* pool, uint64_t steps);

/**
 * @brief Reseed RNG pool with new seed
 * @param pool RNG pool
 * @param seed New seed value
 */
void simd_rng_pool_reseed(simd_rng_pool_t* pool, uint64_t seed);

/**
 * @brief Get RNG quality metrics
 * @param algorithm RNG algorithm
 * @return Quality metrics structure
 */
rng_quality_metrics_t simd_rng_get_quality_metrics(rng_algorithm_t algorithm);

/**
 * @brief Get current performance statistics
 * @param pool RNG pool
 * @return Performance statistics
 */
rng_performance_stats_t simd_rng_pool_get_stats(const simd_rng_pool_t* pool);

/**
 * @brief Reset performance statistics
 * @param pool RNG pool
 */
void simd_rng_pool_reset_stats(simd_rng_pool_t* pool);

/**
 * @brief Check if pool can generate specified number of values without wrapping
 * @param pool RNG pool
 * @param count Number of values to check
 * @return true if safe to generate, false if period exhaustion possible
 */
bool simd_rng_pool_check_period_safety(const simd_rng_pool_t* pool, uint64_t count);

/**
 * @brief Get pool configuration information
 * @param pool RNG pool
 * @param pool_size Output: number of parallel generators
 * @param algorithm Output: RNG algorithm being used
 * @return true on success, false on error
 */
bool simd_rng_pool_get_info(const simd_rng_pool_t* pool, uint32_t* pool_size, rng_algorithm_t* algorithm);

/**
 * @brief Warm up RNG pool (generate and discard initial values)
 * @param pool RNG pool
 * @param warmup_count Number of values to generate and discard
 */
void simd_rng_pool_warmup(simd_rng_pool_t* pool, uint32_t warmup_count);

/**
 * @brief Test RNG pool for basic statistical properties
 * @param pool RNG pool
 * @param test_size Number of samples for testing
 * @return true if tests pass, false otherwise
 */
bool simd_rng_pool_statistical_test(simd_rng_pool_t* pool, uint32_t test_size);

#ifdef __cplusplus
}
#endif

#endif // BAKUHATSU_RNG_POOL_H