#ifndef BAKUHATSU_BOX_MULLER_SIMD_H
#define BAKUHATSU_BOX_MULLER_SIMD_H

#include "bakuhatsu/simd/neon_utils.h"
#include "bakuhatsu/rng/rng_pool.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file box_muller_simd.h
 * @brief SIMD-optimized Box-Muller transformation for generating correlated Gaussian variates
 */

/**
 * @brief Box-Muller transformation state
 */
typedef struct {
    bool has_spare;              // Whether we have a spare Gaussian value
    simd_f32x4_t spare_values;   // Cached spare values from previous call
    uint32_t spare_mask;         // Mask indicating which spare values are valid
    uint64_t total_generated;    // Total Gaussian values generated
} box_muller_state_t;

/**
 * @brief Correlated Gaussian generation parameters
 */
typedef struct {
    float mean;                  // Mean of distribution
    float stddev;               // Standard deviation
    float* correlation_matrix;   // Correlation matrix (must be positive semi-definite)
    uint32_t dimension;         // Number of correlated variables
    float* cholesky_decomp;     // Pre-computed Cholesky decomposition
    bool is_cholesky_valid;     // Whether Cholesky decomposition is up-to-date
} correlated_gaussian_params_t;

/**
 * @brief Gaussian sampling quality metrics
 */
typedef struct {
    double mean_estimate;        // Empirical mean
    double variance_estimate;    // Empirical variance
    double skewness;            // Empirical skewness (should be ~0)
    double kurtosis;            // Empirical kurtosis (should be ~3)
    double kolmogorov_stat;     // Kolmogorov-Smirnov test statistic
    bool normality_test_passed; // Whether distribution passes normality tests
} gaussian_quality_metrics_t;

/**
 * @brief Initialize Box-Muller state
 * @param state Pointer to Box-Muller state
 */
void box_muller_init(box_muller_state_t* state);

/**
 * @brief Generate SIMD batch of 4 standard Gaussian variates N(0,1)
 * @param rng_pool Random number generator pool
 * @param state Box-Muller state (for caching spare values)
 * @return SIMD vector with 4 Gaussian random variates
 */
simd_f32x4_t simd_box_muller_standard(simd_rng_pool_t* rng_pool, box_muller_state_t* state);

/**
 * @brief Generate SIMD batch of 4 Gaussian variates N(mean, stddev^2)
 * @param rng_pool Random number generator pool
 * @param state Box-Muller state
 * @param mean Mean of the distribution
 * @param stddev Standard deviation
 * @return SIMD vector with 4 Gaussian random variates
 */
simd_f32x4_t simd_box_muller_scaled(simd_rng_pool_t* rng_pool, box_muller_state_t* state, 
                                     float mean, float stddev);

/**
 * @brief Generate large array of Gaussian variates using SIMD optimization
 * @param rng_pool Random number generator pool
 * @param state Box-Muller state
 * @param output Output array (must be 16-byte aligned)
 * @param count Number of values to generate (must be multiple of 4)
 * @param mean Mean of distribution
 * @param stddev Standard deviation
 * @return Number of values actually generated
 */
uint32_t simd_box_muller_array(simd_rng_pool_t* rng_pool, box_muller_state_t* state,
                               float* output, uint32_t count, float mean, float stddev);

/**
 * @brief Generate correlated multivariate Gaussian vector
 * @param rng_pool Random number generator pool
 * @param state Box-Muller state
 * @param params Correlation parameters
 * @param output Output vector for correlated Gaussians
 * @return true on success, false on error
 */
bool simd_box_muller_correlated(simd_rng_pool_t* rng_pool, box_muller_state_t* state,
                                const correlated_gaussian_params_t* params, float* output);

/**
 * @brief Initialize correlated Gaussian parameters
 * @param params Parameters structure to initialize
 * @param dimension Number of correlated variables
 * @param mean Mean value for all variables
 * @param stddev Standard deviation for all variables
 * @return true on success, false on memory allocation failure
 */
bool correlated_gaussian_params_init(correlated_gaussian_params_t* params, uint32_t dimension,
                                     float mean, float stddev);

/**
 * @brief Set correlation matrix and compute Cholesky decomposition
 * @param params Parameters structure
 * @param correlation_matrix Correlation matrix (row-major, dimension x dimension)
 * @return true if matrix is positive semi-definite, false otherwise
 */
bool correlated_gaussian_set_correlation(correlated_gaussian_params_t* params,
                                         const float* correlation_matrix);

/**
 * @brief Clean up correlated Gaussian parameters
 * @param params Parameters structure to clean up
 */
void correlated_gaussian_params_cleanup(correlated_gaussian_params_t* params);

/**
 * @brief Validate correlation matrix (positive semi-definite check)
 * @param matrix Correlation matrix
 * @param dimension Matrix dimension
 * @return true if valid, false otherwise
 */
bool validate_correlation_matrix(const float* matrix, uint32_t dimension);

/**
 * @brief Compute Cholesky decomposition using SIMD optimization
 * @param matrix Input correlation matrix
 * @param cholesky Output Cholesky factor (lower triangular)
 * @param dimension Matrix dimension
 * @return true on success, false if matrix is not positive definite
 */
bool simd_cholesky_decomposition(const float* matrix, float* cholesky, uint32_t dimension);

/**
 * @brief Apply Cholesky transformation to independent Gaussians
 * @param independent_gaussians Input independent N(0,1) variates
 * @param cholesky_factor Cholesky decomposition matrix
 * @param dimension Vector dimension
 * @param output Output correlated Gaussians
 */
void simd_cholesky_transform(const float* independent_gaussians, const float* cholesky_factor,
                            uint32_t dimension, float* output);

/**
 * @brief Test Box-Muller implementation for statistical quality
 * @param rng_pool Random number generator pool
 * @param sample_size Number of samples for testing
 * @return Quality metrics structure
 */
gaussian_quality_metrics_t simd_box_muller_quality_test(simd_rng_pool_t* rng_pool, uint32_t sample_size);

/**
 * @brief Generate antithetic Gaussian pairs for variance reduction
 * @param rng_pool Random number generator pool
 * @param state Box-Muller state
 * @param output1 First set of Gaussian variates
 * @param output2 Antithetic pair (negated values)
 * @param count Number of pairs to generate
 * @return Number of pairs actually generated
 */
uint32_t simd_box_muller_antithetic(simd_rng_pool_t* rng_pool, box_muller_state_t* state,
                                    float* output1, float* output2, uint32_t count);

/**
 * @brief Polar form of Box-Muller (alternative implementation)
 * @param rng_pool Random number generator pool
 * @param state Box-Muller state
 * @return SIMD vector with 4 Gaussian variates
 */
simd_f32x4_t simd_box_muller_polar(simd_rng_pool_t* rng_pool, box_muller_state_t* state);

/**
 * @brief Get Box-Muller generation statistics
 * @param state Box-Muller state
 * @return Number of Gaussian values generated
 */
uint64_t box_muller_get_count(const box_muller_state_t* state);

/**
 * @brief Reset Box-Muller state
 * @param state Box-Muller state to reset
 */
void box_muller_reset(box_muller_state_t* state);

#ifdef __cplusplus
}
#endif

#endif // BAKUHATSU_BOX_MULLER_SIMD_H