#ifndef BAKUHATSU_NIG_DISTRIBUTION_H
#define BAKUHATSU_NIG_DISTRIBUTION_H

#include "bakuhatsu/simd/neon_utils.h"
#include "bakuhatsu/rng/rng_pool.h"
#include "bakuhatsu/math/box_muller_simd.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file nig_distribution.h
 * @brief SIMD-optimized Normal Inverse Gaussian (NIG) distribution for realistic fat-tail modeling
 * 
 * The NIG distribution is a four-parameter continuous probability distribution that provides
 * realistic modeling of financial returns with proper fat tails and skewness, addressing
 * the limitations of the Gaussian distribution in financial applications.
 */

/**
 * @brief NIG distribution parameters
 * 
 * The NIG distribution is parameterized by (α, β, δ, μ) where:
 * - α > 0: shape parameter (tail heaviness)
 * - |β| < α: asymmetry parameter (skewness)
 * - δ > 0: scale parameter
 * - μ ∈ ℝ: location parameter (mean)
 */
typedef struct {
    float alpha;                 // Shape parameter (α > 0, controls tail heaviness)
    float beta;                  // Asymmetry parameter (|β| < α, controls skewness)
    float delta;                 // Scale parameter (δ > 0)
    float mu;                    // Location parameter (μ, mean shift)
    
    // Derived parameters for optimization
    float gamma;                 // γ = √(α² - β²)
    float kappa;                 // κ = δ/γ, used in sampling
    float theta;                 // θ = β/γ, asymmetry ratio
    
    // Cached values for efficiency
    bool parameters_valid;       // Whether parameters satisfy constraints
    float pdf_normalization;     // Pre-computed normalization constant
    float moment1;              // First moment (mean)
    float moment2;              // Second moment (variance)
    float moment3;              // Third moment (skewness)
    float moment4;              // Fourth moment (kurtosis)
} nig_params_t;

/**
 * @brief NIG sampling method types
 */
typedef enum {
    NIG_METHOD_INVERSE_GAUSSIAN,    // Via inverse Gaussian subordination (recommended)
    NIG_METHOD_REJECTION,           // Rejection sampling method
    NIG_METHOD_ACCEPTANCE_REJECTION // Acceptance-rejection with envelope
} nig_sampling_method_t;

/**
 * @brief NIG distribution quality metrics
 */
typedef struct {
    double empirical_mean;       // Sample mean
    double empirical_variance;   // Sample variance
    double empirical_skewness;   // Sample skewness
    double empirical_kurtosis;   // Sample kurtosis
    double ks_statistic;        // Kolmogorov-Smirnov test statistic
    double anderson_darling;    // Anderson-Darling test statistic
    bool distribution_valid;    // Whether samples match theoretical distribution
} nig_quality_metrics_t;

/**
 * @brief Initialize NIG parameters with validation
 * @param params NIG parameters structure
 * @param alpha Shape parameter (must be > 0)
 * @param beta Asymmetry parameter (must satisfy |β| < α)
 * @param delta Scale parameter (must be > 0)
 * @param mu Location parameter
 * @return true if parameters are valid, false otherwise
 */
bool nig_params_init(nig_params_t* params, float alpha, float beta, float delta, float mu);

/**
 * @brief Validate NIG parameters
 * @param params NIG parameters
 * @return true if parameters satisfy all constraints
 */
bool nig_params_validate(const nig_params_t* params);

/**
 * @brief Update derived parameters after parameter changes
 * @param params NIG parameters to update
 */
void nig_params_update_derived(nig_params_t* params);

/**
 * @brief Generate SIMD batch of 4 NIG distributed random variates
 * @param rng_pool Random number generator pool
 * @param bm_state Box-Muller state for Gaussian generation
 * @param params NIG distribution parameters
 * @param method Sampling method to use
 * @return SIMD vector with 4 NIG random variates
 */
simd_f32x4_t simd_nig_sample_batch(simd_rng_pool_t* rng_pool, box_muller_state_t* bm_state,
                                   const nig_params_t* params, nig_sampling_method_t method);

/**
 * @brief Generate large array of NIG distributed values
 * @param rng_pool Random number generator pool
 * @param bm_state Box-Muller state
 * @param params NIG parameters
 * @param method Sampling method
 * @param output Output array (must be 16-byte aligned)
 * @param count Number of values to generate (must be multiple of 4)
 * @return Number of values actually generated
 */
uint32_t simd_nig_sample_array(simd_rng_pool_t* rng_pool, box_muller_state_t* bm_state,
                               const nig_params_t* params, nig_sampling_method_t method,
                               float* output, uint32_t count);

/**
 * @brief Sample NIG via Inverse Gaussian subordination (most efficient method)
 * @param rng_pool Random number generator pool
 * @param bm_state Box-Muller state
 * @param params NIG parameters
 * @return SIMD vector with 4 NIG samples
 */
simd_f32x4_t simd_nig_sample_inverse_gaussian(simd_rng_pool_t* rng_pool, box_muller_state_t* bm_state,
                                              const nig_params_t* params);

/**
 * @brief Sample Inverse Gaussian distribution (subordinator for NIG)
 * @param rng_pool Random number generator pool
 * @param mu Mean parameter
 * @param lambda Shape parameter
 * @return SIMD vector with 4 Inverse Gaussian samples
 */
simd_f32x4_t simd_inverse_gaussian_sample(simd_rng_pool_t* rng_pool, float mu, float lambda);

/**
 * @brief Compute NIG probability density function
 * @param x Input value
 * @param params NIG parameters
 * @return PDF value at x
 */
float nig_pdf(float x, const nig_params_t* params);

/**
 * @brief Compute NIG cumulative distribution function (numerical approximation)
 * @param x Input value
 * @param params NIG parameters
 * @return CDF value at x
 */
float nig_cdf(float x, const nig_params_t* params);

/**
 * @brief Compute NIG quantile function (numerical inversion)
 * @param p Probability (0 < p < 1)
 * @param params NIG parameters
 * @return Quantile value
 */
float nig_quantile(float p, const nig_params_t* params);

/**
 * @brief Compute theoretical moments of NIG distribution
 * @param params NIG parameters
 * @param moment1 Output: first moment (mean)
 * @param moment2 Output: second central moment (variance)
 * @param moment3 Output: third central moment (skewness measure)
 * @param moment4 Output: fourth central moment (kurtosis measure)
 */
void nig_compute_moments(const nig_params_t* params, float* moment1, float* moment2,
                        float* moment3, float* moment4);

/**
 * @brief Estimate NIG parameters from sample data using method of moments
 * @param data Sample data array
 * @param count Number of samples
 * @param params Output: estimated parameters
 * @return true on successful estimation, false otherwise
 */
bool nig_estimate_parameters_mom(const float* data, uint32_t count, nig_params_t* params);

/**
 * @brief Estimate NIG parameters using maximum likelihood estimation
 * @param data Sample data array
 * @param count Number of samples
 * @param params Output: estimated parameters
 * @param max_iterations Maximum optimization iterations
 * @param tolerance Convergence tolerance
 * @return true on successful estimation, false otherwise
 */
bool nig_estimate_parameters_mle(const float* data, uint32_t count, nig_params_t* params,
                                 uint32_t max_iterations, float tolerance);

/**
 * @brief Test NIG sampling quality against theoretical distribution
 * @param rng_pool Random number generator pool
 * @param bm_state Box-Muller state
 * @param params NIG parameters
 * @param method Sampling method
 * @param sample_size Number of samples for testing
 * @return Quality metrics structure
 */
nig_quality_metrics_t simd_nig_quality_test(simd_rng_pool_t* rng_pool, box_muller_state_t* bm_state,
                                            const nig_params_t* params, nig_sampling_method_t method,
                                            uint32_t sample_size);

/**
 * @brief Generate correlated NIG variates using Gaussian copula
 * @param rng_pool Random number generator pool
 * @param bm_state Box-Muller state
 * @param params Array of NIG parameters for each variate
 * @param correlation_matrix Gaussian copula correlation matrix
 * @param dimension Number of variates
 * @param output Output array for correlated NIG samples
 * @return true on success, false on error
 */
bool simd_nig_sample_correlated(simd_rng_pool_t* rng_pool, box_muller_state_t* bm_state,
                               const nig_params_t* params, const float* correlation_matrix,
                               uint32_t dimension, float* output);

/**
 * @brief Convert NIG parameters between different parameterizations
 * @param params Input parameters (α, β, δ, μ)
 * @param params_alt Output alternative parameterization
 * @param target_form Target parameterization type
 * @return true on successful conversion
 */
bool nig_convert_parameterization(const nig_params_t* params, nig_params_t* params_alt,
                                 int target_form);

/**
 * @brief Compute Value-at-Risk for NIG distribution
 * @param params NIG parameters
 * @param confidence_level Confidence level (e.g., 0.95, 0.99)
 * @return VaR value
 */
float nig_value_at_risk(const nig_params_t* params, float confidence_level);

/**
 * @brief Compute Expected Shortfall (Conditional VaR) for NIG distribution
 * @param params NIG parameters
 * @param confidence_level Confidence level
 * @return Expected Shortfall value
 */
float nig_expected_shortfall(const nig_params_t* params, float confidence_level);

#ifdef __cplusplus
}
#endif

#endif // BAKUHATSU_NIG_DISTRIBUTION_H