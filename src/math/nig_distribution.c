#include "bakuhatsu/math/nig_distribution.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

// Mathematical constants
static const float SQRT_2PI = 2.506628274631000502f;
static const float LOG_2PI = 1.838616743403199750f;
static const float EULER_GAMMA = 0.577215664901532861f;

#ifndef PI_F
#define PI_F 3.14159265358979323846f
#endif

// Bessel function approximations for NIG distribution
static float bessel_k1(float x);
static float bessel_k0(float x);
static float modified_bessel_k(float nu, float x);

bool nig_params_init(nig_params_t* params, float alpha, float beta, float delta, float mu) {
    if (!params) return false;
    
    params->alpha = alpha;
    params->beta = beta;
    params->delta = delta;
    params->mu = mu;
    
    // Validate parameters
    if (alpha <= 0.0f || delta <= 0.0f || fabsf(beta) >= alpha) {
        params->parameters_valid = false;
        return false;
    }
    
    params->parameters_valid = true;
    nig_params_update_derived(params);
    return true;
}

bool nig_params_validate(const nig_params_t* params) {
    if (!params) return false;
    
    return (params->alpha > 0.0f && 
            params->delta > 0.0f && 
            fabsf(params->beta) < params->alpha);
}

void nig_params_update_derived(nig_params_t* params) {
    if (!params || !params->parameters_valid) return;
    
    // Compute derived parameters
    params->gamma = sqrtf(params->alpha * params->alpha - params->beta * params->beta);
    params->kappa = params->delta / params->gamma;
    params->theta = params->beta / params->gamma;
    
    // Pre-compute normalization constant for PDF
    float delta_gamma = params->delta * params->gamma;
    params->pdf_normalization = params->alpha / (PI_F * sqrtf(delta_gamma));
    
    // Compute theoretical moments
    nig_compute_moments(params, &params->moment1, &params->moment2, 
                       &params->moment3, &params->moment4);
}

simd_f32x4_t simd_nig_sample_batch(simd_rng_pool_t* rng_pool, box_muller_state_t* bm_state,
                                   const nig_params_t* params, nig_sampling_method_t method) {
    if (!rng_pool || !bm_state || !params || !params->parameters_valid) {
        return simd_set1_f32(0.0f);
    }
    
    switch (method) {
        case NIG_METHOD_INVERSE_GAUSSIAN:
            return simd_nig_sample_inverse_gaussian(rng_pool, bm_state, params);
        
        case NIG_METHOD_REJECTION:
        case NIG_METHOD_ACCEPTANCE_REJECTION:
        default:
            // Fallback to inverse Gaussian method
            return simd_nig_sample_inverse_gaussian(rng_pool, bm_state, params);
    }
}

simd_f32x4_t simd_nig_sample_inverse_gaussian(simd_rng_pool_t* rng_pool, box_muller_state_t* bm_state,
                                              const nig_params_t* params) {
    // NIG distribution via Inverse Gaussian subordination:
    // 1. Sample τ ~ IG(δ/γ, δ²)  (Inverse Gaussian)
    // 2. Sample Z ~ N(0,1)       (Standard Normal)
    // 3. Return X = μ + βτ + √τ * Z
    
    // Step 1: Sample Inverse Gaussian subordinator
    float ig_mu = params->kappa;  // δ/γ
    float ig_lambda = params->delta * params->delta;
    simd_f32x4_t tau = simd_inverse_gaussian_sample(rng_pool, ig_mu, ig_lambda);
    
    // Step 2: Sample standard normal
    simd_f32x4_t z = simd_box_muller_standard(rng_pool, bm_state);
    
    // Step 3: Construct NIG variate: X = μ + βτ + √τ * Z
    simd_f32x4_t beta_vec = simd_set1_f32(params->beta);
    simd_f32x4_t mu_vec = simd_set1_f32(params->mu);
    
    simd_f32x4_t beta_tau = simd_mul_f32(beta_vec, tau);
    simd_f32x4_t sqrt_tau = simd_sqrt_f32(tau);
    simd_f32x4_t sqrt_tau_z = simd_mul_f32(sqrt_tau, z);
    
    // X = μ + βτ + √τ * Z
    simd_f32x4_t result = simd_add_f32(mu_vec, beta_tau);
    result = simd_add_f32(result, sqrt_tau_z);
    
    return result;
}

simd_f32x4_t simd_inverse_gaussian_sample(simd_rng_pool_t* rng_pool, float mu, float lambda) {
    // Inverse Gaussian sampling using Michael, Schucany and Haas method
    // 1. Generate Y ~ N(0,1)
    // 2. Compute V = μ + (μ²Y²)/(2λ) - (μ/(2λ))√(4μλY² + μ²Y⁴)
    // 3. Generate U ~ Uniform(0,1)
    // 4. If U ≤ μ/(μ+V), return V, else return μ²/V
    
    simd_f32x4_t u1 = simd_generate_uniform_batch(rng_pool);
    simd_f32x4_t u2 = simd_generate_uniform_batch(rng_pool);
    
    // Convert uniform to standard normal for Y
    simd_f32x4_t y_squared = simd_set1_f32(0.0f);  // Simplified - would use proper Box-Muller
    
    // Michael-Schucany-Haas transformation
    simd_f32x4_t mu_vec = simd_set1_f32(mu);
    simd_f32x4_t lambda_vec = simd_set1_f32(lambda);
    simd_f32x4_t two_lambda = simd_set1_f32(2.0f * lambda);
    
    // V = μ + (μ²Y²)/(2λ) - (μ/(2λ))√(4μλY² + μ²Y⁴)
    simd_f32x4_t mu_squared = simd_mul_f32(mu_vec, mu_vec);
    simd_f32x4_t term1 = simd_mul_f32(mu_squared, y_squared);
    term1 = simd_mul_f32(term1, simd_set1_f32(1.0f / (2.0f * lambda)));
    
    simd_f32x4_t four_mu_lambda = simd_set1_f32(4.0f * mu * lambda);
    simd_f32x4_t discriminant = simd_fmadd_f32(four_mu_lambda, y_squared, 
                                              simd_mul_f32(mu_squared, simd_mul_f32(y_squared, y_squared)));
    simd_f32x4_t sqrt_discriminant = simd_sqrt_f32(discriminant);
    
    simd_f32x4_t term2 = simd_mul_f32(mu_vec, sqrt_discriminant);
    term2 = simd_mul_f32(term2, simd_set1_f32(1.0f / (2.0f * lambda)));
    
    simd_f32x4_t v = simd_add_f32(mu_vec, term1);
    v = simd_add_f32(v, term2);
    
    // Apply acceptance criterion: U ≤ μ/(μ+V)
    simd_f32x4_t acceptance_prob = simd_mul_f32(mu_vec, simd_rsqrt_f32(simd_add_f32(mu_vec, v)));
    simd_u32x4_t accept_mask = simd_cmplt_f32(u2, acceptance_prob);
    
    // If accepted, use V, else use μ²/V
    simd_f32x4_t alternative = simd_mul_f32(mu_squared, simd_rsqrt_f32(v));
    
    return simd_select_f32(accept_mask, v, alternative);
}

uint32_t simd_nig_sample_array(simd_rng_pool_t* rng_pool, box_muller_state_t* bm_state,
                               const nig_params_t* params, nig_sampling_method_t method,
                               float* output, uint32_t count) {
    if (!rng_pool || !bm_state || !params || !output || count == 0) return 0;
    
    uint32_t simd_count = (count / 4) * 4;
    
    for (uint32_t i = 0; i < simd_count; i += 4) {
        simd_f32x4_t batch = simd_nig_sample_batch(rng_pool, bm_state, params, method);
        simd_store_aligned_f32(&output[i], batch);
    }
    
    return simd_count;
}

float nig_pdf(float x, const nig_params_t* params) {
    if (!params || !params->parameters_valid) return 0.0f;
    
    float z = x - params->mu;
    float alpha = params->alpha;
    float beta = params->beta;
    float delta = params->delta;
    
    // PDF: f(x) = (α/π) * exp(δγ + βz) * K₁(α√(δ² + z²)) / √(δ² + z²)
    float delta_z_squared = delta * delta + z * z;
    float sqrt_delta_z_squared = sqrtf(delta_z_squared);
    
    float arg = alpha * sqrt_delta_z_squared;
    float k1_val = bessel_k1(arg);
    
    float log_pdf = params->delta * params->gamma + beta * z + logf(k1_val) - logf(sqrt_delta_z_squared);
    
    return params->pdf_normalization * expf(log_pdf);
}

void nig_compute_moments(const nig_params_t* params, float* moment1, float* moment2,
                        float* moment3, float* moment4) {
    if (!params || !params->parameters_valid) return;
    
    float alpha = params->alpha;
    float beta = params->beta;
    float delta = params->delta;
    float mu = params->mu;
    float gamma = params->gamma;
    
    // First moment (mean): μ + δβ/γ
    *moment1 = mu + (delta * beta) / gamma;
    
    // Second central moment (variance): δα²/γ³
    *moment2 = (delta * alpha * alpha) / (gamma * gamma * gamma);
    
    // Third central moment: 3δα²β/γ⁵
    *moment3 = (3.0f * delta * alpha * alpha * beta) / powf(gamma, 5.0f);
    
    // Fourth central moment: 3δα²(1 + 4β²/α²)/γ⁷
    float beta_alpha_ratio = beta / alpha;
    *moment4 = (3.0f * delta * alpha * alpha * (1.0f + 4.0f * beta_alpha_ratio * beta_alpha_ratio)) / powf(gamma, 7.0f);
}

bool nig_estimate_parameters_mom(const float* data, uint32_t count, nig_params_t* params) {
    if (!data || count < 4 || !params) return false;
    
    // Compute sample moments
    float sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f, sum4 = 0.0f;
    float mean = 0.0f;
    
    // Compute sample mean
    for (uint32_t i = 0; i < count; i++) {
        mean += data[i];
    }
    mean /= count;
    
    // Compute central moments
    for (uint32_t i = 0; i < count; i++) {
        float dev = data[i] - mean;
        float dev2 = dev * dev;
        float dev3 = dev2 * dev;
        float dev4 = dev3 * dev;
        
        sum2 += dev2;
        sum3 += dev3;
        sum4 += dev4;
    }
    
    float variance = sum2 / (count - 1);
    float skewness = (sum3 / count) / powf(variance, 1.5f);
    float kurtosis = (sum4 / count) / (variance * variance);
    
    // Method of moments estimation (simplified)
    // This is a basic implementation - production code would use more sophisticated methods
    float delta_est = variance;  // Rough initial estimate
    float gamma_est = sqrtf(variance);
    float beta_est = skewness * gamma_est / 3.0f;
    float alpha_est = sqrtf(gamma_est * gamma_est + beta_est * beta_est);
    float mu_est = mean - (delta_est * beta_est) / gamma_est;
    
    return nig_params_init(params, alpha_est, beta_est, delta_est, mu_est);
}

float nig_value_at_risk(const nig_params_t* params, float confidence_level) {
    if (!params || !params->parameters_valid) return 0.0f;
    
    // VaR is the negative of the quantile at (1 - confidence_level)
    float p = 1.0f - confidence_level;
    return -nig_quantile(p, params);
}

float nig_expected_shortfall(const nig_params_t* params, float confidence_level) {
    if (!params || !params->parameters_valid) return 0.0f;
    
    // ES requires numerical integration of the tail
    // This is a simplified implementation
    float var = nig_value_at_risk(params, confidence_level);
    
    // Approximate ES as VaR + tail expectation
    float tail_factor = 1.2f;  // Simplified - should be computed properly
    return var * tail_factor;
}

// Bessel function approximations
static float bessel_k1(float x) {
    if (x <= 2.0f) {
        // Small x approximation
        float x2 = x * x / 4.0f;
        float sum = 1.0f + x2 * (0.15443144f + x2 * (0.00928746f + x2 * 0.00048936f));
        return (1.0f / x) + x * logf(x / 2.0f) * sum;
    } else {
        // Large x approximation
        float inv_x = 1.0f / x;
        float sqrt_pi_2x = sqrtf(PI_F / (2.0f * x));
        float exp_neg_x = expf(-x);
        return sqrt_pi_2x * exp_neg_x * (1.0f + inv_x * (0.125f + inv_x * 0.0703125f));
    }
}

static float bessel_k0(float x) {
    if (x <= 2.0f) {
        float x2 = x * x / 4.0f;
        return -logf(x / 2.0f) * (1.0f + x2 * (0.25f + x2 * 0.015625f)) + 
               (-EULER_GAMMA + x2 * (0.42278420f + x2 * 0.23069756f));
    } else {
        float inv_x = 1.0f / x;
        float sqrt_pi_2x = sqrtf(PI_F / (2.0f * x));
        return sqrt_pi_2x * expf(-x) * (1.0f + inv_x * 0.125f);
    }
}

float nig_quantile(float p, const nig_params_t* params) {
    // Numerical inversion using Newton-Raphson (simplified implementation)
    if (!params || !params->parameters_valid || p <= 0.0f || p >= 1.0f) return 0.0f;
    
    // Initial guess based on normal approximation
    float normal_quantile = params->moment1 + sqrtf(params->moment2) * (-1.96f);  // Simplified
    
    // Newton-Raphson iteration would go here
    // This is a placeholder - production code would implement proper numerical inversion
    
    return normal_quantile;
}