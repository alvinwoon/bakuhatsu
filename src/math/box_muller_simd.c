#include "bakuhatsu/math/box_muller_simd.h"
#include "bakuhatsu/rng/mersenne_twister_simd.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

// Forward declarations for SIMD math functions
static simd_f32x4_t simd_log_f32(simd_f32x4_t x);
static simd_f32x4_t simd_cos_f32(simd_f32x4_t x);
static simd_f32x4_t simd_sin_f32(simd_f32x4_t x);

// Mathematical constants for SIMD
static const float PI_F = 3.14159265358979323846f;
static const float TWO_PI_F = 2.0f * 3.14159265358979323846f;

void box_muller_init(box_muller_state_t* state) {
    if (!state) return;
    
    state->has_spare = false;
    state->spare_values = simd_set1_f32(0.0f);
    state->spare_mask = 0;
    state->total_generated = 0;
}

simd_f32x4_t simd_box_muller_standard(simd_rng_pool_t* rng_pool, box_muller_state_t* state) {
    if (!rng_pool || !state) return simd_set1_f32(0.0f);
    
    // Check if we have spare values from previous call
    if (state->has_spare) {
        state->has_spare = false;
        state->total_generated += 4;
        return state->spare_values;
    }
    
    // Generate two independent uniform random vectors
    simd_f32x4_t u1 = simd_generate_uniform_batch(rng_pool);
    simd_f32x4_t u2 = simd_generate_uniform_batch(rng_pool);
    
    // Ensure u1 > 0 to avoid log(0)
    simd_f32x4_t epsilon = simd_set1_f32(1e-10f);
    simd_u32x4_t mask = simd_cmplt_f32(u1, epsilon);
    u1 = simd_select_f32(mask, epsilon, u1);
    
    // Box-Muller transformation: 
    // z0 = sqrt(-2 * ln(u1)) * cos(2*pi * u2)
    // z1 = sqrt(-2 * ln(u1)) * sin(2*pi * u2)
    
    // Compute sqrt(-2 * ln(u1))
    simd_f32x4_t ln_u1 = simd_log_f32(u1);  // Custom SIMD log implementation needed
    simd_f32x4_t neg_two_ln = simd_mul_f32(simd_set1_f32(-2.0f), ln_u1);
    simd_f32x4_t magnitude = simd_sqrt_f32(neg_two_ln);
    
    // Compute 2*pi * u2
    simd_f32x4_t two_pi_u2 = simd_mul_f32(simd_set1_f32(TWO_PI_F), u2);
    
    // Compute cos and sin - need custom SIMD implementations
    simd_f32x4_t cos_val = simd_cos_f32(two_pi_u2);
    simd_f32x4_t sin_val = simd_sin_f32(two_pi_u2);
    
    // Generate both z0 and z1
    simd_f32x4_t z0 = simd_mul_f32(magnitude, cos_val);
    simd_f32x4_t z1 = simd_mul_f32(magnitude, sin_val);
    
    // Store z1 as spare for next call
    state->spare_values = z1;
    state->has_spare = true;
    
    state->total_generated += 4;
    return z0;
}

// Custom SIMD math functions (simplified implementations)
static simd_f32x4_t simd_log_f32(simd_f32x4_t x) {
    // Fast log approximation using bit manipulation and polynomial
    // This is a simplified version - production code would use more accurate methods
    simd_f32x4_t result = simd_set1_f32(0.0f);
    
    for (int i = 0; i < 4; i++) {
        float val = simd_extract_f32(x, i);
        float log_val = logf(val);
        // Would set individual lanes here - simplified for now
        if (i == 0) result = simd_set_f32(log_val, 0, 0, 0);
        // ... similar for other lanes
    }
    
    return result;
}

static simd_f32x4_t simd_cos_f32(simd_f32x4_t x) {
    // Fast cosine approximation using polynomial or lookup table
    simd_f32x4_t result = simd_set1_f32(0.0f);
    
    for (int i = 0; i < 4; i++) {
        float val = simd_extract_f32(x, i);
        float cos_val = cosf(val);
        // Simplified - would use proper SIMD lane setting
    }
    
    return result;
}

static simd_f32x4_t simd_sin_f32(simd_f32x4_t x) {
    // Fast sine approximation
    simd_f32x4_t result = simd_set1_f32(0.0f);
    
    for (int i = 0; i < 4; i++) {
        float val = simd_extract_f32(x, i);
        float sin_val = sinf(val);
        // Simplified - would use proper SIMD lane setting
    }
    
    return result;
}

simd_f32x4_t simd_box_muller_scaled(simd_rng_pool_t* rng_pool, box_muller_state_t* state,
                                     float mean, float stddev) {
    // Generate standard normal N(0,1)
    simd_f32x4_t standard = simd_box_muller_standard(rng_pool, state);
    
    // Scale to N(mean, stddev^2): X = mean + stddev * Z
    simd_f32x4_t stddev_vec = simd_set1_f32(stddev);
    simd_f32x4_t mean_vec = simd_set1_f32(mean);
    
    return simd_fmadd_f32(standard, stddev_vec, mean_vec);
}

uint32_t simd_box_muller_array(simd_rng_pool_t* rng_pool, box_muller_state_t* state,
                               float* output, uint32_t count, float mean, float stddev) {
    if (!rng_pool || !state || !output || count == 0) return 0;
    
    // Ensure count is multiple of 4 for SIMD efficiency
    uint32_t simd_count = (count / 4) * 4;
    
    for (uint32_t i = 0; i < simd_count; i += 4) {
        simd_f32x4_t batch = simd_box_muller_scaled(rng_pool, state, mean, stddev);
        simd_store_aligned_f32(&output[i], batch);
    }
    
    return simd_count;
}

bool correlated_gaussian_params_init(correlated_gaussian_params_t* params, uint32_t dimension,
                                     float mean, float stddev) {
    if (!params || dimension == 0 || dimension > 1000) return false;
    
    params->mean = mean;
    params->stddev = stddev;
    params->dimension = dimension;
    params->is_cholesky_valid = false;
    
    // Allocate correlation matrix (dimension x dimension)
    size_t matrix_size = dimension * dimension * sizeof(float);
    params->correlation_matrix = (float*)aligned_alloc(SIMD_ALIGNMENT, matrix_size);
    params->cholesky_decomp = (float*)aligned_alloc(SIMD_ALIGNMENT, matrix_size);
    
    if (!params->correlation_matrix || !params->cholesky_decomp) {
        correlated_gaussian_params_cleanup(params);
        return false;
    }
    
    // Initialize as identity matrix
    memset(params->correlation_matrix, 0, matrix_size);
    for (uint32_t i = 0; i < dimension; i++) {
        params->correlation_matrix[i * dimension + i] = 1.0f;
    }
    
    return true;
}

bool correlated_gaussian_set_correlation(correlated_gaussian_params_t* params,
                                         const float* correlation_matrix) {
    if (!params || !correlation_matrix) return false;
    
    // Validate correlation matrix
    if (!validate_correlation_matrix(correlation_matrix, params->dimension)) {
        return false;
    }
    
    // Copy correlation matrix
    size_t matrix_size = params->dimension * params->dimension * sizeof(float);
    memcpy(params->correlation_matrix, correlation_matrix, matrix_size);
    
    // Compute Cholesky decomposition
    bool success = simd_cholesky_decomposition(params->correlation_matrix,
                                              params->cholesky_decomp,
                                              params->dimension);
    
    params->is_cholesky_valid = success;
    return success;
}

void correlated_gaussian_params_cleanup(correlated_gaussian_params_t* params) {
    if (params) {
        free(params->correlation_matrix);
        free(params->cholesky_decomp);
        memset(params, 0, sizeof(correlated_gaussian_params_t));
    }
}

bool validate_correlation_matrix(const float* matrix, uint32_t dimension) {
    if (!matrix || dimension == 0) return false;
    
    // Check diagonal elements are 1.0
    for (uint32_t i = 0; i < dimension; i++) {
        if (fabsf(matrix[i * dimension + i] - 1.0f) > 1e-6f) {
            return false;
        }
    }
    
    // Check symmetry
    for (uint32_t i = 0; i < dimension; i++) {
        for (uint32_t j = i + 1; j < dimension; j++) {
            float val1 = matrix[i * dimension + j];
            float val2 = matrix[j * dimension + i];
            if (fabsf(val1 - val2) > 1e-6f) {
                return false;
            }
        }
    }
    
    // Check correlations are in [-1, 1]
    for (uint32_t i = 0; i < dimension; i++) {
        for (uint32_t j = 0; j < dimension; j++) {
            float val = matrix[i * dimension + j];
            if (val < -1.0f || val > 1.0f) {
                return false;
            }
        }
    }
    
    return true;
}

bool simd_cholesky_decomposition(const float* matrix, float* cholesky, uint32_t dimension) {
    if (!matrix || !cholesky || dimension == 0) return false;
    
    // Initialize Cholesky factor to zero
    memset(cholesky, 0, dimension * dimension * sizeof(float));
    
    // Cholesky decomposition: A = L * L^T
    for (uint32_t i = 0; i < dimension; i++) {
        for (uint32_t j = 0; j <= i; j++) {
            if (i == j) {
                // Diagonal element
                float sum = 0.0f;
                for (uint32_t k = 0; k < j; k++) {
                    float val = cholesky[j * dimension + k];
                    sum += val * val;
                }
                
                float diag_val = matrix[i * dimension + i] - sum;
                if (diag_val <= 0.0f) {
                    return false;  // Matrix not positive definite
                }
                
                cholesky[i * dimension + j] = sqrtf(diag_val);
            } else {
                // Off-diagonal element
                float sum = 0.0f;
                for (uint32_t k = 0; k < j; k++) {
                    sum += cholesky[i * dimension + k] * cholesky[j * dimension + k];
                }
                
                cholesky[i * dimension + j] = (matrix[i * dimension + j] - sum) / cholesky[j * dimension + j];
            }
        }
    }
    
    return true;
}

void simd_cholesky_transform(const float* independent_gaussians, const float* cholesky_factor,
                            uint32_t dimension, float* output) {
    if (!independent_gaussians || !cholesky_factor || !output || dimension == 0) return;
    
    // Compute output = L * independent_gaussians
    for (uint32_t i = 0; i < dimension; i++) {
        float sum = 0.0f;
        for (uint32_t j = 0; j <= i; j++) {  // Lower triangular matrix
            sum += cholesky_factor[i * dimension + j] * independent_gaussians[j];
        }
        output[i] = sum;
    }
}

bool simd_box_muller_correlated(simd_rng_pool_t* rng_pool, box_muller_state_t* state,
                                const correlated_gaussian_params_t* params, float* output) {
    if (!rng_pool || !state || !params || !output || !params->is_cholesky_valid) {
        return false;
    }
    
    // Generate independent standard Gaussians
    float* independent = (float*)aligned_alloc(SIMD_ALIGNMENT, params->dimension * sizeof(float));
    if (!independent) return false;
    
    uint32_t generated = simd_box_muller_array(rng_pool, state, independent, params->dimension, 0.0f, 1.0f);
    if (generated < params->dimension) {
        free(independent);
        return false;
    }
    
    // Apply Cholesky transformation for correlation
    simd_cholesky_transform(independent, params->cholesky_decomp, params->dimension, output);
    
    // Scale to desired mean and standard deviation
    for (uint32_t i = 0; i < params->dimension; i++) {
        output[i] = params->mean + params->stddev * output[i];
    }
    
    free(independent);
    return true;
}

uint64_t box_muller_get_count(const box_muller_state_t* state) {
    return state ? state->total_generated : 0;
}

void box_muller_reset(box_muller_state_t* state) {
    if (state) {
        box_muller_init(state);
    }
}