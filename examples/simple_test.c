#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "bakuhatsu/simd/neon_utils.h"
#include "bakuhatsu/rng/rng_pool.h"
#include "bakuhatsu/math/box_muller_simd.h"
#include "bakuhatsu/math/nig_distribution.h"

/**
 * @file simple_test.c
 * @brief Simple test of core Bakuhatsu functionality
 */

int main(void) {
    printf("🔥 Bakuhatsu - Simple Functionality Test\\n");
    printf("==========================================\\n\\n");
    
    // Test SIMD availability
    printf("1. Testing SIMD Support:\\n");
    if (simd_is_neon_available()) {
        printf("   ✅ ARM NEON SIMD: Available\\n");
    } else {
        printf("   ⚠️  ARM NEON SIMD: Not available (using scalar fallback)\\n");
    }
    
    // Test basic SIMD operations
    printf("\\n2. Testing SIMD Operations:\\n");
    simd_f32x4_t test_vec = simd_set_f32(1.0f, 2.0f, 3.0f, 4.0f);
    simd_f32x4_t doubled = simd_mul_f32(test_vec, simd_set1_f32(2.0f));
    
    printf("   Vector [1,2,3,4] * 2 = [%.1f,%.1f,%.1f,%.1f]\\n",
           simd_extract_f32(doubled, 0),
           simd_extract_f32(doubled, 1), 
           simd_extract_f32(doubled, 2),
           simd_extract_f32(doubled, 3));
    
    // Test RNG
    printf("\\n3. Testing Random Number Generation:\\n");
    simd_rng_pool_t* rng = simd_rng_pool_create(4, RNG_MERSENNE_TWISTER, time(NULL));
    if (rng) {
        simd_f32x4_t random_batch = simd_generate_uniform_batch(rng);
        printf("   Random batch: [%.3f,%.3f,%.3f,%.3f]\\n",
               simd_extract_f32(random_batch, 0),
               simd_extract_f32(random_batch, 1),
               simd_extract_f32(random_batch, 2),
               simd_extract_f32(random_batch, 3));
        
        // Test Box-Muller
        printf("\\n4. Testing Gaussian Generation (Box-Muller):\\n");
        box_muller_state_t bm_state;
        box_muller_init(&bm_state);
        
        simd_f32x4_t gaussian_batch = simd_box_muller_standard(rng, &bm_state);
        printf("   Gaussian batch: [%.3f,%.3f,%.3f,%.3f]\\n",
               simd_extract_f32(gaussian_batch, 0),
               simd_extract_f32(gaussian_batch, 1),
               simd_extract_f32(gaussian_batch, 2),
               simd_extract_f32(gaussian_batch, 3));
        
        // Test NIG distribution
        printf("\\n5. Testing NIG Distribution:\\n");
        nig_params_t nig_params;
        if (nig_params_init(&nig_params, 1.5f, -0.3f, 0.5f, 0.0f)) {
            simd_f32x4_t nig_batch = simd_nig_sample_batch(rng, &bm_state, &nig_params, 
                                                           NIG_METHOD_INVERSE_GAUSSIAN);
            printf("   NIG batch: [%.3f,%.3f,%.3f,%.3f]\\n",
                   simd_extract_f32(nig_batch, 0),
                   simd_extract_f32(nig_batch, 1),
                   simd_extract_f32(nig_batch, 2),
                   simd_extract_f32(nig_batch, 3));
            
            printf("   NIG parameters: α=%.2f, β=%.2f, δ=%.2f, μ=%.2f\\n",
                   nig_params.alpha, nig_params.beta, nig_params.delta, nig_params.mu);
        } else {
            printf("   ❌ Failed to initialize NIG parameters\\n");
        }
        
        simd_rng_pool_destroy(rng);
    } else {
        printf("   ❌ Failed to create RNG pool\\n");
    }
    
    printf("\\n🎉 Basic functionality test completed!\\n");
    printf("\\n📊 Bakuhatsu is ready for high-frequency financial computing.\\n");
    
    return 0;
}