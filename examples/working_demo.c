#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "bakuhatsu/simd/neon_utils.h"
#include "bakuhatsu/rng/rng_pool.h"
#include "bakuhatsu/math/box_muller_simd.h"
#include "bakuhatsu/math/nig_distribution.h"

/**
 * @file working_demo.c
 * @brief Working demonstration of Bakuhatsu core features
 */

#define NUM_ASSETS 10
#define NUM_SIMULATIONS 10000

typedef struct {
    char symbol[8];
    float current_price;
    float volatility;
    float weight;
    nig_params_t nig_params;
} asset_t;

static void print_header(void) {
    printf("\n");
    printf("=================================================================\n");
    printf("    BAKUHATSU - Next-Generation Monte Carlo VaR Engine Demo\n");
    printf("=================================================================\n");
    printf("Real-time financial risk management with SIMD acceleration\n\n");
}

static void demonstrate_simd_performance(void) {
    printf("1. SIMD Performance Test\n");
    printf("------------------------\n");
    
    if (simd_is_neon_available()) {
        printf("   Status: ARM NEON SIMD Available\n");
    } else {
        printf("   Status: Using scalar fallback\n");
    }
    
    // Benchmark SIMD vs scalar operations
    const int iterations = 1000000;
    clock_t start, end;
    
    // SIMD benchmark
    start = clock();
    for (int i = 0; i < iterations; i += 4) {
        simd_f32x4_t a = simd_set_f32(1.0f, 2.0f, 3.0f, 4.0f);
        simd_f32x4_t b = simd_set1_f32(1.414f);
        simd_f32x4_t result = simd_mul_f32(a, b);
        (void)result; // Suppress unused variable warning
    }
    end = clock();
    
    double simd_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    printf("   SIMD Operations: %.3f seconds (%d ops)\n", simd_time, iterations);
    printf("   Performance: %.0f MFLOPS\n\n", iterations / simd_time / 1000000.0);
}

static void demonstrate_rng_quality(void) {
    printf("2. Random Number Generation Quality\n");
    printf("-----------------------------------\n");
    
    simd_rng_pool_t* rng = simd_rng_pool_create(4, RNG_MERSENNE_TWISTER, time(NULL));
    if (!rng) {
        printf("   ERROR: Failed to create RNG pool\n\n");
        return;
    }
    
    // Generate samples and compute basic statistics
    const int num_samples = 100000;
    float sum = 0.0f;
    float sum_sq = 0.0f;
    int samples_generated = 0;
    
    for (int i = 0; i < num_samples; i += 4) {
        simd_f32x4_t batch = simd_generate_uniform_batch(rng);
        
        for (int j = 0; j < 4 && samples_generated < num_samples; j++) {
            float val = simd_extract_f32(batch, j);
            if (val >= 0.0f && val <= 1.0f) {  // Valid uniform sample
                sum += val;
                sum_sq += val * val;
                samples_generated++;
            }
        }
    }
    
    if (samples_generated > 0) {
        float mean = sum / samples_generated;
        float variance = (sum_sq / samples_generated) - (mean * mean);
        
        printf("   Samples Generated: %d\n", samples_generated);
        printf("   Mean: %.4f (expected: 0.5000)\n", mean);
        printf("   Variance: %.4f (expected: 0.0833)\n", variance);
        printf("   Quality: %s\n\n", 
               (fabs(mean - 0.5f) < 0.01f && fabs(variance - 0.0833f) < 0.01f) ? "GOOD" : "NEEDS_TUNING");
    }
    
    simd_rng_pool_destroy(rng);
}

static void demonstrate_nig_distribution(void) {
    printf("3. Non-Gaussian NIG Distribution Sampling\n");
    printf("-----------------------------------------\n");
    
    simd_rng_pool_t* rng = simd_rng_pool_create(4, RNG_MERSENNE_TWISTER, time(NULL) + 1);
    if (!rng) {
        printf("   ERROR: Failed to create RNG pool\n\n");
        return;
    }
    
    box_muller_state_t bm_state;
    box_muller_init(&bm_state);
    
    // Initialize NIG parameters for financial returns
    nig_params_t nig_params;
    if (!nig_params_init(&nig_params, 1.5f, -0.3f, 0.2f, 0.0f)) {
        printf("   ERROR: Failed to initialize NIG parameters\n\n");
        simd_rng_pool_destroy(rng);
        return;
    }
    
    printf("   NIG Parameters: alpha=%.2f, beta=%.2f, delta=%.2f, mu=%.2f\n",
           nig_params.alpha, nig_params.beta, nig_params.delta, nig_params.mu);
    
    // Generate samples
    const int nig_samples = 1000;
    float nig_sum = 0.0f;
    float nig_sum_sq = 0.0f;
    int valid_samples = 0;
    
    for (int i = 0; i < nig_samples; i += 4) {
        simd_f32x4_t nig_batch = simd_nig_sample_batch(rng, &bm_state, &nig_params, 
                                                      NIG_METHOD_INVERSE_GAUSSIAN);
        
        for (int j = 0; j < 4 && valid_samples < nig_samples; j++) {
            float val = simd_extract_f32(nig_batch, j);
            if (isfinite(val)) {  // Check for valid sample
                nig_sum += val;
                nig_sum_sq += val * val;
                valid_samples++;
            }
        }
    }
    
    if (valid_samples > 10) {
        float nig_mean = nig_sum / valid_samples;
        float nig_variance = (nig_sum_sq / valid_samples) - (nig_mean * nig_mean);
        
        printf("   Valid Samples: %d\n", valid_samples);
        printf("   Sample Mean: %.4f\n", nig_mean);
        printf("   Sample Variance: %.4f\n", nig_variance);
        printf("   Fat Tails: %s\n\n", (nig_variance > 0.1f) ? "DETECTED" : "NORMAL");
    } else {
        printf("   WARNING: Few valid samples generated - check implementation\n\n");
    }
    
    simd_rng_pool_destroy(rng);
}

static void demonstrate_portfolio_simulation(void) {
    printf("4. Portfolio Risk Simulation\n");
    printf("----------------------------\n");
    
    // Define a simple portfolio
    asset_t portfolio[NUM_ASSETS] = {
        {"AAPL", 150.0f, 0.25f, 0.15f},
        {"MSFT", 300.0f, 0.22f, 0.12f},
        {"GOOGL", 2500.0f, 0.28f, 0.10f},
        {"AMZN", 3000.0f, 0.30f, 0.08f},
        {"TSLA", 200.0f, 0.45f, 0.05f},
        {"META", 280.0f, 0.35f, 0.08f},
        {"NVDA", 400.0f, 0.40f, 0.07f},
        {"JPM", 130.0f, 0.20f, 0.10f},
        {"SPY", 420.0f, 0.15f, 0.15f},
        {"QQQ", 350.0f, 0.18f, 0.10f}
    };
    
    // Initialize NIG parameters for each asset
    for (int i = 0; i < NUM_ASSETS; i++) {
        float alpha = 1.0f + portfolio[i].volatility;
        float beta = -0.2f * alpha;  // Negative skew for financial assets
        float delta = portfolio[i].volatility * 0.5f;
        float mu = 0.0f;
        
        nig_params_init(&portfolio[i].nig_params, alpha, beta, delta, mu);
    }
    
    // Monte Carlo simulation
    simd_rng_pool_t* rng = simd_rng_pool_create(8, RNG_MERSENNE_TWISTER, time(NULL) + 2);
    if (!rng) {
        printf("   ERROR: Failed to create RNG pool\n\n");
        return;
    }
    
    box_muller_state_t bm_state;
    box_muller_init(&bm_state);
    
    float* portfolio_returns = malloc(NUM_SIMULATIONS * sizeof(float));
    if (!portfolio_returns) {
        printf("   ERROR: Memory allocation failed\n\n");
        simd_rng_pool_destroy(rng);
        return;
    }
    
    printf("   Running %d Monte Carlo simulations...\n", NUM_SIMULATIONS);
    
    clock_t sim_start = clock();
    
    // Run simulations
    for (int sim = 0; sim < NUM_SIMULATIONS; sim++) {
        float portfolio_return = 0.0f;
        
        for (int asset = 0; asset < NUM_ASSETS; asset++) {
            // Generate return using NIG distribution
            simd_f32x4_t return_batch = simd_nig_sample_batch(rng, &bm_state, 
                                                             &portfolio[asset].nig_params, 
                                                             NIG_METHOD_INVERSE_GAUSSIAN);
            
            float asset_return = simd_extract_f32(return_batch, 0);
            if (isfinite(asset_return)) {
                portfolio_return += portfolio[asset].weight * asset_return;
            }
        }
        
        portfolio_returns[sim] = portfolio_return;
    }
    
    clock_t sim_end = clock();
    double simulation_time = ((double)(sim_end - sim_start)) / CLOCKS_PER_SEC;
    
    // Sort returns for VaR calculation
    // Simple bubble sort for small arrays
    for (int i = 0; i < NUM_SIMULATIONS - 1; i++) {
        for (int j = 0; j < NUM_SIMULATIONS - i - 1; j++) {
            if (portfolio_returns[j] > portfolio_returns[j + 1]) {
                float temp = portfolio_returns[j];
                portfolio_returns[j] = portfolio_returns[j + 1];
                portfolio_returns[j + 1] = temp;
            }
        }
    }
    
    // Calculate risk metrics
    int var_95_index = (int)(0.05 * NUM_SIMULATIONS);
    int var_99_index = (int)(0.01 * NUM_SIMULATIONS);
    
    float var_95 = -portfolio_returns[var_95_index] * 1000000.0f;  // $1M portfolio
    float var_99 = -portfolio_returns[var_99_index] * 1000000.0f;
    
    printf("   Simulation Time: %.3f seconds\n", simulation_time);
    printf("   Performance: %.0f simulations/second\n", NUM_SIMULATIONS / simulation_time);
    printf("   VaR (95%%): $%.0f\n", var_95);
    printf("   VaR (99%%): $%.0f\n", var_99);
    printf("   Worst Case: $%.0f\n\n", -portfolio_returns[0] * 1000000.0f);
    
    free(portfolio_returns);
    simd_rng_pool_destroy(rng);
}

int main(void) {
    print_header();
    
    demonstrate_simd_performance();
    demonstrate_rng_quality();
    demonstrate_nig_distribution();
    demonstrate_portfolio_simulation();
    
    printf("=================================================================\n");
    printf("    Bakuhatsu core functionality demonstration completed!\n");
    printf("    Ready for high-frequency financial risk management.\n");
    printf("=================================================================\n\n");
    
    return 0;
}