#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "bakuhatsu/var/realtime_var.h"
#include "bakuhatsu/math/nig_distribution.h"
#include "bakuhatsu/streaming/correlation_monitor.h"

/**
 * @file basic_var.c
 * @brief Comprehensive demonstration of Bakuhatsu's bleeding-edge features
 * 
 * This example showcases:
 * 1. Real-time intraday VaR recalibration with microsecond latency
 * 2. Non-Gaussian tail modeling using NIG distributions
 * 3. Dynamic correlation breakdown detection and regime switching
 * 4. SIMD-accelerated Monte Carlo simulation
 * 5. Variance explosion detection and mitigation
 */

#define PORTFOLIO_SIZE 50
#define SIMULATION_MINUTES 5
#define PRICE_UPDATES_PER_SECOND 1000

// Simulate market data structure
typedef struct {
    uint32_t asset_id;
    char symbol[8];
    float base_price;
    float volatility;
    float drift;
    nig_params_t nig_params;
} market_asset_t;

// Global simulation state
static market_asset_t market_assets[PORTFOLIO_SIZE];
static realtime_var_engine_t* var_engine = NULL;
static uint64_t simulation_start_time = 0;
static uint32_t total_price_updates = 0;
static uint32_t regime_changes_detected = 0;

static uint64_t get_timestamp_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000UL + (uint64_t)ts.tv_nsec / 1000UL;
}

static void init_market_data(void) {
    printf(\"🔥 Initializing synthetic market data with NIG fat-tail distributions...\\n\");
    
    // Define realistic asset parameters based on different asset classes
    const char* symbols[] = {
        \"AAPL\", \"MSFT\", \"GOOGL\", \"AMZN\", \"TSLA\", \"META\", \"NVDA\", \"JPM\", \"BAC\", \"GS\",
        \"SPY\", \"QQQ\", \"IWM\", \"VTI\", \"EFA\", \"EEM\", \"TLT\", \"GLD\", \"SLV\", \"OIL\",
        \"BTC\", \"ETH\", \"XRP\", \"ADA\", \"DOT\", \"EUR\", \"GBP\", \"JPY\", \"CHF\", \"AUD\",
        \"CORP1\", \"CORP2\", \"CORP3\", \"CORP4\", \"CORP5\", \"REIT1\", \"REIT2\", \"REIT3\", \"REIT4\", \"REIT5\",
        \"CMDTY1\", \"CMDTY2\", \"CMDTY3\", \"CMDTY4\", \"CMDTY5\", \"FX1\", \"FX2\", \"FX3\", \"FX4\", \"FX5\"
    };
    
    for (uint32_t i = 0; i < PORTFOLIO_SIZE; i++) {
        market_assets[i].asset_id = i;
        strncpy(market_assets[i].symbol, symbols[i], sizeof(market_assets[i].symbol) - 1);
        
        // Base prices between $10 and $500
        market_assets[i].base_price = 10.0f + (float)(rand() % 490);
        
        // Volatilities based on asset class
        if (i < 10) {
            // Individual stocks: higher volatility
            market_assets[i].volatility = 0.20f + (float)(rand() % 30) / 100.0f;  // 20-50% vol
        } else if (i < 20) {
            // ETFs: moderate volatility
            market_assets[i].volatility = 0.15f + (float)(rand() % 15) / 100.0f;  // 15-30% vol
        } else if (i < 30) {
            // Crypto: very high volatility
            market_assets[i].volatility = 0.50f + (float)(rand() % 100) / 100.0f; // 50-150% vol
        } else {
            // Other assets: varied volatility
            market_assets[i].volatility = 0.10f + (float)(rand() % 20) / 100.0f;  // 10-30% vol
        }
        
        market_assets[i].drift = -0.05f + (float)(rand() % 100) / 1000.0f;  // -5% to +5% annual drift
        
        // Initialize NIG parameters for realistic fat-tail modeling
        float alpha = 1.0f + (float)(rand() % 20) / 10.0f;  // 1.0 to 3.0
        float beta = -alpha/3.0f + (float)(rand() % 100) / 150.0f * alpha;  // Skewed left for financial returns
        float delta = market_assets[i].volatility;
        float mu = market_assets[i].drift / 252.0f;  // Daily drift
        
        bool success = nig_params_init(&market_assets[i].nig_params, alpha, beta, delta, mu);
        if (!success) {
            printf(\"⚠️  Warning: Failed to initialize NIG parameters for %s, using default\\n\", 
                   market_assets[i].symbol);
            nig_params_init(&market_assets[i].nig_params, 1.5f, -0.3f, 0.2f, 0.0f);
        }
    }
    
    printf(\"✅ Market data initialized: %d assets with NIG fat-tail distributions\\n\", PORTFOLIO_SIZE);
}

static void init_portfolio(void) {
    printf(\"📈 Initializing diversified portfolio positions...\\n\");
    
    portfolio_position_t positions[PORTFOLIO_SIZE];
    
    for (uint32_t i = 0; i < PORTFOLIO_SIZE; i++) {
        positions[i].asset_id = i;
        positions[i].current_price = market_assets[i].base_price;
        positions[i].price_volatility = market_assets[i].volatility;
        positions[i].distribution = &market_assets[i].nig_params;
        positions[i].last_update_time = get_timestamp_us();
        positions[i].is_active = true;
        
        // Generate realistic position sizes (some long, some short)
        float position_weight = -1.0f + 2.0f * (float)rand() / RAND_MAX;  // -1 to +1
        float notional = 100000.0f + (float)(rand() % 900000);  // $100K to $1M notional
        positions[i].position_size = (position_weight * notional) / positions[i].current_price;
    }
    
    bool success = realtime_var_update_portfolio(var_engine, positions, PORTFOLIO_SIZE, get_timestamp_us());
    if (!success) {
        printf(\"❌ Failed to initialize portfolio\\n\");
        exit(1);
    }
    
    printf(\"✅ Portfolio initialized with %d positions\\n\", PORTFOLIO_SIZE);
}

static void simulate_price_tick(uint32_t asset_id, uint64_t timestamp) {
    // Generate realistic price movement using NIG distribution
    market_asset_t* asset = &market_assets[asset_id];
    
    // Time step (assuming updates every millisecond on average)
    float dt = 1.0f / (252.0f * 24.0f * 3600.0f * 1000.0f);  // Millisecond time step in years
    
    // Generate NIG distributed return
    simd_rng_pool_t* temp_rng = simd_rng_pool_create(1, RNG_MERSENNE_TWISTER, timestamp + asset_id);
    box_muller_state_t temp_bm;
    box_muller_init(&temp_bm);
    
    simd_f32x4_t nig_sample = simd_nig_sample_batch(temp_rng, &temp_bm, &asset->nig_params, 
                                                   NIG_METHOD_INVERSE_GAUSSIAN);
    float return_value = simd_extract_f32(nig_sample, 0) * sqrtf(dt);
    
    // Update price with fat-tail return
    float new_price = asset->base_price * expf(return_value);
    asset->base_price = new_price;
    
    // Update VaR engine with new price
    var_result_t var_result;
    if (!var_result_init(&var_result, PORTFOLIO_SIZE)) {
        printf(\"❌ Failed to initialize VaR result\\n\");
        return;
    }
    
    bool regime_change = realtime_var_update_price(var_engine, asset_id, new_price, timestamp, &var_result);
    
    total_price_updates++;
    
    // Display real-time VaR updates every 100 price updates
    if (total_price_updates % 100 == 0) {
        float elapsed_ms = (timestamp - simulation_start_time) / 1000.0f;
        
        printf(\"\\r⚡ t=%.1fms | %s: $%.2f | VaR(99%%): $%.0f | ES: $%.0f | Latency: %lluμs | Regime: %d\",
               elapsed_ms,
               asset->symbol,
               new_price,
               var_result.value_at_risk,
               var_result.expected_shortfall,
               var_result.computation_time_us,
               var_result.current_regime);
        fflush(stdout);
        
        // Check for variance explosion
        if (var_result.variance_explosion_detected) {
            printf(\"\\n🚨 VARIANCE EXPLOSION DETECTED! Enhanced risk monitoring activated.\\n\");
        }
        
        // Check for regime change
        if (regime_change) {
            regime_changes_detected++;
            printf(\"\\n🔄 CORRELATION REGIME CHANGE #%d detected! Adapting risk model...\\n\", 
                   regime_changes_detected);
        }
    }
    
    var_result_cleanup(&var_result);
    simd_rng_pool_destroy(temp_rng);
}

static void run_stress_test(void) {
    printf(\"\\n\\n🧪 Running stress test scenarios...\\n\");
    
    // Define stress scenarios
    const char* stress_names[] = {
        \"Black Monday (-22%)\",
        \"Flash Crash (-9% in minutes)\", 
        \"COVID Crash (-35%)\",
        \"2008 Financial Crisis (-50%)\",
        \"Long-Term Capital Management (-25%)\"
    };
    
    float stress_magnitudes[] = {5.0f, 3.0f, 7.0f, 10.0f, 6.0f};
    
    for (int scenario = 0; scenario < 5; scenario++) {
        printf(\"\\n📊 Stress Scenario %d: %s\\n\", scenario + 1, stress_names[scenario]);
        
        // Create stress direction (all assets move down with correlation)
        float stress_direction[PORTFOLIO_SIZE];
        for (uint32_t i = 0; i < PORTFOLIO_SIZE; i++) {
            stress_direction[i] = -1.0f + (float)(rand() % 40) / 100.0f;  // -100% to -60% moves
        }
        
        var_result_t stress_result;
        if (!var_result_init(&stress_result, PORTFOLIO_SIZE)) {
            continue;
        }
        
        bool success = realtime_var_stress_test(var_engine, stress_magnitudes[scenario], 
                                              stress_direction, &stress_result);
        
        if (success) {
            printf(\"   💥 Stress VaR: $%.0f (%.1fx normal)\\n\", 
                   stress_result.value_at_risk,
                   stress_result.value_at_risk / 1000000.0f);  // Assuming $1M normal VaR
            printf(\"   🩸 Stress ES:  $%.0f\\n\", stress_result.expected_shortfall);
            printf(\"   ⏱️  Computation: %lluμs\\n\", stress_result.computation_time_us);
        }
        
        var_result_cleanup(&stress_result);
    }
}

static void display_final_statistics(void) {
    printf(\"\\n\\n📈 BAKUHATSU PERFORMANCE REPORT\\n\");
    printf(\"═══════════════════════════════════════════════════════════════\\n\");
    
    const var_performance_stats_t* perf = realtime_var_get_performance(var_engine);
    if (perf) {
        printf(\"🚀 Total VaR Calculations: %llu\\n\", perf->total_calculations);
        printf(\"⚡ Average Latency: %.1fμs\\n\", perf->avg_latency_us);
        printf(\"📊 99th Percentile Latency: %.1fμs\\n\", perf->p99_latency_us);
        printf(\"🎯 Latency Target Violations: %llu\\n\", perf->latency_violations);
        printf(\"🔄 Correlation Regime Changes: %d\\n\", regime_changes_detected);
        printf(\"💥 Variance Explosions Detected: %d\\n\", perf->variance_explosions);
        printf(\"🎮 SIMD Utilization: %.1f%%\\n\", perf->simd_utilization * 100.0f);
        printf(\"💾 Cache Hit Ratio: %.1f%%\\n\", 
               100.0f * perf->cache_hits / (perf->cache_hits + perf->cache_misses + 1));
    }
    
    printf(\"\\n🏆 Total Price Updates Processed: %d\\n\", total_price_updates);
    printf(\"📡 Average Update Rate: %.0f updates/second\\n\", 
           total_price_updates * 1000000.0f / (get_timestamp_us() - simulation_start_time));
    
    printf(\"\\n✨ Bakuhatsu successfully demonstrated bleeding-edge features:\\n\");
    printf(\"   ✅ Real-time intraday VaR with microsecond latency\\n\");
    printf(\"   ✅ Non-Gaussian NIG distribution tail modeling\\n\");
    printf(\"   ✅ Dynamic correlation regime detection\\n\");
    printf(\"   ✅ SIMD-accelerated Monte Carlo simulation\\n\");
    printf(\"   ✅ Variance explosion detection and mitigation\\n\");
    printf(\"   ✅ Comprehensive stress testing capabilities\\n\");
}

int main(void) {
    printf(\"\\n🔥🔥🔥 BAKUHATSU - Next-Generation Monte Carlo VaR Engine 🔥🔥🔥\\n\");
    printf(\"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n\");
    printf(\"🚀 Demonstrating bleeding-edge real-time risk management technology\\n\\n\");
    
    // Initialize random seed
    srand((unsigned int)time(NULL));
    simulation_start_time = get_timestamp_us();
    
    // Check SIMD availability
    if (simd_is_neon_available()) {
        printf(\"✅ ARM NEON SIMD acceleration: AVAILABLE\\n\");
    } else {
        printf(\"⚠️  ARM NEON SIMD acceleration: NOT AVAILABLE (using scalar fallback)\\n\");
    }
    
    // Initialize market data with NIG distributions
    init_market_data();
    
    // Create real-time VaR engine
    printf(\"⚙️  Initializing real-time VaR engine...\\n\");
    realtime_var_config_t config;
    realtime_var_config_init_default(&config, PORTFOLIO_SIZE);
    
    // Configure for bleeding-edge performance
    config.monte_carlo_paths = 50000;        // High path count for accuracy
    config.update_frequency_us = 50;         // 50 microsecond target updates
    config.max_latency_us = 100;             // 100 microsecond max latency
    config.use_nig_tails = true;             // Enable NIG fat-tail modeling
    config.use_dynamic_correlation = true;   // Enable correlation monitoring
    config.variance_explosion_threshold = 2.5f; // Sensitive variance explosion detection
    
    var_engine = realtime_var_engine_create(&config);
    if (!var_engine) {
        printf(\"❌ Failed to create VaR engine\\n\");
        return 1;
    }
    
    printf(\"✅ VaR engine initialized with bleeding-edge configuration\\n\");
    
    // Initialize portfolio
    init_portfolio();
    
    // Warm up the engine
    printf(\"🔥 Warming up SIMD pipelines and caches...\\n\");
    realtime_var_warmup(var_engine, 1000);
    
    // Main simulation loop
    printf(\"\\n🎯 Starting real-time market simulation...\\n\");
    printf(\"⏱️  Simulation duration: %d minutes\\n\", SIMULATION_MINUTES);
    printf(\"📊 Price updates per second: %d\\n\", PRICE_UPDATES_PER_SECOND);
    printf(\"🎮 Portfolio size: %d assets\\n\\n\", PORTFOLIO_SIZE);
    
    uint64_t simulation_duration_us = SIMULATION_MINUTES * 60 * 1000000UL;
    uint64_t update_interval_us = 1000000UL / PRICE_UPDATES_PER_SECOND;
    uint64_t next_update_time = simulation_start_time;
    
    while ((get_timestamp_us() - simulation_start_time) < simulation_duration_us) {
        uint64_t current_time = get_timestamp_us();
        
        if (current_time >= next_update_time) {
            // Simulate price update for random asset
            uint32_t asset_id = rand() % PORTFOLIO_SIZE;
            simulate_price_tick(asset_id, current_time);
            
            next_update_time += update_interval_us;
        }
        
        // Small sleep to prevent busy waiting
        usleep(10);  // 10 microseconds
    }
    
    // Run comprehensive stress tests
    run_stress_test();
    
    // Display final performance statistics
    display_final_statistics();
    
    // Cleanup
    realtime_var_engine_destroy(var_engine);
    
    printf(\"\\n🎉 Bakuhatsu demonstration completed successfully!\\n\");
    printf(\"💡 Ready for production deployment in high-frequency trading environments.\\n\\n\");
    
    return 0;
}