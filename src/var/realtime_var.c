#include "bakuhatsu/var/realtime_var.h"
#include "bakuhatsu/rng/mersenne_twister_simd.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>

/**
 * @brief Real-time VaR engine internal structure
 */
struct realtime_var_engine {
    // Configuration
    realtime_var_config_t config;
    
    // Portfolio data
    portfolio_position_t* positions;
    uint32_t num_positions;
    float* position_weights;           // Normalized position weights
    float* expected_returns;           // Expected return vector
    float* volatilities;              // Volatility vector
    
    // Random number generation
    simd_rng_pool_t* rng_pool;
    box_muller_state_t* bm_states;     // One per thread/path
    
    // Correlation monitoring
    correlation_monitor_t* corr_monitor;
    
    // Simulation state
    float* scenario_buffer;            // Pre-allocated scenario buffer
    float* portfolio_scenarios;       // Portfolio value scenarios
    uint32_t* sorted_indices;         // For quantile calculation
    
    // Caching for performance
    bool* price_changed;              // Track which prices changed
    float* cached_portfolio_value;    // Cached portfolio valuation
    uint64_t last_calculation_time;   // Last full calculation timestamp
    
    // Performance tracking
    var_performance_stats_t performance;
    uint64_t* latency_measurements;   // Ring buffer for latency tracking
    uint32_t latency_ring_pos;        // Current position in ring buffer
    
    // Variance explosion monitoring
    float* volatility_history;        // Historical volatility estimates
    uint32_t volatility_history_size;
    uint32_t volatility_history_pos;
    
    // Threading and synchronization
    bool is_calculating;              // Atomic flag for calculation in progress
    uint32_t calculation_id;          // Monotonic calculation counter
};

static uint64_t get_timestamp_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000UL + (uint64_t)ts.tv_nsec / 1000UL;
}

realtime_var_engine_t* realtime_var_engine_create(const realtime_var_config_t* config) {
    if (!config || config->portfolio_size == 0 || config->monte_carlo_paths == 0) {
        return NULL;
    }
    
    realtime_var_engine_t* engine = (realtime_var_engine_t*)aligned_alloc(
        SIMD_ALIGNMENT, sizeof(realtime_var_engine_t));
    if (!engine) return NULL;
    
    memset(engine, 0, sizeof(realtime_var_engine_t));
    engine->config = *config;
    engine->num_positions = config->portfolio_size;
    
    // Allocate portfolio arrays
    size_t portfolio_size = config->portfolio_size;
    engine->positions = (portfolio_position_t*)calloc(portfolio_size, sizeof(portfolio_position_t));
    engine->position_weights = (float*)aligned_alloc(SIMD_ALIGNMENT, portfolio_size * sizeof(float));
    engine->expected_returns = (float*)aligned_alloc(SIMD_ALIGNMENT, portfolio_size * sizeof(float));
    engine->volatilities = (float*)aligned_alloc(SIMD_ALIGNMENT, portfolio_size * sizeof(float));
    engine->price_changed = (bool*)calloc(portfolio_size, sizeof(bool));
    
    if (!engine->positions || !engine->position_weights || 
        !engine->expected_returns || !engine->volatilities || !engine->price_changed) {
        realtime_var_engine_destroy(engine);
        return NULL;
    }
    
    // Initialize RNG pool
    engine->rng_pool = simd_rng_pool_create(config->rng_pool_size, RNG_MERSENNE_TWISTER, 
                                           get_timestamp_us());
    if (!engine->rng_pool) {
        realtime_var_engine_destroy(engine);
        return NULL;
    }
    
    // Initialize Box-Muller states (one per RNG in pool)
    engine->bm_states = (box_muller_state_t*)calloc(config->rng_pool_size, sizeof(box_muller_state_t));
    if (!engine->bm_states) {
        realtime_var_engine_destroy(engine);
        return NULL;
    }
    
    for (uint32_t i = 0; i < config->rng_pool_size; i++) {
        box_muller_init(&engine->bm_states[i]);
    }
    
    // Initialize correlation monitor if enabled
    if (config->use_dynamic_correlation) {
        correlation_config_t corr_config;
        correlation_config_init_default(&corr_config, portfolio_size);
        engine->corr_monitor = correlation_monitor_create(&corr_config);
        
        if (!engine->corr_monitor) {
            realtime_var_engine_destroy(engine);
            return NULL;
        }
    }
    
    // Allocate simulation buffers
    uint32_t total_scenarios = config->monte_carlo_paths * portfolio_size;
    engine->scenario_buffer = (float*)aligned_alloc(SIMD_ALIGNMENT, total_scenarios * sizeof(float));
    engine->portfolio_scenarios = (float*)aligned_alloc(SIMD_ALIGNMENT, config->monte_carlo_paths * sizeof(float));
    engine->sorted_indices = (uint32_t*)aligned_alloc(SIMD_ALIGNMENT, config->monte_carlo_paths * sizeof(uint32_t));
    
    if (!engine->scenario_buffer || !engine->portfolio_scenarios || !engine->sorted_indices) {
        realtime_var_engine_destroy(engine);
        return NULL;
    }
    
    // Initialize performance tracking
    engine->latency_measurements = (uint64_t*)calloc(1000, sizeof(uint64_t));  // 1000 sample ring buffer
    engine->volatility_history = (float*)calloc(100, sizeof(float));          // 100 period history
    engine->volatility_history_size = 100;
    
    if (!engine->latency_measurements || !engine->volatility_history) {
        realtime_var_engine_destroy(engine);
        return NULL;
    }
    
    return engine;
}

void realtime_var_engine_destroy(realtime_var_engine_t* engine) {
    if (!engine) return;
    
    // Clean up RNG and mathematical components
    simd_rng_pool_destroy(engine->rng_pool);
    correlation_monitor_destroy(engine->corr_monitor);
    
    // Free memory
    free(engine->positions);
    free(engine->position_weights);
    free(engine->expected_returns);
    free(engine->volatilities);
    free(engine->price_changed);
    free(engine->bm_states);
    free(engine->scenario_buffer);
    free(engine->portfolio_scenarios);
    free(engine->sorted_indices);
    free(engine->latency_measurements);
    free(engine->volatility_history);
    
    free(engine);
}

bool realtime_var_update_portfolio(realtime_var_engine_t* engine, const portfolio_position_t* positions,
                                   uint32_t num_positions, uint64_t timestamp) {
    if (!engine || !positions || num_positions != engine->num_positions) {
        return false;
    }
    
    // Update positions and mark changes
    float total_value = 0.0f;
    for (uint32_t i = 0; i < num_positions; i++) {
        bool position_changed = (engine->positions[i].position_size != positions[i].position_size ||
                               engine->positions[i].current_price != positions[i].current_price);
        
        engine->positions[i] = positions[i];
        engine->price_changed[i] = position_changed;
        
        float position_value = positions[i].position_size * positions[i].current_price;
        total_value += fabsf(position_value);
    }
    
    // Compute normalized position weights
    if (total_value > 0.0f) {
        for (uint32_t i = 0; i < num_positions; i++) {
            float position_value = engine->positions[i].position_size * engine->positions[i].current_price;
            engine->position_weights[i] = position_value / total_value;
        }
    }
    
    return true;
}

bool realtime_var_calculate(realtime_var_engine_t* engine, var_result_t* result) {
    if (!engine || !result || engine->is_calculating) {
        return false;
    }
    
    uint64_t start_time = get_timestamp_us();
    engine->is_calculating = true;
    engine->calculation_id++;
    
    // Clear result structure
    memset(result, 0, sizeof(var_result_t));
    result->timestamp = start_time;
    result->method_used = engine->config.method;
    
    bool success = false;
    
    switch (engine->config.method) {
        case VAR_METHOD_MONTE_CARLO:
        case VAR_METHOD_MONTE_CARLO_NIG:
            success = simd_monte_carlo_var(engine, engine->config.monte_carlo_paths,
                                         engine->config.confidence_level, result);
            break;
            
        case VAR_METHOD_PARAMETRIC:
            // Implement parametric VaR calculation
            success = false;  // Placeholder
            break;
            
        default:
            success = false;
    }
    
    // Record performance metrics
    uint64_t end_time = get_timestamp_us();
    result->computation_time_us = end_time - start_time;
    
    // Update performance statistics
    engine->performance.total_calculations++;
    engine->latency_measurements[engine->latency_ring_pos] = result->computation_time_us;
    engine->latency_ring_pos = (engine->latency_ring_pos + 1) % 1000;
    
    if (result->computation_time_us > engine->config.max_latency_us) {
        engine->performance.latency_violations++;
    }
    
    // Update running average latency
    engine->performance.avg_latency_us = 
        (engine->performance.avg_latency_us * 0.95) + (result->computation_time_us * 0.05);
    
    if (result->computation_time_us > engine->performance.max_latency_us) {
        engine->performance.max_latency_us = result->computation_time_us;
    }
    
    engine->is_calculating = false;
    return success;
}

bool simd_monte_carlo_var(realtime_var_engine_t* engine, uint32_t num_paths,
                         float confidence_level, var_result_t* result) {
    if (!engine || !result) return false;
    
    uint32_t portfolio_size = engine->num_positions;
    
    // Generate correlated random scenarios
    for (uint32_t path = 0; path < num_paths; path += 4) {  // Process 4 paths at once with SIMD
        uint32_t batch_size = (path + 4 <= num_paths) ? 4 : (num_paths - path);
        
        // Generate portfolio scenario for this path batch
        float portfolio_pnl[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        
        for (uint32_t asset = 0; asset < portfolio_size; asset++) {
            simd_f32x4_t random_batch;
            
            if (engine->config.method == VAR_METHOD_MONTE_CARLO_NIG && 
                engine->positions[asset].distribution) {
                // Use NIG distribution for this asset
                random_batch = simd_nig_sample_batch(engine->rng_pool, &engine->bm_states[0],
                                                   engine->positions[asset].distribution, 
                                                   NIG_METHOD_INVERSE_GAUSSIAN);
            } else {
                // Use standard Gaussian
                random_batch = simd_box_muller_scaled(engine->rng_pool, &engine->bm_states[0],
                                                    0.0f, engine->volatilities[asset]);
            }
            
            // Apply to portfolio P&L
            float position_value = engine->positions[asset].position_size * engine->positions[asset].current_price;
            
            for (uint32_t i = 0; i < batch_size; i++) {
                float asset_return = simd_extract_f32(random_batch, i);
                portfolio_pnl[i] += position_value * asset_return;
            }
        }
        
        // Store portfolio scenarios
        for (uint32_t i = 0; i < batch_size; i++) {
            engine->portfolio_scenarios[path + i] = portfolio_pnl[i];
        }
    }
    
    // Compute VaR using quantile estimation
    // Sort scenarios to find quantile
    for (uint32_t i = 0; i < num_paths; i++) {
        engine->sorted_indices[i] = i;
    }
    
    // Simple insertion sort for scenarios (could be optimized with SIMD sorting)
    for (uint32_t i = 1; i < num_paths; i++) {
        uint32_t key_idx = engine->sorted_indices[i];
        float key_val = engine->portfolio_scenarios[key_idx];
        int j = i - 1;
        
        while (j >= 0 && engine->portfolio_scenarios[engine->sorted_indices[j]] > key_val) {
            engine->sorted_indices[j + 1] = engine->sorted_indices[j];
            j--;
        }
        engine->sorted_indices[j + 1] = key_idx;
    }
    
    // Calculate VaR as quantile
    uint32_t var_index = (uint32_t)((1.0f - confidence_level) * num_paths);
    if (var_index >= num_paths) var_index = num_paths - 1;
    
    result->value_at_risk = -engine->portfolio_scenarios[engine->sorted_indices[var_index]];
    result->paths_used = num_paths;
    
    // Calculate Expected Shortfall if requested
    if (engine->config.risk_measures == RISK_MEASURE_BOTH || 
        engine->config.risk_measures == RISK_MEASURE_EXPECTED_SHORTFALL) {
        
        float es_sum = 0.0f;
        uint32_t es_count = 0;
        
        for (uint32_t i = 0; i <= var_index; i++) {
            es_sum += engine->portfolio_scenarios[engine->sorted_indices[i]];
            es_count++;
        }
        
        if (es_count > 0) {
            result->expected_shortfall = -es_sum / es_count;
        }
    }
    
    // Compute current portfolio value
    result->portfolio_value = 0.0f;
    for (uint32_t i = 0; i < portfolio_size; i++) {
        result->portfolio_value += engine->positions[i].position_size * engine->positions[i].current_price;
    }
    
    // Estimate Monte Carlo error
    float variance = 0.0f;
    float mean_pnl = 0.0f;
    for (uint32_t i = 0; i < num_paths; i++) {
        mean_pnl += engine->portfolio_scenarios[i];
    }
    mean_pnl /= num_paths;
    
    for (uint32_t i = 0; i < num_paths; i++) {
        float diff = engine->portfolio_scenarios[i] - mean_pnl;
        variance += diff * diff;
    }
    variance /= (num_paths - 1);
    
    result->monte_carlo_error = sqrtf(variance / num_paths);
    
    // Check for variance explosion
    float current_volatility = sqrtf(variance);
    result->variance_explosion_detected = detect_variance_explosion(engine, current_volatility, 
                                                                   engine->config.variance_explosion_threshold);
    
    return true;
}

bool detect_variance_explosion(realtime_var_engine_t* engine, float current_volatility, float threshold) {
    if (!engine || threshold <= 0.0f) return false;
    
    // Add current volatility to history
    engine->volatility_history[engine->volatility_history_pos] = current_volatility;
    engine->volatility_history_pos = (engine->volatility_history_pos + 1) % engine->volatility_history_size;
    
    // Compute rolling average volatility
    float avg_volatility = 0.0f;
    uint32_t count = 0;
    
    for (uint32_t i = 0; i < engine->volatility_history_size; i++) {
        if (engine->volatility_history[i] > 0.0f) {
            avg_volatility += engine->volatility_history[i];
            count++;
        }
    }
    
    if (count == 0) return false;
    avg_volatility /= count;
    
    // Check if current volatility exceeds threshold
    float volatility_ratio = current_volatility / avg_volatility;
    
    if (volatility_ratio > threshold) {
        engine->performance.variance_explosions++;
        return true;
    }
    
    return false;
}

bool realtime_var_update_price(realtime_var_engine_t* engine, uint32_t asset_id, float new_price,
                               uint64_t timestamp, var_result_t* result) {
    if (!engine || asset_id >= engine->num_positions || !result) {
        return false;
    }
    
    // Update single asset price
    engine->positions[asset_id].current_price = new_price;
    engine->positions[asset_id].last_update_time = timestamp;
    engine->price_changed[asset_id] = true;
    
    // Recalculate VaR
    return realtime_var_calculate(engine, result);
}

void realtime_var_config_init_default(realtime_var_config_t* config, uint32_t portfolio_size) {
    if (!config) return;
    
    memset(config, 0, sizeof(realtime_var_config_t));
    
    config->portfolio_size = portfolio_size;
    config->monte_carlo_paths = 10000;
    config->confidence_level = 0.99f;
    config->method = VAR_METHOD_MONTE_CARLO_NIG;
    config->risk_measures = RISK_MEASURE_BOTH;
    
    config->update_frequency_us = 100;      // 100 microsecond updates
    config->max_latency_us = 100;           // 100 microsecond max latency
    config->use_adaptive_paths = true;
    
    config->use_nig_tails = true;
    config->use_dynamic_correlation = true;
    config->variance_explosion_threshold = 3.0f;  // 3x normal volatility
    
    config->use_antithetic_variates = true;
    config->use_control_variates = false;
    config->use_importance_sampling = false;
    
    config->enable_simd = true;
    config->rng_pool_size = 16;             // 16 parallel RNG streams
    config->prefetch_buffer_size = 1024;
    config->cache_scenarios = true;
}

bool var_result_init(var_result_t* result, uint32_t portfolio_size) {
    if (!result || portfolio_size == 0) return false;
    
    memset(result, 0, sizeof(var_result_t));
    
    result->component_var = (float*)calloc(portfolio_size, sizeof(float));
    result->marginal_var = (float*)calloc(portfolio_size, sizeof(float));
    
    return (result->component_var && result->marginal_var);
}

void var_result_cleanup(var_result_t* result) {
    if (result) {
        free(result->component_var);
        free(result->marginal_var);
        memset(result, 0, sizeof(var_result_t));
    }
}

const var_performance_stats_t* realtime_var_get_performance(const realtime_var_engine_t* engine) {
    return engine ? &engine->performance : NULL;
}