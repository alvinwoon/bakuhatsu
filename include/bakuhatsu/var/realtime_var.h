#ifndef BAKUHATSU_REALTIME_VAR_H
#define BAKUHATSU_REALTIME_VAR_H

#include "bakuhatsu/simd/neon_utils.h"
#include "bakuhatsu/rng/rng_pool.h"
#include "bakuhatsu/math/box_muller_simd.h"
#include "bakuhatsu/math/nig_distribution.h"
#include "bakuhatsu/streaming/correlation_monitor.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file realtime_var.h
 * @brief Real-time Value-at-Risk engine with microsecond-latency portfolio risk calculations
 * 
 * This is the core engine that combines SIMD-accelerated Monte Carlo simulation,
 * non-Gaussian tail modeling, and dynamic correlation monitoring to provide
 * real-time intraday VaR updates with sub-100 microsecond latency.
 */

// Forward declarations
typedef struct realtime_var_engine realtime_var_engine_t;

/**
 * @brief VaR calculation methods
 */
typedef enum {
    VAR_METHOD_MONTE_CARLO,          // Pure Monte Carlo simulation
    VAR_METHOD_MONTE_CARLO_NIG,      // Monte Carlo with NIG distributions
    VAR_METHOD_HISTORICAL_SIMULATION, // Historical simulation
    VAR_METHOD_PARAMETRIC,           // Parametric VaR (analytical)
    VAR_METHOD_HYBRID                // Hybrid Monte Carlo + analytical
} var_method_t;

/**
 * @brief Risk measure types
 */
typedef enum {
    RISK_MEASURE_VAR,               // Value-at-Risk
    RISK_MEASURE_EXPECTED_SHORTFALL, // Expected Shortfall (Conditional VaR)
    RISK_MEASURE_BOTH               // Both VaR and ES
} risk_measure_t;

/**
 * @brief Portfolio position structure
 */
typedef struct {
    uint32_t asset_id;              // Unique asset identifier
    float position_size;            // Position size (can be negative for shorts)
    float current_price;            // Current market price
    float price_volatility;         // Current volatility estimate
    nig_params_t* distribution;     // Asset-specific NIG parameters (optional)
    uint64_t last_update_time;      // Last price update timestamp
    bool is_active;                 // Whether position is currently active
} portfolio_position_t;

/**
 * @brief Real-time VaR engine configuration
 */
typedef struct {
    uint32_t portfolio_size;        // Number of assets in portfolio
    uint32_t monte_carlo_paths;     // Number of MC simulation paths
    float confidence_level;         // VaR confidence level (e.g., 0.95, 0.99)
    var_method_t method;           // VaR calculation method
    risk_measure_t risk_measures;  // Which risk measures to compute
    
    // Real-time performance parameters
    uint32_t update_frequency_us;   // Target update frequency (microseconds)
    uint32_t max_latency_us;       // Maximum acceptable latency
    bool use_adaptive_paths;       // Dynamically adjust MC paths for latency
    
    // Distribution modeling
    bool use_nig_tails;            // Use NIG distributions for tail modeling
    bool use_dynamic_correlation;  // Use real-time correlation monitoring
    float variance_explosion_threshold; // Threshold for variance explosion detection
    
    // Variance reduction techniques
    bool use_antithetic_variates;  // Use antithetic variance reduction
    bool use_control_variates;     // Use control variate method
    bool use_importance_sampling;  // Use adaptive importance sampling
    
    // Performance optimization
    bool enable_simd;              // Enable SIMD optimizations
    uint32_t rng_pool_size;       // Size of RNG pool
    uint32_t prefetch_buffer_size; // Size of prefetch buffer
    bool cache_scenarios;          // Cache scenario calculations
} realtime_var_config_t;

/**
 * @brief VaR calculation result
 */
typedef struct {
    float value_at_risk;           // VaR value
    float expected_shortfall;      // Expected Shortfall (if calculated)
    float portfolio_value;         // Current portfolio value
    float portfolio_volatility;    // Portfolio volatility estimate
    
    // Calculation metadata
    uint32_t paths_used;          // Number of MC paths actually used
    uint64_t computation_time_us; // Computation time in microseconds
    uint64_t timestamp;           // Calculation timestamp
    var_method_t method_used;     // Method actually used
    
    // Risk decomposition
    float* component_var;         // Per-asset VaR contributions
    float* marginal_var;          // Marginal VaR for each position
    float diversification_ratio;  // Portfolio diversification ratio
    
    // Statistical quality
    float monte_carlo_error;      // MC standard error estimate
    uint32_t current_regime;      // Current correlation regime
    bool variance_explosion_detected; // Whether variance explosion occurred
    
    // Performance metrics
    float cache_hit_ratio;        // Computation cache hit ratio
    float simd_efficiency;        // SIMD utilization efficiency
} var_result_t;

/**
 * @brief VaR engine performance statistics
 */
typedef struct {
    uint64_t total_calculations;   // Total VaR calculations performed
    double avg_latency_us;        // Average calculation latency
    double max_latency_us;        // Maximum latency observed
    double p99_latency_us;        // 99th percentile latency
    uint64_t latency_violations;  // Number of latency target violations
    
    uint64_t cache_hits;          // Scenario cache hits
    uint64_t cache_misses;        // Scenario cache misses
    double mc_paths_avg;          // Average MC paths used
    double simd_utilization;      // Average SIMD utilization
    
    uint32_t regime_changes;      // Number of correlation regime changes
    uint32_t variance_explosions; // Number of variance explosion events
} var_performance_stats_t;

/**
 * @brief Create real-time VaR engine
 * @param config Engine configuration
 * @return Pointer to VaR engine or NULL on failure
 */
realtime_var_engine_t* realtime_var_engine_create(const realtime_var_config_t* config);

/**
 * @brief Destroy VaR engine
 * @param engine VaR engine to destroy
 */
void realtime_var_engine_destroy(realtime_var_engine_t* engine);

/**
 * @brief Update portfolio positions
 * @param engine VaR engine
 * @param positions Array of portfolio positions
 * @param num_positions Number of positions
 * @param timestamp Update timestamp
 * @return true on success, false on error
 */
bool realtime_var_update_portfolio(realtime_var_engine_t* engine, const portfolio_position_t* positions,
                                   uint32_t num_positions, uint64_t timestamp);

/**
 * @brief Calculate VaR for current portfolio
 * @param engine VaR engine
 * @param result Output VaR result structure
 * @return true on success, false on error
 */
bool realtime_var_calculate(realtime_var_engine_t* engine, var_result_t* result);

/**
 * @brief Update single asset price and recalculate VaR
 * @param engine VaR engine
 * @param asset_id Asset identifier
 * @param new_price New asset price
 * @param timestamp Update timestamp
 * @param result Output VaR result
 * @return true on success, false on error
 */
bool realtime_var_update_price(realtime_var_engine_t* engine, uint32_t asset_id, float new_price,
                               uint64_t timestamp, var_result_t* result);

/**
 * @brief Batch price update for multiple assets
 * @param engine VaR engine
 * @param asset_ids Array of asset identifiers
 * @param new_prices Array of new prices
 * @param num_updates Number of price updates
 * @param timestamp Update timestamp
 * @param result Output VaR result
 * @return true on success, false on error
 */
bool realtime_var_batch_update(realtime_var_engine_t* engine, const uint32_t* asset_ids,
                               const float* new_prices, uint32_t num_updates,
                               uint64_t timestamp, var_result_t* result);

/**
 * @brief Stress test portfolio under extreme scenarios
 * @param engine VaR engine
 * @param stress_magnitude Stress intensity (1.0 = 1 standard deviation move)
 * @param stress_direction Stress direction vector (per asset)
 * @param result Output stress VaR result
 * @return true on success, false on error
 */
bool realtime_var_stress_test(realtime_var_engine_t* engine, float stress_magnitude,
                              const float* stress_direction, var_result_t* result);

/**
 * @brief Get engine performance statistics
 * @param engine VaR engine
 * @return Pointer to performance statistics (read-only)
 */
const var_performance_stats_t* realtime_var_get_performance(const realtime_var_engine_t* engine);

/**
 * @brief Reset performance statistics
 * @param engine VaR engine
 */
void realtime_var_reset_stats(realtime_var_engine_t* engine);

/**
 * @brief Configure adaptive Monte Carlo path adjustment
 * @param engine VaR engine
 * @param target_latency_us Target latency in microseconds
 * @param min_paths Minimum number of MC paths
 * @param max_paths Maximum number of MC paths
 * @return true on success, false on error
 */
bool realtime_var_configure_adaptive_mc(realtime_var_engine_t* engine, uint32_t target_latency_us,
                                        uint32_t min_paths, uint32_t max_paths);

/**
 * @brief Warm up engine (pre-compute scenarios, populate caches)
 * @param engine VaR engine
 * @param warmup_iterations Number of warmup iterations
 */
void realtime_var_warmup(realtime_var_engine_t* engine, uint32_t warmup_iterations);

/**
 * @brief Validate portfolio for VaR calculation
 * @param positions Portfolio positions
 * @param num_positions Number of positions
 * @return true if portfolio is valid, false otherwise
 */
bool realtime_var_validate_portfolio(const portfolio_position_t* positions, uint32_t num_positions);

/**
 * @brief Compute portfolio Greeks (sensitivity measures)
 * @param engine VaR engine
 * @param delta_output Delta values (price sensitivity)
 * @param gamma_output Gamma values (convexity)
 * @param vega_output Vega values (volatility sensitivity)
 * @return true on success, false on error
 */
bool realtime_var_compute_greeks(realtime_var_engine_t* engine, float* delta_output,
                                 float* gamma_output, float* vega_output);

/**
 * @brief Backtest VaR model performance
 * @param engine VaR engine
 * @param historical_returns Historical return data
 * @param num_periods Number of historical periods
 * @param violation_ratio Output: VaR violation ratio
 * @param kupiec_test Output: Kupiec test p-value
 * @return true on successful backtest, false on error
 */
bool realtime_var_backtest(realtime_var_engine_t* engine, const float* historical_returns,
                           uint32_t num_periods, float* violation_ratio, float* kupiec_test);

/**
 * @brief Initialize default VaR engine configuration
 * @param config Configuration structure to initialize
 * @param portfolio_size Number of assets in portfolio
 */
void realtime_var_config_init_default(realtime_var_config_t* config, uint32_t portfolio_size);

/**
 * @brief Initialize VaR result structure
 * @param result Result structure to initialize
 * @param portfolio_size Number of assets (for component arrays)
 */
bool var_result_init(var_result_t* result, uint32_t portfolio_size);

/**
 * @brief Clean up VaR result structure
 * @param result Result structure to clean up
 */
void var_result_cleanup(var_result_t* result);

/**
 * @brief SIMD-optimized Monte Carlo VaR calculation
 * @param engine VaR engine
 * @param num_paths Number of simulation paths
 * @param confidence_level VaR confidence level
 * @param result Output VaR result
 * @return true on success, false on error
 */
bool simd_monte_carlo_var(realtime_var_engine_t* engine, uint32_t num_paths,
                         float confidence_level, var_result_t* result);

/**
 * @brief Variance explosion detection and mitigation
 * @param engine VaR engine
 * @param current_volatility Current portfolio volatility
 * @param threshold Explosion detection threshold
 * @return true if variance explosion detected, false otherwise
 */
bool detect_variance_explosion(realtime_var_engine_t* engine, float current_volatility, float threshold);

#ifdef __cplusplus
}
#endif

#endif // BAKUHATSU_REALTIME_VAR_H