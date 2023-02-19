#ifndef BAKUHATSU_CORRELATION_MONITOR_H
#define BAKUHATSU_CORRELATION_MONITOR_H

#include "bakuhatsu/simd/neon_utils.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file correlation_monitor.h
 * @brief Real-time correlation monitoring and regime detection for dynamic risk management
 * 
 * This module implements streaming correlation estimation with automatic regime change detection,
 * enabling real-time adaptation to market stress conditions and correlation breakdown scenarios.
 */

// Forward declarations
typedef struct correlation_monitor correlation_monitor_t;

/**
 * @brief Correlation estimation methods
 */
typedef enum {
    CORR_METHOD_EXPONENTIAL_WEIGHTED,  // Exponentially weighted moving average
    CORR_METHOD_ROLLING_WINDOW,        // Rolling window correlation
    CORR_METHOD_DCC_GARCH,            // Dynamic Conditional Correlation GARCH
    CORR_METHOD_REGIME_SWITCHING      // Regime-switching correlation model
} correlation_method_t;

/**
 * @brief Regime change detection algorithms
 */
typedef enum {
    REGIME_DETECTOR_CUSUM,            // Cumulative sum control chart
    REGIME_DETECTOR_EWMA,             // Exponentially weighted moving average control
    REGIME_DETECTOR_KALMAN,           // Kalman filter-based detection
    REGIME_DETECTOR_MULTIVARIATE_EWMA // Multivariate EWMA for correlation matrix
} regime_detection_method_t;

/**
 * @brief Correlation regime state
 */
typedef struct {
    uint32_t regime_id;               // Unique regime identifier
    float* correlation_matrix;        // Regime-specific correlation matrix
    float* eigenvalues;              // Principal component eigenvalues
    float* eigenvectors;             // Principal component eigenvectors
    float regime_probability;        // Current probability of being in this regime
    uint64_t regime_duration;        // Time spent in this regime (microseconds)
    uint32_t observation_count;      // Number of observations in this regime
    bool is_stress_regime;           // Whether this is identified as a stress regime
} correlation_regime_t;

/**
 * @brief Correlation monitoring configuration
 */
typedef struct {
    uint32_t dimension;              // Number of assets/variables
    correlation_method_t method;     // Correlation estimation method
    regime_detection_method_t detector; // Regime detection method
    
    // Estimation parameters
    float decay_factor;              // Exponential decay factor (0 < λ < 1)
    uint32_t window_size;           // Rolling window size
    float regularization;           // Matrix regularization parameter
    
    // Regime detection parameters
    float cusum_threshold;          // CUSUM detection threshold
    float regime_min_probability;   // Minimum probability to declare regime change
    uint32_t min_regime_duration;   // Minimum regime duration (observations)
    
    // Performance parameters
    bool use_simd_optimization;     // Enable SIMD acceleration
    uint32_t batch_update_size;     // Batch size for updates
    bool precompute_inverses;       // Pre-compute matrix inverses
} correlation_config_t;

/**
 * @brief Real-time correlation statistics
 */
typedef struct {
    float* current_correlations;     // Current correlation matrix
    float* correlation_volatility;   // Correlation volatility estimates
    float* eigenvalue_series;       // Time series of largest eigenvalue
    float condition_number;         // Current matrix condition number
    float determinant;              // Current matrix determinant
    uint32_t current_regime;        // Current active regime ID
    float regime_change_probability; // Probability of regime change
    uint64_t last_regime_change;    // Timestamp of last regime change
    uint32_t total_regime_changes;  // Total number of regime changes detected
} correlation_statistics_t;

/**
 * @brief Performance metrics for correlation monitoring
 */
typedef struct {
    uint64_t total_updates;         // Total number of updates processed
    uint64_t total_regime_detections; // Total regime changes detected
    double avg_update_time_us;      // Average update time in microseconds
    double max_update_time_us;      // Maximum update time
    uint64_t cache_hits;           // Matrix computation cache hits
    uint64_t cache_misses;         // Matrix computation cache misses
    double matrix_condition_avg;    // Average condition number
    double matrix_condition_max;    // Maximum condition number observed
} correlation_performance_t;

/**
 * @brief Create correlation monitor
 * @param config Configuration parameters
 * @return Pointer to correlation monitor or NULL on failure
 */
correlation_monitor_t* correlation_monitor_create(const correlation_config_t* config);

/**
 * @brief Destroy correlation monitor
 * @param monitor Monitor to destroy
 */
void correlation_monitor_destroy(correlation_monitor_t* monitor);

/**
 * @brief Update correlations with new return vector
 * @param monitor Correlation monitor
 * @param returns Return vector (length = dimension)
 * @param timestamp Timestamp in microseconds
 * @return true if regime change detected, false otherwise
 */
bool correlation_monitor_update(correlation_monitor_t* monitor, const float* returns, uint64_t timestamp);

/**
 * @brief Batch update with multiple return observations
 * @param monitor Correlation monitor
 * @param returns_matrix Matrix of returns (rows = observations, cols = assets)
 * @param num_observations Number of observations
 * @param timestamps Array of timestamps
 * @return Number of regime changes detected
 */
uint32_t correlation_monitor_batch_update(correlation_monitor_t* monitor, const float* returns_matrix,
                                          uint32_t num_observations, const uint64_t* timestamps);

/**
 * @brief Get current correlation matrix
 * @param monitor Correlation monitor
 * @param output Output correlation matrix (dimension x dimension)
 * @return true on success, false on error
 */
bool correlation_monitor_get_correlation_matrix(const correlation_monitor_t* monitor, float* output);

/**
 * @brief Get current correlation statistics
 * @param monitor Correlation monitor
 * @return Pointer to current statistics (read-only)
 */
const correlation_statistics_t* correlation_monitor_get_statistics(const correlation_monitor_t* monitor);

/**
 * @brief Get performance metrics
 * @param monitor Correlation monitor
 * @return Pointer to performance metrics (read-only)
 */
const correlation_performance_t* correlation_monitor_get_performance(const correlation_monitor_t* monitor);

/**
 * @brief Check if correlation matrix is well-conditioned
 * @param monitor Correlation monitor
 * @param max_condition_number Maximum acceptable condition number
 * @return true if well-conditioned, false otherwise
 */
bool correlation_monitor_is_well_conditioned(const correlation_monitor_t* monitor, float max_condition_number);

/**
 * @brief Force regime detection recalibration
 * @param monitor Correlation monitor
 */
void correlation_monitor_recalibrate(correlation_monitor_t* monitor);

/**
 * @brief Get regime information
 * @param monitor Correlation monitor
 * @param regime_id Regime identifier
 * @return Pointer to regime structure or NULL if not found
 */
const correlation_regime_t* correlation_monitor_get_regime(const correlation_monitor_t* monitor, uint32_t regime_id);

/**
 * @brief Get all active regimes
 * @param monitor Correlation monitor
 * @param regimes Output array of regime pointers
 * @param max_regimes Maximum number of regimes to return
 * @return Number of regimes returned
 */
uint32_t correlation_monitor_get_all_regimes(const correlation_monitor_t* monitor, 
                                            const correlation_regime_t** regimes, uint32_t max_regimes);

/**
 * @brief Compute correlation forecast
 * @param monitor Correlation monitor
 * @param forecast_horizon Number of steps ahead
 * @param forecast_correlation Output forecasted correlation matrix
 * @return true on success, false on error
 */
bool correlation_monitor_forecast(const correlation_monitor_t* monitor, uint32_t forecast_horizon,
                                 float* forecast_correlation);

/**
 * @brief Compute stress-test correlation scenario
 * @param monitor Correlation monitor
 * @param stress_magnitude Stress intensity (0.0 = no stress, 1.0 = maximum historical stress)
 * @param stress_correlation Output stress correlation matrix
 * @return true on success, false on error
 */
bool correlation_monitor_stress_scenario(const correlation_monitor_t* monitor, float stress_magnitude,
                                        float* stress_correlation);

/**
 * @brief Validate correlation matrix properties
 * @param correlation_matrix Input correlation matrix
 * @param dimension Matrix dimension
 * @param tolerance Numerical tolerance for validation
 * @return true if valid correlation matrix, false otherwise
 */
bool correlation_validate_matrix(const float* correlation_matrix, uint32_t dimension, float tolerance);

/**
 * @brief Regularize correlation matrix to ensure positive definiteness
 * @param correlation_matrix Input/output correlation matrix
 * @param dimension Matrix dimension
 * @param regularization_factor Regularization strength
 * @return true on success, false on error
 */
bool correlation_regularize_matrix(float* correlation_matrix, uint32_t dimension, float regularization_factor);

/**
 * @brief Compute condition number of correlation matrix
 * @param correlation_matrix Input correlation matrix
 * @param dimension Matrix dimension
 * @return Condition number (ratio of largest to smallest eigenvalue)
 */
float correlation_compute_condition_number(const float* correlation_matrix, uint32_t dimension);

/**
 * @brief SIMD-optimized correlation matrix update
 * @param old_correlation Previous correlation matrix
 * @param returns Current return vector
 * @param decay_factor Exponential decay factor
 * @param dimension Matrix dimension
 * @param new_correlation Output updated correlation matrix
 */
void simd_correlation_exponential_update(const float* old_correlation, const float* returns,
                                        float decay_factor, uint32_t dimension, float* new_correlation);

/**
 * @brief SIMD-optimized eigenvalue computation for correlation monitoring
 * @param correlation_matrix Input correlation matrix
 * @param dimension Matrix dimension
 * @param eigenvalues Output eigenvalues
 * @param eigenvectors Output eigenvectors (optional, can be NULL)
 * @return true on successful computation, false on error
 */
bool simd_correlation_compute_eigenvalues(const float* correlation_matrix, uint32_t dimension,
                                         float* eigenvalues, float* eigenvectors);

/**
 * @brief Detect correlation regime change using multivariate CUSUM
 * @param monitor Correlation monitor
 * @param new_observation New return vector
 * @return Regime change score (> threshold indicates regime change)
 */
float correlation_detect_regime_change_cusum(correlation_monitor_t* monitor, const float* new_observation);

/**
 * @brief Initialize default correlation monitoring configuration
 * @param config Configuration structure to initialize
 * @param dimension Number of assets
 */
void correlation_config_init_default(correlation_config_t* config, uint32_t dimension);

/**
 * @brief Reset correlation monitor state
 * @param monitor Correlation monitor
 */
void correlation_monitor_reset(correlation_monitor_t* monitor);

#ifdef __cplusplus
}
#endif

#endif // BAKUHATSU_CORRELATION_MONITOR_H