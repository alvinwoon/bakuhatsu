#ifndef BAKUHATSU_H
#define BAKUHATSU_H

/**
 * @file bakuhatsu.h
 * @brief Main header file for Bakuhatsu - Next-Generation Monte Carlo VaR Engine
 */

// Core SIMD utilities
#include "bakuhatsu/simd/neon_utils.h"

// Random number generation
#include "bakuhatsu/rng/rng_pool.h"
#include "bakuhatsu/rng/mersenne_twister_simd.h"

// Mathematical distributions and transformations
#include "bakuhatsu/math/box_muller_simd.h"
#include "bakuhatsu/math/nig_distribution.h"

// Real-time streaming components
#include "bakuhatsu/streaming/correlation_monitor.h"

// VaR calculation engine
#include "bakuhatsu/var/realtime_var.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Bakuhatsu library version information
 */
#define BAKUHATSU_VERSION_MAJOR 1
#define BAKUHATSU_VERSION_MINOR 0
#define BAKUHATSU_VERSION_PATCH 0
#define BAKUHATSU_VERSION_STRING "1.0.0"

/**
 * @brief Initialize Bakuhatsu library
 * @return true on successful initialization, false otherwise
 */
bool bakuhatsu_init(void);

/**
 * @brief Cleanup Bakuhatsu library resources
 */
void bakuhatsu_cleanup(void);

/**
 * @brief Get library version string
 * @return Version string
 */
const char* bakuhatsu_get_version(void);

/**
 * @brief Check if SIMD acceleration is available
 * @return true if SIMD is available, false otherwise
 */
bool bakuhatsu_has_simd_support(void);

#ifdef __cplusplus
}
#endif

#endif // BAKUHATSU_H