#ifndef BAKUHATSU_NEON_UTILS_H
#define BAKUHATSU_NEON_UTILS_H

#ifdef HAVE_NEON
#include <arm_neon.h>
#endif

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file neon_utils.h
 * @brief ARM NEON SIMD utility functions for high-performance Monte Carlo calculations
 */

// SIMD vector types
#ifdef HAVE_NEON
typedef float32x4_t simd_f32x4_t;
typedef uint32x4_t simd_u32x4_t;
typedef int32x4_t simd_i32x4_t;
#else
// Fallback scalar implementations
typedef struct { float v[4]; } simd_f32x4_t;
typedef struct { uint32_t v[4]; } simd_u32x4_t;
typedef struct { int32_t v[4]; } simd_i32x4_t;
#endif

// Constants
#define SIMD_VECTOR_SIZE 4
#define SIMD_ALIGNMENT 16

/**
 * @brief Check if ARM NEON is available at runtime
 * @return true if NEON is available, false otherwise
 */
bool simd_is_neon_available(void);

/**
 * @brief Load 4 floats from aligned memory
 * @param ptr Pointer to 16-byte aligned memory
 * @return SIMD vector with loaded values
 */
simd_f32x4_t simd_load_aligned_f32(const float* ptr);

/**
 * @brief Store 4 floats to aligned memory
 * @param ptr Pointer to 16-byte aligned memory
 * @param vec SIMD vector to store
 */
void simd_store_aligned_f32(float* ptr, simd_f32x4_t vec);

/**
 * @brief Create vector with all elements set to same value
 * @param value Value to replicate
 * @return SIMD vector with replicated value
 */
simd_f32x4_t simd_set1_f32(float value);

/**
 * @brief Create vector from 4 individual values
 * @param a, b, c, d Individual float values
 * @return SIMD vector [a, b, c, d]
 */
simd_f32x4_t simd_set_f32(float a, float b, float c, float d);

/**
 * @brief Add two SIMD vectors element-wise
 * @param a First vector
 * @param b Second vector
 * @return Result of a + b
 */
simd_f32x4_t simd_add_f32(simd_f32x4_t a, simd_f32x4_t b);

/**
 * @brief Multiply two SIMD vectors element-wise
 * @param a First vector
 * @param b Second vector
 * @return Result of a * b
 */
simd_f32x4_t simd_mul_f32(simd_f32x4_t a, simd_f32x4_t b);

/**
 * @brief Fused multiply-add: a * b + c
 * @param a First multiplicand
 * @param b Second multiplicand
 * @param c Addend
 * @return Result of a * b + c
 */
simd_f32x4_t simd_fmadd_f32(simd_f32x4_t a, simd_f32x4_t b, simd_f32x4_t c);

/**
 * @brief Square root of vector elements
 * @param a Input vector
 * @return Element-wise square root
 */
simd_f32x4_t simd_sqrt_f32(simd_f32x4_t a);

/**
 * @brief Fast reciprocal square root approximation
 * @param a Input vector
 * @return Approximate 1/sqrt(a)
 */
simd_f32x4_t simd_rsqrt_f32(simd_f32x4_t a);

/**
 * @brief Horizontal sum of vector elements
 * @param a Input vector
 * @return Sum of all 4 elements
 */
float simd_horizontal_sum_f32(simd_f32x4_t a);

/**
 * @brief Compare vectors for less-than
 * @param a First vector
 * @param b Second vector
 * @return Mask vector (0xFFFFFFFF where a[i] < b[i], 0x00000000 otherwise)
 */
simd_u32x4_t simd_cmplt_f32(simd_f32x4_t a, simd_f32x4_t b);

/**
 * @brief Bitwise select based on mask
 * @param mask Selection mask
 * @param a Vector selected where mask bits are 1
 * @param b Vector selected where mask bits are 0
 * @return Blended result
 */
simd_f32x4_t simd_select_f32(simd_u32x4_t mask, simd_f32x4_t a, simd_f32x4_t b);

/**
 * @brief Convert 4 uint32 to 4 float32
 * @param a Input uint32 vector
 * @return Converted float32 vector
 */
simd_f32x4_t simd_cvt_u32_to_f32(simd_u32x4_t a);

/**
 * @brief Extract single float from vector at index
 * @param vec Input vector
 * @param index Index (0-3)
 * @return Float value at index
 */
float simd_extract_f32(simd_f32x4_t vec, int index);

/**
 * @brief Memory barrier for SIMD operations
 */
void simd_memory_barrier(void);

/**
 * @brief Prefetch data into cache
 * @param ptr Pointer to data to prefetch
 */
void simd_prefetch(const void* ptr);

#ifdef __cplusplus
}
#endif

#endif // BAKUHATSU_NEON_UTILS_H