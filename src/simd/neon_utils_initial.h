#ifndef NEON_UTILS_H
#define NEON_UTILS_H

// ARM NEON SIMD utilities for financial computing
// TODO: implement proper NEON intrinsics

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

// Basic vector operations
void simd_init(void);
float* simd_malloc(int size);

#endif