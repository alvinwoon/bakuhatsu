// Testing ARM NEON intrinsics 

#ifdef __ARM_NEON
#include <arm_neon.h>
#include <stdio.h>

void test_neon_basic() {
    printf("Testing NEON SIMD operations...\n");
    
    // Test basic vector operations
    float32x4_t a = vdupq_n_f32(2.0f);
    float32x4_t b = vdupq_n_f32(3.0f);  
    float32x4_t result = vaddq_f32(a, b);
    
    float output[4];
    vst1q_f32(output, result);
    
    printf("2.0 + 3.0 = %.1f %.1f %.1f %.1f\n", 
           output[0], output[1], output[2], output[3]);
}

// TODO: implement proper SIMD RNG here
// Need to figure out how to vectorize MT efficiently

#else 
void test_neon_basic() {
    printf("NEON not available on this platform\n");
}
#endif