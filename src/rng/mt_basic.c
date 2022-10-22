/* 
 * Basic Mersenne Twister implementation 
 * Will optimize with SIMD later
 */

#include <stdio.h>

// MT19937 parameters  
#define N 624
#define M 397

static unsigned long mt[N];
static int mti = N+1;

// Initialize with seed
void mt_init(unsigned long s) {
    mt[0]= s & 0xffffffffUL;
    for (mti=1; mti<N; mti++) {
        mt[mti] = (1812433253UL * (mt[mti-1] ^ (mt[mti-1] >> 30)) + mti); 
        mt[mti] &= 0xffffffffUL;
    }
}

// Generate random number
unsigned long mt_random(void) {
    unsigned long y;
    static unsigned long mag01[2]={0x0UL, 0x9908b0dfUL};
    
    if (mti >= N) { // generate N words at one time
        int kk;
        
        for (kk=0;kk<N-M;kk++) {
            y = (mt[kk]&0x80000000UL)|(mt[kk+1]&0x7fffffffUL);
            mt[kk] = mt[kk+M] ^ (y >> 1) ^ mag01[y & 0x1UL];
        }
        for (;kk<N-1;kk++) {
            y = (mt[kk]&0x80000000UL)|(mt[kk+1]&0x7fffffffUL);
            mt[kk] = mt[kk+(M-N)] ^ (y >> 1) ^ mag01[y & 0x1UL];
        }
        y = (mt[N-1]&0x80000000UL)|(mt[0]&0x7fffffffUL);
        mt[N-1] = mt[M-1] ^ (y >> 1) ^ mag01[y & 0x1UL];
        
        mti = 0;
    }
    
    y = mt[mti++];
    
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    y ^= (y >> 18);
    
    return y;
}