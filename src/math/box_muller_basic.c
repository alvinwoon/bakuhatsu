// Box-Muller transformation for Gaussian random numbers
// Basic scalar version - will vectorize later

#include <math.h>
#include <stdio.h>

static int has_spare = 0;
static double spare_val;

// Generate standard normal random variable
double box_muller_normal() {
    if (has_spare) {
        has_spare = 0;
        return spare_val;
    }
    
    // Generate two uniform random numbers
    // TODO: use proper RNG here instead of rand()
    double u1 = ((double)rand() + 1.0) / ((double)RAND_MAX + 2.0);
    double u2 = ((double)rand() + 1.0) / ((double)RAND_MAX + 2.0);
    
    // Box-Muller transformation
    double mag = sqrt(-2.0 * log(u1));
    double z0 = mag * cos(2.0 * M_PI * u2);
    double z1 = mag * sin(2.0 * M_PI * u2);
    
    spare_val = z1;
    has_spare = 1;
    
    return z0;
}

// TODO: vectorize this with NEON
// Should be able to compute 4 Gaussians at once