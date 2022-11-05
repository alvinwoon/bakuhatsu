# Bakuhatsu

High-performance Monte Carlo VaR engine for real-time risk managment.

## Progress
- [x] Basic Mersenne Twister RNG
- [x] Box-Muller Gaussian sampling  
- [x] ARM NEON testing framework
- [ ] SIMD optimization for RNG
- [ ] Parallel path generation
- [ ] Non-gaussian tail modeling (NIG distribution)
- [ ] Real-time correlation monitoring
- [ ] Variance explosion detection

## Performance Goals
Target: sub-100μs VaR updates for real-time trading systems.

Current bottlenecks:
- RNG generation (need SIMD parallization)
- Gaussian sampling (vectorize Box-Muller)
- Portfolio aggregation