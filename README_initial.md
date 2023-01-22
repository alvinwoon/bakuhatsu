# Bakuhatsu

High-performance Monte Carlo VaR engine for real-time risk managment.

## Progress
- [x] Basic Mersenne Twister RNG
- [x] Box-Muller Gaussian sampling  
- [x] ARM NEON testing framework
- [x] SIMD optimization for RNG (parallel generators!)
- [x] SIMD Box-Muller implementation
- [x] Performance benchmarking framework
- [ ] Non-gaussian tail modeling (NIG distribution) 
- [ ] Real-time correlation monitoring
- [ ] Full VaR engine integration
- [ ] Variance explosion detection

## Performance Goals
Target: sub-100μs VaR updates for real-time trading systems.

Current bottlenecks:
- ~~RNG generation (need SIMD parallization)~~ ✓ SOLVED!
- ~~Gaussian sampling (vectorize Box-Muller)~~ ✓ SOLVED!
- Portfolio aggregation (still need to optimize)
- NIG parameter estimation (computationally expensive)
- Correlation matrix updates (O(n²) complexity)