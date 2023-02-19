# Bakuhatsu - Next-Generation Monte Carlo VaR Engine
# High-performance build configuration

# Compiler and flags
CC = clang
CFLAGS = -std=c11 -O3 -Wall -Wextra -march=native -ffast-math
CFLAGS += -DHAVE_NEON=1
LDFLAGS = -lm

# Detect architecture and set appropriate flags
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_M),arm64)
    # Apple Silicon - NEON is always available
    CFLAGS += -mcpu=apple-m1
else ifeq ($(UNAME_M),aarch64)
    # ARM64 Linux
    CFLAGS += -mfpu=neon
endif

# Directories
SRCDIR = src
INCDIR = include
BUILDDIR = build
EXAMPLEDIR = examples

# Include paths
INCLUDES = -I$(INCDIR) -I$(SRCDIR)

# Source files
SIMD_SOURCES = $(SRCDIR)/simd/neon_utils.c
RNG_SOURCES = $(SRCDIR)/rng/rng_pool.c $(SRCDIR)/rng/mersenne_twister_simd.c
MATH_SOURCES = $(SRCDIR)/math/box_muller_simd.c $(SRCDIR)/math/nig_distribution.c
STREAMING_SOURCES = $(SRCDIR)/streaming/correlation_monitor.c
# VAR_SOURCES = $(SRCDIR)/var/realtime_var.c  # Temporarily disabled due to typedef issues

ALL_SOURCES = $(SIMD_SOURCES) $(RNG_SOURCES) $(MATH_SOURCES) $(STREAMING_SOURCES)

# Object files
OBJECTS = $(ALL_SOURCES:$(SRCDIR)/%.c=$(BUILDDIR)/%.o)

# Targets
LIBRARY = $(BUILDDIR)/libbakuhatsu.a
EXAMPLE = $(BUILDDIR)/bakuhatsu_example
SIMPLE_TEST = $(BUILDDIR)/simple_test
WORKING_DEMO = $(BUILDDIR)/working_demo

.PHONY: all clean example test simple demo working

all: $(LIBRARY) $(SIMPLE_TEST)

simple: $(SIMPLE_TEST)

working: $(WORKING_DEMO)

# Create build directories
$(BUILDDIR):
	mkdir -p $(BUILDDIR)/simd $(BUILDDIR)/rng $(BUILDDIR)/math $(BUILDDIR)/var $(BUILDDIR)/streaming

# Build object files
$(BUILDDIR)/%.o: $(SRCDIR)/%.c | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Build static library
$(LIBRARY): $(OBJECTS)
	ar rcs $@ $^
	@echo "✅ Built Bakuhatsu library: $@"

# Build simple test
$(SIMPLE_TEST): $(EXAMPLEDIR)/simple_test.c $(LIBRARY)
	$(CC) $(CFLAGS) $(INCLUDES) $< -L$(BUILDDIR) -lbakuhatsu $(LDFLAGS) -o $@
	@echo "✅ Built simple test: $@"

# Build working demo
$(WORKING_DEMO): $(EXAMPLEDIR)/working_demo.c $(LIBRARY)
	$(CC) $(CFLAGS) $(INCLUDES) $< -L$(BUILDDIR) -lbakuhatsu $(LDFLAGS) -o $@
	@echo "✅ Built working demo: $@"

# Build example (currently broken due to VaR engine issues)
$(EXAMPLE): $(EXAMPLEDIR)/basic_var.c $(LIBRARY)
	$(CC) $(CFLAGS) $(INCLUDES) $< -L$(BUILDDIR) -lbakuhatsu $(LDFLAGS) -o $@
	@echo "✅ Built example application: $@"

# Convenience target for example
example: $(EXAMPLE)

# Test target (run the simple test)
test: $(SIMPLE_TEST)
	@echo "🚀 Running Bakuhatsu core functionality test..."
	$(SIMPLE_TEST)

# Full demo target (run the comprehensive example when available)
demo: $(EXAMPLE)
	@echo "🚀 Running comprehensive Bakuhatsu demonstration..."
	$(EXAMPLE)

# Clean build artifacts
clean:
	rm -rf $(BUILDDIR)
	@echo "🧹 Cleaned build directory"

# Display build info
info:
	@echo "🔥 Bakuhatsu Build Configuration"
	@echo "Compiler: $(CC)"
	@echo "Flags: $(CFLAGS)"
	@echo "Sources: $(words $(ALL_SOURCES)) files"
	@echo "Target: ARM NEON SIMD optimized"

# Install (placeholder)
install: $(LIBRARY)
	@echo "📦 Installation not implemented yet"