cmake_minimum_required(VERSION 3.20)
project(bakuhatsu VERSION 1.0.0 LANGUAGES C CXX)

# Set C/C++ standards
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG -ffast-math -march=native")
set(CMAKE_C_FLAGS_DEBUG "-O0 -g -Wall -Wextra -fsanitize=address")

# Options
option(ENABLE_NEON "Enable ARM NEON SIMD optimizations" ON)
option(ENABLE_TESTS "Build unit tests" ON)
option(ENABLE_BENCHMARKS "Build benchmarks" ON)

# Check for ARM NEON support
if(ENABLE_NEON)
    include(CheckCSourceCompiles)
    check_c_source_compiles("
        #include <arm_neon.h>
        int main() {
            float32x4_t v = vdupq_n_f32(1.0f);
            return 0;
        }
    " HAVE_NEON)
    
    if(HAVE_NEON)
        add_definitions(-DHAVE_NEON=1)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm" OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mfpu=neon")
        endif()
    else()
        message(WARNING "ARM NEON not available, falling back to scalar implementations")
    endif()
endif()

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

# Source files
file(GLOB_RECURSE BAKUHATSU_SOURCES
    "src/*.c"
    "src/*.cpp"
)

# Main library
add_library(bakuhatsu STATIC ${BAKUHATSU_SOURCES})
target_include_directories(bakuhatsu PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Link math library
target_link_libraries(bakuhatsu m)

# Example executable
add_executable(bakuhatsu_example examples/basic_var.c)
target_link_libraries(bakuhatsu_example bakuhatsu)

# Tests
if(ENABLE_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Benchmarks
if(ENABLE_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Install
install(TARGETS bakuhatsu
    EXPORT bakuhatsu-targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include)

# Package config
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/bakuhatsu-config-version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

export(EXPORT bakuhatsu-targets
    FILE "${CMAKE_CURRENT_BINARY_DIR}/bakuhatsu-targets.cmake"
)

install(EXPORT bakuhatsu-targets
    FILE bakuhatsu-targets.cmake
    DESTINATION lib/cmake/bakuhatsu
)