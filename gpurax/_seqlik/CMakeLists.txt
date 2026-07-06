cmake_minimum_required(VERSION 3.18)
project(gpurax_seqlik LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# GeneRaxCore (source-only: we do NOT add_subdirectory the GeneRaxCore
# project itself, since that pulls in MPI/mpischeduler machinery we don't
# need for A2). We only reuse its coraxlib dependency and compile the
# handful of GeneRaxCore .cpp files needed to link LibpllEvaluation.
set(GRC ${CMAKE_CURRENT_SOURCE_DIR}/../../generax_gpurec/generax/ext/GeneRaxCore)

# --- coraxlib -----------------------------------------------------------
# Keep coraxlib's own tests/benchmarks/docs disabled (they default to Off
# already, but set explicitly so this build never depends on GoogleTest /
# Doxygen being present).
set(CORAX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(CORAX_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(CORAX_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(CORAX_BUILD_DIFFICULTY_PREDICTION OFF CACHE BOOL "" FORCE)

add_subdirectory(${GRC}/ext/coraxlib ${CMAKE_CURRENT_BINARY_DIR}/coraxlib_build EXCLUDE_FROM_ALL)

# --- pybind11 -------------------------------------------------------------
find_package(pybind11 CONFIG REQUIRED)

# Minimal GeneRaxCore source set needed to link LibpllEvaluation and its
# transitive dependencies (discovered by iterating on link errors: each
# undefined-symbol failure named the .cpp defining it, added one at a time).
set(GENERAX_SOURCES
    ${GRC}/src/likelihoods/LibpllEvaluation.cpp
    ${GRC}/src/trees/PLLUnrootedTree.cpp
    ${GRC}/src/trees/PLLRootedTree.cpp
    ${GRC}/src/trees/PLLTreeInfo.cpp
    ${GRC}/src/IO/LibpllParsers.cpp
    ${GRC}/src/IO/Logger.cpp
    ${GRC}/src/IO/NewickParserCommon.cpp
    ${GRC}/src/IO/RootedNewickParser.cpp
    ${GRC}/src/parallelization/ParallelContext.cpp
    ${GRC}/src/maths/Random.cpp
)

pybind11_add_module(_impl
    seqlik.cpp
    ${GENERAX_SOURCES}
)

target_include_directories(_impl PRIVATE
    ${GRC}/src
    ${GRC}/ext/coraxlib/src
)

target_link_libraries(_impl PRIVATE corax)

# Output the compiled extension directly into this source directory so it is
# importable in-place as gpurax/_seqlik/_impl*.so (no install step needed).
set_target_properties(_impl PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
