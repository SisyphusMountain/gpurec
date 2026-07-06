#!/usr/bin/env bash
# Rebuild the gpurax._seqlik pybind11 extension in-place.
#
# The extension links coraxlib + a minimal set of GeneRaxCore sources; see
# CMakeLists.txt for the exact source list. Output goes to
# gpurax/_seqlik/_impl*.so (importable directly, no `pip install` needed).
#
# Usage: gpurax/_seqlik/build.sh [extra cmake args...]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC_DIR="${REPO_ROOT}/gpurax/_seqlik"
BUILD_DIR="${REPO_ROOT}/build/_seqlik"
PYTHON="${REPO_ROOT}/.venv/bin/python"

PYBIND11_CMAKE_DIR="$("${PYTHON}" -m pybind11 --cmakedir)"

mkdir -p "${BUILD_DIR}"

cmake -S "${SRC_DIR}" -B "${BUILD_DIR}" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -Dpybind11_DIR="${PYBIND11_CMAKE_DIR}" \
    -DPYTHON_EXECUTABLE="${PYTHON}" \
    -DPython_EXECUTABLE="${PYTHON}" \
    "$@"

cmake --build "${BUILD_DIR}" --parallel
