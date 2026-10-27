#!/bin/sh
# Build jev-score-v2 (jev_score_v2.cpp) against a llama.cpp checkout you supply.
#   sh build_jev_score.sh /path/to/llama.cpp          -> ./build/jev-score-v2
# Tested with llama.cpp commit 441df11f65ea0b6d0c72965aaf70c8241070ddcb (2026-09-23): macOS / Metal (2B v3) and
# Linux / CUDA (v4, LLAMA_CMAKE_FLAGS="-DGGML_CUDA=ON"):
#   git clone https://github.com/ggml-org/llama.cpp && git -C llama.cpp checkout 441df11f65ea0b6d0c72965aaf70c8241070ddcb
# Other commits need qwen35 support and llama_context_params.n_outputs_max; a warning is printed if the
# checkout is at a different commit (the parity figures in release_config.json are for the tested one).
#
# Stock libllama, no patches. jev-score-v2 creates ONE context with attention_type = NON_CAUSAL and
# n_batch = n_ubatch = 2048 and calls llama_decode once per renderer block (<= 2048 tokens), which is
# exactly the block-causal attention the model was trained with (see the header of jev_score_v2.cpp).
#
# If llama.cpp has no shared-library build yet, it is configured and built first (Metal on macOS; pass
# extra CMake flags in LLAMA_CMAKE_FLAGS, e.g. "-DGGML_CUDA=ON" on Linux/CUDA).
# OUT=/some/path/jev-score-v2 overrides the output file.
set -eu
LC=${1:?usage: sh build_jev_score.sh /path/to/llama.cpp}
LC=$(cd "$LC" && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-"$HERE/build/jev-score-v2"}
TESTED=441df11f65ea0b6d0c72965aaf70c8241070ddcb
if command -v git >/dev/null 2>&1 && git -C "$LC" rev-parse HEAD >/dev/null 2>&1; then
  HEAD=$(git -C "$LC" rev-parse HEAD)
  if [ "$HEAD" != "$TESTED" ]; then
    echo "warning: llama.cpp is at $HEAD; this runtime was tested with $TESTED" >&2
  fi
fi
LIBDIR="$LC/build/bin"
if [ ! -f "$LIBDIR/libllama.dylib" ] && [ ! -f "$LIBDIR/libllama.so" ]; then
  cmake -S "$LC" -B "$LC/build" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF ${LLAMA_CMAKE_FLAGS:-}
  cmake --build "$LC/build" -j 8 --target llama
fi
if [ ! -f "$LIBDIR/libllama.dylib" ] && [ ! -f "$LIBDIR/libllama.so" ]; then
  LIBDIR=$(dirname "$(find "$LC/build" -name 'libllama.so' -o -name 'libllama.dylib' | head -n 1)")
fi
mkdir -p "$(dirname "$OUT")"
c++ -std=c++17 -O2 -Wall -Wno-unused-function \
  -I"$LC/include" -I"$LC/ggml/include" -I"$LC/vendor" \
  "$HERE/jev_score_v2.cpp" -o "$OUT" \
  -L"$LIBDIR" -lllama -lggml -lggml-base -Wl,-rpath,"$LIBDIR"
echo "$OUT"
