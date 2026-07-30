#!/bin/bash

# Builds the Metal experiment tools into build-tools/. These are deliberately not
# in CMake: they are investigation aids, not part of the library. See README.md.
#
# sdl_plane_test additionally needs an SDL build with the per plane IOSurface
# texture properties; point SDL_BUILD_DIR at it, for example:
#   SDL_BUILD_DIR=/path/to/sdl3-build ./tools/metal/build.sh

set -e
cd "$(dirname "$0")/../.."

OUT=build-tools
mkdir -p $OUT

CXX="xcrun -sdk macosx clang++ -std=c++17 -O2 -fobjc-arc -I."
FRAMEWORKS="-framework Metal -framework Foundation"
# pyrowave_metal_common.mm defines the shared device object, so every tool that pulls
# in a backend needs it.
COMMON="pyrowave_metal_common.mm pyrowave_bitstream.cpp"
DECODER="pyrowave_decoder_metal.mm $COMMON"
ENCODER="pyrowave_encoder_metal.mm $COMMON"

echo "bench"
$CXX $FRAMEWORKS tools/metal/bench.mm $DECODER -o $OUT/bench

# The encoder's timing hooks are #ifdef'd out of ordinary builds, so ask for them.
echo "bench_encode"
$CXX $FRAMEWORKS -framework IOSurface -framework CoreFoundation \
	-DPYROWAVE_METAL_BENCH_HOOKS \
	tools/metal/bench_encode.mm $ENCODER -o $OUT/bench_encode

echo "leak_test"
$CXX $FRAMEWORKS -framework IOSurface -framework CoreFoundation \
	tools/metal/leak_test.mm $ENCODER pyrowave_decoder_metal.mm -o $OUT/leak_test

echo "dispatch_cost"
$CXX $FRAMEWORKS tools/metal/dispatch_cost.mm -o $OUT/dispatch_cost

echo "counter_probe"
$CXX $FRAMEWORKS tools/metal/counter_probe.mm -o $OUT/counter_probe

echo "iosurface_array_probe"
$CXX $FRAMEWORKS -framework CoreVideo -framework IOSurface \
	tools/metal/iosurface_array_probe.mm -o $OUT/iosurface_array_probe

echo "iosurface_format_probe"
$CXX $FRAMEWORKS -framework CoreVideo -framework IOSurface \
	tools/metal/iosurface_format_probe.mm -o $OUT/iosurface_format_probe

echo "iosurface_444_probe"
xcrun -sdk macosx clang++ -std=c++17 -O2 -framework CoreVideo -framework IOSurface \
	-framework CoreFoundation tools/metal/iosurface_444_probe.cpp -o $OUT/iosurface_444_probe

echo "gen_stream"
xcrun -sdk macosx clang++ -std=c++17 -O2 -I. \
	tools/metal/gen_stream.cpp pyrowave_bitstream.cpp -o $OUT/gen_stream

echo "y4m_probe"
xcrun -sdk macosx clang++ -std=c++17 -O2 -I. \
	tools/metal/y4m_probe.cpp yuv4mpeg.cpp -o $OUT/y4m_probe

if [ -n "$SDL_BUILD_DIR" ]; then
	echo "sdl_plane_test"
	xcrun -sdk macosx clang -std=c11 -O2 \
		-IGranite/third_party/sdl3/include -I"$SDL_BUILD_DIR/include-config-release" \
		-framework IOSurface -framework CoreFoundation \
		-L"$SDL_BUILD_DIR" -lSDL3 -Wl,-rpath,"$SDL_BUILD_DIR" \
		tools/metal/sdl_plane_test.c -o $OUT/sdl_plane_test
else
	echo "sdl_plane_test  (skipped: set SDL_BUILD_DIR to build it)"
fi

echo
echo "Built into $OUT/"
