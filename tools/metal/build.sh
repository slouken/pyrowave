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

if [ ! -f metal-cpp/Metal/Metal.hpp ]; then
	echo "metal-cpp missing; run ./checkout_metal_cpp.sh first." >&2
	exit 1
fi

CXX="xcrun -sdk macosx clang++ -std=c++17 -O2 -fblocks -I. -Imetal-cpp"
FRAMEWORKS="-framework Metal -framework Foundation"
DECODER="pyrowave_decoder_metal.cpp pyrowave_bitstream.cpp"

echo "bench"
$CXX $FRAMEWORKS tools/metal/bench.cpp $DECODER -o $OUT/bench

echo "dispatch_cost"
$CXX $FRAMEWORKS tools/metal/dispatch_cost.cpp -o $OUT/dispatch_cost

echo "counter_probe"
$CXX $FRAMEWORKS tools/metal/counter_probe.cpp -o $OUT/counter_probe

echo "iosurface_array_probe"
$CXX $FRAMEWORKS -framework CoreVideo -framework IOSurface \
	tools/metal/iosurface_array_probe.cpp -o $OUT/iosurface_array_probe

echo "iosurface_format_probe"
$CXX $FRAMEWORKS -framework CoreVideo -framework IOSurface \
	tools/metal/iosurface_format_probe.cpp -o $OUT/iosurface_format_probe

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
