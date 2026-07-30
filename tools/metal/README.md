# Metal port experiment tools

Tools written while porting the decoder and encoder to Metal and tuning them. They
are investigation aids, not part of the library: no CMake wiring, minimal error
handling, hardcoded assumptions. They are here because re-deriving the measurements
is much more work than re-reading the code.

Build everything with `./tools/metal/build.sh`, which drops binaries in
`build-tools/`. Some tools need `samples/` (see the "Sample data" section at the
bottom) and `sdl_plane_test` needs an SDL with the per plane IOSurface texture
properties.

The cross backend comparison below additionally uses `pyrowave-bench`, which is a
normal `PYROWAVE_DEVEL` target rather than one of these tools.

`bench_encode` is the one tool that needs something from the library itself: the
encoder owns its command buffer and never hands it out, so reading `GPUStartTime`
or swapping the dispatch type has to happen inside the backend. Those two hooks sit
behind `PYROWAVE_METAL_BENCH_HOOKS` in `pyrowave_encoder_metal.cpp` and are
compiled out of every ordinary build; `build.sh` passes the define.

## Performance tools

### `bench.cpp` — decoder GPU timing
Decodes the first frame of a stream repeatedly and reports GPU time from
`GPUStartTime`/`GPUEndTime`, as min / p10 / median over 360 samples.

    build-tools/bench samples/stream_1080p_high.bin
    PYROWAVE_PRECISION=2 build-tools/bench samples/stream_1080p_444.bin

**Use min, and interleave runs.** This is a live desktop with background GPU
work; single medians vary enough to invent a 1.4x speedup that is not real. Every
performance claim in the commit messages comes from min-of-400 over three
interleaved rounds.

### `bench_encode.cpp` — encoder GPU timing, and the dispatch mode A/B
Encodes a synthetic frame at six configurations and reports GPU time, the wall
clock latency an application actually sees from the synchronous encode API, and the
CPU cost of `packetize`. It also measures the concurrent compute encoder against a
serial one, interleaving both modes inside a single run.

    build-tools/bench_encode [iterations]

Apple M1, min of 400 interleaved iterations, rate target `width*height/8`:

    case                gpu min  wall min  packetize
    640x480 420           0.392     0.774      0.006
    1280x720 420          0.730     1.035      0.018
    1920x1080 420         1.334     1.653      0.038
    1920x1080 444         2.270     2.588      0.045
    3840x2160 420         4.730     5.103      0.150

1080p 4:2:0 encodes in ~1.33 ms of GPU time, about **2.2x the decoder's 0.627 ms**,
which is what five stages against two ought to cost. `packetize` is a CPU memcpy and
never matters. The gap between gpu and wall is command buffer submission and
completion notification, which the synchronous API cannot avoid.

Serial against concurrent, GPU minimum: **2.47x at 640x480**, 1.81x at 720p, 1.44x
at 1080p 4:2:0, 1.29x at 1080p 4:4:4, 1.14x at 4K. Same shape as the decoder, and
the same cause — see `dispatch_cost.cpp` below.

**Interleave the A/B inside one run.** Measuring the two dispatch modes in separate
runs inflated the 480p speedup from 2.47x to 2.70x, purely from background load
drifting between runs. The serial toggle is re-read per encode specifically so this
is possible.

**Unexpected: the IOSurface input path costs ~12% more GPU time than feeding from
host memory** (1.334 vs 1.184 ms at 1080p 4:2:0, consistently). Most likely because
IOSurface backed textures are forced linear while Metal allocated ones can be tiled,
which the DWT's gather reads care about. IOSurface still wins overall on wall clock,
1.653 against 2.407 ms, because the host path pays a 3 MB upload first — but
zero-copy is not free here.

### `dispatch_cost.cpp` — marginal cost of a compute dispatch
Sweeps dispatch count with a trivial kernel, serial vs concurrent encoder. The
slope is the per dispatch cost.

Measured on an M1: **1.31 us serial, 0.05 us concurrent.**

The important conclusion is what this *rules out*. 42 dequant dispatches at
1.31 us is only 55 us, but making them concurrent saved 220 us at 640x480. So
barrier latency was never the bottleneck — the cost was **underutilization**,
because each dispatch is far too small to fill the GPU on its own (the coarsest
480p iDWT level is 12 threadgroups, 768 threads). That is why concurrency pays
enormously at 480p and not at all at 1080p 4:4:4.

### `counter_probe.cpp` — counter sampling support
Prints which `MTLCounterSamplingPoint` values the device supports.

On an M1: **only `AtStageBoundary`.** There is no dispatch boundary sampling, so
per dispatch timestamps cannot be collected on this hardware; use
`dispatch_cost` or temporary phase skipping instead.

## Cross backend comparison: Metal vs Vulkan on the same GPU

The Vulkan build does run on this Mac, on the KosmicKrisp Vulkan driver, so the two
backends can be compared on identical hardware rather than against a reference from a
different GPU.

    cmake -B build-vk -G Ninja -DCMAKE_BUILD_TYPE=Release -DPYROWAVE_DEVEL=ON -DPYROWAVE_METAL=OFF
    ninja -C build-vk pyrowave-bench
    export VK_DRIVER_FILES=/usr/local/share/vulkan/icd.d/libkosmickrisp_icd.json
    export GRANITE_VULKAN_LIBRARY=/usr/local/lib/libvulkan.1.dylib
    PYROWAVE_BENCH_ITERATIONS=300 ./build-vk/pyrowave-bench input.y4m

Two things are needed to get that far, both outside this repo. Granite's
`third_party/CMakeLists.txt` adds its vendored pyrowave unconditionally under
`GRANITE_FFMPEG`, which collides with the top level target; guard it with
`if (NOT TARGET pyrowave)`. And `Granite/util/timer.cpp` uses `clock_nanosleep` with
`TIMER_ABSTIME`, neither of which exists on macOS.

**`GRANITE_VULKAN_LIBRARY` is mandatory.** Granite's loader does a bare
`dlopen("libvulkan.1.dylib")`, which does not resolve `/usr/local/lib` here, and then
returns false *silently* — the symptom is a clean exit 0 with no output and no
message.

### Results

GPU time per frame, same M1, ~500 KB/frame:

    config            encode Vk   encode Mtl   decode Vk   decode Mtl
    640x480 420           2.467        0.356       0.740       0.172
    1920x1080 420         3.972        1.301       1.376       0.541
    3840x2160 420         8.603        4.51        2.964       1.572
    3840x2160 444        13.513        7.81        5.018       3.046

Metal is **6.9x** ahead at 480p narrowing to **1.7x** at 4K 4:4:4 for encode, and
**4.3x** to **1.65x** for decode. That shape matches the serial versus concurrent
dispatch measurement almost exactly, and Vulkan is still 2.6x / 2.1x / 1.6x slower
than *serialized* Metal, so dispatch concurrency explains part of the gap and
KosmicKrisp's shader codegen the rest.

### Reading the Vulkan numbers

KosmicKrisp reports no timestamp support, so `write_timestamp` yields nothing and the
repository root `bench.cpp` behind `pyrowave-bench` — not `tools/metal/bench.cpp` —
also wall clocks its loops. It reports four segments per frame, because a
single bracket around the loop body cannot tell a busy CPU from one **blocked** inside
`next_frame_context()` — Granite's frame contexts are a ring, so that call waits for an
older context to retire, which is where GPU time actually shows up. My first attempt
made exactly that mistake and reported "cpu-in-loop is 100% of wall", which looks like
a CPU bound loop and is not.

**Do not subtract the CPU cost from the wall time.** For 4K 4:2:0 encode the segments
are record 0.210, submit 0.491, frame-context 7.874. `record` is KosmicKrisp
translating the ~170 dispatches and is *constant* across resolution, as expected since
the dispatch count barely varies with frame size. `PYROWAVE_BENCH_NULL=1` submits the
same command buffers with the encode work removed and costs 0.083 ms/frame, so the
harness is free. The CPU work is overlapped with GPU execution, so wall per frame
already *is* GPU time per frame; subtracting would double count. An earlier guess that
there was a ~2 ms fixed CPU floor, inferred from 480p to 1080p scaling only 1.6x
against a 6.75x area ratio, was wrong — the 480p figure is genuine GPU time.

### Caveats

- The Vulkan decoder takes fallback paths here: it logs "Using texel buffers instead
  of SSBO" and "Using linear textures instead of texel buffers", so this is not the
  code path a desktop GPU would run.
- Shader compiles are slow, 240-280 ms per pipeline on first use.
- Decoding the same stream with both decoders at `PYROWAVE_PRECISION=2` **on the same
  GPU** still differs by 30 pixels over 4 frames at 480p, all off by one. The residual
  1 LSB is therefore independent float codegen, not the hardware difference it was
  originally attributed to.

## SDL / IOSurface investigation

These back the SDL Metal renderer change (per plane IOSurface texture
properties). Keep them with any upstream bug report.

### `iosurface_array_probe.cpp` — the silent aliasing bug
Fills plane 1 of a `y420` CVPixelBuffer with `0xAA` and plane 2 with `0xBB`, wraps
plane 1 in the 2 layer array texture SDL uses for IYUV/YV12/P408/P416, and reads
both slices back.

Both slices return `0xAA`. The texture is created successfully, with no
assertion and no validation warning even under `METAL_DEVICE_WRAPPER_TYPE=1` —
**slice 1 silently aliases plane 1, so V renders as U.** An IOSurface backed
texture must be `MTLTextureType2D`; array textures are not allowed, and this path
fails quietly rather than loudly.

### `iosurface_444_probe.cpp` — why P408 needs per plane surfaces
Builds a 3 plane 4:4:4 IOSurface by hand (which works) and tries to wrap it in a
CVPixelBuffer.

`CVPixelBufferCreateWithIOSurface` fails with **-6661**
(`kCVReturnInvalidPixelFormat`). CoreVideo has no 3 plane 8-bit 4:4:4 format, so
`SDL_PROP_TEXTURE_CREATE_METAL_PIXELBUFFER_POINTER` cannot express P408 at all.

### `iosurface_format_probe.cpp` — what Metal accepts per CV format
Enumerates 2D and array texture creation against several CVPixelBuffer formats.
Note this one reports the array case as "OK" — that result is misleading, and
`iosurface_array_probe` is what shows the data is wrong.

### `sdl_plane_test.c` — end to end U/V correctness
Renders a solid `Y=128 U=200 V=60` through the per plane IOSurface path for
IYUV, YV12 and P408 and reads pixels back. Under BT.709 limited that should come
out blue (`R~9 G~151 B~255`); a U/V swap comes out red instead, which is
impossible to miss.

Needs SDL built with the per plane properties, and `-lSDL3` pointed at it.

## Sample data helpers

### `gen_stream.cpp` — synthetic bitstream
Writes a PYROWAVE container whose blocks are all coded with an empty ballot. The
decoded result is a known constant, which makes it useful for exercising the
payload path and the subgroup prefix sums without needing the Vulkan encoder.

### `y4m_probe.cpp` — validate a y4m against the repo's own reader
Opens a file with `YUV4MPEGFile` exactly as `pyrowave-encode` does and reports
geometry, format detection and frame count. Written to settle whether a failing
encode was caused by bad input; it was not.

## Sample data

None of the media is in git. To regenerate the inputs:

    ffmpeg -f lavfi -i testsrc2=size=1920x1080:rate=60 -frames:v 8 -pix_fmt yuv420p samples/sample_1080p_420.y4m
    ffmpeg -f lavfi -i testsrc2=size=1920x1080:rate=60 -frames:v 8 -pix_fmt yuv444p samples/sample_1080p_444.y4m
    ffmpeg -f lavfi -i testsrc2=size=640x480:rate=60   -frames:v 4 -pix_fmt yuv420p samples/sample_480p_420.y4m

Bitstreams and reference decodes need the Vulkan build, which does not compile on
macOS, so they were produced on Linux:

    pyrowave-encode samples/sample_1080p_420.y4m stream_1080p_high.bin 200000
    pyrowave-decode stream_1080p_high.bin stream_1080p_high.bin.y4m
    PYROWAVE_PRECISION=2 pyrowave-decode stream_1080p_high.bin stream_1080p_high.bin-precision2.y4m

`Configuration` reads `PYROWAVE_PRECISION` at runtime, so the second reference
needs no rebuild. The default build is `PRECISION=1`, not FP32 — comparing the
Metal FP32 output against a default reference shows a spurious ~1 LSB
disagreement that is entirely the precision difference.
