# Verifying correctness after a change

Every check here has a stated expected result. Where a number is given it was measured
on an Apple M1; treat a deviation as something to explain, not to re-baseline.

## Always

```
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja -C build pyrowave-metal pyrowave-metal-test pyrowave-metal-encode \
                pyrowave-metal-decode pyrowave-bitstream-test
./build/pyrowave-bitstream-test      # "All tests passed."           (< 1 s)
./build/pyrowave-metal-test          # "All Metal tests passed."     (~1.7 s)
```

The build must be warning free. Both binaries exit non-zero on failure and print
`FAIL <file>:<line>` for each problem, so they are usable from a script.

## What to run for what you touched

| Changed | Run |
|---|---|
| `pyrowave_bitstream.*` | bitstream test — it differentially checks `BlockLayout` against a verbatim copy of the original `init_block_meta` over ~1200 resolutions and both chroma modes |
| Anything in the Metal decoder | metal test, then **reference streams** below |
| Anything in the Metal encoder | metal test, then **encoder against Vulkan** below |
| A shader, or `transpile.sh` | both of the above; a shader change can be silently wrong in ways the smoke test's invariants do not reach |
| The SDL viewer or the SDL prototype | **display path** below |
| The Vulkan encoder or decoder | **cross backend** below |
| Precision handling | every check above, at each of `PYROWAVE_PRECISION=0,1,2` |

## Reference streams (decoder)

Needs `samples/`, which is gitignored; see `tools/metal/README.md` for how it is
produced. Compare against the reference decoded **at the same precision**:

```
# Metal default (precision 1) against the Vulkan default reference
./build/pyrowave-metal-decode samples/stream_480p.bin --reference samples/stream_480p.bin.y4m

# Both at precision 2
PYROWAVE_PRECISION=2 ./build/pyrowave-metal-decode samples/stream_480p.bin \
    --reference samples/stream_480p.bin-precision2.y4m
```

Total differing pixels over all frames and planes, which should be stable to the pixel:

| stream | default (1) | precision 2 |
|---|---|---|
| `stream_480p` | 724 | 22 |
| `stream_1080p_high` | 623 | 57 |
| `stream_1080p_low` | 22776 | 481 |
| `stream_1080p_444` | 6298 | 183 |

**`max |diff|` must be 1 everywhere.** That is the real assertion. The counts move with
any change to rounding, but a single pixel off by more than one means a structural
error — a band mapped to the wrong layer, a bad block offset, a mis-sized dispatch —
and no amount of "it is only a few pixels" excuses it.

## Encoder against Vulkan

Encode the same source at the same rate as a committed Vulkan stream and compare both
against the *original*, not against each other:

```
./build/pyrowave-metal-encode samples/sample_480p_420.y4m --output /tmp/m.bin --size 30000
./build/pyrowave-metal-decode /tmp/m.bin              --reference samples/sample_480p_420.y4m
./build/pyrowave-metal-decode samples/stream_480p.bin --reference samples/sample_480p_420.y4m
```

Expect PSNR within 0.02 dB of each other; measured 37.25 / 41.02 / 41.58 dB against
37.25 / 41.02 / 41.59 dB. For a deeper check, parse the two bitstreams and confirm they
code the *identical set* of 32x32 blocks with the *identical size* for each — that
pins quantization, RDO and rate control rather than just the average outcome.

## Display path

Needs the SDL prototype in the `sdl3` submodule (per plane MTLTexture properties, the
shader write opt-in, and the renderer device and queue properties).

```
ninja -C build pyrowave-metal-viewer
./build/pyrowave-metal-viewer samples/stream_1080p_high.bin --screenshot /tmp/s.bmp
./build/pyrowave-metal-decode samples/stream_1080p_high.bin --output /tmp/s.y4m
python3 tools/metal/checkshot.py /tmp/s.bmp /tmp/s.y4m
```

Expect ~34 dB for 4:2:0 and ~53 dB for 4:4:4, against ~4 dB with Cb and Cr swapped.
The gap is the assertion; the absolute 4:2:0 figure is limited by the script's nearest
neighbour chroma upsampling, not by the decoder.

## Cross backend

Needs KosmicKrisp and two local Granite fixes; recipe in `tools/metal/README.md`. Both
decoders should read each other's streams, and decoding one stream with both at
`PYROWAVE_PRECISION=2` should differ by tens of pixels, all off by one — about 30 over
four frames at 480p.

## Under validation layers

Run these before believing a change that touches resources or synchronization. Both
are clean and both should stay clean:

```
METAL_DEVICE_WRAPPER_TYPE=1 ./build/pyrowave-metal-test
MTL_DEBUG_LAYER=1           ./build/pyrowave-metal-test
```

**Do not use `MTL_SHADER_VALIDATION=1` as a correctness gate.** With GPU shader
validation enabled the decoder produces garbage — the metal test reports about 90
failures and a real stream decodes at ~12 dB instead of ~88 dB — and the evidence says
this is an artifact of the instrumentation rather than a defect it has found:

- The failures are all encode round trips, but the encoder is not at fault. Encoding
  under validation and decoding without gives the normal 87.95 dB; encoding without and
  decoding under validation gives 11.99 dB.
- The decoder's own smoke tests still pass, because the empty bitstream and empty
  ballot cases never read coefficient payload and their subgroup scans are all zero.
- **Every** one of 300 32x32 blocks is corrupt, minimum mean error 6.3. A wrong block
  offset, a missing barrier or an out of bounds read would leave some blocks clean.
- Validation prints no diagnostic at all, which it would for an access violation.
- It is deterministic, content independent, identical at all three precisions, and
  unchanged by switching the compute encoders from concurrent to serial.
- Isolated minimal repros of each mechanism the coefficient path relies on are clean
  under validation: the `device void*` buffer aliasing SPIRV-Cross emits, including
  reads spread across a 1 MB buffer; `simd_ballot`, `simd_prefix_inclusive_sum`,
  `simd_prefix_exclusive_sum` and `simd_shuffle`; and
  `simdgroups_per_threadgroup`/`simdgroup_index_in_threadgroup` with a 128 thread
  group. Pipeline limits stay adequate too, dequant reporting 832 then 704 threads
  against the 128 it needs.
- The decoder agrees with an independent implementation to within 1 LSB on every plane
  of four reference streams, and separately with the Vulkan decoder running on the same
  GPU. A latent decoder defect that severe would not survive that.

This has not been proven positively, only narrowed. If you find the mechanism, replace
this section.

## Traps

These each produced a wrong conclusion at least once.

**Never compare bitstreams byte for byte.** Two runs of one binary in one
configuration already differ: the unused bits of a block's final sign byte come from
`shared_sign_bank` in `block_packing.comp`, which is never initialized and is updated
with `atomicAnd`/`atomicOr` that preserve bits outside the write mask. A `cmp` will
report a difference that means nothing, in either direction. **Compare decoded output**,
which is deterministic because the decoder reads only as many sign bits as there are
significant coefficients.

**"Pixels differing from a reference decode" is agreement, not accuracy.** It measures
how closely one rounding path tracks another. A change once moved that count from 623
to 85209 while absolute PSNR was identical to 0.01 dB. For quality, measure end to end
PSNR against the *original source*.

**Match precision on both sides.** `PYROWAVE_PRECISION` is runtime on both backends,
for both directions. Comparing a precision 1 decode against a precision 2 reference
shows thousands of differing pixels and means nothing.

**A test that cannot fail is worth nothing.** After adding or changing a check, break
the thing it covers and confirm it fails. Swapping Red and Green in the NV12 chroma
swizzle, for instance, makes 76794 of 76800 chroma samples differ and drops every NV12
case to 10.2 dB — that is what makes the passing result meaningful.

**Benchmarks are not correctness, and they lie more easily.** Numbers drift 10-15%
between runs from GPU clock state, and up to 35% if a heavy workload ran just before.
Only compare within a single interleaved run. Verify a graphics microbenchmark scales
with the work before trusting it: repeated opaque full screen draws are collapsed by
tile based hidden surface removal, so one such benchmark was measuring almost nothing
until blending was enabled.
