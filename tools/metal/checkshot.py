# Copyright (c) 2026 Hans-Kristian Arntzen
# SPDX-License-Identifier: MIT

# Checks a viewer screenshot against a reference decode of the same stream.
#
#   ./build/pyrowave-metal-viewer stream.bin --screenshot shot.bmp
#   ./build/pyrowave-metal-decode stream.bin --output ref.y4m
#   python3 tools/metal/checkshot.py shot.bmp ref.y4m
#
# Converts the reference YCbCr to RGB with BT.709 full range and reports PSNR against
# the screenshot, then repeats it with Cb and Cr deliberately swapped. The second
# number is the point: a chroma mix-up in the display path is easy to miss by eye and
# survives a loose PSNR threshold, but it cannot survive being 25+ dB worse than the
# correct mapping. Expect roughly 32-34 dB correct against 4 dB swapped for 4:2:0, and
# 53 dB against 4 dB for 4:4:4 -- 4:2:0 is limited by this script upsampling chroma
# with nearest neighbour where SDL uses bilinear, not by the decoder.

import struct, sys, math

bmp_path, y4m_path = sys.argv[1], sys.argv[2]

# --- BMP (SDL_SaveBMP: 24bpp BGR, bottom-up) ---
d = open(bmp_path, 'rb').read()
assert d[:2] == b'BM'
pix_off = struct.unpack('<I', d[10:14])[0]
w, h = struct.unpack('<ii', d[18:26])
bpp = struct.unpack('<H', d[28:30])[0]
assert bpp in (24, 32), f"expected 24 or 32 bpp, got {bpp}"
nch = bpp // 8
flip = h > 0
h = abs(h)
row_stride = ((w * nch + 3) // 4) * 4
rgb = bytearray(w * h * 3)
for y in range(h):
    src_y = (h - 1 - y) if flip else y
    row = d[pix_off + src_y * row_stride: pix_off + src_y * row_stride + w * nch]
    for x in range(w):
        b, g, r = row[nch * x], row[nch * x + 1], row[nch * x + 2]
        o = (y * w + x) * 3
        rgb[o], rgb[o + 1], rgb[o + 2] = r, g, b

# --- Y4M frame 0 ---
f = open(y4m_path, 'rb')
hdr = b''
while not hdr.endswith(b'\n'):
    hdr += f.read(1)
assert b'FRAME' not in hdr
line = b''
while not line.endswith(b'\n'):
    line += f.read(1)
assert line.startswith(b'FRAME')
is444 = b'C444' in hdr
Y = f.read(w * h)
cw, ch = (w, h) if is444 else (w // 2, h // 2)
sub = 1 if is444 else 2
Cb = f.read(cw * ch)
Cr = f.read(cw * ch)

def convert(swap_uv):
    # BT.709, full range, nearest chroma upsample.
    err = 0
    n = 0
    for y in range(0, h, 3):          # subsample rows for speed
        for x in range(0, w, 3):
            yy = Y[y * w + x]
            ci = (y // sub) * cw + (x // sub)
            u, v = Cb[ci], Cr[ci]
            if swap_uv:
                u, v = v, u
            cb, cr = u - 128.0, v - 128.0
            r = yy + 1.5748 * cr
            g = yy - 0.1873 * cb - 0.4681 * cr
            b = yy + 1.8556 * cb
            o = (y * w + x) * 3
            for got, want in ((rgb[o], r), (rgb[o + 1], g), (rgb[o + 2], b)):
                want = min(255.0, max(0.0, want))
                err += (got - want) ** 2
                n += 1
    mse = err / n
    return 10 * math.log10(255.0 * 255.0 / mse) if mse else 999.0

ok = convert(False)
swapped = convert(True)
print(f"  correct U/V : {ok:6.2f} dB")
print(f"  swapped U/V : {swapped:6.2f} dB")
print(f"  -> {'PASS' if ok > swapped + 6 else 'FAIL'}: correct mapping is "
      f"{ok - swapped:.1f} dB better than the swap")
