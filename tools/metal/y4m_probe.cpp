// Reads a y4m with the repo's own YUV4MPEGFile, exactly as pyrowave-encode does.
#include "yuv4mpeg.hpp"
#include <stdio.h>
#include <vector>
int main(int argc, char **argv) {
    YUV4MPEGFile f;
    if (!f.open_read(argv[1])) { printf("open_read FAILED\n"); return 1; }
    printf("open_read ok: %dx%d fmt=%d bpc=%d subsampled=%d fps=%d/%d full_range=%d\n",
           f.get_width(), f.get_height(), int(f.get_format()),
           YUV4MPEGFile::format_to_bytes_per_component(f.get_format()),
           int(YUV4MPEGFile::format_has_subsampling(f.get_format())),
           f.get_frame_rate_num(), f.get_frame_rate_den(), int(f.is_full_range()));
    printf("params: [%s]\n", f.get_params().c_str());
    int w = f.get_width(), h = f.get_height();
    bool sub = YUV4MPEGFile::format_has_subsampling(f.get_format());
    int bpc = YUV4MPEGFile::format_to_bytes_per_component(f.get_format());
    size_t luma = size_t(w) * h * bpc;
    size_t chroma = sub ? luma / 4 : luma;
    std::vector<unsigned char> buf(luma);
    int frames = 0;
    while (f.begin_frame()) {
        if (!f.read(buf.data(), luma)) { printf("  frame %d: luma read FAILED\n", frames); break; }
        buf.resize(chroma > luma ? chroma : luma);
        if (!f.read(buf.data(), chroma)) { printf("  frame %d: Cb read FAILED\n", frames); break; }
        if (!f.read(buf.data(), chroma)) { printf("  frame %d: Cr read FAILED\n", frames); break; }
        frames++;
    }
    printf("frames read: %d\n", frames);
    return frames ? 0 : 1;
}
