// Emits a synthetic 2-frame stream: every block coded with an empty ballot.
#include "pyrowave_bitstream.hpp"
#include <stdio.h>
#include <vector>
using namespace PyroWave;
int main(int argc, char **argv) {
    const int W = 640, H = 480;
    BlockLayout layout;
    layout.init(W, H, ChromaSubsampling::Chroma420);
    std::vector<uint8_t> s;
    auto app = [&](const void *d, size_t n) {
        auto *p = (const uint8_t *)d; s.insert(s.end(), p, p + n);
    };
    for (uint32_t frame = 0; frame < 2; frame++) {
        BitstreamSequenceHeader seq = {};
        seq.width_minus_1 = W - 1; seq.height_minus_1 = H - 1;
        seq.sequence = frame; seq.extended = 1;
        seq.total_blocks = uint32_t(layout.block_count_32x32);
        seq.code = BITSTREAM_EXTENDED_CODE_START_OF_FRAME;
        app(&seq, sizeof(seq));
        for (int i = 0; i < layout.block_count_32x32; i++) {
            BitstreamHeader h = {};
            h.ballot = 0; h.payload_words = 2; h.sequence = frame;
            h.quant_code = 0x20; h.block_index = uint32_t(i);
            app(&h, sizeof(h));
        }
    }
    FILE *f = fopen(argv[1], "wb");
    fwrite(s.data(), 1, s.size(), f); fclose(f);
    printf("wrote %zu bytes, %d blocks/frame\n", s.size(), layout.block_count_32x32);
}
