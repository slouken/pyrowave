// Differential + behavioural test for pyrowave_bitstream.{hpp,cpp}.
// Reference side is a verbatim transcription of WaveletBuffers::init_block_meta /
// accumulate_block_mapping from pyrowave_common.cpp, driven by a stand-in for the
// Granite image that reproduces Vulkan::Image::get_width/get_height(lod) exactly.
#include "pyrowave_bitstream.hpp"
#include <algorithm>
#include <stdio.h>
#include <string.h>
#include <vector>

using namespace PyroWave;

static int g_failures;

#define CHECK(cond, ...) do { \
	if (!(cond)) { \
		printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); \
		g_failures++; \
	} \
} while (0)

// ---------------------------------------------------------------- reference

struct FakeImage
{
	uint32_t width, height;
	// Matches Vulkan::Image::get_width/get_height in Granite/vulkan/image.hpp.
	uint32_t get_width(uint32_t lod = 0) const { return std::max<uint32_t>(1u, width >> lod); }
	uint32_t get_height(uint32_t lod = 0) const { return std::max<uint32_t>(1u, height >> lod); }
};

struct RefLayout
{
	struct BlockInfo { int block_offset_8x8, block_stride_8x8, block_offset_32x32, block_stride_32x32; };
	struct BlockMapping { int block_offset_8x8, block_stride_8x8, block_width_8x8, block_height_8x8; };

	BlockInfo block_meta[NumComponents][DecompositionLevels][4] = {};
	std::vector<BlockMapping> block_32x32_to_8x8_mapping;
	int block_count_8x8 = 0, block_count_32x32 = 0;
	int aligned_width = 0, aligned_height = 0;
	ChromaSubsampling chroma{};
	FakeImage wavelet_img_high_res{};

	void accumulate_block_mapping(int blocks_x_8x8, int blocks_y_8x8)
	{
		int blocks_x_32x32 = (blocks_x_8x8 + 3) / 4;
		int blocks_y_32x32 = (blocks_y_8x8 + 3) / 4;

		for (int y = 0; y < blocks_y_32x32; y++)
		{
			for (int x = 0; x < blocks_x_32x32; x++)
			{
				BlockMapping mapping = {};
				mapping.block_offset_8x8 = block_count_8x8 + 4 * y * blocks_x_8x8 + 4 * x;
				mapping.block_stride_8x8 = blocks_x_8x8;
				mapping.block_width_8x8 = std::min<int>(4, blocks_x_8x8 - 4 * x);
				mapping.block_height_8x8 = std::min<int>(4, blocks_y_8x8 - 4 * y);
				block_32x32_to_8x8_mapping.push_back(mapping);
				block_count_32x32++;
			}
		}

		block_count_8x8 += blocks_x_8x8 * blocks_y_8x8;
	}

	void init(int width, int height, ChromaSubsampling chroma_)
	{
		chroma = chroma_;
		aligned_width = align(width, Alignment);
		aligned_height = align(height, Alignment);
		aligned_width = std::max<int>(aligned_width, MinimumImageSize);
		aligned_height = std::max<int>(aligned_height, MinimumImageSize);

		// allocate_images() creates the wavelet image at half the aligned size.
		wavelet_img_high_res.width = aligned_width / 2;
		wavelet_img_high_res.height = aligned_height / 2;

		for (int level = DecompositionLevels - 1; level >= 0; level--)
		{
			for (int component = 0; component < NumComponents; component++)
			{
				if (level == 0 && component != 0 && chroma == ChromaSubsampling::Chroma420)
					continue;

				for (int band = (level == DecompositionLevels - 1 ? 0 : 1); band < 4; band++)
				{
					uint32_t level_width = wavelet_img_high_res.get_width(level);
					uint32_t level_height = wavelet_img_high_res.get_height(level);

					int blocks_x_8x8 = (level_width + 7) / 8;
					int blocks_y_8x8 = (level_height + 7) / 8;
					int blocks_x_32x32 = (level_width + 31) / 32;

					block_meta[component][level][band] = {
						block_count_8x8, blocks_x_8x8,
						block_count_32x32, blocks_x_32x32,
					};

					accumulate_block_mapping(blocks_x_8x8, blocks_y_8x8);
				}
			}
		}
	}
};

// ---------------------------------------------------------------- layout test

static void test_layout(int w, int h, ChromaSubsampling chroma)
{
	RefLayout ref;
	ref.init(w, h, chroma);

	BlockLayout got;
	CHECK(got.init(w, h, chroma), "init failed for %dx%d", w, h);

	const char *cs = chroma == ChromaSubsampling::Chroma420 ? "420" : "444";

	CHECK(got.aligned_width == ref.aligned_width && got.aligned_height == ref.aligned_height,
	      "%dx%d %s aligned mismatch: (%d,%d) vs (%d,%d)", w, h, cs,
	      got.aligned_width, got.aligned_height, ref.aligned_width, ref.aligned_height);
	CHECK(got.block_count_8x8 == ref.block_count_8x8,
	      "%dx%d %s block_count_8x8 %d vs %d", w, h, cs, got.block_count_8x8, ref.block_count_8x8);
	CHECK(got.block_count_32x32 == ref.block_count_32x32,
	      "%dx%d %s block_count_32x32 %d vs %d", w, h, cs, got.block_count_32x32, ref.block_count_32x32);

	for (int level = 0; level < DecompositionLevels; level++)
	{
		CHECK(got.level_width(level) == int(ref.wavelet_img_high_res.get_width(level)),
		      "%dx%d %s level_width(%d) %d vs %u", w, h, cs, level,
		      got.level_width(level), ref.wavelet_img_high_res.get_width(level));
		CHECK(got.level_height(level) == int(ref.wavelet_img_high_res.get_height(level)),
		      "%dx%d %s level_height(%d) %d vs %u", w, h, cs, level,
		      got.level_height(level), ref.wavelet_img_high_res.get_height(level));

		for (int comp = 0; comp < NumComponents; comp++)
		{
			for (int band = 0; band < 4; band++)
			{
				auto &a = got.block_meta[comp][level][band];
				auto &b = ref.block_meta[comp][level][band];
				CHECK(a.block_offset_8x8 == b.block_offset_8x8 && a.block_stride_8x8 == b.block_stride_8x8 &&
				      a.block_offset_32x32 == b.block_offset_32x32 && a.block_stride_32x32 == b.block_stride_32x32,
				      "%dx%d %s block_meta[%d][%d][%d] (%d,%d,%d,%d) vs (%d,%d,%d,%d)", w, h, cs, comp, level, band,
				      a.block_offset_8x8, a.block_stride_8x8, a.block_offset_32x32, a.block_stride_32x32,
				      b.block_offset_8x8, b.block_stride_8x8, b.block_offset_32x32, b.block_stride_32x32);
			}
		}
	}

	CHECK(got.block_32x32_to_8x8_mapping.size() == ref.block_32x32_to_8x8_mapping.size(),
	      "%dx%d %s mapping size %zu vs %zu", w, h, cs,
	      got.block_32x32_to_8x8_mapping.size(), ref.block_32x32_to_8x8_mapping.size());

	size_t n = std::min(got.block_32x32_to_8x8_mapping.size(), ref.block_32x32_to_8x8_mapping.size());
	for (size_t i = 0; i < n; i++)
	{
		auto &a = got.block_32x32_to_8x8_mapping[i];
		auto &b = ref.block_32x32_to_8x8_mapping[i];
		CHECK(a.block_offset_8x8 == b.block_offset_8x8 && a.block_stride_8x8 == b.block_stride_8x8 &&
		      a.block_width_8x8 == b.block_width_8x8 && a.block_height_8x8 == b.block_height_8x8,
		      "%dx%d %s mapping[%zu] mismatch", w, h, cs, i);
	}

	// Invariants that must hold regardless of the reference.
	CHECK(int(got.block_32x32_to_8x8_mapping.size()) == got.block_count_32x32,
	      "%dx%d %s mapping size != block_count_32x32", w, h, cs);

	long covered = 0;
	for (auto &m : got.block_32x32_to_8x8_mapping)
	{
		CHECK(m.block_width_8x8 >= 1 && m.block_width_8x8 <= 4 &&
		      m.block_height_8x8 >= 1 && m.block_height_8x8 <= 4,
		      "%dx%d %s degenerate mapping extent %dx%d", w, h, cs, m.block_width_8x8, m.block_height_8x8);
		covered += long(m.block_width_8x8) * m.block_height_8x8;
	}
	CHECK(covered == got.block_count_8x8,
	      "%dx%d %s 32x32 mapping covers %ld 8x8 blocks, expected %d", w, h, cs, covered, got.block_count_8x8);
}

// ---------------------------------------------------------------- parser test

static BitstreamHeader make_packet_header(uint32_t block_index, uint32_t payload_words, uint32_t sequence)
{
	BitstreamHeader h = {};
	h.ballot = 0xabcd;
	h.payload_words = payload_words;
	h.sequence = sequence;
	h.extended = 0;
	h.quant_code = 0x5a;
	h.block_index = block_index;
	return h;
}

static BitstreamSequenceHeader make_seq_header(int w, int h, uint32_t sequence, uint32_t total_blocks,
                                               ChromaSubsampling chroma)
{
	BitstreamSequenceHeader s = {};
	s.width_minus_1 = w - 1;
	s.height_minus_1 = h - 1;
	s.sequence = sequence;
	s.extended = 1;
	s.total_blocks = total_blocks;
	s.code = BITSTREAM_EXTENDED_CODE_START_OF_FRAME;
	s.chroma_resolution = uint32_t(chroma);
	return s;
}

static void append(std::vector<uint8_t> &buf, const void *data, size_t size)
{
	auto *p = static_cast<const uint8_t *>(data);
	buf.insert(buf.end(), p, p + size);
}

// Appends a packet whose payload words after the header are filled with `fill`.
static void append_packet(std::vector<uint8_t> &buf, uint32_t block_index, uint32_t payload_words,
                          uint32_t sequence, uint32_t fill)
{
	auto h = make_packet_header(block_index, payload_words, sequence);
	append(buf, &h, sizeof(h));
	for (uint32_t i = 2; i < payload_words; i++)
		append(buf, &fill, sizeof(fill));
}

static void test_parser()
{
	const int W = 1920, H = 1080;
	BlockLayout layout;
	CHECK(layout.init(W, H, ChromaSubsampling::Chroma420), "layout init failed");

	BitstreamParser parser;
	parser.init(&layout);

	// Fresh state: everything unset, nothing ready.
	CHECK(int(parser.dequant_offsets().size()) == layout.block_count_32x32, "offset table wrong size");
	for (auto v : parser.dequant_offsets())
		CHECK(v == UINT32_MAX, "offset table not cleared");
	CHECK(parser.payload().empty(), "payload not empty");
	CHECK(!parser.decode_is_ready(false), "ready with no sequence");
	CHECK(!parser.decode_is_ready(true), "ready (partial) with no sequence");

	// One sequence header + three distinct blocks.
	std::vector<uint8_t> buf;
	auto seq = make_seq_header(W, H, 0, 3, ChromaSubsampling::Chroma420);
	append(buf, &seq, sizeof(seq));
	append_packet(buf, 0, 4, 0, 0x11111111u);
	append_packet(buf, 5, 2, 0, 0);
	append_packet(buf, 9, 6, 0, 0x33333333u);

	CHECK(parser.push_packet(buf.data(), buf.size()), "push_packet failed");
	CHECK(parser.get_decoded_blocks() == 3, "decoded_blocks %d != 3", parser.get_decoded_blocks());
	CHECK(parser.get_total_blocks_in_sequence() == 3, "total_blocks %d != 3", parser.get_total_blocks_in_sequence());

	// Offsets point at consecutive payload regions in arrival order.
	auto &off = parser.dequant_offsets();
	CHECK(off[0] == 0, "block 0 offset %u != 0", off[0]);
	CHECK(off[5] == 4, "block 5 offset %u != 4", off[5]);
	CHECK(off[9] == 6, "block 9 offset %u != 6", off[9]);
	CHECK(parser.payload().size() == 12, "payload size %zu != 12", parser.payload().size());

	// Payload must start with the packet header itself, then the fill words.
	auto &pay = parser.payload();
	CHECK(pay[2] == 0x11111111u && pay[3] == 0x11111111u, "block 0 payload wrong");
	CHECK(pay[8] == 0x33333333u && pay[11] == 0x33333333u, "block 9 payload wrong");
	{
		auto h0 = make_packet_header(0, 4, 0);
		uint32_t expect[2];
		memcpy(expect, &h0, sizeof(h0));
		CHECK(pay[0] == expect[0] && pay[1] == expect[1], "block 0 header not copied into payload");
	}

	CHECK(parser.decode_is_ready(false), "should be ready: all blocks in");
	parser.mark_frame_decoded();
	CHECK(!parser.decode_is_ready(false), "should not re-decode same sequence");

	// A duplicate block index is ignored rather than double counted.
	{
		BlockLayout l2;
		l2.init(W, H, ChromaSubsampling::Chroma420);
		BitstreamParser p2;
		p2.init(&l2);
		std::vector<uint8_t> b2;
		auto s2 = make_seq_header(W, H, 0, 2, ChromaSubsampling::Chroma420);
		append(b2, &s2, sizeof(s2));
		append_packet(b2, 7, 4, 0, 0xaaaaaaaau);
		append_packet(b2, 7, 4, 0, 0xbbbbbbbbu);
		CHECK(p2.push_packet(b2.data(), b2.size()), "dup push failed");
		CHECK(p2.get_decoded_blocks() == 1, "duplicate block counted twice (%d)", p2.get_decoded_blocks());
		CHECK(p2.payload().size() == 4, "duplicate block appended payload (%zu)", p2.payload().size());
	}

	// Out-of-range block index is rejected.
	{
		BlockLayout l3;
		l3.init(W, H, ChromaSubsampling::Chroma420);
		BitstreamParser p3;
		p3.init(&l3);
		std::vector<uint8_t> b3;
		auto s3 = make_seq_header(W, H, 0, 1, ChromaSubsampling::Chroma420);
		append(b3, &s3, sizeof(s3));
		append_packet(b3, uint32_t(l3.block_count_32x32), 4, 0, 0);
		printf("  (expect one bounds error below)\n");
		CHECK(!p3.push_packet(b3.data(), b3.size()), "out-of-bounds block_index accepted");
	}

	// payload_words < 2 cannot even hold the header.
	{
		BlockLayout l4;
		l4.init(W, H, ChromaSubsampling::Chroma420);
		BitstreamParser p4;
		p4.init(&l4);
		std::vector<uint8_t> b4;
		auto s4 = make_seq_header(W, H, 0, 1, ChromaSubsampling::Chroma420);
		append(b4, &s4, sizeof(s4));
		append_packet(b4, 3, 1, 0, 0);
		printf("  (expect one payload_words error below)\n");
		CHECK(!p4.push_packet(b4.data(), b4.size()), "payload_words=1 accepted");
	}

	// Partial frame gating: more than half decoded is required.
	{
		BlockLayout l5;
		l5.init(W, H, ChromaSubsampling::Chroma420);
		BitstreamParser p5;
		p5.init(&l5);
		std::vector<uint8_t> b5;
		auto s5 = make_seq_header(W, H, 0, 10, ChromaSubsampling::Chroma420);
		append(b5, &s5, sizeof(s5));
		for (uint32_t i = 0; i < 5; i++)
			append_packet(b5, i, 2, 0, 0);
		CHECK(p5.push_packet(b5.data(), b5.size()), "partial push failed");
		CHECK(!p5.decode_is_ready(false), "partial accepted without allow_partial");
		CHECK(!p5.decode_is_ready(true), "exactly half accepted as partial");

		std::vector<uint8_t> b5b;
		append_packet(b5b, 5, 2, 0, 0);
		CHECK(p5.push_packet(b5b.data(), b5b.size()), "partial push 2 failed");
		CHECK(p5.decode_is_ready(true), "6/10 not accepted as partial");
		CHECK(!p5.decode_is_ready(false), "6/10 accepted as complete");
	}

	// A new sequence number resets accumulated state.
	{
		BlockLayout l6;
		l6.init(W, H, ChromaSubsampling::Chroma420);
		BitstreamParser p6;
		p6.init(&l6);
		std::vector<uint8_t> b6;
		auto s6 = make_seq_header(W, H, 0, 2, ChromaSubsampling::Chroma420);
		append(b6, &s6, sizeof(s6));
		append_packet(b6, 1, 4, 0, 0);
		append_packet(b6, 2, 4, 0, 0);
		CHECK(p6.push_packet(b6.data(), b6.size()), "seq0 push failed");
		CHECK(p6.get_decoded_blocks() == 2, "seq0 blocks");

		std::vector<uint8_t> b7;
		auto s7 = make_seq_header(W, H, 1, 2, ChromaSubsampling::Chroma420);
		append(b7, &s7, sizeof(s7));
		append_packet(b7, 4, 4, 1, 0);
		CHECK(p6.push_packet(b7.data(), b7.size()), "seq1 push failed");
		CHECK(p6.get_decoded_blocks() == 1, "new sequence did not reset (%d)", p6.get_decoded_blocks());
		CHECK(p6.dequant_offsets()[1] == UINT32_MAX, "old block still present after reset");
		CHECK(p6.dequant_offsets()[4] == 0, "new block not at payload start");
	}

	// Stale (older) sequence is dropped without disturbing current state.
	{
		BlockLayout l8;
		l8.init(W, H, ChromaSubsampling::Chroma420);
		BitstreamParser p8;
		p8.init(&l8);
		std::vector<uint8_t> b8;
		auto s8 = make_seq_header(W, H, 4, 2, ChromaSubsampling::Chroma420);
		append(b8, &s8, sizeof(s8));
		append_packet(b8, 1, 4, 4, 0);
		CHECK(p8.push_packet(b8.data(), b8.size()), "seq4 push failed");
		CHECK(p8.get_decoded_blocks() == 1, "seq4 blocks");

		// sequence 0 is 4 behind 4 -> diff = 4 > 3, treated as stale.
		std::vector<uint8_t> b9;
		append_packet(b9, 2, 4, 0, 0);
		CHECK(p8.push_packet(b9.data(), b9.size()), "stale push should return true");
		CHECK(p8.get_decoded_blocks() == 1, "stale packet mutated state (%d)", p8.get_decoded_blocks());
	}

	// Chroma mismatch in the sequence header is rejected.
	{
		BlockLayout l10;
		l10.init(W, H, ChromaSubsampling::Chroma420);
		BitstreamParser p10;
		p10.init(&l10);
		std::vector<uint8_t> b10;
		auto s10 = make_seq_header(W, H, 0, 1, ChromaSubsampling::Chroma444);
		append(b10, &s10, sizeof(s10));
		printf("  (expect one chroma error below)\n");
		CHECK(!p10.push_packet(b10.data(), b10.size()), "chroma mismatch accepted");
	}

	// Dimension mismatch is rejected.
	{
		BlockLayout l11;
		l11.init(W, H, ChromaSubsampling::Chroma420);
		BitstreamParser p11;
		p11.init(&l11);
		std::vector<uint8_t> b11;
		auto s11 = make_seq_header(1280, 720, 0, 1, ChromaSubsampling::Chroma420);
		append(b11, &s11, sizeof(s11));
		printf("  (expect one dimension error below)\n");
		CHECK(!p11.push_packet(b11.data(), b11.size()), "dimension mismatch accepted");
	}

	// Truncated packet is rejected.
	{
		BlockLayout l12;
		l12.init(W, H, ChromaSubsampling::Chroma420);
		BitstreamParser p12;
		p12.init(&l12);
		std::vector<uint8_t> b12;
		auto s12 = make_seq_header(W, H, 0, 1, ChromaSubsampling::Chroma420);
		append(b12, &s12, sizeof(s12));
		append_packet(b12, 1, 8, 0, 0);
		b12.resize(b12.size() - 8); // claim 8 words, deliver 6
		printf("  (expect one truncation error below)\n");
		CHECK(!p12.push_packet(b12.data(), b12.size()), "truncated packet accepted");
	}

	// Trailing bytes that cannot form a header are rejected.
	{
		BlockLayout l13;
		l13.init(W, H, ChromaSubsampling::Chroma420);
		BitstreamParser p13;
		p13.init(&l13);
		std::vector<uint8_t> b13;
		auto s13 = make_seq_header(W, H, 0, 1, ChromaSubsampling::Chroma420);
		append(b13, &s13, sizeof(s13));
		uint32_t junk = 0;
		append(b13, &junk, sizeof(junk));
		printf("  (expect one consume error below)\n");
		CHECK(!p13.push_packet(b13.data(), b13.size()), "trailing junk accepted");
	}

	// Empty input is a no-op success.
	{
		BlockLayout l14;
		l14.init(W, H, ChromaSubsampling::Chroma420);
		BitstreamParser p14;
		p14.init(&l14);
		CHECK(p14.push_packet(nullptr, 0), "empty push rejected");
		CHECK(p14.get_decoded_blocks() == 0, "empty push decoded something");
	}
}

int main()
{
	static const int dims[][2] = {
		{ 1920, 1080 }, { 1280, 720 }, { 3840, 2160 }, { 640, 480 },
		{ 128, 128 }, { 129, 129 }, { 1, 1 }, { 33, 17 }, { 160, 90 },
		{ 4096, 4096 }, { 16384, 16384 }, { 1023, 767 }, { 2560, 1440 },
		{ 7680, 4320 }, { 31, 31 }, { 8192, 128 }, { 128, 8192 },
	};

	for (auto &d : dims)
	{
		test_layout(d[0], d[1], ChromaSubsampling::Chroma420);
		test_layout(d[0], d[1], ChromaSubsampling::Chroma444);
	}

	// Sweep a contiguous range to catch off-by-one behaviour around alignment.
	for (int w = 1; w <= 600; w++)
	{
		test_layout(w, 256, ChromaSubsampling::Chroma420);
		test_layout(256, w, ChromaSubsampling::Chroma444);
	}

	// Out-of-range dimensions must be refused.
	{
		BlockLayout l;
		printf("  (expect range errors below)\n");
		CHECK(!l.init(0, 100, ChromaSubsampling::Chroma420), "width 0 accepted");
		CHECK(!l.init(100, 0, ChromaSubsampling::Chroma420), "height 0 accepted");
		CHECK(!l.init(16385, 100, ChromaSubsampling::Chroma420), "width 16385 accepted");
		CHECK(!l.init(-4, 100, ChromaSubsampling::Chroma420), "negative width accepted");
	}

	test_parser();

	if (g_failures)
		printf("\n%d FAILURE(S)\n", g_failures);
	else
		printf("\nAll tests passed.\n");
	return g_failures ? 1 : 0;
}
