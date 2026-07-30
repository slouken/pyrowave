// End-to-end GPU smoke test for the Metal decoder.
//
// With no packets pushed, every 32x32 block is marked missing, so the dequant
// shader zero-fills the wavelet pyramid and the iDWT chain reduces to a constant.
// Every write into an output plane happens with the DCShift function constant
// enabled, which adds 0.5, so a correct decode must leave every output pixel at
// exactly 0.5. That single expectation covers dequant, all five iDWT levels, the
// texture views, hazard tracking and the final stores.
#include <Metal/Metal.hpp>

#include <IOSurface/IOSurfaceRef.h>

#include "pyrowave_metal.h"
#include "pyrowave_bitstream.hpp"

#include <algorithm>
#include <cmath>
#include <stdio.h>
#include <string.h>
#include <vector>

static int g_failures;

#define CHECK(cond, ...) do { \
	if (!(cond)) { \
		printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); \
		g_failures++; \
	} \
} while (0)

static void message_cb(void *, const char *msg)
{
	printf("  [pyrowave] %s\n", msg);
}

static MTL::Texture *make_plane(MTL::Device *dev, int width, int height)
{
	auto *desc = MTL::TextureDescriptor::alloc()->init();
	desc->setTextureType(MTL::TextureType2D);
	desc->setPixelFormat(MTL::PixelFormatR32Float);
	desc->setWidth(width);
	desc->setHeight(height);
	desc->setUsage(MTL::TextureUsageShaderWrite | MTL::TextureUsageShaderRead);
	desc->setStorageMode(MTL::StorageModeShared);
	auto *tex = dev->newTexture(desc);
	desc->release();

	// Poison the contents so an unwritten pixel is detectable.
	std::vector<float> poison(size_t(width) * height, -1.0f);
	tex->replaceRegion(MTL::Region(0, 0, width, height), 0, poison.data(), width * sizeof(float));
	return tex;
}

static void check_plane(MTL::Texture *tex, const char *label, int index)
{
	const int width = int(tex->width());
	const int height = int(tex->height());
	std::vector<float> data(size_t(width) * height, -12345.0f);
	tex->getBytes(data.data(), width * sizeof(float), MTL::Region(0, 0, width, height), 0);

	size_t bad = 0;
	float first_bad = 0.0f;
	size_t first_bad_index = 0;
	for (size_t i = 0; i < data.size(); i++)
	{
		if (std::fabs(data[i] - 0.5f) > 1.0f / 512.0f)
		{
			if (!bad)
			{
				first_bad = data[i];
				first_bad_index = i;
			}
			bad++;
		}
	}

	CHECK(bad == 0, "%s plane %d: %zu/%zu pixels wrong, first at (%zu,%zu) = %f (expected 0.5)",
	      label, index, bad, data.size(), first_bad_index % width, first_bad_index / width, first_bad);
}

static void test_resolution(pyrowave_device device, MTL::Device *mtl, MTL::CommandQueue *queue,
                            int width, int height, pyrowave_chroma_subsampling chroma)
{
	const bool is_420 = chroma == PYROWAVE_CHROMA_SUBSAMPLING_420;
	const char *label = is_420 ? "420" : "444";
	printf("-- %dx%d %s\n", width, height, label);

	pyrowave_decoder_create_info info = {};
	info.device = device;
	info.width = width;
	info.height = height;
	info.chroma = chroma;

	pyrowave_decoder decoder = nullptr;
	auto res = pyrowave_decoder_create(&info, &decoder);
	CHECK(res == PYROWAVE_SUCCESS, "%dx%d %s: create failed: %s",
	      width, height, label, pyrowave_result_to_string(res));
	if (res != PYROWAVE_SUCCESS)
		return;

	const int chroma_width = is_420 ? width / 2 : width;
	const int chroma_height = is_420 ? height / 2 : height;

	MTL::Texture *planes[3] = {
		make_plane(mtl, width, height),
		make_plane(mtl, chroma_width, chroma_height),
		make_plane(mtl, chroma_width, chroma_height),
	};

	pyrowave_gpu_buffers buffers = {};
	for (int i = 0; i < 3; i++)
		buffers.planes[i] = planes[i];

	auto *cmd = queue->commandBuffer();
	res = pyrowave_decoder_decode(decoder, cmd, &buffers);
	CHECK(res == PYROWAVE_SUCCESS, "%dx%d %s: decode failed: %s",
	      width, height, label, pyrowave_result_to_string(res));

	cmd->commit();
	cmd->waitUntilCompleted();

	CHECK(cmd->status() == MTL::CommandBufferStatusCompleted,
	      "%dx%d %s: command buffer status %d", width, height, label, int(cmd->status()));
	if (cmd->error())
	{
		printf("FAIL %dx%d %s: GPU error: %s\n", width, height, label,
		       cmd->error()->localizedDescription()->utf8String());
		g_failures++;
	}

	char full_label[64];
	snprintf(full_label, sizeof(full_label), "%dx%d %s", width, height, label);
	for (int i = 0; i < 3; i++)
		check_plane(planes[i], full_label, i);

	for (auto *plane : planes)
		plane->release();
	pyrowave_decoder_destroy(decoder);
}

//////
// Encoder tests. The encoder has no equivalent of the decoder's "every pixel must
// be 0.5" invariant, so these round trip a synthetic frame through encode and
// decode and check the result resembles what went in. Content is deliberately
// smooth so a generous rate target is effectively lossless and the threshold can
// be strict: anything structurally wrong (a band mapped to the wrong layer, a
// mis-sized dispatch, a scrambled buffer binding) costs far more than a few dB.

struct TestFrame
{
	std::vector<uint8_t> planes[3];
	std::vector<uint8_t> chroma_interleaved;
	int width = 0, height = 0, chroma_width = 0, chroma_height = 0;

	void init(int width_, int height_, bool is_420)
	{
		width = width_;
		height = height_;
		chroma_width = is_420 ? width / 2 : width;
		chroma_height = is_420 ? height / 2 : height;

		for (int i = 0; i < 3; i++)
		{
			const int w = i == 0 ? width : chroma_width;
			const int h = i == 0 ? height : chroma_height;
			planes[i].resize(size_t(w) * h);
			for (int y = 0; y < h; y++)
			{
				for (int x = 0; x < w; x++)
				{
					const double fx = double(x) / double(w);
					const double fy = double(y) / double(h);
					const double v = 128.0 + 90.0 * std::sin(6.0 * fx + 2.0 * double(i)) * std::cos(4.0 * fy);
					planes[i][size_t(y) * w + x] = uint8_t(std::min(255.0, std::max(0.0, v)));
				}
			}
		}

		chroma_interleaved.resize(planes[1].size() * 2);
		for (size_t i = 0; i < planes[1].size(); i++)
		{
			chroma_interleaved[2 * i + 0] = planes[1][i];
			chroma_interleaved[2 * i + 1] = planes[2][i];
		}
	}
};

static double plane_psnr(MTL::Texture *tex, const std::vector<uint8_t> &reference)
{
	const int width = int(tex->width());
	const int height = int(tex->height());
	std::vector<float> data(size_t(width) * height);
	tex->getBytes(data.data(), width * sizeof(float), MTL::Region(0, 0, width, height), 0);

	double error = 0.0;
	for (size_t i = 0; i < data.size() && i < reference.size(); i++)
	{
		const double diff = double(data[i]) * 255.0 - double(reference[i]);
		error += diff * diff;
	}

	const double mse = error / double(data.size());
	// A bit-exact round trip is not expected, but guard against log10(0) anyway.
	return mse == 0.0 ? 1000.0 : 10.0 * std::log10(255.0 * 255.0 / mse);
}

enum class EncodeInput { Planar, NV12, Surfaces };

// Walks a packet's concatenated block headers, skipping any sequence header.
static size_t count_blocks_in_packet(const uint8_t *data, size_t size)
{
	size_t offset = 0;
	size_t blocks = 0;

	while (offset + sizeof(PyroWave::BitstreamHeader) <= size)
	{
		auto *header = reinterpret_cast<const PyroWave::BitstreamHeader *>(data + offset);
		if (header->extended)
		{
			offset += sizeof(PyroWave::BitstreamHeader);
			continue;
		}

		const size_t block_size = size_t(header->payload_words) * sizeof(uint32_t);
		if (block_size < sizeof(PyroWave::BitstreamHeader))
			break;

		offset += block_size;
		blocks++;
	}

	return blocks;
}

static void dict_set_int(CFMutableDictionaryRef dict, const void *key, int value)
{
	CFNumberRef number = CFNumberCreate(kCFAllocatorDefault, kCFNumberIntType, &value);
	CFDictionarySetValue(dict, key, number);
	CFRelease(number);
}

static IOSurfaceRef make_r8_surface(const uint8_t *src, int width, int height)
{
	CFMutableDictionaryRef props = CFDictionaryCreateMutable(kCFAllocatorDefault, 4,
	                                                         &kCFTypeDictionaryKeyCallBacks,
	                                                         &kCFTypeDictionaryValueCallBacks);
	dict_set_int(props, kIOSurfaceWidth, width);
	dict_set_int(props, kIOSurfaceHeight, height);
	dict_set_int(props, kIOSurfaceBytesPerElement, 1);
	dict_set_int(props, kIOSurfacePixelFormat, int('L008'));
	IOSurfaceRef surface = IOSurfaceCreate(props);
	CFRelease(props);

	if (!surface)
		return nullptr;

	IOSurfaceLock(surface, 0, nullptr);
	auto *dst = static_cast<uint8_t *>(IOSurfaceGetBaseAddress(surface));
	const size_t stride = IOSurfaceGetBytesPerRow(surface);
	for (int y = 0; y < height; y++)
		memcpy(dst + size_t(y) * stride, src + size_t(y) * width, size_t(width));
	IOSurfaceUnlock(surface, 0, nullptr);

	return surface;
}

static void test_encode_round_trip(pyrowave_device device, MTL::Device *mtl, MTL::CommandQueue *queue,
                                   int width, int height, pyrowave_chroma_subsampling chroma,
                                   EncodeInput input_kind, double min_psnr)
{
	const bool is_420 = chroma == PYROWAVE_CHROMA_SUBSAMPLING_420;
	static const char *kind_names[] = { "planar", "nv12", "iosurface" };
	const char *kind = kind_names[int(input_kind)];
	char label[64];
	snprintf(label, sizeof(label), "%dx%d %s %s", width, height, is_420 ? "420" : "444", kind);
	printf("-- encode round trip: %s\n", label);

	pyrowave_encoder_create_info einfo = {};
	einfo.device = device;
	einfo.width = width;
	einfo.height = height;
	einfo.chroma = chroma;

	pyrowave_encoder encoder = nullptr;
	auto res = pyrowave_encoder_create(&einfo, &encoder);
	CHECK(res == PYROWAVE_SUCCESS, "%s: encoder create failed: %s", label,
	      pyrowave_result_to_string(res));
	if (res != PYROWAVE_SUCCESS)
		return;

	TestFrame frame;
	frame.init(width, height, is_420);

	pyrowave_rate_control rate_control = {};
	// Half a byte per luma pixel is far more than this content needs.
	rate_control.maximum_bitstream_size = size_t(width) * height / 2;

	IOSurfaceRef surfaces[3] = {};
	if (input_kind == EncodeInput::Surfaces)
	{
		for (int i = 0; i < 3; i++)
		{
			const int w = i == 0 ? width : frame.chroma_width;
			const int h = i == 0 ? height : frame.chroma_height;
			surfaces[i] = make_r8_surface(frame.planes[i].data(), w, h);
			CHECK(surfaces[i] != nullptr, "%s: IOSurface %d creation failed", label, i);
		}

		pyrowave_gpu_input input = {};
		for (int i = 0; i < 3; i++)
			input.planes[i] = surfaces[i];
		res = pyrowave_encoder_encode_gpu_synchronous(encoder, &input, &rate_control);
	}
	else
	{
		pyrowave_cpu_buffer buffer = {};
		buffer.width = width;
		buffer.height = height;

		if (input_kind == EncodeInput::NV12)
		{
			buffer.format = PYROWAVE_CPU_BUFFER_FORMAT_NV12;
			buffer.data[0] = frame.planes[0].data();
			buffer.row_stride_in_bytes[0] = size_t(width);
			buffer.plane_size_in_bytes[0] = frame.planes[0].size();
			buffer.data[1] = frame.chroma_interleaved.data();
			buffer.row_stride_in_bytes[1] = size_t(frame.chroma_width) * 2;
			buffer.plane_size_in_bytes[1] = frame.chroma_interleaved.size();
		}
		else
		{
			buffer.format = is_420 ? PYROWAVE_CPU_BUFFER_FORMAT_YUV420P :
			                         PYROWAVE_CPU_BUFFER_FORMAT_YUV444P;
			for (int i = 0; i < 3; i++)
			{
				buffer.data[i] = frame.planes[i].data();
				buffer.row_stride_in_bytes[i] = size_t(i == 0 ? width : frame.chroma_width);
				buffer.plane_size_in_bytes[i] = frame.planes[i].size();
			}
		}

		res = pyrowave_encoder_encode_cpu_synchronous(encoder, &buffer, &rate_control);
	}

	CHECK(res == PYROWAVE_SUCCESS, "%s: encode failed: %s", label, pyrowave_result_to_string(res));

	// A deliberately small boundary, so packetize has to split the frame and the
	// decoder has to reassemble it from many packets.
	constexpr size_t PacketBoundary = 1300;
	size_t num_packets = 0;
	res = pyrowave_encoder_compute_num_packets(encoder, PacketBoundary, &num_packets);
	CHECK(res == PYROWAVE_SUCCESS, "%s: compute_num_packets failed: %s", label,
	      pyrowave_result_to_string(res));
	CHECK(num_packets > 1, "%s: expected the frame to need several packets, got %zu",
	      label, num_packets);

	std::vector<pyrowave_packet> packets(num_packets);
	std::vector<uint8_t> bitstream(rate_control.maximum_bitstream_size);
	size_t written = 0;
	res = pyrowave_encoder_packetize(encoder, packets.data(), PacketBoundary, &written,
	                                 bitstream.data(), bitstream.size());
	CHECK(res == PYROWAVE_SUCCESS, "%s: packetize failed: %s", label,
	      pyrowave_result_to_string(res));
	CHECK(written > 0 && written <= num_packets,
	      "%s: packetize wrote %zu packets, compute_num_packets promised at most %zu",
	      label, written, num_packets);

	for (size_t i = 0; i < written; i++)
	{
		if (packets[i].size <= PacketBoundary)
			continue;

		// A 32x32 block is the smallest unit a packet can carry, so a single block
		// bigger than the boundary has to overshoot it. Anything else must fit.
		CHECK(count_blocks_in_packet(bitstream.data() + packets[i].offset, packets[i].size) == 1,
		      "%s: packet %zu is %zu bytes, over the %zu byte boundary, and holds more than "
		      "one block", label, i, packets[i].size, PacketBoundary);
	}

	// Now decode it back.
	pyrowave_decoder_create_info dinfo = {};
	dinfo.device = device;
	dinfo.width = width;
	dinfo.height = height;
	dinfo.chroma = chroma;

	pyrowave_decoder decoder = nullptr;
	CHECK(pyrowave_decoder_create(&dinfo, &decoder) == PYROWAVE_SUCCESS,
	      "%s: decoder create failed", label);

	for (size_t i = 0; i < written; i++)
	{
		auto push = pyrowave_decoder_push_packet(decoder, bitstream.data() + packets[i].offset,
		                                         packets[i].size);
		CHECK(push == PYROWAVE_SUCCESS, "%s: push_packet %zu failed: %s", label, i,
		      pyrowave_result_to_string(push));
	}

	CHECK(pyrowave_decoder_decode_is_ready(decoder, false),
	      "%s: every packet was pushed, so the frame should be ready", label);

	MTL::Texture *planes[3] = {
		make_plane(mtl, width, height),
		make_plane(mtl, frame.chroma_width, frame.chroma_height),
		make_plane(mtl, frame.chroma_width, frame.chroma_height),
	};
	pyrowave_gpu_buffers buffers = {};
	for (int i = 0; i < 3; i++)
		buffers.planes[i] = planes[i];

	auto *cmd = queue->commandBuffer();
	CHECK(pyrowave_decoder_decode(decoder, cmd, &buffers) == PYROWAVE_SUCCESS,
	      "%s: decode failed", label);
	cmd->commit();
	cmd->waitUntilCompleted();
	CHECK(cmd->status() == MTL::CommandBufferStatusCompleted, "%s: decode status %d", label,
	      int(cmd->status()));

	for (int i = 0; i < 3; i++)
	{
		const double psnr = plane_psnr(planes[i], frame.planes[i]);
		printf("   plane %d: %.2f dB\n", i, psnr);
		CHECK(psnr >= min_psnr, "%s plane %d: PSNR %.2f dB is below the %.1f dB threshold",
		      label, i, psnr, min_psnr);
	}

	for (auto *plane : planes)
		plane->release();
	for (auto surface : surfaces)
		if (surface)
			CFRelease(surface);
	pyrowave_decoder_destroy(decoder);
	pyrowave_encoder_destroy(encoder);
}

int main()
{
	auto *pool = NS::AutoreleasePool::alloc()->init();

	auto *mtl = MTL::CreateSystemDefaultDevice();
	if (!mtl)
	{
		printf("No Metal device.\n");
		return 1;
	}
	printf("Device: %s\n", mtl->name()->utf8String());

	CHECK(pyrowave_device_is_supported(mtl), "device reported unsupported");
	CHECK(!pyrowave_device_is_supported(nullptr), "NULL device reported supported");

	uint32_t major = 0, minor = 0, patch = 0;
	pyrowave_get_api_version(&major, &minor, &patch);
	printf("API version %u.%u.%u\n", major, minor, patch);

	pyrowave_device_create_info dinfo = {};
	dinfo.mtl_device = mtl;
	dinfo.message_callback = message_cb;

	pyrowave_device device = nullptr;
	auto res = pyrowave_device_create(&dinfo, &device);
	CHECK(res == PYROWAVE_SUCCESS, "device create failed: %s", pyrowave_result_to_string(res));
	if (res != PYROWAVE_SUCCESS)
		return 1;

	auto *queue = mtl->newCommandQueue();

	test_resolution(device, mtl, queue, 1920, 1080, PYROWAVE_CHROMA_SUBSAMPLING_420);
	test_resolution(device, mtl, queue, 1920, 1080, PYROWAVE_CHROMA_SUBSAMPLING_444);
	test_resolution(device, mtl, queue, 1280, 720, PYROWAVE_CHROMA_SUBSAMPLING_420);
	test_resolution(device, mtl, queue, 640, 480, PYROWAVE_CHROMA_SUBSAMPLING_444);
	test_resolution(device, mtl, queue, 128, 128, PYROWAVE_CHROMA_SUBSAMPLING_420);
	test_resolution(device, mtl, queue, 3840, 2160, PYROWAVE_CHROMA_SUBSAMPLING_420);

	// Argument validation.
	printf("-- validation\n");
	{
		pyrowave_decoder_create_info info = {};
		info.device = device;
		info.width = 1921;
		info.height = 1080;
		info.chroma = PYROWAVE_CHROMA_SUBSAMPLING_420;
		pyrowave_decoder decoder = nullptr;
		printf("  (expect one odd-dimension error)\n");
		CHECK(pyrowave_decoder_create(&info, &decoder) == PYROWAVE_ERROR_INVALID_ARGUMENT,
		      "odd width accepted for 420");

		info.width = 1920;
		CHECK(pyrowave_decoder_create(&info, &decoder) == PYROWAVE_SUCCESS, "valid create failed");

		// Wrong sized output plane must be rejected rather than corrupting memory.
		MTL::Texture *planes[3] = {
			make_plane(mtl, 1920, 1080), make_plane(mtl, 960, 540), make_plane(mtl, 320, 240),
		};
		pyrowave_gpu_buffers buffers = {};
		for (int i = 0; i < 3; i++)
			buffers.planes[i] = planes[i];

		auto *cmd = queue->commandBuffer();
		printf("  (expect one plane-size error)\n");
		CHECK(pyrowave_decoder_decode(decoder, cmd, &buffers) == PYROWAVE_ERROR_INVALID_ARGUMENT,
		      "mismatched plane size accepted");

		buffers.planes[1] = nullptr;
		printf("  (expect one NULL plane error)\n");
		CHECK(pyrowave_decoder_decode(decoder, cmd, &buffers) == PYROWAVE_ERROR_INVALID_ARGUMENT,
		      "NULL plane accepted");

		CHECK(pyrowave_decoder_decode(decoder, nullptr, &buffers) == PYROWAVE_ERROR_INVALID_ARGUMENT,
		      "NULL command buffer accepted");

		for (auto *plane : planes)
			plane->release();
		pyrowave_decoder_destroy(decoder);
	}

	// Repeated decodes must cycle upload slots without tripping over in-flight buffers.
	printf("-- upload slot reuse (30 frames in flight)\n");
	{
		pyrowave_decoder_create_info info = {};
		info.device = device;
		info.width = 640;
		info.height = 480;
		info.chroma = PYROWAVE_CHROMA_SUBSAMPLING_420;
		pyrowave_decoder decoder = nullptr;
		CHECK(pyrowave_decoder_create(&info, &decoder) == PYROWAVE_SUCCESS, "create failed");

		MTL::Texture *planes[3] = {
			make_plane(mtl, 640, 480), make_plane(mtl, 320, 240), make_plane(mtl, 320, 240),
		};
		pyrowave_gpu_buffers buffers = {};
		for (int i = 0; i < 3; i++)
			buffers.planes[i] = planes[i];

		std::vector<MTL::CommandBuffer *> pending;
		for (int i = 0; i < 30; i++)
		{
			pyrowave_decoder_clear(decoder);
			auto *cmd = queue->commandBuffer();
			cmd->retain();
			CHECK(pyrowave_decoder_decode(decoder, cmd, &buffers) == PYROWAVE_SUCCESS,
			      "decode %d failed", i);
			cmd->commit();
			pending.push_back(cmd);
		}
		for (auto *cmd : pending)
		{
			cmd->waitUntilCompleted();
			CHECK(cmd->status() == MTL::CommandBufferStatusCompleted, "frame status %d", int(cmd->status()));
			cmd->release();
		}
		check_plane(planes[0], "reuse", 0);

		for (auto *plane : planes)
			plane->release();
		pyrowave_decoder_destroy(decoder);
	}

	// Push real packets so the dequant shader takes its payload path rather than the
	// missing-block shortcut. Every packet has ballot == 0, meaning no 8x8 block is
	// coded, so the decoded coefficients are still zero and the output is still 0.5 --
	// but now the shader actually reads the payload buffer, walks the offset table and
	// runs the subgroup prefix sums on the way there.
	printf("-- coded packets with empty ballot\n");
	{
		const int W = 640, H = 480;
		PyroWave::BlockLayout layout;
		CHECK(layout.init(W, H, PyroWave::ChromaSubsampling::Chroma420), "layout init failed");

		pyrowave_decoder_create_info info = {};
		info.device = device;
		info.width = W;
		info.height = H;
		info.chroma = PYROWAVE_CHROMA_SUBSAMPLING_420;
		pyrowave_decoder decoder = nullptr;
		CHECK(pyrowave_decoder_create(&info, &decoder) == PYROWAVE_SUCCESS, "create failed");

		std::vector<uint8_t> stream;
		auto append = [&stream](const void *data, size_t size) {
			auto *p = static_cast<const uint8_t *>(data);
			stream.insert(stream.end(), p, p + size);
		};

		PyroWave::BitstreamSequenceHeader seq = {};
		seq.width_minus_1 = W - 1;
		seq.height_minus_1 = H - 1;
		seq.sequence = 0;
		seq.extended = 1;
		seq.total_blocks = uint32_t(layout.block_count_32x32);
		seq.code = PyroWave::BITSTREAM_EXTENDED_CODE_START_OF_FRAME;
		seq.chroma_resolution = 0;
		append(&seq, sizeof(seq));

		// Code every block, each with an empty ballot and a nonzero quant code.
		for (int i = 0; i < layout.block_count_32x32; i++)
		{
			PyroWave::BitstreamHeader header = {};
			header.ballot = 0;
			header.payload_words = 2;
			header.sequence = 0;
			header.extended = 0;
			header.quant_code = 0x20;
			header.block_index = uint32_t(i);
			append(&header, sizeof(header));
		}

		auto push_res = pyrowave_decoder_push_packet(decoder, stream.data(), stream.size());
		CHECK(push_res == PYROWAVE_SUCCESS, "push_packet failed: %s",
		      pyrowave_result_to_string(push_res));
		CHECK(pyrowave_decoder_decode_is_ready(decoder, false),
		      "frame with every block coded should be ready");

		MTL::Texture *planes[3] = {
			make_plane(mtl, W, H), make_plane(mtl, W / 2, H / 2), make_plane(mtl, W / 2, H / 2),
		};
		pyrowave_gpu_buffers buffers = {};
		for (int i = 0; i < 3; i++)
			buffers.planes[i] = planes[i];

		auto *cmd = queue->commandBuffer();
		CHECK(pyrowave_decoder_decode(decoder, cmd, &buffers) == PYROWAVE_SUCCESS, "decode failed");
		cmd->commit();
		cmd->waitUntilCompleted();
		CHECK(cmd->status() == MTL::CommandBufferStatusCompleted, "status %d", int(cmd->status()));
		if (cmd->error())
		{
			printf("FAIL GPU error: %s\n", cmd->error()->localizedDescription()->utf8String());
			g_failures++;
		}

		// Decoding twice for the same sequence must be refused.
		CHECK(!pyrowave_decoder_decode_is_ready(decoder, false), "same sequence decoded twice");

		for (int i = 0; i < 3; i++)
			check_plane(planes[i], "coded-empty-ballot", i);

		for (auto *plane : planes)
			plane->release();
		pyrowave_decoder_destroy(decoder);
	}

	// Encoder. 40 dB is far below what this smooth content achieves at half a byte
	// per pixel, so the threshold catches structural breakage without being brittle.
	test_encode_round_trip(device, mtl, queue, 640, 480,
	                       PYROWAVE_CHROMA_SUBSAMPLING_420, EncodeInput::Planar, 40.0);
	test_encode_round_trip(device, mtl, queue, 640, 480,
	                       PYROWAVE_CHROMA_SUBSAMPLING_420, EncodeInput::NV12, 40.0);
	test_encode_round_trip(device, mtl, queue, 640, 480,
	                       PYROWAVE_CHROMA_SUBSAMPLING_420, EncodeInput::Surfaces, 40.0);
	test_encode_round_trip(device, mtl, queue, 640, 480,
	                       PYROWAVE_CHROMA_SUBSAMPLING_444, EncodeInput::Planar, 40.0);
	// Neither dimension is a multiple of 32, so the DWT has to mirror the input up to
	// the aligned extent.
	test_encode_round_trip(device, mtl, queue, 1918, 1078,
	                       PYROWAVE_CHROMA_SUBSAMPLING_420, EncodeInput::Planar, 40.0);
	test_encode_round_trip(device, mtl, queue, 1919, 1077,
	                       PYROWAVE_CHROMA_SUBSAMPLING_444, EncodeInput::Surfaces, 40.0);

	printf("-- encoder validation\n");
	{
		pyrowave_encoder_create_info info = {};
		info.device = device;
		info.width = 640;
		info.height = 481;
		info.chroma = PYROWAVE_CHROMA_SUBSAMPLING_420;
		pyrowave_encoder encoder = nullptr;
		printf("  (expect one odd-dimension error)\n");
		CHECK(pyrowave_encoder_create(&info, &encoder) == PYROWAVE_ERROR_INVALID_ARGUMENT,
		      "odd height accepted for a 420 encoder");

		info.height = 480;
		CHECK(pyrowave_encoder_create(&info, &encoder) == PYROWAVE_SUCCESS,
		      "valid encoder create failed");

		pyrowave_rate_control rate_control = {};
		rate_control.maximum_bitstream_size = 32 * 1024;

		// Nothing has been encoded yet, so there is no frame to packetize.
		size_t num_packets = 0;
		printf("  (expect one no-frame error)\n");
		CHECK(pyrowave_encoder_compute_num_packets(encoder, 1300, &num_packets) ==
		      PYROWAVE_ERROR_GENERIC, "compute_num_packets succeeded before any encode");

		TestFrame frame;
		frame.init(640, 480, true);

		pyrowave_cpu_buffer buffer = {};
		buffer.width = 640;
		buffer.height = 480;
		buffer.format = PYROWAVE_CPU_BUFFER_FORMAT_YUV444P;
		for (int i = 0; i < 3; i++)
		{
			buffer.data[i] = frame.planes[i].data();
			buffer.row_stride_in_bytes[i] = size_t(i == 0 ? 640 : 320);
			buffer.plane_size_in_bytes[i] = frame.planes[i].size();
		}

		// 4:4:4 input into a 4:2:0 encoder.
		printf("  (expect one format mismatch error)\n");
		CHECK(pyrowave_encoder_encode_cpu_synchronous(encoder, &buffer, &rate_control) ==
		      PYROWAVE_ERROR_INVALID_ARGUMENT, "444 input accepted by a 420 encoder");

		buffer.format = PYROWAVE_CPU_BUFFER_FORMAT_YUV420P;
		buffer.width = 320;
		printf("  (expect one dimension mismatch error)\n");
		CHECK(pyrowave_encoder_encode_cpu_synchronous(encoder, &buffer, &rate_control) ==
		      PYROWAVE_ERROR_INVALID_ARGUMENT, "mismatched input dimensions accepted");
		buffer.width = 640;

		// A stride that cannot hold a row.
		buffer.row_stride_in_bytes[0] = 320;
		printf("  (expect one stride error)\n");
		CHECK(pyrowave_encoder_encode_cpu_synchronous(encoder, &buffer, &rate_control) ==
		      PYROWAVE_ERROR_INVALID_ARGUMENT, "too small a row stride accepted");
		buffer.row_stride_in_bytes[0] = 640;

		auto zero_rate = rate_control;
		zero_rate.maximum_bitstream_size = 0;
		CHECK(pyrowave_encoder_encode_cpu_synchronous(encoder, &buffer, &zero_rate) ==
		      PYROWAVE_ERROR_INVALID_ARGUMENT, "a zero byte rate target was accepted");

		CHECK(pyrowave_encoder_encode_cpu_synchronous(encoder, nullptr, &rate_control) ==
		      PYROWAVE_ERROR_INVALID_ARGUMENT, "NULL input accepted");
		CHECK(pyrowave_encoder_encode_cpu_synchronous(encoder, &buffer, nullptr) ==
		      PYROWAVE_ERROR_INVALID_ARGUMENT, "NULL rate control accepted");

		pyrowave_gpu_input gpu_input = {};
		printf("  (expect one NULL surface error)\n");
		CHECK(pyrowave_encoder_encode_gpu_synchronous(encoder, &gpu_input, &rate_control) ==
		      PYROWAVE_ERROR_INVALID_ARGUMENT, "an all NULL gpu input was accepted");

		// Re-encoding without reading the result out is allowed; the last frame wins.
		for (int i = 0; i < 5; i++)
		{
			CHECK(pyrowave_encoder_encode_cpu_synchronous(encoder, &buffer, &rate_control) ==
			      PYROWAVE_SUCCESS, "back to back encode %d failed", i);
		}
		CHECK(pyrowave_encoder_compute_num_packets(encoder, 1300, &num_packets) == PYROWAVE_SUCCESS,
		      "compute_num_packets failed after repeated encodes");
		CHECK(num_packets > 0, "repeated encodes produced no packets");

		pyrowave_encoder_destroy(encoder);
	}

	queue->release();
	pyrowave_device_destroy(device);
	mtl->release();
	pool->release();

	if (g_failures)
		printf("\n%d FAILURE(S)\n", g_failures);
	else
		printf("\nAll Metal tests passed.\n");
	return g_failures ? 1 : 0;
}
