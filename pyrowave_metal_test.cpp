// End-to-end GPU smoke test for the Metal decoder.
//
// With no packets pushed, every 32x32 block is marked missing, so the dequant
// shader zero-fills the wavelet pyramid and the iDWT chain reduces to a constant.
// Every write into an output plane happens with the DCShift function constant
// enabled, which adds 0.5, so a correct decode must leave every output pixel at
// exactly 0.5. That single expectation covers dequant, all five iDWT levels, the
// texture views, hazard tracking and the final stores.
#include <Metal/Metal.hpp>

#include "pyrowave_metal.h"
#include "pyrowave_bitstream.hpp"

#include <cmath>
#include <stdio.h>
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
