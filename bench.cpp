// Copyright (c) 2025 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT

#include <string.h>

#include "global_managers_init.hpp"
#include "device.hpp"
#include "context.hpp"
#include "pyrowave_encoder.hpp"
#include <chrono>
#include "pyrowave_decoder.hpp"
#include "yuv4mpeg.hpp"
#include "shaders/slangmosh.hpp"

using namespace Granite;
using namespace Vulkan;

static std::vector<uint8_t> example_payload;

static void run_decoder_test(Device &device, PyroWave::Decoder &dec, const PyroWave::ViewBuffers &output)
{
	// Same instrumentation as the encoder loop: KosmicKrisp has no timestamps, and a
	// single bracket around the loop body cannot tell CPU-busy from CPU-blocked-in-
	// next_frame_context, so the segments are timed separately.
	uint32_t iterations = 10000;
	if (const char *env = getenv("PYROWAVE_BENCH_ITERATIONS"))
		iterations = uint32_t(strtoul(env, nullptr, 0));
	double parse_ms = 0.0, record_ms = 0.0, submit_ms = 0.0, context_ms = 0.0;
	auto wall_start = std::chrono::steady_clock::now();

	for (uint32_t i = 0; i < iterations; i++)
	{
		auto t_a = std::chrono::steady_clock::now();
		dec.clear();
		dec.push_packet(example_payload.data(), example_payload.size());
		if (!dec.decode_is_ready(false))
			return;
		auto t_b = std::chrono::steady_clock::now();

		auto cmd = device.request_command_buffer();

		cmd->begin_barrier_batch();
		for (auto &plane: output.planes)
		{
			if (PyroWave::Decoder::device_prefers_fragment_path(device))
			{
				cmd->image_barrier(plane->get_image(), VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
				                   VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
				                   VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
				                   VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
			}
			else
			{
				cmd->image_barrier(plane->get_image(), VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
				                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
				                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
			}
		}
		cmd->end_barrier_batch();

		auto start_ts = cmd->write_timestamp(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
		dec.decode(*cmd, output);
		auto end_ts = cmd->write_timestamp(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
		device.register_time_interval("GPU", std::move(start_ts), std::move(end_ts), "Overall Decode");
		auto t_c = std::chrono::steady_clock::now();
		device.submit(cmd);
		auto t_d = std::chrono::steady_clock::now();
		device.next_frame_context();
		auto t_e = std::chrono::steady_clock::now();
		parse_ms += std::chrono::duration<double, std::milli>(t_b - t_a).count();
		record_ms += std::chrono::duration<double, std::milli>(t_c - t_b).count();
		submit_ms += std::chrono::duration<double, std::milli>(t_d - t_c).count();
		context_ms += std::chrono::duration<double, std::milli>(t_e - t_d).count();
	}

	{
		double ms = std::chrono::duration<double, std::milli>(
				std::chrono::steady_clock::now() - wall_start).count();
		double n = double(iterations);
		LOGI("Decode: wall %.3f | parse %.3f  record %.3f  submit %.3f  frame-context(GPU wait) %.3f ms/frame\n",
		     ms / n, parse_ms / n, record_ms / n, submit_ms / n, context_ms / n);
	}
}

static void run_encoder_test(Device &device,
                             PyroWave::Encoder &enc,
                             const PyroWave::ViewBuffers &inputs)
{
	BufferCreateInfo buffer_info = {};
	buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

	constexpr uint32_t bitstream_size = 500000;

	buffer_info.size = enc.get_meta_required_size();
	buffer_info.domain = BufferDomain::Device;
	auto meta = device.create_buffer(buffer_info);
	buffer_info.domain = BufferDomain::CachedHost;
	auto meta_host = device.create_buffer(buffer_info);

	buffer_info.size = bitstream_size + 2 * enc.get_meta_required_size();
	buffer_info.domain = BufferDomain::Device;
	auto bitstream = device.create_buffer(buffer_info);
	buffer_info.domain = BufferDomain::CachedHost;
	auto bitstream_host = device.create_buffer(buffer_info);

	PyroWave::Encoder::BitstreamBuffers buffers = {};
	buffers.meta.buffer = meta.get();
	buffers.meta.size = meta->get_create_info().size;
	buffers.bitstream.buffer = bitstream.get();
	buffers.bitstream.size = bitstream->get_create_info().size;
	buffers.target_size = bitstream_size;

	Fence fence;

	// KosmicKrisp reports no timestamp support, so also wall clock the whole
	// saturated loop; on a GPU bound workload that is GPU seconds per frame.
	uint32_t iterations = 10000;
	if (const char *env = getenv("PYROWAVE_BENCH_ITERATIONS"))
		iterations = uint32_t(strtoul(env, nullptr, 0));
	// PYROWAVE_BENCH_NULL submits the same command buffers with no encode work in
	// them, which prices the loop's own overhead: driver command buffer translation,
	// submission and Granite's frame contexts.
	const bool null_mode = getenv("PYROWAVE_BENCH_NULL") != nullptr;
	// Accumulating the time spent inside the loop body separates "the CPU cannot feed
	// the GPU fast enough" from "the GPU is the limit". If this approaches the total
	// wall time the loop is CPU bound and the GPU time is hidden, in which case
	// subtracting an overhead figure from the wall time is meaningless.
	double record_ms = 0.0, submit_ms = 0.0, context_ms = 0.0;
	auto wall_start = std::chrono::steady_clock::now();

	for (uint32_t i = 0; i < iterations; i++)
	{
		auto t_a = std::chrono::steady_clock::now();
		auto cmd = device.request_command_buffer(CommandBuffer::Type::AsyncCompute);
		auto start_ts = cmd->write_timestamp(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
		if (!null_mode)
		{
			enc.encode(*cmd, inputs, buffers);
			cmd->barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
			             VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
		}
		auto end_ts = cmd->write_timestamp(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
		device.register_time_interval("GPU", std::move(start_ts), std::move(end_ts), "Overall Encode");
		start_ts = cmd->write_timestamp(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
		if (!null_mode)
		{
			cmd->copy_buffer(*bitstream_host, *bitstream);
			cmd->copy_buffer(*meta_host, *meta);
			cmd->barrier(VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
			             VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
			             VK_ACCESS_HOST_READ_BIT);
		}
		end_ts = cmd->write_timestamp(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
		device.register_time_interval("GPU", std::move(start_ts), std::move(end_ts), "Bitstream Readback");
		auto t_b = std::chrono::steady_clock::now();
		fence.reset();
		device.submit(cmd, &fence);
		auto t_c = std::chrono::steady_clock::now();
		// Granite's frame contexts are a ring, so this blocks until an older context
		// retires -- i.e. this is where waiting on the GPU actually shows up.
		device.next_frame_context();
		auto t_d = std::chrono::steady_clock::now();
		record_ms += std::chrono::duration<double, std::milli>(t_b - t_a).count();
		submit_ms += std::chrono::duration<double, std::milli>(t_c - t_b).count();
		context_ms += std::chrono::duration<double, std::milli>(t_d - t_c).count();
	}

	fence->wait();

	{
		auto wall_end = std::chrono::steady_clock::now();
		double ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
		double n = double(iterations);
		LOGI("Encode%s: wall %.3f | record %.3f  submit %.3f  frame-context(GPU wait) %.3f ms/frame\n",
		     null_mode ? " (NULL)" : "",
		     ms / n, record_ms / n, submit_ms / n, context_ms / n);
	}

	PyroWave::Encoder::Packet packet = {};
	example_payload.resize(500000);
	auto num_packets = enc.compute_num_packets(device.map_host_buffer(*meta_host, MEMORY_ACCESS_READ_BIT), 500000);
	(void)num_packets;
	assert(num_packets == 1);
	enc.packetize(&packet, 500000, example_payload.data(), example_payload.size(),
	              device.map_host_buffer(*meta_host, MEMORY_ACCESS_READ_BIT),
	              device.map_host_buffer(*bitstream_host, MEMORY_ACCESS_READ_BIT));
	example_payload.resize(packet.size);
}

struct YCbCrImages
{
	Vulkan::ImageHandle images[3];
	PyroWave::ViewBuffers views;
};

static YCbCrImages create_ycbcr_images(Device &device, int width, int height, VkFormat fmt, PyroWave::ChromaSubsampling chroma)
{
	YCbCrImages images;
	auto info = ImageCreateInfo::immutable_2d_image(width, height, fmt);
	info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
	             VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
	info.initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;

	images.images[0] = device.create_image(info);
	device.set_name(*images.images[0], "Y");

	if (chroma == PyroWave::ChromaSubsampling::Chroma420)
	{
		info.width >>= 1;
		info.height >>= 1;
	}

	images.images[1] = device.create_image(info);
	device.set_name(*images.images[1], "Cb");

	images.images[2] = device.create_image(info);
	device.set_name(*images.images[2], "Cr");

	for (int i = 0; i < 3; i++)
		images.views.planes[i] = &images.images[i]->get_view();

	return images;
}

static void run_vulkan_test(Device &device, const char *in_path)
{
	YUV4MPEGFile input;

	if (!input.open_read(in_path))
		return;

	auto width = input.get_width();
	auto height = input.get_height();

	auto fmt = YUV4MPEGFile::format_to_bytes_per_component(input.get_format()) == 2 ? VK_FORMAT_R16_UNORM : VK_FORMAT_R8_UNORM;
	auto chroma = YUV4MPEGFile::format_has_subsampling(input.get_format()) ? PyroWave::ChromaSubsampling::Chroma420 : PyroWave::ChromaSubsampling::Chroma444;
	auto inputs = create_ycbcr_images(device, width, height, fmt, chroma);

	PyroWave::Encoder enc;
	PyroWave::Decoder dec;
	if (!enc.init(&device, width, height, chroma))
		return;

	if (!dec.init(&device, width, height, chroma, PyroWave::Decoder::device_prefers_fragment_path(device)))
		return;

	if (!input.begin_frame())
		return;

	auto cmd = device.request_command_buffer();

	for (int i = 0; i < 3; i++)
	{
		cmd->image_barrier(*inputs.images[i], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		                   0, 0,
		                   VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
	}

	for (int i = 0; i < 3; i++)
	{
		auto *y = cmd->update_image(*inputs.images[i]);
		if (!input.read(y, inputs.images[i]->get_width() * inputs.images[i]->get_height()))
		{
			LOGE("Failed to read plane.\n");
			device.submit_discard(cmd);
			return;
		}
	}

	for (int i = 0; i < 3; i++)
	{
		cmd->image_barrier(*inputs.images[i], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL,
		                   VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
		                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
	}

	device.submit(cmd);

	run_encoder_test(device, enc, inputs.views);
	for (int i = 0; i < 4; i++)
		device.next_frame_context();
	device.timestamp_log([](const std::string &tag, const TimestampIntervalReport &report)
	{
		LOGI("%s -> %.3f us avg\n", tag.c_str(), report.time_per_frame_context * 1e6);
	});
	device.timestamp_log_reset();

	run_decoder_test(device, dec, inputs.views);
	for (int i = 0; i < 4; i++)
		device.next_frame_context();
	device.timestamp_log([](const std::string &tag, const TimestampIntervalReport &report)
	{
		LOGI("%s -> %.3f us avg\n", tag.c_str(), report.time_per_frame_context * 1e6);
	});
	device.timestamp_log_reset();
}

static void run_vulkan_test(const char *in_path)
{
	if (!Context::init_loader(nullptr))
		return;

	Context ctx;

	if (!ctx.init_instance_and_device(nullptr, 0, nullptr, 0, CONTEXT_CREATION_ENABLE_PUSH_DESCRIPTOR_BIT))
		return;

	Device dev;
	dev.set_context(ctx);

	run_vulkan_test(dev, in_path);
}

int main(int argc, char **argv)
{
	if (argc != 2)
		return EXIT_FAILURE;

	run_vulkan_test(argv[1]);
}
