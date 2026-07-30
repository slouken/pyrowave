// Copyright (c) 2026 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT

// Decodes a PyroWave bitstream with the Metal backend and optionally compares the
// result against a reference decode, so the Metal path can be validated against the
// Vulkan one.
//
//   pyrowave-metal-decode stream.bin --output out.y4m
//   pyrowave-metal-decode stream.bin --reference vulkan.y4m
//
// Frame dimensions and chroma mode are read from the bitstream's sequence header.

#import <Metal/Metal.h>

#include "pyrowave_metal.h"
#include "pyrowave_bitstream.hpp"
#include "yuv4mpeg.hpp"

#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

using namespace PyroWave;

namespace
{
struct Plane
{
	id<MTLTexture> texture;
	int width = 0;
	int height = 0;
	std::vector<uint8_t> pixels;
};

void message_cb(void *, const char *msg)
{
	fprintf(stderr, "pyrowave: %s\n", msg);
}

bool read_file(const char *path, std::vector<uint8_t> &out)
{
	FILE *file = fopen(path, "rb");
	if (!file)
	{
		fprintf(stderr, "Failed to open %s\n", path);
		return false;
	}

	fseek(file, 0, SEEK_END);
	long size = ftell(file);
	fseek(file, 0, SEEK_SET);

	if (size < 0)
	{
		fclose(file);
		return false;
	}

	out.resize(size_t(size));
	bool ok = out.empty() || fread(out.data(), 1, out.size(), file) == out.size();
	fclose(file);

	if (!ok)
		fprintf(stderr, "Failed to read %s\n", path);
	return ok;
}

// pyrowave-encode wraps the bitstream in a small container: the magic, eight
// int32 parameters, then per frame a uint32 byte count followed by that frame's
// packets. A raw packet stream with no container is also accepted.
constexpr char ContainerMagic[8] = { 'P', 'Y', 'R', 'O', 'W', 'A', 'V', 'E' };
constexpr size_t ContainerHeaderSize = sizeof(ContainerMagic) + 8 * sizeof(int32_t);

bool is_container(const std::vector<uint8_t> &stream)
{
	return stream.size() >= ContainerHeaderSize &&
	       memcmp(stream.data(), ContainerMagic, sizeof(ContainerMagic)) == 0;
}

bool parse_container_header(const std::vector<uint8_t> &stream, int &width, int &height,
                            pyrowave_chroma_subsampling &chroma)
{
	int32_t params[8];
	memcpy(params, stream.data() + sizeof(ContainerMagic), sizeof(params));

	width = params[0];
	height = params[1];
	// params[3] is PyroWave::ChromaSubsampling: 0 is 420, 1 is 444.
	chroma = params[3] == 0 ? PYROWAVE_CHROMA_SUBSAMPLING_420 : PYROWAVE_CHROMA_SUBSAMPLING_444;

	if (width <= 0 || height <= 0)
	{
		fprintf(stderr, "Container declares bogus dimensions %dx%d.\n", width, height);
		return false;
	}

	if (stream.size() == ContainerHeaderSize)
	{
		fprintf(stderr,
		        "Container has a header but no frames -- the encoder wrote no output.\n");
		return false;
	}

	return true;
}

// Finds the first start-of-frame header so the stream can be self describing.
bool probe_sequence_header(const std::vector<uint8_t> &stream, int &width, int &height,
                           pyrowave_chroma_subsampling &chroma)
{
	size_t offset = 0;
	while (offset + sizeof(BitstreamHeader) <= stream.size())
	{
		auto *header = reinterpret_cast<const BitstreamHeader *>(stream.data() + offset);
		if (header->extended != 0)
		{
			auto *seq = reinterpret_cast<const BitstreamSequenceHeader *>(header);
			if (seq->code == BITSTREAM_EXTENDED_CODE_START_OF_FRAME)
			{
				width = int(seq->width_minus_1) + 1;
				height = int(seq->height_minus_1) + 1;
				chroma = seq->chroma_resolution == 0 ?
				         PYROWAVE_CHROMA_SUBSAMPLING_420 : PYROWAVE_CHROMA_SUBSAMPLING_444;
				return true;
			}
			offset += sizeof(BitstreamHeader);
		}
		else
		{
			size_t packet_size = size_t(header->payload_words) * sizeof(uint32_t);
			if (packet_size < sizeof(BitstreamHeader))
				return false;
			offset += packet_size;
		}
	}

	fprintf(stderr, "No start-of-frame header found in stream.\n");
	return false;
}

id<MTLTexture> create_plane(id<MTLDevice> device, int width, int height)
{
	auto desc = [MTLTextureDescriptor new];
	desc.textureType = MTLTextureType2D;
	// 8-bit unorm matches how the Vulkan decoder is normally consumed, and makes
	// the readback directly comparable to a Y4M reference.
	desc.pixelFormat = MTLPixelFormatR8Unorm;
	desc.width = width;
	desc.height = height;
	desc.usage = MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead;
	desc.storageMode = MTLStorageModeShared;
	auto texture = [device newTextureWithDescriptor:desc];
	return texture;
}

void read_plane(Plane &plane)
{
	plane.pixels.resize(size_t(plane.width) * plane.height);
	[plane.texture getBytes:plane.pixels.data() bytesPerRow:plane.width fromRegion:MTLRegionMake2D(0, 0, plane.width, plane.height) mipmapLevel:0];
}

struct PlaneStats
{
	uint32_t max_abs_diff = 0;
	uint64_t differing = 0;
	double mse = 0.0;
};

PlaneStats compare_plane(const std::vector<uint8_t> &a, const std::vector<uint8_t> &b)
{
	PlaneStats stats = {};
	uint64_t accum = 0;
	for (size_t i = 0; i < a.size(); i++)
	{
		int diff = int(a[i]) - int(b[i]);
		uint32_t abs_diff = uint32_t(diff < 0 ? -diff : diff);
		if (abs_diff)
		{
			stats.differing++;
			if (abs_diff > stats.max_abs_diff)
				stats.max_abs_diff = abs_diff;
		}
		accum += uint64_t(diff * diff);
	}
	stats.mse = a.empty() ? 0.0 : double(accum) / double(a.size());
	return stats;
}
}

int main(int argc, char **argv)
{
	const char *stream_path = nullptr;
	const char *output_path = nullptr;
	const char *reference_path = nullptr;
	bool allow_partial = false;

	for (int i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "--output") == 0 && i + 1 < argc)
			output_path = argv[++i];
		else if (strcmp(argv[i], "--reference") == 0 && i + 1 < argc)
			reference_path = argv[++i];
		else if (strcmp(argv[i], "--allow-partial") == 0)
			allow_partial = true;
		else if (argv[i][0] == '-')
		{
			fprintf(stderr, "Unknown argument %s\n", argv[i]);
			return EXIT_FAILURE;
		}
		else
			stream_path = argv[i];
	}

	if (!stream_path)
	{
		fprintf(stderr, "Usage: %s stream.bin [--output out.y4m] [--reference ref.y4m] [--allow-partial]\n",
		        argv[0]);
		return EXIT_FAILURE;
	}

	std::vector<uint8_t> stream;
	if (!read_file(stream_path, stream))
		return EXIT_FAILURE;

	int width = 0, height = 0;
	pyrowave_chroma_subsampling chroma = PYROWAVE_CHROMA_SUBSAMPLING_420;
	const bool containerized = is_container(stream);

	if (containerized)
	{
		if (!parse_container_header(stream, width, height, chroma))
			return EXIT_FAILURE;
	}
	else if (!probe_sequence_header(stream, width, height, chroma))
	{
		return EXIT_FAILURE;
	}

	const bool is_420 = chroma == PYROWAVE_CHROMA_SUBSAMPLING_420;
	printf("Stream: %dx%d %s, %zu bytes (%s)\n", width, height, is_420 ? "420" : "444",
	       stream.size(), containerized ? "PYROWAVE container" : "raw packets");


	id<MTLDevice> mtl = MTLCreateSystemDefaultDevice();
	if (!mtl)
	{
		fprintf(stderr, "No Metal device available.\n");
		return EXIT_FAILURE;
	}
	printf("Device: %s\n", mtl.name.UTF8String);

	if (!pyrowave_device_is_supported((__bridge void *)mtl))
	{
		fprintf(stderr, "Metal device is not supported by PyroWave.\n");
		return EXIT_FAILURE;
	}

	pyrowave_device_create_info device_info = {};
	device_info.mtl_device = (__bridge void *)mtl;
	device_info.message_callback = message_cb;

	pyrowave_device device = nullptr;
	auto res = pyrowave_device_create(&device_info, &device);
	if (res != PYROWAVE_SUCCESS)
	{
		fprintf(stderr, "Failed to create device: %s\n", pyrowave_result_to_string(res));
		return EXIT_FAILURE;
	}

	pyrowave_decoder_create_info decoder_info = {};
	decoder_info.device = device;
	decoder_info.width = width;
	decoder_info.height = height;
	decoder_info.chroma = chroma;

	pyrowave_decoder decoder = nullptr;
	res = pyrowave_decoder_create(&decoder_info, &decoder);
	if (res != PYROWAVE_SUCCESS)
	{
		fprintf(stderr, "Failed to create decoder: %s\n", pyrowave_result_to_string(res));
		return EXIT_FAILURE;
	}

	Plane planes[3];
	planes[0].width = width;
	planes[0].height = height;
	planes[1].width = planes[2].width = is_420 ? width / 2 : width;
	planes[1].height = planes[2].height = is_420 ? height / 2 : height;
	for (auto &plane : planes)
	{
		plane.texture = create_plane(mtl, plane.width, plane.height);
		if (!plane.texture)
		{
			fprintf(stderr, "Failed to allocate output planes.\n");
			return EXIT_FAILURE;
		}
	}

	YUV4MPEGFile output;
	if (output_path)
	{
		// open_write() emits the YUV4MPEG2 magic itself, so params carries only the
		// tags, and must end in a newline or the FRAME header runs into the header line.
		char params[128];
		snprintf(params, sizeof(params), "W%d H%d F60:1 Ip A1:1 C%s\n",
		         width, height, is_420 ? "420mpeg2" : "444");
		if (!output.open_write(output_path, params))
		{
			fprintf(stderr, "Failed to open %s for writing.\n", output_path);
			return EXIT_FAILURE;
		}
	}

	YUV4MPEGFile reference;
	if (reference_path)
	{
		if (!reference.open_read(reference_path))
		{
			fprintf(stderr, "Failed to open reference %s.\n", reference_path);
			return EXIT_FAILURE;
		}
		if (reference.get_width() != width || reference.get_height() != height)
		{
			fprintf(stderr, "Reference is %dx%d, stream is %dx%d.\n",
			        reference.get_width(), reference.get_height(), width, height);
			return EXIT_FAILURE;
		}
	}

	auto queue = [mtl newCommandQueue];

	pyrowave_gpu_buffers buffers = {};
	for (int i = 0; i < 3; i++)
		buffers.planes[i] = (__bridge void *)planes[i].texture;

	// Feed one packet at a time so multi-frame streams decode frame by frame.
	size_t offset = 0;
	int frames = 0;
	int mismatched_frames = 0;
	bool failed = false;

	auto decode_frame = [&]() -> bool {
		auto cmd = [queue commandBuffer];
		auto decode_res = pyrowave_decoder_decode(decoder, (__bridge void *)cmd, &buffers);
		if (decode_res != PYROWAVE_SUCCESS)
		{
			fprintf(stderr, "Frame %d: decode failed: %s\n", frames,
			        pyrowave_result_to_string(decode_res));
			return false;
		}

		[cmd commit];
		[cmd waitUntilCompleted];

		if (cmd.error)
		{
			fprintf(stderr, "Frame %d: GPU error: %s\n", frames,
			        cmd.error.localizedDescription.UTF8String);
			return false;
		}

		for (auto &plane : planes)
			read_plane(plane);

		if (output_path)
		{
			if (!output.begin_frame())
				return false;
			for (auto &plane : planes)
				if (!output.write(plane.pixels.data(), plane.pixels.size()))
					return false;
		}

		if (reference_path)
		{
			// Each Y4M frame is preceded by a FRAME header that has to be consumed,
			// otherwise its bytes are read as pixel data and every plane shifts.
			if (!reference.begin_frame())
			{
				fprintf(stderr, "Frame %d: reference has no further frames.\n", frames);
				return false;
			}

			bool frame_matches = true;
			for (int i = 0; i < 3; i++)
			{
				std::vector<uint8_t> ref(planes[i].pixels.size());
				if (!reference.read(ref.data(), ref.size()))
				{
					fprintf(stderr, "Frame %d: reference ran out of data.\n", frames);
					return false;
				}

				auto stats = compare_plane(planes[i].pixels, ref);
				if (stats.differing)
				{
					frame_matches = false;
					double psnr = stats.mse > 0.0 ? 10.0 * log10(255.0 * 255.0 / stats.mse) : 99.0;
					printf("  frame %d plane %d: %llu/%zu pixels differ, max |diff| %u, PSNR %.2f dB\n",
					       frames, i, (unsigned long long)stats.differing,
					       planes[i].pixels.size(), stats.max_abs_diff, psnr);
				}
			}
			if (!frame_matches)
				mismatched_frames++;
		}

		frames++;
		return true;
	};

	if (containerized)
	{
		// Each container record holds one frame's worth of packets.
		offset = ContainerHeaderSize;
		while (offset + sizeof(uint32_t) <= stream.size())
		{
			uint32_t frame_size = 0;
			memcpy(&frame_size, stream.data() + offset, sizeof(frame_size));
			offset += sizeof(frame_size);

			if (offset + frame_size > stream.size())
			{
				fprintf(stderr, "Frame %d claims %u bytes but only %zu remain.\n",
				        frames, frame_size, stream.size() - offset);
				failed = true;
				break;
			}

			res = pyrowave_decoder_push_packet(decoder, stream.data() + offset, frame_size);
			if (res != PYROWAVE_SUCCESS)
			{
				fprintf(stderr, "Frame %d: push_packet failed: %s\n", frames,
				        pyrowave_result_to_string(res));
				failed = true;
				break;
			}

			offset += frame_size;

			if (!pyrowave_decoder_decode_is_ready(decoder, false))
			{
				fprintf(stderr, "Frame %d: incomplete after its packets were pushed.\n", frames);
				if (!allow_partial)
				{
					failed = true;
					break;
				}
			}

			if (!decode_frame())
			{
				failed = true;
				break;
			}
		}
	}
	else while (offset + sizeof(BitstreamHeader) <= stream.size())
	{
		auto *header = reinterpret_cast<const BitstreamHeader *>(stream.data() + offset);
		size_t packet_size = header->extended != 0 ?
		                     sizeof(BitstreamHeader) :
		                     size_t(header->payload_words) * sizeof(uint32_t);

		if (packet_size < sizeof(BitstreamHeader) || offset + packet_size > stream.size())
		{
			fprintf(stderr, "Malformed packet at byte %zu.\n", offset);
			failed = true;
			break;
		}

		res = pyrowave_decoder_push_packet(decoder, stream.data() + offset, packet_size);
		if (res != PYROWAVE_SUCCESS)
		{
			fprintf(stderr, "push_packet failed at byte %zu: %s\n", offset,
			        pyrowave_result_to_string(res));
			failed = true;
			break;
		}

		offset += packet_size;

		if (pyrowave_decoder_decode_is_ready(decoder, false) && !decode_frame())
		{
			failed = true;
			break;
		}
	}

	// A trailing frame may be incomplete if the capture was cut short.
	if (!failed && allow_partial && pyrowave_decoder_decode_is_ready(decoder, true))
		failed = !decode_frame();

	printf("Decoded %d frame(s).\n", frames);

	if (reference_path)
	{
		if (mismatched_frames == 0 && frames > 0)
			printf("Bit exact against reference across all %d frame(s).\n", frames);
		else if (frames > 0)
			printf("%d of %d frame(s) differ from reference.\n", mismatched_frames, frames);
	}

	pyrowave_decoder_destroy(decoder);
	pyrowave_device_destroy(device);

	if (failed || frames == 0)
		return EXIT_FAILURE;
	return (reference_path && mismatched_frames) ? EXIT_FAILURE : EXIT_SUCCESS;
}
