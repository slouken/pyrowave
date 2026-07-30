// Copyright (c) 2026 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT

// Minimal SDL3 viewer for the Metal decoder.
//
//   pyrowave-metal-viewer stream.bin [--loop] [--fps N]
//
// Fully zero-copy, and without allocating any texture storage of its own: SDL
// creates the three plane textures, this borrows their MTLTextures back out, and
// the decoder's compute shaders write straight into them. SDL then samples the
// very same textures to do the YCbCr to RGB conversion. No pixel data is copied
// or read back to the CPU.
//
// Because the planes are one set of MTLTexture objects rather than two aliasing
// views of shared memory, ordering the decode against the render needs nothing
// but submission order on a single queue -- so this borrows the renderer's own
// MTLCommandQueue too, and commits the decode before issuing the draw.
//
// This needs three things from the Metal renderer: writable plane textures
// (SDL_PROP_TEXTURE_CREATE_METAL_SHADER_WRITE_BOOLEAN), the per plane MTLTexture
// properties to fetch them, and the renderer's device and queue.
//
// Both chroma modes map onto a 3 plane SDL format, which is exactly what the
// decoder emits: SDL_PIXELFORMAT_IYUV for 4:2:0 (half resolution chroma) and
// SDL_PIXELFORMAT_P408 for 4:4:4 (full resolution chroma).

#include <Metal/Metal.hpp>

#include <SDL3/SDL.h>

#include "pyrowave_metal.h"
#include "pyrowave_bitstream.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <vector>

using namespace PyroWave;

namespace
{
struct StreamInfo
{
	std::vector<uint8_t> data;
	std::vector<std::pair<size_t, size_t>> frames;
	int width = 0;
	int height = 0;
	pyrowave_chroma_subsampling chroma = PYROWAVE_CHROMA_SUBSAMPLING_420;
	int fps_num = 60;
	int fps_den = 1;
	bool full_range = false;
	bool bt2020 = false;
};

constexpr char ContainerMagic[8] = { 'P', 'Y', 'R', 'O', 'W', 'A', 'V', 'E' };
constexpr size_t ContainerHeaderSize = sizeof(ContainerMagic) + 8 * sizeof(int32_t);

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
	return ok;
}

// Colour metadata is not part of the decode API, so read it off the sequence header.
void probe_color_info(const std::vector<uint8_t> &data, size_t offset, StreamInfo &info)
{
	while (offset + sizeof(BitstreamHeader) <= data.size())
	{
		auto *header = reinterpret_cast<const BitstreamHeader *>(data.data() + offset);
		if (header->extended != 0)
		{
			auto *seq = reinterpret_cast<const BitstreamSequenceHeader *>(header);
			if (seq->code == BITSTREAM_EXTENDED_CODE_START_OF_FRAME)
			{
				info.full_range = seq->ycbcr_range == 0;
				info.bt2020 = seq->ycbcr_transform != 0;
				return;
			}
			offset += sizeof(BitstreamHeader);
		}
		else
		{
			size_t packet_size = size_t(header->payload_words) * sizeof(uint32_t);
			if (packet_size < sizeof(BitstreamHeader))
				return;
			offset += packet_size;
		}
	}
}

bool load_stream(const char *path, StreamInfo &info)
{
	if (!read_file(path, info.data))
		return false;

	const bool containerized = info.data.size() >= ContainerHeaderSize &&
	                           memcmp(info.data.data(), ContainerMagic, sizeof(ContainerMagic)) == 0;

	if (containerized)
	{
		int32_t params[8];
		memcpy(params, info.data.data() + sizeof(ContainerMagic), sizeof(params));
		info.width = params[0];
		info.height = params[1];
		info.chroma = params[3] == 0 ? PYROWAVE_CHROMA_SUBSAMPLING_420 :
		                               PYROWAVE_CHROMA_SUBSAMPLING_444;
		info.fps_num = params[5] > 0 ? params[5] : 60;
		info.fps_den = params[6] > 0 ? params[6] : 1;

		size_t offset = ContainerHeaderSize;
		while (offset + sizeof(uint32_t) <= info.data.size())
		{
			uint32_t frame_size = 0;
			memcpy(&frame_size, info.data.data() + offset, sizeof(frame_size));
			offset += sizeof(frame_size);
			if (offset + frame_size > info.data.size())
			{
				fprintf(stderr, "Truncated frame record at byte %zu.\n", offset);
				break;
			}
			info.frames.emplace_back(offset, frame_size);
			offset += frame_size;
		}

		if (!info.frames.empty())
			probe_color_info(info.data, info.frames.front().first, info);
	}
	else
	{
		// Raw packet stream: the whole file is treated as one frame.
		size_t offset = 0;
		bool found = false;
		while (offset + sizeof(BitstreamHeader) <= info.data.size())
		{
			auto *header = reinterpret_cast<const BitstreamHeader *>(info.data.data() + offset);
			if (header->extended != 0)
			{
				auto *seq = reinterpret_cast<const BitstreamSequenceHeader *>(header);
				if (seq->code == BITSTREAM_EXTENDED_CODE_START_OF_FRAME)
				{
					info.width = int(seq->width_minus_1) + 1;
					info.height = int(seq->height_minus_1) + 1;
					info.chroma = seq->chroma_resolution == 0 ?
					              PYROWAVE_CHROMA_SUBSAMPLING_420 :
					              PYROWAVE_CHROMA_SUBSAMPLING_444;
					found = true;
					break;
				}
				offset += sizeof(BitstreamHeader);
			}
			else
			{
				size_t packet_size = size_t(header->payload_words) * sizeof(uint32_t);
				if (packet_size < sizeof(BitstreamHeader))
					break;
				offset += packet_size;
			}
		}

		if (!found)
		{
			fprintf(stderr, "No start-of-frame header found in %s.\n", path);
			return false;
		}

		info.frames.emplace_back(size_t(0), info.data.size());
		probe_color_info(info.data, 0, info);
	}

	if (info.frames.empty())
	{
		fprintf(stderr, "%s contains no frames.\n", path);
		return false;
	}

	return true;
}
}

int main(int argc, char **argv)
{
	const char *stream_path = nullptr;
	const char *screenshot_path = nullptr;
	bool report_timing = false;
	bool sync_decode = false;
	int bench_frames = 0;
	bool loop = false;
	double fps_override = 0.0;

	for (int i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "--loop") == 0)
			loop = true;
		else if (strcmp(argv[i], "--fps") == 0 && i + 1 < argc)
			fps_override = atof(argv[++i]);
		// Dump the composited result after the first frame and exit, so the whole
		// decode-into-SDL's-textures path can be checked without a human looking.
		else if (strcmp(argv[i], "--screenshot") == 0 && i + 1 < argc)
			screenshot_path = argv[++i];
		// Reports decode GPU time, which we can read straight off our own command
		// buffer even though it was created from the renderer's queue.
		else if (strcmp(argv[i], "--timing") == 0)
			report_timing = true;
		// Runs uncapped for N frames and reports wall clock throughput.
		else if (strcmp(argv[i], "--bench") == 0 && i + 1 < argc)
			bench_frames = atoi(argv[++i]);
		// Restores the CPU stall the IOSurface based flow needed, for comparison.
		else if (strcmp(argv[i], "--sync-decode") == 0)
			sync_decode = true;
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
		fprintf(stderr, "Usage: %s stream.bin [--loop] [--fps N] [--screenshot out.bmp]\n", argv[0]);
		return EXIT_FAILURE;
	}

	StreamInfo info;
	if (!load_stream(stream_path, info))
		return EXIT_FAILURE;

	const bool is_420 = info.chroma == PYROWAVE_CHROMA_SUBSAMPLING_420;
	// Both are 3 plane formats; they differ only in chroma resolution.
	const SDL_PixelFormat sdl_format = is_420 ? SDL_PIXELFORMAT_IYUV : SDL_PIXELFORMAT_P408;

	double fps = fps_override > 0.0 ? fps_override : double(info.fps_num) / double(info.fps_den);
	printf("%dx%d %s, %zu frame(s), %.2f fps, %s range, %s\n",
	       info.width, info.height, is_420 ? "420" : "444", info.frames.size(), fps,
	       info.full_range ? "full" : "limited", info.bt2020 ? "BT.2020" : "BT.709");

	if (!SDL_Init(SDL_INIT_VIDEO))
	{
		fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
		return EXIT_FAILURE;
	}

	SDL_SetHint(SDL_HINT_RENDER_DRIVER, "metal");

	SDL_Window *window = nullptr;
	SDL_Renderer *renderer = nullptr;
	if (!SDL_CreateWindowAndRenderer("PyroWave Metal", info.width, info.height,
	                                 SDL_WINDOW_RESIZABLE, &window, &renderer))
	{
		fprintf(stderr, "SDL_CreateWindowAndRenderer failed: %s\n", SDL_GetError());
		return EXIT_FAILURE;
	}

	const char *driver = SDL_GetRendererName(renderer);
	if (!driver || strcmp(driver, "metal") != 0)
	{
		fprintf(stderr, "Expected the metal renderer, got '%s'.\n", driver ? driver : "none");
		return EXIT_FAILURE;
	}

	SDL_SetRenderLogicalPresentation(renderer, info.width, info.height,
	                                 SDL_LOGICAL_PRESENTATION_LETTERBOX);

	// Otherwise the throughput measurement is just the display's refresh rate.
	if (bench_frames)
		SDL_SetRenderVSync(renderer, SDL_RENDERER_VSYNC_DISABLED);

	// Share the renderer's own device and queue rather than making our own, so the
	// decode and the render are the same device's resources and are ordered against
	// each other by nothing more than command buffer submission order.
	SDL_PropertiesID renderer_props = SDL_GetRendererProperties(renderer);
	auto *mtl = static_cast<MTL::Device *>(
			SDL_GetPointerProperty(renderer_props, SDL_PROP_RENDERER_METAL_DEVICE_POINTER, nullptr));
	auto *queue = static_cast<MTL::CommandQueue *>(
			SDL_GetPointerProperty(renderer_props, SDL_PROP_RENDERER_METAL_COMMAND_QUEUE_POINTER, nullptr));

	if (!mtl || !queue)
	{
		fprintf(stderr, "The metal renderer did not publish its device and command queue.\n");
		return EXIT_FAILURE;
	}

	if (!pyrowave_device_is_supported(mtl))
	{
		fprintf(stderr, "No supported Metal device.\n");
		return EXIT_FAILURE;
	}

	SDL_Colorspace colorspace;
	if (info.bt2020)
		colorspace = info.full_range ? SDL_COLORSPACE_BT2020_FULL : SDL_COLORSPACE_BT2020_LIMITED;
	else
		colorspace = info.full_range ? SDL_COLORSPACE_BT709_FULL : SDL_COLORSPACE_BT709_LIMITED;

	SDL_PropertiesID props = SDL_CreateProperties();
	SDL_SetNumberProperty(props, SDL_PROP_TEXTURE_CREATE_FORMAT_NUMBER, sdl_format);
	SDL_SetNumberProperty(props, SDL_PROP_TEXTURE_CREATE_ACCESS_NUMBER, SDL_TEXTUREACCESS_STATIC);
	SDL_SetNumberProperty(props, SDL_PROP_TEXTURE_CREATE_WIDTH_NUMBER, info.width);
	SDL_SetNumberProperty(props, SDL_PROP_TEXTURE_CREATE_HEIGHT_NUMBER, info.height);
	SDL_SetNumberProperty(props, SDL_PROP_TEXTURE_CREATE_COLORSPACE_NUMBER, colorspace);
	// The decoder writes into these planes with a compute shader.
	SDL_SetBooleanProperty(props, SDL_PROP_TEXTURE_CREATE_METAL_SHADER_WRITE_BOOLEAN, true);
	SDL_Texture *texture = SDL_CreateTextureWithProperties(renderer, props);
	SDL_DestroyProperties(props);

	if (!texture)
	{
		fprintf(stderr, "SDL_CreateTextureWithProperties failed: %s\n", SDL_GetError());
		return EXIT_FAILURE;
	}

	// SDL allocated the planes; borrow them rather than supplying our own storage.
	// Named by component, so the decoder's Cb and Cr map straight across.
	static const char *plane_props[3] = {
		SDL_PROP_TEXTURE_METAL_TEXTURE_POINTER,
		SDL_PROP_TEXTURE_METAL_TEXTURE_U_POINTER,
		SDL_PROP_TEXTURE_METAL_TEXTURE_V_POINTER,
	};
	SDL_PropertiesID texture_props = SDL_GetTextureProperties(texture);
	MTL::Texture *planes[3] = {};
	for (int i = 0; i < 3; i++)
	{
		planes[i] = static_cast<MTL::Texture *>(
				SDL_GetPointerProperty(texture_props, plane_props[i], nullptr));
		if (!planes[i])
		{
			fprintf(stderr, "SDL did not publish a Metal texture for plane %d.\n", i);
			return EXIT_FAILURE;
		}
		printf("  plane %d: %ux%u, usage %#x\n", i,
		       unsigned(planes[i]->width()), unsigned(planes[i]->height()),
		       unsigned(planes[i]->usage()));
	}

	pyrowave_device_create_info device_info = {};
	device_info.mtl_device = mtl;
	pyrowave_device device = nullptr;
	auto res = pyrowave_device_create(&device_info, &device);
	if (res != PYROWAVE_SUCCESS)
	{
		fprintf(stderr, "pyrowave_device_create failed: %s\n", pyrowave_result_to_string(res));
		return EXIT_FAILURE;
	}

	pyrowave_decoder_create_info decoder_info = {};
	decoder_info.device = device;
	decoder_info.width = info.width;
	decoder_info.height = info.height;
	decoder_info.chroma = info.chroma;

	pyrowave_decoder decoder = nullptr;
	res = pyrowave_decoder_create(&decoder_info, &decoder);
	if (res != PYROWAVE_SUCCESS)
	{
		fprintf(stderr, "pyrowave_decoder_create failed: %s\n", pyrowave_result_to_string(res));
		return EXIT_FAILURE;
	}

	pyrowave_gpu_buffers buffers = {};
	for (int i = 0; i < 3; i++)
		buffers.planes[i] = planes[i];

	const uint64_t frame_time_ns = uint64_t(1e9 / (fps > 0.0 ? fps : 60.0));
	size_t frame_index = 0;
	std::vector<double> decode_ms;
	std::vector<double> wall_ms;
	bool running = true;
	bool have_frame = false;

	while (running)
	{
		uint64_t frame_start = SDL_GetTicksNS();

		SDL_Event event;
		while (SDL_PollEvent(&event))
		{
			if (event.type == SDL_EVENT_QUIT)
				running = false;
			else if (event.type == SDL_EVENT_KEY_DOWN &&
			         (event.key.key == SDLK_ESCAPE || event.key.key == SDLK_Q))
				running = false;
		}

		if (!running)
			break;

		if (frame_index < info.frames.size())
		{
			auto *pool = NS::AutoreleasePool::alloc()->init();

			auto range = info.frames[frame_index++];
			pyrowave_decoder_clear(decoder);
			res = pyrowave_decoder_push_packet(decoder, info.data.data() + range.first, range.second);

			if (res != PYROWAVE_SUCCESS)
			{
				fprintf(stderr, "push_packet failed: %s\n", pyrowave_result_to_string(res));
				running = false;
			}
			else
			{
				auto *cmd = queue->commandBuffer();
				res = pyrowave_decoder_decode(decoder, cmd, &buffers);
				if (res != PYROWAVE_SUCCESS)
				{
					fprintf(stderr, "decode failed: %s\n", pyrowave_result_to_string(res));
					running = false;
				}
				else
				{
					// This command buffer came from the renderer's own queue, so
					// committing it here orders it ahead of everything SDL submits
					// for this frame. No CPU wait, and no event or fence either.
					cmd->commit();
					have_frame = true;

					if (sync_decode)
						cmd->waitUntilCompleted();

					if (report_timing)
					{
						// Only a benchmark blocks; this is not part of the normal path.
						cmd->waitUntilCompleted();
						decode_ms.push_back((cmd->GPUEndTime() - cmd->GPUStartTime()) * 1000.0);
						if (decode_ms.size() >= 400)
							running = false;
						if (cmd->error())
						{
							fprintf(stderr, "GPU error: %s\n",
							        cmd->error()->localizedDescription()->utf8String());
							running = false;
						}
					}
				}
			}

			pool->release();

			if (frame_index >= info.frames.size() && loop)
				frame_index = 0;
		}

		if (!running)
			break;

		SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
		SDL_RenderClear(renderer);
		if (have_frame)
			SDL_RenderTexture(renderer, texture, nullptr, nullptr);
		if (screenshot_path && have_frame)
		{
			SDL_Surface *shot = SDL_RenderReadPixels(renderer, nullptr);
			if (!shot || !SDL_SaveBMP(shot, screenshot_path))
			{
				fprintf(stderr, "Screenshot failed: %s\n", SDL_GetError());
				return EXIT_FAILURE;
			}
			printf("Wrote %s (%dx%d)\n", screenshot_path, shot->w, shot->h);
			SDL_DestroySurface(shot);
			running = false;
		}

		SDL_RenderPresent(renderer);

		uint64_t elapsed = SDL_GetTicksNS() - frame_start;
		if (bench_frames)
		{
			wall_ms.push_back(double(elapsed) / 1.0e6);
			if (int(wall_ms.size()) >= bench_frames)
				running = false;
		}
		else if (elapsed < frame_time_ns)
			SDL_DelayNS(frame_time_ns - elapsed);
	}

	if (!wall_ms.empty())
	{
		std::sort(wall_ms.begin(), wall_ms.end());
		printf("wall frame time over %zu frames: min %.3f  p10 %.3f  med %.3f ms  (%.0f fps at med)\n",
		       wall_ms.size(), wall_ms.front(), wall_ms[wall_ms.size() / 10],
		       wall_ms[wall_ms.size() / 2], 1000.0 / wall_ms[wall_ms.size() / 2]);
	}

	if (!decode_ms.empty())
	{
		std::sort(decode_ms.begin(), decode_ms.end());
		printf("decode GPU time over %zu frames: min %.3f  p10 %.3f  med %.3f ms\n",
		       decode_ms.size(), decode_ms.front(),
		       decode_ms[decode_ms.size() / 10], decode_ms[decode_ms.size() / 2]);
	}

	// planes, queue and mtl all belong to the renderer; only the decoder is ours.
	pyrowave_decoder_destroy(decoder);
	pyrowave_device_destroy(device);
	SDL_DestroyTexture(texture);
	SDL_DestroyRenderer(renderer);
	SDL_DestroyWindow(window);
	SDL_Quit();
	return EXIT_SUCCESS;
}
