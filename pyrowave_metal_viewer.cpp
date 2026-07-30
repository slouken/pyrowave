// Copyright (c) 2026 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT

// Minimal SDL3 viewer for the Metal decoder.
//
//   pyrowave-metal-viewer stream.bin [--loop] [--fps N]
//
// Fully zero-copy: each of the three decoded planes lives in its own IOSurface,
// the decoder's compute shaders write straight into MTLTextures wrapping those
// surfaces, and SDL's Metal renderer wraps the same surfaces to do the YCbCr to
// RGB conversion. No pixel data is ever copied or read back to the CPU.
//
// This needs the per plane IOSurface texture properties, since Metal does not
// allow an IOSurface backed texture to be an array texture, which is how SDL
// otherwise stores the two chroma planes.
//
// Both chroma modes map onto a 3 plane SDL format, which is exactly what the
// decoder emits: SDL_PIXELFORMAT_IYUV for 4:2:0 (half resolution chroma) and
// SDL_PIXELFORMAT_P408 for 4:4:4 (full resolution chroma).

#include <Metal/Metal.hpp>

#include <IOSurface/IOSurfaceRef.h>
#include <SDL3/SDL.h>

#include "pyrowave_metal.h"
#include "pyrowave_bitstream.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

void dict_set_int(CFMutableDictionaryRef dict, CFStringRef key, int value)
{
	CFNumberRef number = CFNumberCreate(kCFAllocatorDefault, kCFNumberIntType, &value);
	CFDictionarySetValue(dict, key, number);
	CFRelease(number);
}

// One single channel 8-bit surface per plane. Kept separate rather than using a
// multi plane surface so each can back its own 2D texture.
IOSurfaceRef create_plane_surface(int width, int height)
{
	CFMutableDictionaryRef props = CFDictionaryCreateMutable(kCFAllocatorDefault, 5,
	                                                         &kCFTypeDictionaryKeyCallBacks,
	                                                         &kCFTypeDictionaryValueCallBacks);
	dict_set_int(props, kIOSurfaceWidth, width);
	dict_set_int(props, kIOSurfaceHeight, height);
	dict_set_int(props, kIOSurfaceBytesPerElement, 1);
	dict_set_int(props, kIOSurfacePixelFormat, 'L008');
	IOSurfaceRef surface = IOSurfaceCreate(props);
	CFRelease(props);
	return surface;
}

MTL::Texture *create_plane_texture(MTL::Device *device, IOSurfaceRef surface, int width, int height)
{
	auto *desc = MTL::TextureDescriptor::alloc()->init();
	desc->setTextureType(MTL::TextureType2D);
	desc->setPixelFormat(MTL::PixelFormatR8Unorm);
	desc->setWidth(width);
	desc->setHeight(height);
	desc->setUsage(MTL::TextureUsageShaderWrite | MTL::TextureUsageShaderRead);
	// IOSurface backed textures cannot be private.
	desc->setStorageMode(MTL::StorageModeShared);
	auto *texture = device->newTexture(desc, surface, NS::UInteger(0));
	desc->release();
	return texture;
}
}

int main(int argc, char **argv)
{
	const char *stream_path = nullptr;
	bool loop = false;
	double fps_override = 0.0;

	for (int i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "--loop") == 0)
			loop = true;
		else if (strcmp(argv[i], "--fps") == 0 && i + 1 < argc)
			fps_override = atof(argv[++i]);
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
		fprintf(stderr, "Usage: %s stream.bin [--loop] [--fps N]\n", argv[0]);
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

	auto *mtl = MTL::CreateSystemDefaultDevice();
	if (!mtl || !pyrowave_device_is_supported(mtl))
	{
		fprintf(stderr, "No supported Metal device.\n");
		return EXIT_FAILURE;
	}

	const int chroma_width = is_420 ? info.width / 2 : info.width;
	const int chroma_height = is_420 ? info.height / 2 : info.height;

	IOSurfaceRef surfaces[3] = {};
	MTL::Texture *planes[3] = {};
	for (int i = 0; i < 3; i++)
	{
		const int w = i == 0 ? info.width : chroma_width;
		const int h = i == 0 ? info.height : chroma_height;

		surfaces[i] = create_plane_surface(w, h);
		if (!surfaces[i])
		{
			fprintf(stderr, "IOSurfaceCreate failed for plane %d.\n", i);
			return EXIT_FAILURE;
		}
		planes[i] = create_plane_texture(mtl, surfaces[i], w, h);
		if (!planes[i])
		{
			fprintf(stderr, "Failed to wrap plane %d in a Metal texture.\n", i);
			return EXIT_FAILURE;
		}
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
	// Named by component, so the decoder's Cb and Cr map straight across.
	SDL_SetPointerProperty(props, SDL_PROP_TEXTURE_CREATE_METAL_IOSURFACE_POINTER, surfaces[0]);
	SDL_SetPointerProperty(props, SDL_PROP_TEXTURE_CREATE_METAL_IOSURFACE_U_POINTER, surfaces[1]);
	SDL_SetPointerProperty(props, SDL_PROP_TEXTURE_CREATE_METAL_IOSURFACE_V_POINTER, surfaces[2]);
	SDL_Texture *texture = SDL_CreateTextureWithProperties(renderer, props);
	SDL_DestroyProperties(props);

	if (!texture)
	{
		fprintf(stderr, "SDL_CreateTextureWithProperties failed: %s\n", SDL_GetError());
		return EXIT_FAILURE;
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

	auto *queue = mtl->newCommandQueue();

	const uint64_t frame_time_ns = uint64_t(1e9 / (fps > 0.0 ? fps : 60.0));
	size_t frame_index = 0;
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
					cmd->commit();
					// SDL renders on its own command buffer, so make sure the
					// writes into the surfaces have landed before handing over.
					cmd->waitUntilCompleted();

					if (cmd->error())
					{
						fprintf(stderr, "GPU error: %s\n",
						        cmd->error()->localizedDescription()->utf8String());
						running = false;
					}
					else
					{
						have_frame = true;
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
		SDL_RenderPresent(renderer);

		uint64_t elapsed = SDL_GetTicksNS() - frame_start;
		if (elapsed < frame_time_ns)
			SDL_DelayNS(frame_time_ns - elapsed);
	}

	for (int i = 0; i < 3; i++)
	{
		planes[i]->release();
		CFRelease(surfaces[i]);
	}
	queue->release();
	pyrowave_decoder_destroy(decoder);
	pyrowave_device_destroy(device);
	mtl->release();
	SDL_DestroyTexture(texture);
	SDL_DestroyRenderer(renderer);
	SDL_DestroyWindow(window);
	SDL_Quit();
	return EXIT_SUCCESS;
}
