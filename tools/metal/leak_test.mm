// Copyright (c) 2026 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT

// Long running encode + decode loop that watches the process footprint.
//
//   build-tools/leak_test [iterations] [--no-pool]
//
// Every iteration encodes a frame, packetizes it, feeds the packets back through the
// decoder and decodes. Resident footprint is sampled as it goes; a flat trend means
// nothing is accumulating. Worth running under macOS's own detector too:
//
//   leaks --atExit -- build-tools/leak_test 400
//
// By default each iteration runs inside its own @autoreleasepool, which is what a
// frame loop in a real application looks like. --no-pool removes it, so anything the
// library autoreleases without draining shows up as growth; that distinguishes an
// accumulation inside PyroWave from one belonging to the caller.

#import <Metal/Metal.h>
#import <IOSurface/IOSurfaceRef.h>

#include "pyrowave_metal.h"

#include <cmath>
#include <libproc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

namespace
{
// What Activity Monitor calls "memory": the same figure jetsam looks at.
double footprint_mb()
{
	rusage_info_current info = {};
	if (proc_pid_rusage(getpid(), RUSAGE_INFO_CURRENT, (rusage_info_t *)&info) != 0)
		return 0.0;
	return double(info.ri_phys_footprint) / (1024.0 * 1024.0);
}

void dict_set_int(CFMutableDictionaryRef dict, const void *key, int value)
{
	CFNumberRef number = CFNumberCreate(kCFAllocatorDefault, kCFNumberIntType, &value);
	CFDictionarySetValue(dict, key, number);
	CFRelease(number);
}

IOSurfaceRef make_surface(const uint8_t *src, int width, int height)
{
	CFMutableDictionaryRef props = CFDictionaryCreateMutable(kCFAllocatorDefault, 4,
			&kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
	dict_set_int(props, kIOSurfaceWidth, width);
	dict_set_int(props, kIOSurfaceHeight, height);
	dict_set_int(props, kIOSurfaceBytesPerElement, 1);
	dict_set_int(props, kIOSurfacePixelFormat, int('L008'));
	IOSurfaceRef surface = IOSurfaceCreate(props);
	CFRelease(props);

	IOSurfaceLock(surface, 0, nullptr);
	auto *dst = static_cast<uint8_t *>(IOSurfaceGetBaseAddress(surface));
	const size_t stride = IOSurfaceGetBytesPerRow(surface);
	for (int y = 0; y < height; y++)
		memcpy(dst + size_t(y) * stride, src + size_t(y) * width, size_t(width));
	IOSurfaceUnlock(surface, 0, nullptr);
	return surface;
}

id<MTLTexture> make_plane(id<MTLDevice> dev, int w, int h)
{
	MTLTextureDescriptor *desc = [MTLTextureDescriptor new];
	desc.textureType = MTLTextureType2D;
	desc.pixelFormat = MTLPixelFormatR8Unorm;
	desc.width = w;
	desc.height = h;
	desc.usage = MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead;
	desc.storageMode = MTLStorageModePrivate;
	return [dev newTextureWithDescriptor:desc];
}

void message_cb(void *, const char *msg) { fprintf(stderr, "pyrowave: %s\n", msg); }
}

int main(int argc, char **argv)
{
	int iterations = 400;
	bool use_pool = true;
	for (int i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "--no-pool") == 0)
			use_pool = false;
		else
			iterations = atoi(argv[i]);
	}

	constexpr int W = 1920, H = 1080;
	const int cw = W / 2, ch = H / 2;

	id<MTLDevice> mtl = MTLCreateSystemDefaultDevice();
	pyrowave_device_create_info dinfo = {};
	dinfo.mtl_device = (__bridge void *)mtl;
	dinfo.message_callback = message_cb;
	pyrowave_device device = nullptr;
	if (pyrowave_device_create(&dinfo, &device) != PYROWAVE_SUCCESS)
		return 1;

	pyrowave_encoder_create_info einfo = {};
	einfo.device = device;
	einfo.width = W;
	einfo.height = H;
	einfo.chroma = PYROWAVE_CHROMA_SUBSAMPLING_420;
	pyrowave_encoder encoder = nullptr;
	if (pyrowave_encoder_create(&einfo, &encoder) != PYROWAVE_SUCCESS)
		return 1;

	pyrowave_decoder_create_info cinfo = {};
	cinfo.device = device;
	cinfo.width = W;
	cinfo.height = H;
	cinfo.chroma = PYROWAVE_CHROMA_SUBSAMPLING_420;
	pyrowave_decoder decoder = nullptr;
	if (pyrowave_decoder_create(&cinfo, &decoder) != PYROWAVE_SUCCESS)
		return 1;

	std::vector<uint8_t> planes[3];
	IOSurfaceRef surf[3];
	for (int i = 0; i < 3; i++)
	{
		const int w = i ? cw : W, h = i ? ch : H;
		planes[i].resize(size_t(w) * h);
		for (int y = 0; y < h; y++)
			for (int x = 0; x < w; x++)
				planes[i][size_t(y) * w + x] = uint8_t(std::fmin(255.0, std::fmax(0.0,
						128.0 + 70.0 * std::sin(x * 0.013 + i) * std::cos(y * 0.011))));
		surf[i] = make_surface(planes[i].data(), w, h);
	}

	pyrowave_gpu_input input = {};
	for (int i = 0; i < 3; i++)
		input.planes[i] = surf[i];
	pyrowave_rate_control rc = {};
	rc.maximum_bitstream_size = 250000;

	id<MTLTexture> out[3] = { make_plane(mtl, W, H), make_plane(mtl, cw, ch), make_plane(mtl, cw, ch) };
	pyrowave_gpu_buffers bufs = {};
	for (int i = 0; i < 3; i++)
		bufs.planes[i] = (__bridge void *)out[i];

	id<MTLCommandQueue> queue = [mtl newCommandQueue];
	std::vector<uint8_t> bitstream(rc.maximum_bitstream_size);
	std::vector<pyrowave_packet> packets;

	printf("1920x1080 420, %d iterations, per-iteration pool: %s\n",
	       iterations, use_pool ? "yes" : "NO");

	double first = 0.0, last = 0.0;
	for (int i = 0; i < iterations; i++)
	{
		auto body = [&]() {
			if (pyrowave_encoder_encode_gpu_synchronous(encoder, &input, &rc) != PYROWAVE_SUCCESS)
				exit(1);
			size_t n = 0;
			if (pyrowave_encoder_compute_num_packets(encoder, 1200, &n) != PYROWAVE_SUCCESS)
				exit(1);
			packets.resize(n + 1);
			size_t written = 0;
			if (pyrowave_encoder_packetize(encoder, packets.data(), 1200, &written,
			                               bitstream.data(), bitstream.size()) != PYROWAVE_SUCCESS)
				exit(1);

			pyrowave_decoder_clear(decoder);
			for (size_t p = 0; p < written; p++)
				pyrowave_decoder_push_packet(decoder, bitstream.data() + packets[p].offset,
				                             packets[p].size);

			id<MTLCommandBuffer> cmd = [queue commandBuffer];
			if (pyrowave_decoder_decode(decoder, (__bridge void *)cmd, &bufs) != PYROWAVE_SUCCESS)
				exit(1);
			[cmd commit];
			[cmd waitUntilCompleted];
		};

		if (use_pool)
			@autoreleasepool { body(); }
		else
			body();

		// Sample after a warmup so one-off pipeline and buffer allocation is excluded.
		if (i == iterations / 10)
			first = footprint_mb();
		if ((i + 1) % (iterations / 8 ? iterations / 8 : 1) == 0)
		{
			last = footprint_mb();
			printf("  iter %5d  footprint %7.1f MB\n", i + 1, last);
		}
	}

	const double growth = last - first;
	printf("\nfootprint after warmup %.1f MB -> %.1f MB, growth %+.1f MB over %d iterations\n",
	       first, last, growth, iterations);
	// A real leak of command buffers or encoders at 1080p would run to tens of MB
	// across a few hundred frames, so a couple of MB of allocator slack is not it.
	const bool ok = growth < 8.0;
	printf("%s\n", ok ? "PASS: footprint is flat" : "FAIL: footprint is growing");

	pyrowave_decoder_destroy(decoder);
	pyrowave_encoder_destroy(encoder);
	pyrowave_device_destroy(device);
	for (int i = 0; i < 3; i++)
		CFRelease(surf[i]);
	return ok ? 0 : 1;
}
