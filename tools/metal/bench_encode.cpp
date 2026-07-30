// Copyright (c) 2026 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT

// Benchmark for the Metal encoder.
//
// Built by ./tools/metal/build.sh like the rest of these tools, which passes
// -DPYROWAVE_METAL_BENCH_HOOKS because the backend compiles the timing hooks out of
// ordinary builds:
//
//   ./tools/metal/build.sh
//   build-tools/bench_encode [iterations]
//
// Reports three things per resolution: GPU execution time, the wall clock latency
// an application actually sees from the "synchronous" encode API, and the CPU cost
// of packetize. It also measures the concurrent compute encoder against a serial
// one, which is the reason the hooks have to live inside the backend at all.
//
// Methodology, learned the hard way on this hardware: report the minimum over many
// iterations, and interleave every case and both dispatch modes within a single
// run. Absolute numbers drift by 10-15% between runs from background GPU load, so a
// single median, or an A/B measured in two separate runs, can invent a result that
// is not there.

#include <Metal/Metal.hpp>
#include <IOSurface/IOSurfaceRef.h>

#include "pyrowave_metal.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#ifndef PYROWAVE_METAL_BENCH_HOOKS
#error "Configure with -DPYROWAVE_METAL_BENCH_HOOKS=ON; without it the backend has no timing hooks."
#endif

// Defined by pyrowave_encoder_metal.cpp under the same macro. Deliberately not in
// pyrowave_metal.h: the encoder owns its command buffer and does not hand it out,
// so a caller has no other way to time the GPU work.
extern "C" double pyrowave_bench_last_gpu_ms(pyrowave_encoder encoder);

namespace
{
using Clock = std::chrono::steady_clock;

// The serial fallback is selected per encode, so both modes can be interleaved.
void set_serial_dispatch(bool serial)
{
	setenv("PYROWAVE_BENCH_SERIAL", serial ? "1" : "0", 1);
}

void message_cb(void *, const char *msg)
{
	fprintf(stderr, "pyrowave: %s\n", msg);
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

// A biplanar '420v' surface: R8 luma in plane 0, interleaved RG8 chroma in plane 1.
IOSurfaceRef make_nv12_surface(const uint8_t *luma, const uint8_t *chroma, int width, int height)
{
	CFMutableDictionaryRef plane0 = CFDictionaryCreateMutable(kCFAllocatorDefault, 3,
			&kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
	dict_set_int(plane0, kIOSurfacePlaneWidth, width);
	dict_set_int(plane0, kIOSurfacePlaneHeight, height);
	dict_set_int(plane0, kIOSurfacePlaneBytesPerElement, 1);

	CFMutableDictionaryRef plane1 = CFDictionaryCreateMutable(kCFAllocatorDefault, 3,
			&kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
	dict_set_int(plane1, kIOSurfacePlaneWidth, width / 2);
	dict_set_int(plane1, kIOSurfacePlaneHeight, height / 2);
	dict_set_int(plane1, kIOSurfacePlaneBytesPerElement, 2);

	const void *planes[] = { plane0, plane1 };
	CFArrayRef plane_array = CFArrayCreate(kCFAllocatorDefault, planes, 2, &kCFTypeArrayCallBacks);

	CFMutableDictionaryRef props = CFDictionaryCreateMutable(kCFAllocatorDefault, 4,
			&kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
	dict_set_int(props, kIOSurfaceWidth, width);
	dict_set_int(props, kIOSurfaceHeight, height);
	dict_set_int(props, kIOSurfacePixelFormat, int('420v'));
	CFDictionarySetValue(props, kIOSurfacePlaneInfo, plane_array);

	IOSurfaceRef surface = IOSurfaceCreate(props);
	CFRelease(props);
	CFRelease(plane_array);
	CFRelease(plane1);
	CFRelease(plane0);

	if (!surface)
		return nullptr;

	IOSurfaceLock(surface, 0, nullptr);
	for (int plane = 0; plane < 2; plane++)
	{
		// Both planes are `width` bytes per row: luma is width x 1 byte, chroma is
		// width/2 Cb,Cr pairs at 2 bytes each.
		const uint8_t *src = plane == 0 ? luma : chroma;
		const int rows = plane == 0 ? height : height / 2;
		auto *dst = static_cast<uint8_t *>(IOSurfaceGetBaseAddressOfPlane(surface, plane));
		const size_t stride = IOSurfaceGetBytesPerRowOfPlane(surface, plane);
		for (int y = 0; y < rows; y++)
			memcpy(dst + size_t(y) * stride, src + size_t(y) * width, size_t(width));
	}
	IOSurfaceUnlock(surface, 0, nullptr);

	return surface;
}

struct Samples
{
	std::vector<double> gpu_ms;
	std::vector<double> wall_ms;
	std::vector<double> packetize_ms;

	// Minimum, then a low quantile as a sanity check that the minimum is not a
	// lone outlier.
	double quantile(std::vector<double> &v, double q)
	{
		if (v.empty())
			return 0.0;
		std::sort(v.begin(), v.end());
		return v[size_t(q * double(v.size() - 1))];
	}
};

// How the frame reaches the encoder. NV12Surface is one biplanar '420v' surface
// whose chroma the library reads through a pair of swizzled RG8 views; the others
// are three single plane R8 surfaces, or host memory.
enum InputKind { PlanarSurface, NV12Surface, CpuPlanar, CpuNV12 };

struct Case
{
	const char *name;
	int width, height;
	pyrowave_chroma_subsampling chroma;
	InputKind input;

	pyrowave_encoder encoder = nullptr;
	IOSurfaceRef surfaces[3] = {};
	std::vector<uint8_t> planes[3];
	std::vector<uint8_t> chroma_interleaved;
	pyrowave_rate_control rate_control = {};
	std::vector<uint8_t> bitstream;
	std::vector<pyrowave_packet> packets;
	size_t frame_bytes = 0;

	// Indexed by dispatch mode: 0 concurrent, 1 serial.
	Samples samples[2];

	int chroma_width() const
	{
		return chroma == PYROWAVE_CHROMA_SUBSAMPLING_420 ? width / 2 : width;
	}

	int chroma_height() const
	{
		return chroma == PYROWAVE_CHROMA_SUBSAMPLING_420 ? height / 2 : height;
	}

	bool init(pyrowave_device device)
	{
		// Detailed but compressible, so rate control has real decisions to make
		// rather than trivially coding everything or nothing.
		for (int i = 0; i < 3; i++)
		{
			const int w = i == 0 ? width : chroma_width();
			const int h = i == 0 ? height : chroma_height();
			planes[i].resize(size_t(w) * h);
			for (int y = 0; y < h; y++)
			{
				for (int x = 0; x < w; x++)
				{
					const double v = 128.0 + 60.0 * std::sin(x * 0.05 + i) * std::cos(y * 0.04) +
					                 25.0 * std::sin(x * 0.9 + y * 0.7);
					planes[i][size_t(y) * w + x] = uint8_t(std::min(255.0, std::max(0.0, v)));
				}
			}

		}

		// Interleaved chroma for the NV12 paths.
		chroma_interleaved.resize(planes[1].size() * 2);
		for (size_t i = 0; i < planes[1].size(); i++)
		{
			chroma_interleaved[2 * i + 0] = planes[1][i];
			chroma_interleaved[2 * i + 1] = planes[2][i];
		}

		if (input == NV12Surface)
		{
			surfaces[0] = make_nv12_surface(planes[0].data(), chroma_interleaved.data(),
			                                width, height);
			if (!surfaces[0])
			{
				fprintf(stderr, "%s: NV12 IOSurface creation failed\n", name);
				return false;
			}
		}
		else if (input == PlanarSurface)
		{
			for (int i = 0; i < 3; i++)
			{
				surfaces[i] = make_surface(planes[i].data(), i == 0 ? width : chroma_width(),
				                           i == 0 ? height : chroma_height());
				if (!surfaces[i])
				{
					fprintf(stderr, "%s: IOSurface %d creation failed\n", name, i);
					return false;
				}
			}
		}

		// Roughly one bit per luma pixel, a plausible streaming target.
		rate_control.maximum_bitstream_size = size_t(width) * height / 8;
		bitstream.resize(rate_control.maximum_bitstream_size);

		pyrowave_encoder_create_info info = {};
		info.device = device;
		info.width = width;
		info.height = height;
		info.chroma = chroma;

		if (pyrowave_encoder_create(&info, &encoder) != PYROWAVE_SUCCESS)
		{
			fprintf(stderr, "%s: encoder create failed\n", name);
			return false;
		}

		return true;
	}

	void deinit()
	{
		if (encoder)
			pyrowave_encoder_destroy(encoder);
		for (auto surface : surfaces)
			if (surface)
				CFRelease(surface);
	}

	bool iterate(int mode, bool collect)
	{
		set_serial_dispatch(mode != 0);

		const auto t0 = Clock::now();

		pyrowave_result res;
		if (input == CpuPlanar || input == CpuNV12)
		{
			pyrowave_cpu_buffer buffer = {};
			buffer.width = width;
			buffer.height = height;

			if (input == CpuNV12)
			{
				buffer.format = PYROWAVE_CPU_BUFFER_FORMAT_NV12;
				buffer.data[0] = planes[0].data();
				buffer.row_stride_in_bytes[0] = size_t(width);
				buffer.plane_size_in_bytes[0] = planes[0].size();
				buffer.data[1] = chroma_interleaved.data();
				buffer.row_stride_in_bytes[1] = size_t(chroma_width()) * 2;
				buffer.plane_size_in_bytes[1] = chroma_interleaved.size();
			}
			else
			{
				buffer.format = chroma == PYROWAVE_CHROMA_SUBSAMPLING_420 ?
				                PYROWAVE_CPU_BUFFER_FORMAT_YUV420P :
				                PYROWAVE_CPU_BUFFER_FORMAT_YUV444P;
				for (int i = 0; i < 3; i++)
				{
					buffer.data[i] = planes[i].data();
					buffer.row_stride_in_bytes[i] = size_t(i == 0 ? width : chroma_width());
					buffer.plane_size_in_bytes[i] = planes[i].size();
				}
			}
			res = pyrowave_encoder_encode_cpu_synchronous(encoder, &buffer, &rate_control);
		}
		else
		{
			pyrowave_gpu_input gpu = {};
			// The NV12 case leaves planes[1] and [2] NULL on purpose.
			gpu.planes[0] = surfaces[0];
			if (input == PlanarSurface)
			{
				gpu.planes[1] = surfaces[1];
				gpu.planes[2] = surfaces[2];
			}
			res = pyrowave_encoder_encode_gpu_synchronous(encoder, &gpu, &rate_control);
		}

		if (res != PYROWAVE_SUCCESS)
			return false;

		// Blocks on the command buffer, so the GPU time lands inside this call.
		size_t num_packets = 0;
		if (pyrowave_encoder_compute_num_packets(encoder, PacketBoundary, &num_packets) !=
		    PYROWAVE_SUCCESS)
			return false;

		const auto t1 = Clock::now();

		packets.resize(num_packets + 1);
		size_t written = 0;
		if (pyrowave_encoder_packetize(encoder, packets.data(), PacketBoundary, &written,
		                               bitstream.data(), bitstream.size()) != PYROWAVE_SUCCESS)
			return false;

		const auto t2 = Clock::now();

		if (collect)
		{
			auto &s = samples[mode];
			s.gpu_ms.push_back(pyrowave_bench_last_gpu_ms(encoder));
			s.wall_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
			s.packetize_ms.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count());

			frame_bytes = 0;
			for (size_t i = 0; i < written; i++)
				frame_bytes += packets[i].size;
		}

		return true;
	}

	static constexpr size_t PacketBoundary = 1200;
};
}

int main(int argc, char **argv)
{
	int iterations = 400;
	if (argc > 1)
		iterations = atoi(argv[1]);
	if (iterations <= 0)
	{
		fprintf(stderr, "Usage: %s [iterations]\n", argv[0]);
		return EXIT_FAILURE;
	}

	auto *pool = NS::AutoreleasePool::alloc()->init();

	auto *mtl = MTL::CreateSystemDefaultDevice();
	if (!mtl)
	{
		fprintf(stderr, "No Metal device.\n");
		return EXIT_FAILURE;
	}

	printf("Device: %s\n", mtl->name()->utf8String());
	printf("%d iterations, all cases and both dispatch modes interleaved.\n\n", iterations);

	pyrowave_device_create_info dinfo = {};
	dinfo.mtl_device = mtl;
	dinfo.message_callback = message_cb;

	pyrowave_device device = nullptr;
	if (pyrowave_device_create(&dinfo, &device) != PYROWAVE_SUCCESS)
	{
		fprintf(stderr, "pyrowave_device_create failed.\n");
		return EXIT_FAILURE;
	}

	std::vector<Case> cases = {
		{ "640x480 420",   640,  480,  PYROWAVE_CHROMA_SUBSAMPLING_420, PlanarSurface },
		{ "1280x720 420",  1280, 720,  PYROWAVE_CHROMA_SUBSAMPLING_420, PlanarSurface },
		{ "1920x1080 420", 1920, 1080, PYROWAVE_CHROMA_SUBSAMPLING_420, PlanarSurface },
		{ "1920x1080 444", 1920, 1080, PYROWAVE_CHROMA_SUBSAMPLING_444, PlanarSurface },
		{ "3840x2160 420", 3840, 2160, PYROWAVE_CHROMA_SUBSAMPLING_420, PlanarSurface },
		{ "3840x2160 444", 3840, 2160, PYROWAVE_CHROMA_SUBSAMPLING_444, PlanarSurface },
		// The remaining cases repeat a geometry already above so the input paths can be
		// priced against each other rather than against nothing. Biplanar NV12 reads
		// chroma through swizzled RG8 views instead of two R8 textures, which is the
		// difference worth measuring.
		{ "640x480 420 nv12-surf",   640,  480,  PYROWAVE_CHROMA_SUBSAMPLING_420, NV12Surface },
		{ "1920x1080 420 nv12-surf", 1920, 1080, PYROWAVE_CHROMA_SUBSAMPLING_420, NV12Surface },
		{ "3840x2160 420 nv12-surf", 3840, 2160, PYROWAVE_CHROMA_SUBSAMPLING_420, NV12Surface },
		{ "1920x1080 420 cpu",       1920, 1080, PYROWAVE_CHROMA_SUBSAMPLING_420, CpuPlanar },
		{ "1920x1080 420 cpu-nv12",  1920, 1080, PYROWAVE_CHROMA_SUBSAMPLING_420, CpuNV12 },
	};

	for (auto &c : cases)
		if (!c.init(device))
			return EXIT_FAILURE;

	// The first encodes pay lazy pipeline compilation and first-touch page faults.
	for (auto &c : cases)
		for (int mode = 0; mode < 2; mode++)
			for (int i = 0; i < 20; i++)
				if (!c.iterate(mode, false))
				{
					fprintf(stderr, "%s: warmup failed\n", c.name);
					return EXIT_FAILURE;
				}

	for (int round = 0; round < iterations; round++)
		for (auto &c : cases)
			for (int mode = 0; mode < 2; mode++)
				if (!c.iterate(mode, true))
				{
					fprintf(stderr, "%s: iteration failed\n", c.name);
					return EXIT_FAILURE;
				}

	printf("%-26s %8s %8s %8s %8s %10s %9s\n",
	       "case", "gpu min", "gpu p10", "wall min", "wall p10", "packetize", "KB/frame");
	for (auto &c : cases)
	{
		auto &s = c.samples[0];
		printf("%-26s %8.3f %8.3f %8.3f %8.3f %10.3f %9.1f\n",
		       c.name,
		       s.quantile(s.gpu_ms, 0.0), s.quantile(s.gpu_ms, 0.10),
		       s.quantile(s.wall_ms, 0.0), s.quantile(s.wall_ms, 0.10),
		       s.quantile(s.packetize_ms, 0.5), double(c.frame_bytes) / 1024.0);
	}

	printf("\n  gpu       MTLCommandBuffer GPUEndTime - GPUStartTime\n"
	       "  wall      submit + GPU + block until the result is readable\n"
	       "  packetize CPU only, median\n");

	printf("\nDispatch mode, GPU minimum:\n");
	printf("%-26s %10s %12s %9s\n", "case", "serial", "concurrent", "speedup");
	for (auto &c : cases)
	{
		const double concurrent = c.samples[0].quantile(c.samples[0].gpu_ms, 0.0);
		const double serial = c.samples[1].quantile(c.samples[1].gpu_ms, 0.0);
		printf("%-26s %10.3f %12.3f %8.2fx\n", c.name, serial, concurrent,
		       concurrent > 0.0 ? serial / concurrent : 0.0);
	}

	printf("\nThe concurrent encoder wins most where the ~170 dispatches are individually\n"
	       "too small to fill the GPU, and tapers off as they grow.\n");

	for (auto &c : cases)
		c.deinit();

	pyrowave_device_destroy(device);
	mtl->release();
	pool->release();
	return EXIT_SUCCESS;
}
