// Copyright (c) 2026 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT

// Encodes a Y4M file with the Metal encoder, writing the same PYROWAVE container
// pyrowave-encode produces, so the result can be fed straight to
// pyrowave-metal-decode or the Vulkan pyrowave-decode.
//
//   pyrowave-metal-encode in.y4m --output stream.bin [--size BYTES] [--frames N]
//                                [--input cpu|nv12|iosurface|nv12-iosurface]
//
// The --input modes exist to cover all four ways a frame can reach the encoder:
// planar or interleaved chroma, from host memory or from an IOSurface.

#include <Metal/Metal.hpp>

#include <IOSurface/IOSurfaceRef.h>

#include "pyrowave_metal.h"
#include "pyrowave_bitstream.hpp"
#include "yuv4mpeg.hpp"

#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

using namespace PyroWave;

namespace
{
void message_cb(void *, const char *msg)
{
	fprintf(stderr, "pyrowave: %s\n", msg);
}

enum class InputMode
{
	Cpu,
	CpuNV12,
	IOSurface,
	IOSurfaceNV12,
};

bool mode_is_nv12(InputMode mode)
{
	return mode == InputMode::CpuNV12 || mode == InputMode::IOSurfaceNV12;
}

bool mode_is_iosurface(InputMode mode)
{
	return mode == InputMode::IOSurface || mode == InputMode::IOSurfaceNV12;
}

// Same container pyrowave-encode writes: the magic, eight int32 parameters, then
// per frame a uint32 byte count followed by that frame's packets.
bool write_container_header(FILE *file, const YUV4MPEGFile &input, ChromaSubsampling chroma)
{
	if (fwrite("PYROWAVE", 1, 8, file) != 8)
		return false;

	const int32_t params[8] = {
		input.get_width(), input.get_height(), int(input.get_format()), int(chroma),
		input.is_full_range(), input.get_frame_rate_num(), input.get_frame_rate_den(),
		0 /* placeholder for unknown chroma siting */
	};

	return fwrite(params, sizeof(params), 1, file) == 1;
}

void dict_set_int(CFMutableDictionaryRef dict, const void *key, int value)
{
	CFNumberRef number = CFNumberCreate(kCFAllocatorDefault, kCFNumberIntType, &value);
	CFDictionarySetValue(dict, key, number);
	CFRelease(number);
}

IOSurfaceRef create_surface(int width, int height, int bytes_per_element, uint32_t pixel_format)
{
	CFMutableDictionaryRef props = CFDictionaryCreateMutable(kCFAllocatorDefault, 5,
	                                                         &kCFTypeDictionaryKeyCallBacks,
	                                                         &kCFTypeDictionaryValueCallBacks);
	dict_set_int(props, kIOSurfaceWidth, width);
	dict_set_int(props, kIOSurfaceHeight, height);
	dict_set_int(props, kIOSurfaceBytesPerElement, bytes_per_element);
	dict_set_int(props, kIOSurfacePixelFormat, int(pixel_format));
	IOSurfaceRef surface = IOSurfaceCreate(props);
	CFRelease(props);
	return surface;
}

// A biplanar NV12 surface, which is what a camera or a hardware decoder hands out.
IOSurfaceRef create_nv12_surface(int width, int height)
{
	CFMutableDictionaryRef plane0 = CFDictionaryCreateMutable(kCFAllocatorDefault, 4,
	                                                          &kCFTypeDictionaryKeyCallBacks,
	                                                          &kCFTypeDictionaryValueCallBacks);
	dict_set_int(plane0, kIOSurfacePlaneWidth, width);
	dict_set_int(plane0, kIOSurfacePlaneHeight, height);
	dict_set_int(plane0, kIOSurfacePlaneBytesPerElement, 1);

	CFMutableDictionaryRef plane1 = CFDictionaryCreateMutable(kCFAllocatorDefault, 4,
	                                                          &kCFTypeDictionaryKeyCallBacks,
	                                                          &kCFTypeDictionaryValueCallBacks);
	dict_set_int(plane1, kIOSurfacePlaneWidth, width / 2);
	dict_set_int(plane1, kIOSurfacePlaneHeight, height / 2);
	dict_set_int(plane1, kIOSurfacePlaneBytesPerElement, 2);

	const void *planes[] = { plane0, plane1 };
	CFArrayRef plane_array = CFArrayCreate(kCFAllocatorDefault, planes, 2, &kCFTypeArrayCallBacks);

	CFMutableDictionaryRef props = CFDictionaryCreateMutable(kCFAllocatorDefault, 5,
	                                                         &kCFTypeDictionaryKeyCallBacks,
	                                                         &kCFTypeDictionaryValueCallBacks);
	dict_set_int(props, kIOSurfaceWidth, width);
	dict_set_int(props, kIOSurfaceHeight, height);
	dict_set_int(props, kIOSurfacePixelFormat, int('420v'));
	CFDictionarySetValue(props, kIOSurfacePlaneInfo, plane_array);

	IOSurfaceRef surface = IOSurfaceCreate(props);

	CFRelease(props);
	CFRelease(plane_array);
	CFRelease(plane1);
	CFRelease(plane0);
	return surface;
}

// Copies a tightly packed plane into an IOSurface, honouring its row stride.
void upload_surface_plane(IOSurfaceRef surface, int plane, const uint8_t *src,
                          int width_in_bytes, int height)
{
	IOSurfaceLock(surface, 0, nullptr);
	auto *dst = static_cast<uint8_t *>(IOSurfaceGetBaseAddressOfPlane(surface, plane));
	const size_t stride = IOSurfaceGetBytesPerRowOfPlane(surface, plane);

	for (int y = 0; y < height; y++)
		memcpy(dst + size_t(y) * stride, src + size_t(y) * width_in_bytes, size_t(width_in_bytes));

	IOSurfaceUnlock(surface, 0, nullptr);
}

struct Frame
{
	// Y, Cb, Cr as read from the Y4M file, tightly packed.
	std::vector<uint8_t> planes[3];
	// Cb and Cr interleaved, for the NV12 modes.
	std::vector<uint8_t> chroma_interleaved;

	int width = 0, height = 0;
	int chroma_width = 0, chroma_height = 0;

	void init(int width_, int height_, bool chroma_420)
	{
		width = width_;
		height = height_;
		chroma_width = chroma_420 ? width / 2 : width;
		chroma_height = chroma_420 ? height / 2 : height;

		planes[0].resize(size_t(width) * height);
		for (int i = 1; i < 3; i++)
			planes[i].resize(size_t(chroma_width) * chroma_height);
		chroma_interleaved.resize(size_t(chroma_width) * chroma_height * 2);
	}

	bool read(YUV4MPEGFile &input)
	{
		for (auto &plane : planes)
			if (!input.read(plane.data(), plane.size()))
				return false;

		for (size_t i = 0; i < planes[1].size(); i++)
		{
			chroma_interleaved[2 * i + 0] = planes[1][i];
			chroma_interleaved[2 * i + 1] = planes[2][i];
		}

		return true;
	}
};

bool encode_cpu(pyrowave_encoder encoder, const Frame &frame, InputMode mode,
                const pyrowave_rate_control &rate_control)
{
	pyrowave_cpu_buffer buffer = {};
	buffer.width = frame.width;
	buffer.height = frame.height;

	if (mode == InputMode::CpuNV12)
	{
		buffer.format = PYROWAVE_CPU_BUFFER_FORMAT_NV12;
		buffer.data[0] = frame.planes[0].data();
		buffer.row_stride_in_bytes[0] = frame.width;
		buffer.plane_size_in_bytes[0] = frame.planes[0].size();
		buffer.data[1] = frame.chroma_interleaved.data();
		buffer.row_stride_in_bytes[1] = size_t(frame.chroma_width) * 2;
		buffer.plane_size_in_bytes[1] = frame.chroma_interleaved.size();
	}
	else
	{
		buffer.format = frame.chroma_width == frame.width ?
		                PYROWAVE_CPU_BUFFER_FORMAT_YUV444P : PYROWAVE_CPU_BUFFER_FORMAT_YUV420P;
		for (int i = 0; i < 3; i++)
		{
			buffer.data[i] = frame.planes[i].data();
			buffer.row_stride_in_bytes[i] = i == 0 ? frame.width : frame.chroma_width;
			buffer.plane_size_in_bytes[i] = frame.planes[i].size();
		}
	}

	auto res = pyrowave_encoder_encode_cpu_synchronous(encoder, &buffer, &rate_control);
	if (res != PYROWAVE_SUCCESS)
		fprintf(stderr, "encode_cpu failed: %s\n", pyrowave_result_to_string(res));
	return res == PYROWAVE_SUCCESS;
}

// Holds the surfaces for the IOSurface input modes, so they are created once.
struct SurfaceInput
{
	IOSurfaceRef surfaces[3] = {};
	pyrowave_gpu_input input = {};

	~SurfaceInput()
	{
		for (auto surface : surfaces)
			if (surface)
				CFRelease(surface);
	}

	bool init(const Frame &frame, InputMode mode)
	{
		if (mode == InputMode::IOSurfaceNV12)
		{
			surfaces[0] = create_nv12_surface(frame.width, frame.height);
			if (!surfaces[0])
			{
				fprintf(stderr, "Failed to create an NV12 IOSurface.\n");
				return false;
			}
		}
		else
		{
			for (int i = 0; i < 3; i++)
			{
				const int width = i == 0 ? frame.width : frame.chroma_width;
				const int height = i == 0 ? frame.height : frame.chroma_height;
				surfaces[i] = create_surface(width, height, 1, 'L008');
				if (!surfaces[i])
				{
					fprintf(stderr, "Failed to create IOSurface %d.\n", i);
					return false;
				}
			}
		}

		for (int i = 0; i < 3; i++)
			input.planes[i] = surfaces[i];

		return true;
	}

	void upload(const Frame &frame, InputMode mode)
	{
		if (mode == InputMode::IOSurfaceNV12)
		{
			upload_surface_plane(surfaces[0], 0, frame.planes[0].data(), frame.width, frame.height);
			upload_surface_plane(surfaces[0], 1, frame.chroma_interleaved.data(),
			                     frame.chroma_width * 2, frame.chroma_height);
		}
		else
		{
			upload_surface_plane(surfaces[0], 0, frame.planes[0].data(), frame.width, frame.height);
			for (int i = 1; i < 3; i++)
			{
				upload_surface_plane(surfaces[i], 0, frame.planes[i].data(),
				                     frame.chroma_width, frame.chroma_height);
			}
		}
	}
};

bool write_frame(FILE *file, pyrowave_encoder encoder, size_t bitstream_size,
                 std::vector<uint8_t> &scratch, std::vector<pyrowave_packet> &packets)
{
	// One packet per frame, matching pyrowave-encode's container: the boundary is
	// the whole frame budget.
	size_t num_packets = 0;
	auto res = pyrowave_encoder_compute_num_packets(encoder, bitstream_size, &num_packets);
	if (res != PYROWAVE_SUCCESS)
	{
		fprintf(stderr, "compute_num_packets failed: %s\n", pyrowave_result_to_string(res));
		return false;
	}

	packets.resize(num_packets ? num_packets : 1);
	scratch.resize(bitstream_size);

	size_t written_packets = 0;
	res = pyrowave_encoder_packetize(encoder, packets.data(), bitstream_size, &written_packets,
	                                 scratch.data(), scratch.size());
	if (res != PYROWAVE_SUCCESS)
	{
		fprintf(stderr, "packetize failed: %s\n", pyrowave_result_to_string(res));
		return false;
	}

	if (written_packets != 1)
	{
		fprintf(stderr, "Expected the frame to fit in one packet, got %zu.\n", written_packets);
		return false;
	}

	const uint32_t size = uint32_t(packets[0].size);
	if (fwrite(&size, sizeof(size), 1, file) != 1)
		return false;
	return fwrite(scratch.data() + packets[0].offset, 1, size, file) == size;
}
}

int main(int argc, char **argv)
{
	const char *input_path = nullptr;
	const char *output_path = nullptr;
	size_t bitstream_size = 200 * 1024;
	int max_frames = 0;
	InputMode mode = InputMode::Cpu;

	for (int i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "--output") == 0 && i + 1 < argc)
			output_path = argv[++i];
		else if (strcmp(argv[i], "--size") == 0 && i + 1 < argc)
			bitstream_size = strtoull(argv[++i], nullptr, 0);
		else if (strcmp(argv[i], "--frames") == 0 && i + 1 < argc)
			max_frames = atoi(argv[++i]);
		else if (strcmp(argv[i], "--input") == 0 && i + 1 < argc)
		{
			const char *value = argv[++i];
			if (strcmp(value, "cpu") == 0)
				mode = InputMode::Cpu;
			else if (strcmp(value, "nv12") == 0)
				mode = InputMode::CpuNV12;
			else if (strcmp(value, "iosurface") == 0)
				mode = InputMode::IOSurface;
			else if (strcmp(value, "nv12-iosurface") == 0)
				mode = InputMode::IOSurfaceNV12;
			else
			{
				fprintf(stderr, "Unknown input mode %s.\n", value);
				return EXIT_FAILURE;
			}
		}
		else if (argv[i][0] == '-')
		{
			fprintf(stderr, "Unknown argument %s\n", argv[i]);
			return EXIT_FAILURE;
		}
		else
			input_path = argv[i];
	}

	if (!input_path || !output_path)
	{
		fprintf(stderr, "Usage: %s in.y4m --output stream.bin [--size BYTES] [--frames N] "
		                "[--input cpu|nv12|iosurface|nv12-iosurface]\n", argv[0]);
		return EXIT_FAILURE;
	}

	YUV4MPEGFile input;
	if (!input.open_read(input_path))
	{
		fprintf(stderr, "Failed to open %s\n", input_path);
		return EXIT_FAILURE;
	}

	if (YUV4MPEGFile::format_to_bytes_per_component(input.get_format()) != 1)
	{
		fprintf(stderr, "Only 8-bit Y4M input is supported; the encoder API takes 8-bit planes.\n");
		return EXIT_FAILURE;
	}

	const bool chroma_420 = YUV4MPEGFile::format_has_subsampling(input.get_format());
	const auto chroma = chroma_420 ? ChromaSubsampling::Chroma420 : ChromaSubsampling::Chroma444;

	if (mode_is_nv12(mode) && !chroma_420)
	{
		fprintf(stderr, "NV12 input modes have half resolution chroma, so they need a 4:2:0 source.\n");
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

	pyrowave_device_create_info device_info = {};
	device_info.mtl_device = mtl;
	device_info.message_callback = message_cb;

	pyrowave_device device = nullptr;
	auto res = pyrowave_device_create(&device_info, &device);
	if (res != PYROWAVE_SUCCESS)
	{
		fprintf(stderr, "pyrowave_device_create failed: %s\n", pyrowave_result_to_string(res));
		return EXIT_FAILURE;
	}

	pyrowave_encoder_create_info encoder_info = {};
	encoder_info.device = device;
	encoder_info.width = input.get_width();
	encoder_info.height = input.get_height();
	encoder_info.chroma = chroma_420 ?
	                      PYROWAVE_CHROMA_SUBSAMPLING_420 : PYROWAVE_CHROMA_SUBSAMPLING_444;

	pyrowave_encoder encoder = nullptr;
	res = pyrowave_encoder_create(&encoder_info, &encoder);
	if (res != PYROWAVE_SUCCESS)
	{
		fprintf(stderr, "pyrowave_encoder_create failed: %s\n", pyrowave_result_to_string(res));
		return EXIT_FAILURE;
	}

	struct FileDeleter { void operator()(FILE *f) { if (f) fclose(f); } };
	std::unique_ptr<FILE, FileDeleter> out{ fopen(output_path, "wb") };
	if (!out)
	{
		fprintf(stderr, "Failed to open %s for writing.\n", output_path);
		return EXIT_FAILURE;
	}

	if (!write_container_header(out.get(), input, chroma))
	{
		fprintf(stderr, "Failed to write the container header.\n");
		return EXIT_FAILURE;
	}

	Frame frame;
	frame.init(input.get_width(), input.get_height(), chroma_420);

	SurfaceInput surfaces;
	if (mode_is_iosurface(mode) && !surfaces.init(frame, mode))
		return EXIT_FAILURE;

	pyrowave_rate_control rate_control = {};
	rate_control.maximum_bitstream_size = bitstream_size;

	std::vector<uint8_t> scratch;
	std::vector<pyrowave_packet> packets;
	int frames = 0;

	while (input.begin_frame())
	{
		if (max_frames && frames >= max_frames)
			break;

		if (!frame.read(input))
		{
			fprintf(stderr, "Truncated frame %d.\n", frames);
			break;
		}

		if (mode_is_iosurface(mode))
		{
			surfaces.upload(frame, mode);
			res = pyrowave_encoder_encode_gpu_synchronous(encoder, &surfaces.input, &rate_control);
			if (res != PYROWAVE_SUCCESS)
			{
				fprintf(stderr, "encode_gpu failed: %s\n", pyrowave_result_to_string(res));
				return EXIT_FAILURE;
			}
		}
		else if (!encode_cpu(encoder, frame, mode, rate_control))
			return EXIT_FAILURE;

		if (!write_frame(out.get(), encoder, bitstream_size, scratch, packets))
		{
			fprintf(stderr, "Failed to write frame %d.\n", frames);
			return EXIT_FAILURE;
		}

		frames++;
	}

	printf("Encoded %d frame(s) at up to %zu bytes each.\n", frames, bitstream_size);

	pyrowave_encoder_destroy(encoder);
	pyrowave_device_destroy(device);
	mtl->release();
	pool->release();

	return frames > 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
