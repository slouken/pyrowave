// Copyright (c) 2026 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT

// Metal backend for the PyroWave decoder. Mirrors the compute path of
// pyrowave_decoder.cpp; the fragment iDWT path and the encoder are not ported.
//
// This is the single translation unit that instantiates metal-cpp.

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Metal/Metal.hpp>

#include "pyrowave_metal.h"
#include "pyrowave_bitstream.hpp"
#include "shaders/metal/pyrowave_msl.h"

#include <atomic>
#include <memory>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <vector>

using namespace PyroWave;

namespace
{
// Push constant layouts. These must match the Registers structs SPIRV-Cross
// emitted into shaders/metal/*.metal. MSL gives int2 an 8 byte alignment, so
// the dequant struct is padded out to 24 bytes even though only 20 are used.
struct DequantPush
{
	int32_t resolution[2];
	int32_t output_layer;
	int32_t block_offset_32x32;
	int32_t block_stride_32x32;
	int32_t padding;
};
static_assert(sizeof(DequantPush) == 24, "DequantPush layout mismatch.");

struct IdwtPush
{
	int32_t resolution[2];
	float inv_resolution[2];
};
static_assert(sizeof(IdwtPush) == 16, "IdwtPush layout mismatch.");

// Threadgroup sizes are baked into the shaders (local_size_x).
constexpr uint32_t DequantThreadgroupSize = 128;
constexpr uint32_t IdwtThreadgroupSize = 64;

// Same meaning and same env var as the Vulkan build:
//   2  FP32 storage and math
//   1  (default) FP16 storage, FP32 lifting math, half sized threadgroup array
//   0  FP16 throughout
//
// 1 is the default because it is both the fastest and essentially free: on an
// M1 it decodes 1080p roughly 1.4x quicker than FP32 while staying within 1 LSB
// of the Vulkan decoder. The win is the halved memory traffic through the R16F
// wavelet pyramid, which is what the decode is bound by above 480p -- 0 is
// actually slower than 1 despite the half precision arithmetic, and costs
// accuracy, so it is not recommended.
//
// Note 1 is not quite identical to the Vulkan build's: that one additionally
// splits the low frequency bands into a separate FP32 image, which is not
// implemented here.
constexpr int DefaultPrecision = 1;

int requested_precision()
{
	const char *env = getenv("PYROWAVE_PRECISION");
	if (!env)
		return DefaultPrecision;
	int precision = atoi(env);
	return (precision < 0 || precision > 2) ? DefaultPrecision : precision;
}

MTL::PixelFormat wavelet_format(int precision)
{
	return precision == 2 ? MTL::PixelFormatR32Float : MTL::PixelFormatR16Float;
}

const char *idwt_source_for(int precision)
{
	switch (precision)
	{
	case 0: return idwt_fp16_msl_source;
	case 1: return idwt_fp16_storage_msl_source;
	default: return idwt_msl_source;
	}
}

// A pair of upload buffers. The GPU may still be reading a previous frame's
// buffers, so decode cycles through slots and only reuses one once its command
// buffer has completed.
struct UploadSlot
{
	MTL::Buffer *offsets = nullptr;
	MTL::Buffer *payload = nullptr;
	std::atomic<bool> in_flight{false};

	~UploadSlot()
	{
		if (offsets)
			offsets->release();
		if (payload)
			payload->release();
	}
};

const char *result_string(pyrowave_result result)
{
	switch (result)
	{
	case PYROWAVE_SUCCESS: return "success";
	case PYROWAVE_ERROR_GENERIC: return "generic error";
	case PYROWAVE_ERROR_INVALID_ARGUMENT: return "invalid argument";
	case PYROWAVE_ERROR_OUT_OF_HOST_MEMORY: return "out of host memory";
	case PYROWAVE_ERROR_OUT_OF_DEVICE_MEMORY: return "out of device memory";
	case PYROWAVE_ERROR_UNSUPPORTED_DEVICE: return "unsupported device";
	case PYROWAVE_ERROR_SHADER_COMPILATION: return "shader compilation failed";
	case PYROWAVE_ERROR_CORRUPT_BITSTREAM: return "corrupt bitstream";
	default: return "unknown error";
	}
}
}

struct pyrowave_device_opaque
{
	MTL::Device *mtl = nullptr;
	MTL::ComputePipelineState *dequant_pipeline = nullptr;
	// Indexed by the DCShift function constant.
	MTL::ComputePipelineState *idwt_pipeline[2] = {};
	MTL::SamplerState *mirror_repeat_sampler = nullptr;

	pyrowave_message_cb message_cb = nullptr;
	void *message_userdata = nullptr;
	int precision = DefaultPrecision;

	void log(const char *fmt, ...) const __attribute__((format(printf, 2, 3)));

	~pyrowave_device_opaque()
	{
		for (auto *pipeline : idwt_pipeline)
			if (pipeline)
				pipeline->release();
		if (dequant_pipeline)
			dequant_pipeline->release();
		if (mirror_repeat_sampler)
			mirror_repeat_sampler->release();
		if (mtl)
			mtl->release();
	}
};

void pyrowave_device_opaque::log(const char *fmt, ...) const
{
	char buffer[512];
	va_list args;
	va_start(args, fmt);
	vsnprintf(buffer, sizeof(buffer), fmt, args);
	va_end(args);

	if (message_cb)
		message_cb(message_userdata, buffer);
	else
		fprintf(stderr, "pyrowave: %s\n", buffer);
}

struct pyrowave_decoder_opaque
{
	pyrowave_device device = nullptr;

	BlockLayout layout;
	BitstreamParser parser;

	// 2D array, NumFrequencyBandsPerLevel * NumComponents layers, DecompositionLevels mips.
	MTL::Texture *wavelet = nullptr;
	// 4 layer array views used by dequant (write) and iDWT (sample).
	MTL::Texture *component_layer_views[NumComponents][DecompositionLevels] = {};
	// Single layer views of band 0 (LL), used as iDWT output.
	MTL::Texture *component_ll_views[NumComponents][DecompositionLevels] = {};

	std::vector<std::unique_ptr<UploadSlot>> upload_slots;

	~pyrowave_decoder_opaque()
	{
		for (auto &views : component_layer_views)
			for (auto *view : views)
				if (view)
					view->release();
		for (auto &views : component_ll_views)
			for (auto *view : views)
				if (view)
					view->release();
		if (wavelet)
			wavelet->release();
	}
};

namespace
{
bool device_is_supported(MTL::Device *mtl)
{
	if (!mtl)
		return false;
	// Apple7 (M1 / A14) and up. This guarantees a 32 wide SIMD group, which the
	// dequant shader's subgroup fast path depends on.
	if (!mtl->supportsFamily(MTL::GPUFamilyApple7))
		return false;
	if (mtl->maxThreadsPerThreadgroup().width < DequantThreadgroupSize)
		return false;
	return true;
}

MTL::Library *compile_library(pyrowave_device device, const char *source, const char *label)
{
	NS::Error *error = nullptr;
	auto *options = MTL::CompileOptions::alloc()->init();
	// Default (fast) math is kept deliberately. MathModeSafe was measured and does
	// not reduce the residual ~1 LSB disagreement with the Vulkan decoder, so the
	// difference does not come from FMA contraction or reassociation.
	auto *string = NS::String::string(source, NS::UTF8StringEncoding);
	auto *library = device->mtl->newLibrary(string, options, &error);
	options->release();

	if (!library)
	{
		device->log("Failed to compile %s: %s", label,
		            error && error->localizedDescription() ?
		            error->localizedDescription()->utf8String() : "unknown error");
	}

	return library;
}

MTL::ComputePipelineState *create_pipeline(pyrowave_device device, MTL::Library *library,
                                           const char *entry_point, const bool *dc_shift,
                                           uint32_t required_threads)
{
	NS::Error *error = nullptr;
	auto *name = NS::String::string(entry_point, NS::UTF8StringEncoding);

	MTL::Function *function;
	if (dc_shift)
	{
		auto *constants = MTL::FunctionConstantValues::alloc()->init();
		constants->setConstantValue(dc_shift, MTL::DataTypeBool, NS::UInteger(0));
		function = library->newFunction(name, constants, &error);
		constants->release();
	}
	else
	{
		function = library->newFunction(name);
	}

	if (!function)
	{
		device->log("Failed to look up %s: %s", entry_point,
		            error && error->localizedDescription() ?
		            error->localizedDescription()->utf8String() : "not found");
		return nullptr;
	}

	auto *pipeline = device->mtl->newComputePipelineState(function, &error);
	function->release();

	if (!pipeline)
	{
		device->log("Failed to create pipeline for %s: %s", entry_point,
		            error && error->localizedDescription() ?
		            error->localizedDescription()->utf8String() : "unknown error");
		return nullptr;
	}

	if (pipeline->maxTotalThreadsPerThreadgroup() < required_threads)
	{
		device->log("%s only supports %u threads per threadgroup, needs %u.",
		            entry_point, unsigned(pipeline->maxTotalThreadsPerThreadgroup()), required_threads);
		pipeline->release();
		return nullptr;
	}

	return pipeline;
}

bool create_wavelet_resources(pyrowave_decoder decoder)
{
	auto &layout = decoder->layout;
	auto *mtl = decoder->device->mtl;

	auto *desc = MTL::TextureDescriptor::alloc()->init();
	desc->setTextureType(MTL::TextureType2DArray);
	desc->setPixelFormat(wavelet_format(decoder->device->precision));
	desc->setWidth(layout.aligned_width / 2);
	desc->setHeight(layout.aligned_height / 2);
	desc->setArrayLength(NumFrequencyBandsPerLevel * NumComponents);
	desc->setMipmapLevelCount(DecompositionLevels);
	// PixelFormatView is required to take the per component/per level views below.
	desc->setUsage(MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite |
	               MTL::TextureUsagePixelFormatView);
	desc->setStorageMode(MTL::StorageModePrivate);

	decoder->wavelet = mtl->newTexture(desc);
	desc->release();

	if (!decoder->wavelet)
	{
		decoder->device->log("Failed to allocate wavelet texture.");
		return false;
	}

	decoder->wavelet->setLabel(NS::String::string("pyrowave-wavelet", NS::UTF8StringEncoding));

	for (int level = 0; level < DecompositionLevels; level++)
	{
		for (int component = 0; component < NumComponents; component++)
		{
			decoder->component_layer_views[component][level] =
					decoder->wavelet->newTextureView(
							wavelet_format(decoder->device->precision), MTL::TextureType2DArray,
							NS::Range(level, 1),
							NS::Range(NumFrequencyBandsPerLevel * component, NumFrequencyBandsPerLevel));

			decoder->component_ll_views[component][level] =
					decoder->wavelet->newTextureView(
							wavelet_format(decoder->device->precision), MTL::TextureType2D,
							NS::Range(level, 1),
							NS::Range(NumFrequencyBandsPerLevel * component, 1));

			if (!decoder->component_layer_views[component][level] ||
			    !decoder->component_ll_views[component][level])
			{
				decoder->device->log("Failed to create wavelet texture views.");
				return false;
			}
		}
	}

	return true;
}

// Grows a shared storage buffer to at least `size` bytes, keeping some slack so
// that a steadily sized stream stops reallocating.
bool ensure_buffer(pyrowave_device device, MTL::Buffer **buffer, size_t size)
{
	if (*buffer && (*buffer)->length() >= size)
		return true;

	if (*buffer)
		(*buffer)->release();

	size_t allocate = size * 2;
	if (allocate < 64 * 1024)
		allocate = 64 * 1024;

	*buffer = device->mtl->newBuffer(allocate, MTL::ResourceStorageModeShared);
	if (!*buffer)
	{
		device->log("Failed to allocate a %zu byte upload buffer.", allocate);
		return false;
	}

	return true;
}

UploadSlot *acquire_upload_slot(pyrowave_decoder decoder, size_t offsets_size, size_t payload_size)
{
	UploadSlot *slot = nullptr;

	for (auto &candidate : decoder->upload_slots)
	{
		if (!candidate->in_flight.load(std::memory_order_acquire))
		{
			slot = candidate.get();
			break;
		}
	}

	if (!slot)
	{
		decoder->upload_slots.emplace_back(new UploadSlot);
		slot = decoder->upload_slots.back().get();
	}

	if (!ensure_buffer(decoder->device, &slot->offsets, offsets_size) ||
	    !ensure_buffer(decoder->device, &slot->payload, payload_size))
		return nullptr;

	return slot;
}

void encode_dequant(pyrowave_decoder decoder, MTL::ComputeCommandEncoder *enc, UploadSlot *slot)
{
	auto &layout = decoder->layout;

	enc->setComputePipelineState(decoder->device->dequant_pipeline);
	// The u8/u16/u32 aliases of the payload collapse into a single binding in MSL.
	enc->setBuffer(slot->payload, 0, 0);
	enc->setBuffer(slot->offsets, 0, 2);

	for (int level = 0; level < DecompositionLevels; level++)
	{
		for (int component = 0; component < NumComponents; component++)
		{
			// Ignore top-level CbCr when doing 420 subsampling.
			if (level == 0 && component != 0 && layout.chroma == ChromaSubsampling::Chroma420)
				continue;

			enc->setTexture(decoder->component_layer_views[component][level], 0);

			for (int band = (level == DecompositionLevels - 1 ? 0 : 1); band < 4; band++)
			{
				DequantPush push = {};
				push.resolution[0] = layout.level_width(level);
				push.resolution[1] = layout.level_height(level);
				push.output_layer = band;
				push.block_offset_32x32 = layout.block_meta[component][level][band].block_offset_32x32;
				push.block_stride_32x32 = layout.block_meta[component][level][band].block_stride_32x32;
				enc->setBytes(&push, sizeof(push), 1);

				enc->dispatchThreadgroups(
						MTL::Size((push.resolution[0] + 31) / 32, (push.resolution[1] + 31) / 32, 1),
						MTL::Size(DequantThreadgroupSize, 1, 1));
			}
		}
	}
}

void encode_idwt_dispatch(pyrowave_decoder decoder, MTL::ComputeCommandEncoder *enc,
                          const IdwtPush &push, MTL::Texture *input, MTL::Texture *output,
                          bool dc_shift)
{
	enc->setComputePipelineState(decoder->device->idwt_pipeline[dc_shift ? 1 : 0]);
	enc->setBytes(&push, sizeof(push), 0);
	enc->setTexture(input, 0);
	enc->setTexture(output, 1);
	enc->setSamplerState(decoder->device->mirror_repeat_sampler, 0);
	enc->dispatchThreadgroups(
			MTL::Size((push.resolution[0] + 15) / 16, (push.resolution[1] + 15) / 16, 1),
			MTL::Size(IdwtThreadgroupSize, 1, 1));
}

void encode_idwt(pyrowave_decoder decoder, MTL::ComputeCommandEncoder *enc,
                 MTL::Texture *const planes[3])
{
	auto &layout = decoder->layout;
	const bool chroma_420 = layout.chroma == ChromaSubsampling::Chroma420;

	for (int input_level = DecompositionLevels - 1; input_level >= 0; input_level--)
	{
		// Levels are a dependent chain: this one reads the LL band the previous
		// one produced. Within a level the three components are independent, so
		// the encoder runs concurrently and only the level boundaries barrier.
		if (input_level != DecompositionLevels - 1)
			enc->memoryBarrier(MTL::BarrierScopeTextures);

		IdwtPush push = {};
		// The shader transposes on load, so resolution is swapped here.
		push.resolution[0] = layout.level_height(input_level);
		push.resolution[1] = layout.level_width(input_level);
		push.inv_resolution[0] = 1.0f / float(push.resolution[0]);
		push.inv_resolution[1] = 1.0f / float(push.resolution[1]);

		if (input_level == 0)
		{
			// Final level writes the output planes directly. Under 420 the chroma
			// planes were already finished one level earlier.
			const int components = chroma_420 ? 1 : NumComponents;
			for (int c = 0; c < components; c++)
			{
				encode_idwt_dispatch(decoder, enc, push,
				                     decoder->component_layer_views[c][input_level],
				                     planes[c], true);
			}
		}
		else
		{
			for (int c = 0; c < NumComponents; c++)
			{
				const bool final_chroma = chroma_420 && c != 0 && input_level == 1;
				MTL::Texture *output = final_chroma ?
				                       planes[c] :
				                       decoder->component_ll_views[c][input_level - 1];

				encode_idwt_dispatch(decoder, enc, push,
				                     decoder->component_layer_views[c][input_level],
				                     output, final_chroma);
			}
		}
	}
}

// Checks a caller supplied output plane against what the shaders will write.
bool validate_plane(pyrowave_device device, MTL::Texture *texture, int index, int width, int height)
{
	if (!texture)
	{
		device->log("Output plane %d is NULL.", index);
		return false;
	}

	if (texture->textureType() != MTL::TextureType2D)
	{
		device->log("Output plane %d must be MTLTextureType2D.", index);
		return false;
	}

	if (int(texture->width()) != width || int(texture->height()) != height)
	{
		device->log("Output plane %d is %ux%u, expected %dx%d.",
		            index, unsigned(texture->width()), unsigned(texture->height()), width, height);
		return false;
	}

	if ((texture->usage() & MTL::TextureUsageShaderWrite) == 0)
	{
		device->log("Output plane %d was not created with MTLTextureUsageShaderWrite.", index);
		return false;
	}

	return true;
}
}

//////
// Public API

void pyrowave_get_api_version(uint32_t *major, uint32_t *minor, uint32_t *patch)
{
	if (major)
		*major = PYROWAVE_API_VERSION_MAJOR;
	if (minor)
		*minor = PYROWAVE_API_VERSION_MINOR;
	if (patch)
		*patch = PYROWAVE_API_VERSION_PATCH;
}

const char *pyrowave_result_to_string(pyrowave_result result)
{
	return result_string(result);
}

bool pyrowave_device_is_supported(pyrowave_mtl_device mtl_device)
{
	return device_is_supported(static_cast<MTL::Device *>(mtl_device));
}

pyrowave_result pyrowave_device_create(const pyrowave_device_create_info *info, pyrowave_device *device)
{
	if (!info || !device || !info->mtl_device)
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	auto *pool = NS::AutoreleasePool::alloc()->init();
	auto release_pool = std::unique_ptr<NS::AutoreleasePool, void (*)(NS::AutoreleasePool *)>(
			pool, [](NS::AutoreleasePool *p) { p->release(); });

	auto *mtl = static_cast<MTL::Device *>(info->mtl_device);
	if (!device_is_supported(mtl))
		return PYROWAVE_ERROR_UNSUPPORTED_DEVICE;

	auto created = std::unique_ptr<pyrowave_device_opaque>(new (std::nothrow) pyrowave_device_opaque);
	if (!created)
		return PYROWAVE_ERROR_OUT_OF_HOST_MEMORY;

	created->message_cb = info->message_callback;
	created->message_userdata = info->message_userdata;
	created->mtl = mtl->retain();

	auto *dequant_library = compile_library(created.get(), wavelet_dequant_msl_source, "wavelet_dequant");
	created->precision = requested_precision();
	char idwt_label[32];
	snprintf(idwt_label, sizeof(idwt_label), "idwt (precision %d)", created->precision);
	auto *idwt_library = compile_library(created.get(),
	                                     idwt_source_for(created->precision), idwt_label);
	if (!dequant_library || !idwt_library)
	{
		if (dequant_library)
			dequant_library->release();
		if (idwt_library)
			idwt_library->release();
		return PYROWAVE_ERROR_SHADER_COMPILATION;
	}

	created->dequant_pipeline = create_pipeline(created.get(), dequant_library,
	                                            "pyrowave_wavelet_dequant", nullptr,
	                                            DequantThreadgroupSize);

	for (int i = 0; i < 2 && created->dequant_pipeline; i++)
	{
		const bool dc_shift = i != 0;
		created->idwt_pipeline[i] = create_pipeline(created.get(), idwt_library,
		                                            "pyrowave_idwt", &dc_shift,
		                                            IdwtThreadgroupSize);
		if (!created->idwt_pipeline[i])
			break;
	}

	dequant_library->release();
	idwt_library->release();

	if (!created->dequant_pipeline || !created->idwt_pipeline[0] || !created->idwt_pipeline[1])
		return PYROWAVE_ERROR_SHADER_COMPILATION;

	// The dequant shader's subgroup path assumes a 32 wide SIMD group.
	if (created->dequant_pipeline->threadExecutionWidth() != 32)
	{
		created->log("Unexpected SIMD width %u, expected 32.",
		             unsigned(created->dequant_pipeline->threadExecutionWidth()));
		return PYROWAVE_ERROR_UNSUPPORTED_DEVICE;
	}

	auto *sampler_desc = MTL::SamplerDescriptor::alloc()->init();
	sampler_desc->setMinFilter(MTL::SamplerMinMagFilterNearest);
	sampler_desc->setMagFilter(MTL::SamplerMinMagFilterNearest);
	sampler_desc->setMipFilter(MTL::SamplerMipFilterNearest);
	sampler_desc->setSAddressMode(MTL::SamplerAddressModeMirrorRepeat);
	sampler_desc->setTAddressMode(MTL::SamplerAddressModeMirrorRepeat);
	sampler_desc->setRAddressMode(MTL::SamplerAddressModeMirrorRepeat);
	created->mirror_repeat_sampler = created->mtl->newSamplerState(sampler_desc);
	sampler_desc->release();

	if (!created->mirror_repeat_sampler)
		return PYROWAVE_ERROR_OUT_OF_DEVICE_MEMORY;

	*device = created.release();
	return PYROWAVE_SUCCESS;
}

void pyrowave_device_destroy(pyrowave_device device)
{
	delete device;
}

pyrowave_result pyrowave_decoder_create(const pyrowave_decoder_create_info *info, pyrowave_decoder *decoder)
{
	if (!info || !decoder || !info->device)
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	if (info->chroma != PYROWAVE_CHROMA_SUBSAMPLING_420 &&
	    info->chroma != PYROWAVE_CHROMA_SUBSAMPLING_444)
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	const bool chroma_420 = info->chroma == PYROWAVE_CHROMA_SUBSAMPLING_420;
	if (chroma_420 && ((info->width & 1) != 0 || (info->height & 1) != 0))
	{
		info->device->log("420 subsampling requires even dimensions, got %dx%d.",
		                  info->width, info->height);
		return PYROWAVE_ERROR_INVALID_ARGUMENT;
	}

	auto *pool = NS::AutoreleasePool::alloc()->init();
	auto release_pool = std::unique_ptr<NS::AutoreleasePool, void (*)(NS::AutoreleasePool *)>(
			pool, [](NS::AutoreleasePool *p) { p->release(); });

	auto created = std::unique_ptr<pyrowave_decoder_opaque>(new (std::nothrow) pyrowave_decoder_opaque);
	if (!created)
		return PYROWAVE_ERROR_OUT_OF_HOST_MEMORY;

	created->device = info->device;

	if (!created->layout.init(info->width, info->height,
	                          chroma_420 ? ChromaSubsampling::Chroma420 : ChromaSubsampling::Chroma444))
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	created->parser.init(&created->layout);

	if (!create_wavelet_resources(created.get()))
		return PYROWAVE_ERROR_OUT_OF_DEVICE_MEMORY;

	*decoder = created.release();
	return PYROWAVE_SUCCESS;
}

void pyrowave_decoder_destroy(pyrowave_decoder decoder)
{
	delete decoder;
}

void pyrowave_decoder_clear(pyrowave_decoder decoder)
{
	if (decoder)
		decoder->parser.clear();
}

pyrowave_result pyrowave_decoder_push_packet(pyrowave_decoder decoder, const void *data, size_t size)
{
	if (!decoder || (!data && size != 0))
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	if (!decoder->parser.push_packet(data, size))
		return PYROWAVE_ERROR_CORRUPT_BITSTREAM;

	return PYROWAVE_SUCCESS;
}

bool pyrowave_decoder_decode_is_ready(pyrowave_decoder decoder, bool allow_partial_frame)
{
	return decoder && decoder->parser.decode_is_ready(allow_partial_frame);
}

pyrowave_result pyrowave_decoder_decode(pyrowave_decoder decoder,
                                        pyrowave_mtl_command_buffer command_buffer,
                                        const pyrowave_gpu_buffers *buffers)
{
	if (!decoder || !command_buffer || !buffers)
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	auto &layout = decoder->layout;
	auto *device = decoder->device;

	MTL::Texture *planes[3];
	const int chroma_width = layout.chroma == ChromaSubsampling::Chroma420 ?
	                         layout.width / 2 : layout.width;
	const int chroma_height = layout.chroma == ChromaSubsampling::Chroma420 ?
	                          layout.height / 2 : layout.height;

	for (int i = 0; i < 3; i++)
	{
		planes[i] = static_cast<MTL::Texture *>(buffers->planes[i]);
		const int expected_width = i == 0 ? layout.width : chroma_width;
		const int expected_height = i == 0 ? layout.height : chroma_height;
		if (!validate_plane(device, planes[i], i, expected_width, expected_height))
			return PYROWAVE_ERROR_INVALID_ARGUMENT;
	}

	auto *pool = NS::AutoreleasePool::alloc()->init();
	auto release_pool = std::unique_ptr<NS::AutoreleasePool, void (*)(NS::AutoreleasePool *)>(
			pool, [](NS::AutoreleasePool *p) { p->release(); });

	const auto &offsets = decoder->parser.dequant_offsets();
	const auto &payload = decoder->parser.payload();

	const size_t offsets_size = offsets.size() * sizeof(uint32_t);
	// The dequant shader can read slightly past the end of the payload, so pad.
	const size_t payload_size = payload.size() * sizeof(uint32_t) + 16;

	auto *slot = acquire_upload_slot(decoder, offsets_size, payload_size);
	if (!slot)
		return PYROWAVE_ERROR_OUT_OF_DEVICE_MEMORY;

	if (offsets_size)
		memcpy(slot->offsets->contents(), offsets.data(), offsets_size);
	if (!payload.empty())
		memcpy(slot->payload->contents(), payload.data(), payload.size() * sizeof(uint32_t));

	auto *cmd = static_cast<MTL::CommandBuffer *>(command_buffer);

	// Every dequant dispatch writes a distinct (component, level, band) region of
	// the pyramid and none reads another's output, so they can all run at once. A
	// serial encoder would put a full barrier between each of the ~42 of them,
	// which dominates the frame at low resolutions where the dispatches are tiny.
	auto *dequant_enc = cmd->computeCommandEncoder(MTL::DispatchTypeConcurrent);
	if (!dequant_enc)
		return PYROWAVE_ERROR_GENERIC;

	dequant_enc->setLabel(NS::String::string("pyrowave dequant", NS::UTF8StringEncoding));
	encode_dequant(decoder, dequant_enc, slot);
	dequant_enc->endEncoding();

	// The iDWT is a dependent chain across levels, but the three components within
	// a level are independent, so this is also concurrent with explicit barriers
	// at the level boundaries only. Ordering against the dequant work above comes
	// from the encoder boundary, which Metal tracks automatically.
	auto *idwt_enc = cmd->computeCommandEncoder(MTL::DispatchTypeConcurrent);
	if (!idwt_enc)
		return PYROWAVE_ERROR_GENERIC;

	idwt_enc->setLabel(NS::String::string("pyrowave idwt", NS::UTF8StringEncoding));
	encode_idwt(decoder, idwt_enc, planes);
	idwt_enc->endEncoding();

	slot->in_flight.store(true, std::memory_order_release);
	cmd->addCompletedHandler([slot](MTL::CommandBuffer *) {
		slot->in_flight.store(false, std::memory_order_release);
	});

	decoder->parser.mark_frame_decoded();
	return PYROWAVE_SUCCESS;
}
