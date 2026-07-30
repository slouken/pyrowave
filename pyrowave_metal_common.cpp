// Copyright (c) 2026 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT

// Device object, shader helpers and wavelet pyramid shared by the Metal encoder
// and decoder. This is the single translation unit that instantiates metal-cpp.

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Metal/Metal.hpp>

#include "pyrowave_metal_common.hpp"
#include "shaders/metal/pyrowave_msl.h"

#include <memory>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

namespace PyroWave
{
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
                                           const char *entry_point, uint32_t required_threads,
                                           MTL::FunctionConstantValues *constants)
{
	NS::Error *error = nullptr;
	auto *name = NS::String::string(entry_point, NS::UTF8StringEncoding);

	MTL::Function *function = constants ?
	                          library->newFunction(name, constants, &error) :
	                          library->newFunction(name);

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

MTL::ComputePipelineState *create_pipeline_bool_constant(pyrowave_device device, MTL::Library *library,
                                                         const char *entry_point, uint32_t required_threads,
                                                         uint32_t index, bool value)
{
	auto *constants = MTL::FunctionConstantValues::alloc()->init();
	constants->setConstantValue(&value, MTL::DataTypeBool, NS::UInteger(index));
	auto *pipeline = create_pipeline(device, library, entry_point, required_threads, constants);
	constants->release();
	return pipeline;
}

bool WaveletPyramid::init(pyrowave_device device, const BlockLayout &layout)
{
	auto *mtl = device->mtl;
	const auto format = wavelet_format(device->precision);

	auto *desc = MTL::TextureDescriptor::alloc()->init();
	desc->setTextureType(MTL::TextureType2DArray);
	desc->setPixelFormat(format);
	desc->setWidth(layout.aligned_width / 2);
	desc->setHeight(layout.aligned_height / 2);
	desc->setArrayLength(NumFrequencyBandsPerLevel * NumComponents);
	desc->setMipmapLevelCount(DecompositionLevels);
	// PixelFormatView is required to take the per component/per level views below.
	desc->setUsage(MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite |
	               MTL::TextureUsagePixelFormatView);
	desc->setStorageMode(MTL::StorageModePrivate);

	texture = mtl->newTexture(desc);
	desc->release();

	if (!texture)
	{
		device->log("Failed to allocate wavelet texture.");
		return false;
	}

	texture->setLabel(NS::String::string("pyrowave-wavelet", NS::UTF8StringEncoding));

	for (int level = 0; level < DecompositionLevels; level++)
	{
		for (int component = 0; component < NumComponents; component++)
		{
			component_layer_views[component][level] =
					texture->newTextureView(
							format, MTL::TextureType2DArray,
							NS::Range(level, 1),
							NS::Range(NumFrequencyBandsPerLevel * component, NumFrequencyBandsPerLevel));

			component_ll_views[component][level] =
					texture->newTextureView(
							format, MTL::TextureType2D,
							NS::Range(level, 1),
							NS::Range(NumFrequencyBandsPerLevel * component, 1));

			if (!component_layer_views[component][level] || !component_ll_views[component][level])
			{
				device->log("Failed to create wavelet texture views.");
				return false;
			}
		}
	}

	return true;
}

WaveletPyramid::~WaveletPyramid()
{
	for (auto &views : component_layer_views)
		for (auto *view : views)
			if (view)
				view->release();
	for (auto &views : component_ll_views)
		for (auto *view : views)
			if (view)
				view->release();
	if (texture)
		texture->release();
}
}

using namespace PyroWave;

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

pyrowave_device_opaque::~pyrowave_device_opaque()
{
	MTL::ComputePipelineState *const pipelines[] = {
		dequant_pipeline, idwt_pipeline[0], idwt_pipeline[1],
		dwt_pipeline[0], dwt_pipeline[1], quant_pipeline, analyze_pipeline,
		analyze_finalize_pipeline, resolve_pipeline, block_packing_pipeline,
	};

	for (auto *pipeline : pipelines)
		if (pipeline)
			pipeline->release();

	if (mirror_repeat_sampler)
		mirror_repeat_sampler->release();
	if (border_sampler)
		border_sampler->release();
	if (mtl)
		mtl->release();
}

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
	if (mtl->maxThreadsPerThreadgroup().width < AnalyzeFinalizeThreadgroupSize)
		return false;
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
	const char *idwt_source;
	switch (created->precision)
	{
	case 0: idwt_source = idwt_fp16_msl_source; break;
	case 1: idwt_source = idwt_fp16_storage_msl_source; break;
	default: idwt_source = idwt_msl_source; break;
	}
	auto *idwt_library = compile_library(created.get(), idwt_source, idwt_label);
	if (!dequant_library || !idwt_library)
	{
		if (dequant_library)
			dequant_library->release();
		if (idwt_library)
			idwt_library->release();
		return PYROWAVE_ERROR_SHADER_COMPILATION;
	}

	created->dequant_pipeline = create_pipeline(created.get(), dequant_library,
	                                            "pyrowave_wavelet_dequant", DequantThreadgroupSize);

	for (int i = 0; i < 2 && created->dequant_pipeline; i++)
	{
		created->idwt_pipeline[i] = create_pipeline_bool_constant(created.get(), idwt_library,
		                                                          "pyrowave_idwt", IdwtThreadgroupSize,
		                                                          0, i != 0);
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
