// Copyright (c) 2026 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT

// Metal backend for the PyroWave encoder. Mirrors the compute pipeline of
// pyrowave_encoder.cpp: dwt -> quant -> analyze_rdo -> resolve_rdo ->
// block_packing on the GPU, then packetize on the CPU.
//
// Two things differ structurally from the Vulkan encoder. It owns its command
// queue rather than taking a caller command buffer, because the packet queries
// have to block on completion. And the result buffers are shared storage, read
// directly by the CPU: the Vulkan side stages them through device local memory
// because writing a bitstream into host visible memory is slow on discrete GPUs,
// which does not apply to unified memory.

#include "pyrowave_metal_common.hpp"
#include "shaders/metal/pyrowave_msl.h"

#include <IOSurface/IOSurfaceRef.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <string.h>

using namespace PyroWave;

namespace
{
// Push constant layouts, matching the Registers structs SPIRV-Cross emitted into
// shaders/metal/*.metal. MSL aligns int2 and float2 to 8 bytes, so several of
// these need explicit tail padding; the sizes were checked against the generated
// MSL with a static_assert on the Metal compiler's own layout.
struct DwtPush
{
	int32_t resolution[2];
	float inv_resolution[2];
	int32_t aligned_resolution[2];
};
static_assert(sizeof(DwtPush) == 24, "DwtPush layout mismatch.");

struct QuantPush
{
	int32_t resolution[2];
	int32_t resolution_8x8_blocks[2];
	float inv_resolution[2];
	float input_layer;
	float quant_resolution;
	int32_t block_offset;
	int32_t block_stride;
	float rdo_distortion_scale;
	float padding;
};
static_assert(sizeof(QuantPush) == 48, "QuantPush layout mismatch.");

struct AnalyzePush
{
	int32_t resolution[2];
	int32_t resolution_8x8_blocks[2];
	int32_t block_offset_8x8;
	int32_t block_stride_8x8;
	int32_t block_offset_32x32;
	int32_t block_stride_32x32;
	uint32_t total_wg_count;
	uint32_t num_blocks_aligned;
	uint32_t block_index_shamt;
	uint32_t padding;
};
static_assert(sizeof(AnalyzePush) == 48, "AnalyzePush layout mismatch.");

struct ResolvePush
{
	uint32_t target_payload_size;
	uint32_t num_blocks_per_subdivision;
};
static_assert(sizeof(ResolvePush) == 8, "ResolvePush layout mismatch.");

struct BlockPackingPush
{
	int32_t resolution[2];
	int32_t resolution_32x32_blocks[2];
	int32_t resolution_8x8_blocks[2];
	uint32_t quant_resolution_code;
	uint32_t sequence_code;
	int32_t block_offset_32x32;
	int32_t block_stride_32x32;
	int32_t block_offset_8x8;
	int32_t block_stride_8x8;
};
static_assert(sizeof(BlockPackingPush) == 48, "BlockPackingPush layout mismatch.");

//////
// Initial quantization resolution. Pure float math lifted from
// pyrowave_encoder.cpp; the only change is that the precision and chroma mode are
// passed in rather than read off Configuration and WaveletBuffers.

float get_noise_power_normalized_quant_resolution(int level, int component, int band, int precision)
{
	// The initial quantization resolution aims for a flat spectrum with noise power
	// normalization. The low-pass gain for CDF 9/7 is 6 dB (1 bit). Every
	// decomposition level subtracts 6 dB.
	int bits = precision >= 1 ? 8 : 6;

	if (band == 0)
		bits += 2;
	else if (band < 3)
		bits += 1;

	bits += level;

	// Chroma starts at level 1, subtract one bit.
	if (component != 0)
		bits--;

	return float(1 << bits);
}

float get_quant_resolution(int level, int component, int band, int precision)
{
	// FP16 range is limited, and this is more than a good enough initial estimate.
	return std::min<float>(precision >= 1 ? 4096.0f : 512.0f,
	                       get_noise_power_normalized_quant_resolution(level, component, band, precision));
}

float get_quant_rdo_distortion_scale(int level, int component, int band, int precision,
                                     ChromaSubsampling chroma)
{
	// From my Linelet master thesis. Copy paste 11 years later, ah yes :D
	float horiz_midpoint = (band & 1) ? 0.75f : 0.25f;
	float vert_midpoint = (band & 2) ? 0.75f : 0.25f;

	// Normal PC monitors.
	constexpr float dpi = 96.0f;
	// Compromise between couch gaming and desktop.
	constexpr float viewing_distance = 1.0f;
	constexpr float cpd_nyquist = 0.34f * viewing_distance * dpi;

	float cpd = std::sqrt(horiz_midpoint * horiz_midpoint + vert_midpoint * vert_midpoint) *
	            cpd_nyquist * std::exp2(-float(level));

	// Don't allow a situation where we're quantizing LL band hard.
	cpd = std::max(cpd, 8.0f);

	float csf = 2.6f * (0.0192f + 0.114f * cpd) * std::exp(-std::pow(0.114f * cpd, 1.1f));

	// Heavily discount chroma quality.
	if (component != 0 && level != DecompositionLevels - 1)
	{
		// Consider chroma a little more important if we're not subsampling.
		if (chroma == ChromaSubsampling::Chroma420)
			csf *= 0.6f;
	}

	// Due to filtering, distortion in lower bands will result in more noise power.
	// By scaling the distortion by this factor, we ensure uniform results.
	float resolution = get_noise_power_normalized_quant_resolution(level, component, band, precision);
	float weighted_resolution = csf * resolution;

	// The distortion is scaled in terms of power, not amplitude.
	return weighted_resolution * weighted_resolution;
}

uint32_t floor_log2(uint32_t v)
{
	uint32_t result = 0;
	while (v > 1)
	{
		v >>= 1;
		result++;
	}
	return result;
}

// The three single channel textures the DWT samples at level 0. For NV12 input
// two of them are swizzled views of one interleaved chroma texture, so the
// objects to release are tracked separately from the sampled views.
struct InputTextures
{
	MTL::Texture *sampled[NumComponents] = {};
	MTL::Texture *owned[4] = {};
	int num_owned = 0;

	MTL::Texture *adopt(MTL::Texture *texture)
	{
		if (texture)
			owned[num_owned++] = texture;
		return texture;
	}

	void release_all()
	{
		for (int i = 0; i < num_owned; i++)
			owned[i]->release();
		num_owned = 0;
		for (auto *&texture : sampled)
			texture = nullptr;
	}

	~InputTextures() { release_all(); }
	InputTextures() = default;
	InputTextures(const InputTextures &) = delete;
	void operator=(const InputTextures &) = delete;
};
}

struct pyrowave_encoder_opaque
{
	pyrowave_device device = nullptr;

	BlockLayout layout;
	WaveletPyramid wavelet;

	MTL::CommandQueue *queue = nullptr;

	// Device local scratch, sized once from the block layout.
	MTL::Buffer *bucket_buffer = nullptr;
	MTL::Buffer *meta_buffer = nullptr;
	MTL::Buffer *block_stat_buffer = nullptr;
	MTL::Buffer *payload_data = nullptr;
	MTL::Buffer *quant_buffer = nullptr;

	// Results. Shared storage so the packet queries can read them without a copy.
	MTL::Buffer *bitstream_meta = nullptr;
	MTL::Buffer *bitstream = nullptr;

	// Input textures for the CPU entry point, reused across frames. The GPU entry
	// point wraps the caller's IOSurfaces per call instead.
	InputTextures cpu_input;
	pyrowave_cpu_buffer_format cpu_input_format = PYROWAVE_CPU_BUFFER_FORMAT_INT_MAX;

	// Retained until the next encode replaces it, so the packet queries have
	// something to block on.
	MTL::CommandBuffer *pending = nullptr;
	bool have_result = false;

	uint32_t sequence_count = 0;

	~pyrowave_encoder_opaque()
	{
		MTL::Buffer *const buffers[] = {
			bucket_buffer, meta_buffer, block_stat_buffer, payload_data, quant_buffer,
			bitstream_meta, bitstream,
		};

		for (auto *buffer : buffers)
			if (buffer)
				buffer->release();

		// Metal keeps anything a command buffer references alive on its own, so an
		// in-flight encode does not have to be waited out here.
		if (pending)
			pending->release();
		if (queue)
			queue->release();
	}
};

namespace
{
size_t bucket_buffer_size(const BlockLayout &layout)
{
	size_t size = RDOBucketOffset;
	size += size_t(NumRDOBuckets) * BlockSpaceSubdivision * sizeof(uint32_t);
	size += size_t(NumRDOBuckets) * compute_block_count_per_subdivision(layout.block_count_32x32) *
	        BlockSpaceSubdivision * sizeof(RDOperation);
	return size;
}

bool ensure_encode_pipelines(pyrowave_device device)
{
	std::lock_guard<std::mutex> holder{device->encode_pipeline_lock};
	if (device->encode_pipelines_ready)
		return true;

	// Each shader is its own MTLLibrary. They cannot be concatenated: SPIRV-Cross
	// emits its own copy of the spv* helpers and a differently shaped Registers
	// struct into every source.
	const char *dwt_source;
	switch (device->precision)
	{
	case 0: dwt_source = dwt_fp16_msl_source; break;
	case 1: dwt_source = dwt_fp16_storage_msl_source; break;
	default: dwt_source = dwt_msl_source; break;
	}

	char dwt_label[32];
	snprintf(dwt_label, sizeof(dwt_label), "dwt (precision %d)", device->precision);

	if (auto *library = compile_library(device, dwt_source, dwt_label))
	{
		// Indexed by DCShift, which converts the unorm input planes into the signed
		// range the wavelet transform works in.
		for (int i = 0; i < 2; i++)
		{
			device->dwt_pipeline[i] = create_pipeline_bool_constant(
					device, library, "pyrowave_dwt", DwtThreadgroupSize, 0, i != 0);
		}
		library->release();
	}

	// SkipQuantScale is function constant 1 and wants its default of false, but
	// Metal will not build a pipeline from an unspecialized function that declares
	// any constants, so it has to be set explicitly.
	if (auto *library = compile_library(device, wavelet_quant_msl_source, "wavelet_quant"))
	{
		device->quant_pipeline = create_pipeline_bool_constant(
				device, library, "pyrowave_wavelet_quant", QuantThreadgroupSize, 1, false);
		library->release();
	}

	struct
	{
		MTL::ComputePipelineState **pipeline;
		const char *source;
		const char *entry_point;
		uint32_t threads;
	} const simple[] = {
		{ &device->analyze_pipeline, analyze_rate_control_msl_source,
		  "pyrowave_analyze_rate_control", AnalyzeThreadgroupSize },
		{ &device->analyze_finalize_pipeline, analyze_rate_control_finalize_msl_source,
		  "pyrowave_analyze_rate_control_finalize", AnalyzeFinalizeThreadgroupSize },
		{ &device->block_packing_pipeline, block_packing_msl_source,
		  "pyrowave_block_packing", BlockPackingThreadgroupSize },
	};

	for (auto &shader : simple)
	{
		if (auto *library = compile_library(device, shader.source, shader.entry_point))
		{
			*shader.pipeline = create_pipeline(device, library, shader.entry_point, shader.threads);
			library->release();
		}
	}

	// resolve_rate_control declares its workgroup size as a specialization
	// constant and requires it to equal the SIMD width, so the whole threadgroup
	// is one SIMD group and its subgroup scan needs no cross-group communication.
	if (auto *library = compile_library(device, resolve_rate_control_msl_source, "resolve_rate_control"))
	{
		uint32_t threadgroup_size = ResolveThreadgroupSize;
		auto *constants = MTL::FunctionConstantValues::alloc()->init();
		constants->setConstantValue(&threadgroup_size, MTL::DataTypeUInt, NS::UInteger(0));
		device->resolve_pipeline = create_pipeline(device, library, "pyrowave_resolve_rate_control",
		                                           ResolveThreadgroupSize, constants);
		constants->release();
		library->release();
	}

	MTL::ComputePipelineState *const required[] = {
		device->dwt_pipeline[0], device->dwt_pipeline[1], device->quant_pipeline,
		device->analyze_pipeline, device->analyze_finalize_pipeline,
		device->resolve_pipeline, device->block_packing_pipeline,
	};

	for (auto *pipeline : required)
		if (!pipeline)
			return false;

	if (device->resolve_pipeline->threadExecutionWidth() != ResolveThreadgroupSize)
	{
		device->log("resolve_rate_control needs a SIMD width of %u, but the device reports %u.",
		            ResolveThreadgroupSize,
		            unsigned(device->resolve_pipeline->threadExecutionWidth()));
		return false;
	}

	// The quantizer samples with a border so coefficients past the edge of a band
	// read as zero, rather than the DWT's mirror repeat.
	auto *sampler_desc = MTL::SamplerDescriptor::alloc()->init();
	sampler_desc->setMinFilter(MTL::SamplerMinMagFilterNearest);
	sampler_desc->setMagFilter(MTL::SamplerMinMagFilterNearest);
	sampler_desc->setMipFilter(MTL::SamplerMipFilterNearest);
	sampler_desc->setSAddressMode(MTL::SamplerAddressModeClampToBorderColor);
	sampler_desc->setTAddressMode(MTL::SamplerAddressModeClampToBorderColor);
	sampler_desc->setRAddressMode(MTL::SamplerAddressModeClampToBorderColor);
	sampler_desc->setBorderColor(MTL::SamplerBorderColorTransparentBlack);
	device->border_sampler = device->mtl->newSamplerState(sampler_desc);
	sampler_desc->release();

	if (!device->border_sampler)
		return false;

	device->encode_pipelines_ready = true;
	return true;
}

MTL::Buffer *create_scratch_buffer(pyrowave_device device, size_t size, MTL::ResourceOptions options,
                                   const char *label)
{
	auto *buffer = device->mtl->newBuffer(size, options);
	if (!buffer)
	{
		device->log("Failed to allocate a %zu byte %s buffer.", size, label);
		return nullptr;
	}
	buffer->setLabel(NS::String::string(label, NS::UTF8StringEncoding));
	return buffer;
}

bool create_encode_buffers(pyrowave_encoder encoder)
{
	auto *device = encoder->device;
	const auto &layout = encoder->layout;

	struct
	{
		MTL::Buffer **buffer;
		size_t size;
		const char *label;
	} const scratch[] = {
		{ &encoder->block_stat_buffer,
		  size_t(layout.block_count_8x8) * sizeof(BlockStatsBlock), "pyrowave-block-stats" },
		{ &encoder->meta_buffer,
		  size_t(layout.block_count_8x8) * sizeof(BlockMeta), "pyrowave-block-meta" },
		// Worst case estimate, same as the Vulkan encoder's. The first two words are
		// allocation counters and the coefficient payload starts at byte 8.
		{ &encoder->payload_data,
		  size_t(layout.aligned_width) * size_t(layout.aligned_height) * 2, "pyrowave-payload" },
		{ &encoder->quant_buffer,
		  size_t(layout.block_count_32x32) * sizeof(uint32_t), "pyrowave-quant" },
		{ &encoder->bucket_buffer, bucket_buffer_size(layout), "pyrowave-buckets" },
	};

	for (auto &entry : scratch)
	{
		*entry.buffer = create_scratch_buffer(device, entry.size,
		                                      MTL::ResourceStorageModePrivate, entry.label);
		if (!*entry.buffer)
			return false;
	}

	encoder->bitstream_meta = create_scratch_buffer(
			device, size_t(layout.block_count_32x32) * sizeof(BitstreamPacket),
			MTL::ResourceStorageModeShared, "pyrowave-bitstream-meta");

	return encoder->bitstream_meta != nullptr;
}

// The bitstream buffer is sized per encode, since the rate control target can
// change from frame to frame. Releasing the old one mid-flight is safe: Metal
// keeps a buffer alive as long as a command buffer references it.
bool ensure_bitstream_buffer(pyrowave_encoder encoder, size_t size)
{
	if (encoder->bitstream && encoder->bitstream->length() >= size)
		return true;

	if (encoder->bitstream)
		encoder->bitstream->release();

	encoder->bitstream = create_scratch_buffer(encoder->device, size,
	                                           MTL::ResourceStorageModeShared, "pyrowave-bitstream");
	return encoder->bitstream != nullptr;
}

MTL::Texture *create_input_texture(pyrowave_device device, MTL::PixelFormat format,
                                   int width, int height, bool needs_view)
{
	auto *desc = MTL::TextureDescriptor::alloc()->init();
	desc->setTextureType(MTL::TextureType2D);
	desc->setPixelFormat(format);
	desc->setWidth(width);
	desc->setHeight(height);
	desc->setUsage(MTL::TextureUsageShaderRead |
	               (needs_view ? MTL::TextureUsagePixelFormatView : 0));
	// Written by the CPU, and IOSurface backed textures cannot be private anyway.
	desc->setStorageMode(MTL::StorageModeShared);
	desc->setMipmapLevelCount(1);

	auto *texture = device->mtl->newTexture(desc);
	desc->release();

	if (!texture)
		device->log("Failed to allocate a %dx%d input texture.", width, height);

	return texture;
}

MTL::Texture *wrap_surface_plane(pyrowave_device device, IOSurfaceRef surface, int plane,
                                 MTL::PixelFormat format, int width, int height, bool needs_view)
{
	auto *desc = MTL::TextureDescriptor::alloc()->init();
	desc->setTextureType(MTL::TextureType2D);
	desc->setPixelFormat(format);
	desc->setWidth(width);
	desc->setHeight(height);
	desc->setUsage(MTL::TextureUsageShaderRead |
	               (needs_view ? MTL::TextureUsagePixelFormatView : 0));
	desc->setStorageMode(MTL::StorageModeShared);
	desc->setMipmapLevelCount(1);

	auto *texture = device->mtl->newTexture(desc, surface, NS::UInteger(plane));
	desc->release();

	if (!texture)
		device->log("Failed to wrap plane %d of the input IOSurface.", plane);

	return texture;
}

// dwt gathers the red channel of whatever it is handed, so an interleaved chroma
// plane is bound twice with the wanted component broadcast into red. This is what
// the Vulkan API documents for NV12 too ("pass in the same plane for Cb and Cr,
// but use swizzle"). Note an R8 format view of an RG8 texture is impossible, 8
// versus 16 bits per pixel, so the format stays and only the swizzle changes.
bool make_interleaved_chroma_views(pyrowave_device device, InputTextures &input, MTL::Texture *chroma)
{
	static const MTL::TextureSwizzle components[2] = { MTL::TextureSwizzleRed, MTL::TextureSwizzleGreen };

	for (int i = 0; i < 2; i++)
	{
		const auto c = components[i];
		input.sampled[i + 1] = input.adopt(chroma->newTextureView(
				chroma->pixelFormat(), MTL::TextureType2D, NS::Range(0, 1), NS::Range(0, 1),
				MTL::TextureSwizzleChannels::Make(c, c, c, c)));

		if (!input.sampled[i + 1])
		{
			device->log("Failed to create a swizzled view of the interleaved chroma plane.");
			return false;
		}
	}

	return true;
}

// Reuses the textures from the previous CPU encode when the format has not
// changed, and rebuilds them when it has.
bool ensure_cpu_input(pyrowave_encoder encoder, pyrowave_cpu_buffer_format format)
{
	if (encoder->cpu_input_format == format && encoder->cpu_input.sampled[0])
		return true;

	encoder->cpu_input.release_all();
	encoder->cpu_input_format = PYROWAVE_CPU_BUFFER_FORMAT_INT_MAX;

	auto *device = encoder->device;
	auto &input = encoder->cpu_input;
	const auto &layout = encoder->layout;
	const bool chroma_420 = layout.chroma == ChromaSubsampling::Chroma420;
	const int chroma_width = chroma_420 ? layout.width / 2 : layout.width;
	const int chroma_height = chroma_420 ? layout.height / 2 : layout.height;

	input.sampled[0] = input.adopt(create_input_texture(device, MTL::PixelFormatR8Unorm,
	                                                    layout.width, layout.height, false));
	if (!input.sampled[0])
		return false;

	if (format == PYROWAVE_CPU_BUFFER_FORMAT_NV12)
	{
		auto *chroma = input.adopt(create_input_texture(device, MTL::PixelFormatRG8Unorm,
		                                                chroma_width, chroma_height, true));
		if (!chroma || !make_interleaved_chroma_views(device, input, chroma))
			return false;
	}
	else
	{
		for (int i = 1; i < NumComponents; i++)
		{
			input.sampled[i] = input.adopt(create_input_texture(device, MTL::PixelFormatR8Unorm,
			                                                    chroma_width, chroma_height, false));
			if (!input.sampled[i])
				return false;
		}
	}

	encoder->cpu_input_format = format;
	return true;
}

//////
// GPU dispatch, one function per stage of pyrowave_encoder.cpp's pipeline.

void dispatch_dwt(pyrowave_encoder encoder, MTL::ComputeCommandEncoder *enc,
                MTL::Texture *const planes[NumComponents])
{
	const auto &layout = encoder->layout;
	auto &wavelet = encoder->wavelet;
	const bool chroma_420 = layout.chroma == ChromaSubsampling::Chroma420;

	enc->setSamplerState(encoder->device->mirror_repeat_sampler, 0);

	for (int output_level = 0; output_level < DecompositionLevels; output_level++)
	{
		// Each level transforms the LL band the previous one produced, so the levels
		// are a dependent chain. The components within a level are independent.
		if (output_level != 0)
			enc->memoryBarrier(MTL::BarrierScopeTextures);

		DwtPush level_push = {};
		if (output_level == 0)
		{
			// The input planes are only `width` wide, but the transform covers the
			// aligned extent and mirrors the source to fill it.
			level_push.resolution[0] = layout.width;
			level_push.resolution[1] = layout.height;
			level_push.aligned_resolution[0] = layout.aligned_width;
			level_push.aligned_resolution[1] = layout.aligned_height;
		}
		else
		{
			level_push.resolution[0] = layout.level_width(output_level - 1);
			level_push.resolution[1] = layout.level_height(output_level - 1);
			level_push.aligned_resolution[0] = level_push.resolution[0];
			level_push.aligned_resolution[1] = level_push.resolution[1];
		}

		// Under 420, chroma is half resolution and so enters the pyramid at level 1
		// rather than level 0.
		const int components = (output_level == 0 && chroma_420) ? 1 : NumComponents;

		for (int component = 0; component < components; component++)
		{
			DwtPush push = level_push;
			MTL::Texture *input;
			// DCShift subtracts 0.5 to centre unorm input on zero, so it applies
			// exactly when the source is an input plane rather than a previous level.
			bool reads_input_plane = output_level == 0;

			if (output_level == 0)
			{
				input = planes[component];
			}
			else if (chroma_420 && component != 0 && output_level == 1)
			{
				input = planes[component];
				reads_input_plane = true;
				push.resolution[0] = layout.width / 2;
				push.resolution[1] = layout.height / 2;
				push.aligned_resolution[0] = layout.aligned_width >> output_level;
				push.aligned_resolution[1] = layout.aligned_height >> output_level;
			}
			else
			{
				input = wavelet.component_ll_views[component][output_level - 1];
			}

			push.inv_resolution[0] = 1.0f / float(push.resolution[0]);
			push.inv_resolution[1] = 1.0f / float(push.resolution[1]);

			enc->setComputePipelineState(encoder->device->dwt_pipeline[reads_input_plane ? 1 : 0]);
			enc->setBytes(&push, sizeof(push), 0);
			enc->setTexture(input, 0);
			enc->setTexture(wavelet.component_layer_views[component][output_level], 1);

			// Each threadgroup consumes a 32x32 source tile.
			enc->dispatchThreadgroups(
					MTL::Size((push.aligned_resolution[0] + 31) / 32,
					          (push.aligned_resolution[1] + 31) / 32, 1),
					MTL::Size(DwtThreadgroupSize, 1, 1));
		}
	}
}

// The remaining stages all walk the same (level, component, band) space as the
// decoder's dequant pass.
template <typename Op>
void for_each_band(const BlockLayout &layout, Op op)
{
	for (int level = 0; level < DecompositionLevels; level++)
	{
		for (int component = 0; component < NumComponents; component++)
		{
			// Ignore top-level CbCr when doing 420 subsampling.
			if (level == 0 && component != 0 && layout.chroma == ChromaSubsampling::Chroma420)
				continue;

			for (int band = (level == DecompositionLevels - 1 ? 0 : 1); band < 4; band++)
				op(level, component, band);
		}
	}
}

void dispatch_quant(pyrowave_encoder encoder, MTL::ComputeCommandEncoder *enc)
{
	const auto &layout = encoder->layout;
	const int precision = encoder->device->precision;

	enc->setComputePipelineState(encoder->device->quant_pipeline);
	enc->setBuffer(encoder->meta_buffer, 0, 1);
	enc->setBuffer(encoder->block_stat_buffer, 0, 2);
	enc->setBuffer(encoder->payload_data, 0, 3);
	enc->setSamplerState(encoder->device->border_sampler, 0);

	for_each_band(layout, [&](int level, int component, int band) {
		QuantPush push = {};
		push.resolution[0] = layout.level_width(level);
		push.resolution[1] = layout.level_height(level);
		push.resolution_8x8_blocks[0] = (push.resolution[0] + 7) / 8;
		push.resolution_8x8_blocks[1] = (push.resolution[1] + 7) / 8;
		push.inv_resolution[0] = 1.0f / float(push.resolution[0]);
		push.inv_resolution[1] = 1.0f / float(push.resolution[1]);
		push.input_layer = float(band);

		// The round trip through the 8 bit quant code is deliberate: the decoder
		// only ever sees the code, so quantize against the value it will dequantize
		// with rather than the ideal one.
		const float quant_res = get_quant_resolution(level, component, band, precision);
		push.quant_resolution = 1.0f / decode_quant(encode_quant(1.0f / quant_res));
		push.rdo_distortion_scale =
				get_quant_rdo_distortion_scale(level, component, band, precision, layout.chroma) *
				(1.0f / 256.0f);

		push.block_offset = layout.block_meta[component][level][band].block_offset_8x8;
		push.block_stride = layout.block_meta[component][level][band].block_stride_8x8;

		enc->setTexture(encoder->wavelet.component_layer_views[component][level], 0);
		enc->setBytes(&push, sizeof(push), 0);
		enc->dispatchThreadgroups(
				MTL::Size((push.resolution[0] + 31) / 32, (push.resolution[1] + 31) / 32, 1),
				MTL::Size(QuantThreadgroupSize, 1, 1));
	});
}

void dispatch_analyze_rdo(pyrowave_encoder encoder, MTL::ComputeCommandEncoder *enc)
{
	const auto &layout = encoder->layout;
	const int per_subdivision = compute_block_count_per_subdivision(layout.block_count_32x32);

	enc->setComputePipelineState(encoder->device->analyze_pipeline);
	enc->setBuffer(encoder->bucket_buffer, 0, 0);
	enc->setBuffer(encoder->block_stat_buffer, 0, 2);

	for_each_band(layout, [&](int level, int component, int band) {
		AnalyzePush push = {};
		push.resolution[0] = layout.level_width(level);
		push.resolution[1] = layout.level_height(level);
		push.resolution_8x8_blocks[0] = (push.resolution[0] + 7) / 8;
		push.resolution_8x8_blocks[1] = (push.resolution[1] + 7) / 8;

		const auto &meta = layout.block_meta[component][level][band];
		push.block_offset_8x8 = meta.block_offset_8x8;
		push.block_stride_8x8 = meta.block_stride_8x8;
		push.block_offset_32x32 = meta.block_offset_32x32;
		push.block_stride_32x32 = meta.block_stride_32x32;
		push.total_wg_count = uint32_t(layout.block_count_32x32);
		push.num_blocks_aligned = uint32_t(per_subdivision * BlockSpaceSubdivision);
		push.block_index_shamt = floor_log2(uint32_t(per_subdivision));

		enc->setBytes(&push, sizeof(push), 1);
		enc->dispatchThreadgroups(
				MTL::Size((push.resolution[0] + 31) / 32, (push.resolution[1] + 31) / 32, 1),
				MTL::Size(AnalyzeThreadgroupSize, 1, 1));
	});

	// The finalize pass prefix sums the buckets the dispatches above filled.
	enc->memoryBarrier(MTL::BarrierScopeBuffers);
	enc->setComputePipelineState(encoder->device->analyze_finalize_pipeline);
	enc->setBuffer(encoder->bucket_buffer, 0, 0);
	enc->dispatchThreadgroups(MTL::Size(1, 1, 1),
	                          MTL::Size(AnalyzeFinalizeThreadgroupSize, 1, 1));
}

void dispatch_resolve_rdo(pyrowave_encoder encoder, MTL::ComputeCommandEncoder *enc, size_t target_size)
{
	const auto &layout = encoder->layout;

	// The sequence header is part of the frame's budget, so the payload target is
	// what is left after it. Guard the subtraction for absurdly small targets.
	if (target_size >= sizeof(BitstreamSequenceHeader))
		target_size -= sizeof(BitstreamSequenceHeader);

	ResolvePush push = {};
	push.target_payload_size = uint32_t(target_size / sizeof(uint32_t));
	push.num_blocks_per_subdivision = uint32_t(compute_block_count_per_subdivision(layout.block_count_32x32));

	enc->setComputePipelineState(encoder->device->resolve_pipeline);
	enc->setBuffer(encoder->bucket_buffer, 0, 0);
	enc->setBytes(&push, sizeof(push), 1);
	enc->setBuffer(encoder->quant_buffer, 0, 2);
	enc->dispatchThreadgroups(MTL::Size(NumRDOBuckets * BlockSpaceSubdivision, 1, 1),
	                          MTL::Size(ResolveThreadgroupSize, 1, 1));
}

void dispatch_block_packing(pyrowave_encoder encoder, MTL::ComputeCommandEncoder *enc)
{
	const auto &layout = encoder->layout;
	const int precision = encoder->device->precision;

	// SPIRV-Cross renumbered these; the GLSL binding order is 1, 6, 4, 0, 5, 3.
	enc->setComputePipelineState(encoder->device->block_packing_pipeline);
	enc->setBuffer(encoder->payload_data, 0, 0);
	enc->setBuffer(encoder->bitstream, 0, 1);
	enc->setBuffer(encoder->quant_buffer, 0, 3);
	enc->setBuffer(encoder->meta_buffer, 0, 4);
	enc->setBuffer(encoder->block_stat_buffer, 0, 5);
	enc->setBuffer(encoder->bitstream_meta, 0, 6);

	for_each_band(layout, [&](int level, int component, int band) {
		BlockPackingPush push = {};
		push.resolution[0] = layout.level_width(level);
		push.resolution[1] = layout.level_height(level);
		push.resolution_32x32_blocks[0] = (push.resolution[0] + 31) / 32;
		push.resolution_32x32_blocks[1] = (push.resolution[1] + 31) / 32;
		push.resolution_8x8_blocks[0] = (push.resolution[0] + 7) / 8;
		push.resolution_8x8_blocks[1] = (push.resolution[1] + 7) / 8;

		// The code itself here, not the reciprocal the quantizer scales by.
		const float quant_res = get_quant_resolution(level, component, band, precision);
		push.quant_resolution_code = encode_quant(1.0f / quant_res);
		push.sequence_code = encoder->sequence_count;

		const auto &meta = layout.block_meta[component][level][band];
		push.block_offset_32x32 = meta.block_offset_32x32;
		push.block_stride_32x32 = meta.block_stride_32x32;
		push.block_offset_8x8 = meta.block_offset_8x8;
		push.block_stride_8x8 = meta.block_stride_8x8;

		enc->setBytes(&push, sizeof(push), 2);
		// Note the /2: unlike the other stages a threadgroup covers 2x2 of the
		// 32x32 blocks, one per 16 lanes.
		enc->dispatchThreadgroups(
				MTL::Size((push.resolution_32x32_blocks[0] + 1) / 2,
				          (push.resolution_32x32_blocks[1] + 1) / 2, 1),
				MTL::Size(BlockPackingThreadgroupSize, 1, 1));
	});
}

pyrowave_result encode_frame(pyrowave_encoder encoder, MTL::Texture *const planes[NumComponents],
                             const pyrowave_rate_control *rate_control)
{
	// Block packing writes u32 words, so round the budget down like the Vulkan
	// encoder does.
	const size_t target_size = rate_control->maximum_bitstream_size & ~size_t(3);
	if (target_size == 0 || target_size > UINT32_MAX)
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	// Same slack the Vulkan encoder leaves for the packer to overshoot into.
	const size_t meta_size = size_t(encoder->layout.block_count_32x32) * sizeof(BitstreamPacket);
	if (!ensure_bitstream_buffer(encoder, target_size + meta_size))
		return PYROWAVE_ERROR_OUT_OF_DEVICE_MEMORY;

	encoder->sequence_count = (encoder->sequence_count + 1) & SequenceCountMask;

	auto *cmd = encoder->queue->commandBuffer();
	if (!cmd)
		return PYROWAVE_ERROR_GENERIC;
	cmd->setLabel(NS::String::string("pyrowave encode", NS::UTF8StringEncoding));

	auto *blit = cmd->blitCommandEncoder();
	if (!blit)
		return PYROWAVE_ERROR_GENERIC;
	// The payload allocation counters, the RDO buckets and the per block quant
	// decisions are all accumulated into, so they start from zero. The rest of the
	// payload buffer is fully overwritten by whatever is coded.
	blit->fillBuffer(encoder->payload_data, NS::Range(0, 2 * sizeof(uint32_t)), 0);
	blit->fillBuffer(encoder->bucket_buffer, NS::Range(0, encoder->bucket_buffer->length()), 0);
	blit->fillBuffer(encoder->quant_buffer, NS::Range(0, encoder->quant_buffer->length()), 0);
	// A block's payload rarely ends on a word boundary, and the packer never writes
	// the leftover bytes of its final word. They still go out on the wire, so clear
	// them rather than shipping whatever this buffer last held.
	//
	// This does not make the bitstream byte for byte reproducible. The unused bits of
	// a block's last sign byte come from shared_sign_bank in block_packing.comp,
	// which is never initialized and is updated with atomicAnd/atomicOr that
	// preserve bits outside the write mask, so those bits are leftover threadgroup
	// memory. The decoder reads only as many sign bits as there are significant
	// coefficients and ignores the rest; the Vulkan encoder behaves the same way.
	blit->fillBuffer(encoder->bitstream, NS::Range(0, target_size + meta_size), 0);
	blit->endEncoding();

	// One concurrent encoder for the whole pipeline, with explicit barriers exactly
	// where the Vulkan encoder has them. Everything between two barriers is
	// mutually independent -- each dispatch owns a distinct (component, level, band)
	// region -- and a serial encoder would barrier between all ~170 of them.
	auto *enc = cmd->computeCommandEncoder(MTL::DispatchTypeConcurrent);
	if (!enc)
		return PYROWAVE_ERROR_GENERIC;
	enc->setLabel(NS::String::string("pyrowave encode", NS::UTF8StringEncoding));

	dispatch_dwt(encoder, enc, planes);
	// The quantizer samples the coefficients the DWT just wrote.
	enc->memoryBarrier(MTL::BarrierScopeTextures);
	dispatch_quant(encoder, enc);
	// Rate control analysis reads the per block statistics and payload sizes.
	enc->memoryBarrier(MTL::BarrierScopeBuffers);
	dispatch_analyze_rdo(encoder, enc);
	enc->memoryBarrier(MTL::BarrierScopeBuffers);
	dispatch_resolve_rdo(encoder, enc, target_size);
	// Packing needs the quant decisions resolve just made.
	enc->memoryBarrier(MTL::BarrierScopeBuffers);
	dispatch_block_packing(encoder, enc);

	enc->endEncoding();
	cmd->commit();

	if (encoder->pending)
		encoder->pending->release();
	encoder->pending = cmd->retain();
	encoder->have_result = true;

	return PYROWAVE_SUCCESS;
}

// Blocks until the last encode has landed, so the result buffers can be read.
pyrowave_result wait_for_result(pyrowave_encoder encoder)
{
	if (!encoder->have_result)
	{
		encoder->device->log("No frame has been encoded yet.");
		return PYROWAVE_ERROR_GENERIC;
	}

	if (encoder->pending)
	{
		encoder->pending->waitUntilCompleted();

		if (encoder->pending->status() == MTL::CommandBufferStatusError)
		{
			auto *error = encoder->pending->error();
			encoder->device->log("Encode command buffer failed: %s",
			                     error && error->localizedDescription() ?
			                     error->localizedDescription()->utf8String() : "unknown error");
			encoder->have_result = false;
			return PYROWAVE_ERROR_GENERIC;
		}

		encoder->pending->release();
		encoder->pending = nullptr;
	}

	return PYROWAVE_SUCCESS;
}

// Validates the caller's CPU buffer and copies it into the input textures.
pyrowave_result upload_cpu_input(pyrowave_encoder encoder, const pyrowave_cpu_buffer *input)
{
	auto *device = encoder->device;
	const auto &layout = encoder->layout;
	const bool chroma_420 = layout.chroma == ChromaSubsampling::Chroma420;
	const int num_planes = input->format == PYROWAVE_CPU_BUFFER_FORMAT_NV12 ? 2 : 3;

	if (input->width != layout.width || input->height != layout.height)
	{
		device->log("Input is %dx%d, but the encoder was created for %dx%d.",
		            input->width, input->height, layout.width, layout.height);
		return PYROWAVE_ERROR_INVALID_ARGUMENT;
	}

	if (chroma_420 == (input->format == PYROWAVE_CPU_BUFFER_FORMAT_YUV444P))
	{
		device->log("Input format %d does not match the encoder's chroma subsampling.",
		            int(input->format));
		return PYROWAVE_ERROR_INVALID_ARGUMENT;
	}

	for (int plane = 0; plane < num_planes; plane++)
	{
		const int plane_width = plane != 0 && chroma_420 ? layout.width / 2 : layout.width;
		const int plane_height = plane != 0 && chroma_420 ? layout.height / 2 : layout.height;
		// NV12 packs Cb and Cr into one 16bpp plane.
		const size_t bytes_per_pixel = num_planes == 2 && plane == 1 ? 2 : 1;

		if (!input->data[plane] ||
		    input->row_stride_in_bytes[plane] < size_t(plane_width) * bytes_per_pixel ||
		    input->row_stride_in_bytes[plane] * size_t(plane_height) > input->plane_size_in_bytes[plane])
		{
			device->log("Input plane %d is NULL or has an inconsistent stride and size.", plane);
			return PYROWAVE_ERROR_INVALID_ARGUMENT;
		}
	}

	if (!ensure_cpu_input(encoder, input->format))
		return PYROWAVE_ERROR_OUT_OF_DEVICE_MEMORY;

	// The textures are reused across frames, so the previous encode has to be done
	// reading them. In practice this is free: the packet queries have already
	// waited by the time a caller asks for another frame.
	if (encoder->pending)
		encoder->pending->waitUntilCompleted();

	// Both NV12 chroma views alias owned[1], so upload through the backing objects
	// rather than the sampled views.
	for (int plane = 0; plane < num_planes; plane++)
	{
		auto *texture = encoder->cpu_input.owned[plane];
		texture->replaceRegion(MTL::Region(0, 0, texture->width(), texture->height()),
		                       0, input->data[plane], input->row_stride_in_bytes[plane]);
	}

	return PYROWAVE_SUCCESS;
}

// Wraps the caller's IOSurfaces: either one biplanar NV12 surface, or three
// single plane R8 surfaces.
pyrowave_result wrap_gpu_input(pyrowave_encoder encoder, const pyrowave_gpu_input *input,
                               InputTextures &wrapped)
{
	auto *device = encoder->device;
	const auto &layout = encoder->layout;
	const bool chroma_420 = layout.chroma == ChromaSubsampling::Chroma420;
	const int chroma_width = chroma_420 ? layout.width / 2 : layout.width;
	const int chroma_height = chroma_420 ? layout.height / 2 : layout.height;

	auto surface = [&](int i) { return static_cast<IOSurfaceRef>(input->planes[i]); };

	if (!surface(0))
	{
		device->log("Input planes[0] is NULL.");
		return PYROWAVE_ERROR_INVALID_ARGUMENT;
	}

	const bool biplanar = !surface(1) && !surface(2);
	if (!biplanar && (!surface(1) || !surface(2)))
	{
		device->log("Expected either one biplanar surface or three single plane surfaces.");
		return PYROWAVE_ERROR_INVALID_ARGUMENT;
	}

	// Checks a surface plane against the geometry the shaders will read.
	auto plane_matches = [&](IOSurfaceRef surf, int plane, int width, int height) {
		if (int(IOSurfaceGetPlaneCount(surf)) <= plane && !(plane == 0 && IOSurfaceGetPlaneCount(surf) == 0))
		{
			device->log("Input surface has %zu planes, needs at least %d.",
			            size_t(IOSurfaceGetPlaneCount(surf)), plane + 1);
			return false;
		}

		// A non-planar surface reports 0 planes but still answers plane 0 queries.
		const int surface_width = int(IOSurfaceGetWidthOfPlane(surf, plane));
		const int surface_height = int(IOSurfaceGetHeightOfPlane(surf, plane));

		if (surface_width != width || surface_height != height)
		{
			device->log("Input surface plane %d is %dx%d, expected %dx%d.",
			            plane, surface_width, surface_height, width, height);
			return false;
		}

		return true;
	};

	if (biplanar)
	{
		if (!chroma_420)
		{
			device->log("Biplanar NV12 input has half resolution chroma, "
			            "so it cannot feed a 4:4:4 encoder.");
			return PYROWAVE_ERROR_INVALID_ARGUMENT;
		}

		if (IOSurfaceGetPlaneCount(surface(0)) != 2)
		{
			device->log("Expected a biplanar surface in planes[0], but it has %zu planes.",
			            size_t(IOSurfaceGetPlaneCount(surface(0))));
			return PYROWAVE_ERROR_INVALID_ARGUMENT;
		}

		if (!plane_matches(surface(0), 0, layout.width, layout.height) ||
		    !plane_matches(surface(0), 1, chroma_width, chroma_height))
			return PYROWAVE_ERROR_INVALID_ARGUMENT;

		wrapped.sampled[0] = wrapped.adopt(wrap_surface_plane(
				device, surface(0), 0, MTL::PixelFormatR8Unorm, layout.width, layout.height, false));

		auto *chroma = wrapped.adopt(wrap_surface_plane(
				device, surface(0), 1, MTL::PixelFormatRG8Unorm, chroma_width, chroma_height, true));

		if (!wrapped.sampled[0] || !chroma)
			return PYROWAVE_ERROR_OUT_OF_DEVICE_MEMORY;

		if (!make_interleaved_chroma_views(device, wrapped, chroma))
			return PYROWAVE_ERROR_OUT_OF_DEVICE_MEMORY;
	}
	else
	{
		for (int i = 0; i < NumComponents; i++)
		{
			const int width = i == 0 ? layout.width : chroma_width;
			const int height = i == 0 ? layout.height : chroma_height;

			if (!plane_matches(surface(i), 0, width, height))
				return PYROWAVE_ERROR_INVALID_ARGUMENT;

			wrapped.sampled[i] = wrapped.adopt(wrap_surface_plane(
					device, surface(i), 0, MTL::PixelFormatR8Unorm, width, height, false));

			if (!wrapped.sampled[i])
				return PYROWAVE_ERROR_OUT_OF_DEVICE_MEMORY;
		}
	}

	return PYROWAVE_SUCCESS;
}

std::unique_ptr<NS::AutoreleasePool, void (*)(NS::AutoreleasePool *)> scoped_pool()
{
	return { NS::AutoreleasePool::alloc()->init(), [](NS::AutoreleasePool *p) { p->release(); } };
}
}

//////
// Public API

pyrowave_result pyrowave_encoder_create(const pyrowave_encoder_create_info *info,
                                        pyrowave_encoder *encoder)
{
	if (!info || !encoder || !info->device)
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

	auto pool = scoped_pool();

	// Deferred rather than done in pyrowave_device_create(), so that decode-only
	// users do not pay for six extra shader compiles.
	if (!ensure_encode_pipelines(info->device))
		return PYROWAVE_ERROR_SHADER_COMPILATION;

	auto created = std::unique_ptr<pyrowave_encoder_opaque>(new (std::nothrow) pyrowave_encoder_opaque);
	if (!created)
		return PYROWAVE_ERROR_OUT_OF_HOST_MEMORY;

	created->device = info->device;

	if (!created->layout.init(info->width, info->height,
	                          chroma_420 ? ChromaSubsampling::Chroma420 : ChromaSubsampling::Chroma444))
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	created->queue = created->device->mtl->newCommandQueue();
	if (!created->queue)
		return PYROWAVE_ERROR_OUT_OF_DEVICE_MEMORY;

	if (!created->wavelet.init(created->device, created->layout) ||
	    !create_encode_buffers(created.get()))
		return PYROWAVE_ERROR_OUT_OF_DEVICE_MEMORY;

	*encoder = created.release();
	return PYROWAVE_SUCCESS;
}

void pyrowave_encoder_destroy(pyrowave_encoder encoder)
{
	delete encoder;
}

pyrowave_result pyrowave_encoder_encode_gpu_synchronous(pyrowave_encoder encoder,
                                                        const pyrowave_gpu_input *input,
                                                        const pyrowave_rate_control *rate_control)
{
	if (!encoder || !input || !rate_control)
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	auto pool = scoped_pool();

	InputTextures wrapped;
	auto result = wrap_gpu_input(encoder, input, wrapped);
	if (result != PYROWAVE_SUCCESS)
		return result;

	// The command buffer retains the wrapping textures, so letting them go here is
	// safe even though the GPU has not run yet.
	return encode_frame(encoder, wrapped.sampled, rate_control);
}

pyrowave_result pyrowave_encoder_encode_cpu_synchronous(pyrowave_encoder encoder,
                                                        const pyrowave_cpu_buffer *input,
                                                        const pyrowave_rate_control *rate_control)
{
	if (!encoder || !input || !rate_control)
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	if (input->format != PYROWAVE_CPU_BUFFER_FORMAT_NV12 &&
	    input->format != PYROWAVE_CPU_BUFFER_FORMAT_YUV420P &&
	    input->format != PYROWAVE_CPU_BUFFER_FORMAT_YUV444P)
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	auto pool = scoped_pool();

	auto result = upload_cpu_input(encoder, input);
	if (result != PYROWAVE_SUCCESS)
		return result;

	return encode_frame(encoder, encoder->cpu_input.sampled, rate_control);
}

pyrowave_result pyrowave_encoder_compute_num_packets(pyrowave_encoder encoder, size_t packet_boundary,
                                                     size_t *num_packets)
{
	if (!encoder || !num_packets || packet_boundary < sizeof(BitstreamSequenceHeader))
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	auto result = wait_for_result(encoder);
	if (result != PYROWAVE_SUCCESS)
		return result;

	*num_packets = compute_num_packets(encoder->layout, encoder->bitstream_meta->contents(),
	                                   packet_boundary);
	return PYROWAVE_SUCCESS;
}

pyrowave_result pyrowave_encoder_packetize(pyrowave_encoder encoder, pyrowave_packet *packets,
                                           size_t packet_boundary, size_t *out_packets,
                                           void *bitstream, size_t size)
{
	if (!encoder || !packets || !out_packets || !bitstream ||
	    packet_boundary < sizeof(BitstreamSequenceHeader))
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	auto result = wait_for_result(encoder);
	if (result != PYROWAVE_SUCCESS)
		return result;

	static_assert(sizeof(pyrowave_packet) == sizeof(Packet), "pyrowave_packet layout mismatch.");
	*out_packets = packetize(encoder->layout, reinterpret_cast<Packet *>(packets), packet_boundary,
	                         bitstream, size,
	                         encoder->bitstream_meta->contents(), encoder->bitstream->contents());
	return PYROWAVE_SUCCESS;
}
