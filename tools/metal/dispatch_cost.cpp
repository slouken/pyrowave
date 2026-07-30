// Measures the marginal cost of a compute dispatch by sweeping dispatch count
// with a trivial kernel, serial vs concurrent encoder. Slope = per-dispatch cost.
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Metal/Metal.hpp>
#include <algorithm>
#include <stdio.h>
#include <vector>

static const char *src = R"(
#include <metal_stdlib>
using namespace metal;
kernel void nop(device uint *out [[buffer(0)]],
                constant uint &slot [[buffer(1)]],
                uint tid [[thread_position_in_grid]])
{
    if (tid == 0) out[slot] = slot;   // disjoint write per dispatch
}
)";

int main(){
    auto *dev = MTL::CreateSystemDefaultDevice();
    NS::Error *err=nullptr;
    auto *lib = dev->newLibrary(NS::String::string(src, NS::UTF8StringEncoding), nullptr, &err);
    auto *fn = lib->newFunction(NS::String::string("nop", NS::UTF8StringEncoding));
    auto *pipe = dev->newComputePipelineState(fn, &err);
    auto *buf = dev->newBuffer(4096*4, MTL::ResourceStorageModePrivate);
    auto *q = dev->newCommandQueue();

    printf("%-10s %14s %14s\n", "dispatches", "serial (ms)", "concurrent (ms)");
    std::vector<int> counts = {1, 10, 20, 40, 80, 160};
    double s_first=0, s_last=0, c_first=0, c_last=0;
    for (size_t k=0;k<counts.size();k++) {
        int n = counts[k];
        double best[2] = {1e9, 1e9};
        for (int mode=0; mode<2; mode++) {
            for (int rep=0; rep<80; rep++) {
                auto *pool = NS::AutoreleasePool::alloc()->init();
                auto *cb = q->commandBuffer();
                auto *enc = mode==0 ? cb->computeCommandEncoder()
                                    : cb->computeCommandEncoder(MTL::DispatchTypeConcurrent);
                enc->setComputePipelineState(pipe);
                enc->setBuffer(buf, 0, 0);
                for (int i=0;i<n;i++) {
                    uint32_t slot = i;
                    enc->setBytes(&slot, sizeof(slot), 1);
                    enc->dispatchThreadgroups(MTL::Size(1,1,1), MTL::Size(64,1,1));
                }
                enc->endEncoding();
                cb->commit(); cb->waitUntilCompleted();
                if (rep>=20) best[mode] = std::min(best[mode], (cb->GPUEndTime()-cb->GPUStartTime())*1000.0);
                pool->release();
            }
        }
        printf("%-10d %14.4f %14.4f\n", n, best[0], best[1]);
        if (k==0) { s_first=best[0]; c_first=best[1]; }
        if (k==counts.size()-1) { s_last=best[0]; c_last=best[1]; }
    }
    int span = counts.back()-counts.front();
    printf("\nmarginal cost per dispatch:\n");
    printf("  serial:     %.2f us\n", (s_last-s_first)*1000.0/span);
    printf("  concurrent: %.2f us\n", (c_last-c_first)*1000.0/span);
    return 0;
}
