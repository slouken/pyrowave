// Measures the marginal cost of a compute dispatch by sweeping dispatch count
// with a trivial kernel, serial vs concurrent encoder. Slope = per-dispatch cost.
#import <Metal/Metal.h>
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
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSError *err=nil;
    auto lib = [dev newLibraryWithSource:@(src) options:nil error:&err];
    auto fn = [lib newFunctionWithName:@"nop"];
    auto pipe = [dev newComputePipelineStateWithFunction:fn error:&err];
    auto buf = [dev newBufferWithLength:4096*4 options:MTLResourceStorageModePrivate];
    auto q = [dev newCommandQueue];

    printf("%-10s %14s %14s\n", "dispatches", "serial (ms)", "concurrent (ms)");
    std::vector<int> counts = {1, 10, 20, 40, 80, 160};
    double s_first=0, s_last=0, c_first=0, c_last=0;
    for (size_t k=0;k<counts.size();k++) {
        int n = counts[k];
        double best[2] = {1e9, 1e9};
        for (int mode=0; mode<2; mode++) {
            for (int rep=0; rep<80; rep++) {
                auto cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> enc = mode==0 ? [cb computeCommandEncoder]
                                    : [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
                [enc setComputePipelineState:pipe];
                [enc setBuffer:buf offset:0 atIndex:0];
                for (int i=0;i<n;i++) {
                    uint32_t slot = i;
                    [enc setBytes:&slot length:sizeof(slot) atIndex:1];
                    [enc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
                }
                [enc endEncoding];
                [cb commit]; [cb waitUntilCompleted];
                if (rep>=20) best[mode] = std::min(best[mode], (cb.GPUEndTime-cb.GPUStartTime)*1000.0);
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
