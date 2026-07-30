#import <Metal/Metal.h>
#include <CoreVideo/CoreVideo.h>
#include <stdio.h>

static CVPixelBufferRef make(int w, int h, OSType fmt) {
    CFDictionaryRef io = CFDictionaryCreate(kCFAllocatorDefault,nullptr,nullptr,0,
        &kCFTypeDictionaryKeyCallBacks,&kCFTypeDictionaryValueCallBacks);
    CFMutableDictionaryRef a = CFDictionaryCreateMutable(kCFAllocatorDefault,1,
        &kCFTypeDictionaryKeyCallBacks,&kCFTypeDictionaryValueCallBacks);
    CFDictionarySetValue(a, kCVPixelBufferIOSurfacePropertiesKey, io);
    CVPixelBufferRef pb=nullptr;
    CVReturn r = CVPixelBufferCreate(kCFAllocatorDefault,w,h,fmt,a,&pb);
    CFRelease(a); CFRelease(io);
    if (r != kCVReturnSuccess) { printf("  CVPixelBufferCreate failed: %d\n", r); return nullptr; }
    return pb;
}

static void try_fmt(id<MTLDevice> dev, const char *name, OSType fmt, int w, int h) {
    printf("%s (%.4s):\n", name, (const char*)&fmt);
    auto *pb = make(w,h,fmt);
    if (!pb) return;
    IOSurfaceRef s = CVPixelBufferGetIOSurface(pb);
    printf("  planes=%zu iosurface=%s\n", CVPixelBufferGetPlaneCount(pb), s?"yes":"NO");
    if (!s) { CVPixelBufferRelease(pb); return; }
    for (size_t p = 0; p < CVPixelBufferGetPlaneCount(pb); p++)
        printf("    plane %zu: %zux%zu\n", p, CVPixelBufferGetWidthOfPlane(pb,p), CVPixelBufferGetHeightOfPlane(pb,p));

    // 2D single-slice view of each plane
    for (size_t p = 0; p < CVPixelBufferGetPlaneCount(pb); p++) {
        auto d = [MTLTextureDescriptor new];
        d.textureType = MTLTextureType2D;
        d.pixelFormat = MTLPixelFormatR8Unorm;
        d.width = CVPixelBufferGetWidthOfPlane(pb,p);
        d.height = CVPixelBufferGetHeightOfPlane(pb,p);
        d.usage = MTLTextureUsageShaderWrite|MTLTextureUsageShaderRead;
        auto t = [dev newTextureWithDescriptor:d iosurface:s plane:p];
        printf("    2D R8 texture from plane %zu: %s\n", p, t?"OK":"FAILED");
    }

    // What SDL does for P408/IYUV: 2-layer array from plane 1
    {
        auto d = [MTLTextureDescriptor new];
        d.textureType = MTLTextureType2DArray;
        d.pixelFormat = MTLPixelFormatR8Unorm;
        d.width = CVPixelBufferGetWidthOfPlane(pb,1);
        d.height = CVPixelBufferGetHeightOfPlane(pb,1);
        d.arrayLength = 2;
        d.usage = MTLTextureUsageShaderRead;
        auto t = [dev newTextureWithDescriptor:d iosurface:s plane:1];
        printf("    2DArray(len=2) from plane 1 [what SDL does]: %s\n", t?"OK":"FAILED");
    }
    CVPixelBufferRelease(pb);
    printf("\n");
}

int main() {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    try_fmt(dev, "420 3-plane planar", kCVPixelFormatType_420YpCbCr8Planar, 640, 480);
    try_fmt(dev, "420 biplanar video", kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange, 640, 480);
    try_fmt(dev, "444 biplanar video", kCVPixelFormatType_444YpCbCr8BiPlanarVideoRange, 640, 480);
}
