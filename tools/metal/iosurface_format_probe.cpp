#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Metal/Metal.hpp>
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

static void try_fmt(MTL::Device *dev, const char *name, OSType fmt, int w, int h) {
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
        auto *d = MTL::TextureDescriptor::alloc()->init();
        d->setTextureType(MTL::TextureType2D);
        d->setPixelFormat(MTL::PixelFormatR8Unorm);
        d->setWidth(CVPixelBufferGetWidthOfPlane(pb,p));
        d->setHeight(CVPixelBufferGetHeightOfPlane(pb,p));
        d->setUsage(MTL::TextureUsageShaderWrite|MTL::TextureUsageShaderRead);
        auto *t = dev->newTexture(d, s, p);
        printf("    2D R8 texture from plane %zu: %s\n", p, t?"OK":"FAILED");
        if (t) t->release();
        d->release();
    }

    // What SDL does for P408/IYUV: 2-layer array from plane 1
    {
        auto *d = MTL::TextureDescriptor::alloc()->init();
        d->setTextureType(MTL::TextureType2DArray);
        d->setPixelFormat(MTL::PixelFormatR8Unorm);
        d->setWidth(CVPixelBufferGetWidthOfPlane(pb,1));
        d->setHeight(CVPixelBufferGetHeightOfPlane(pb,1));
        d->setArrayLength(2);
        d->setUsage(MTL::TextureUsageShaderRead);
        auto *t = dev->newTexture(d, s, 1);
        printf("    2DArray(len=2) from plane 1 [what SDL does]: %s\n", t?"OK":"FAILED");
        if (t) t->release();
        d->release();
    }
    CVPixelBufferRelease(pb);
    printf("\n");
}

int main() {
    auto *dev = MTL::CreateSystemDefaultDevice();
    try_fmt(dev, "420 3-plane planar", kCVPixelFormatType_420YpCbCr8Planar, 640, 480);
    try_fmt(dev, "420 biplanar video", kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange, 640, 480);
    try_fmt(dev, "444 biplanar video", kCVPixelFormatType_444YpCbCr8BiPlanarVideoRange, 640, 480);
    dev->release();
}
