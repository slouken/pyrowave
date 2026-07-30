#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Metal/Metal.hpp>
#include <CoreVideo/CoreVideo.h>
#include <stdio.h>
#include <string.h>
#include <vector>

int main() {
    const int W=640,H=480;
    auto *dev = MTL::CreateSystemDefaultDevice();

    CFDictionaryRef io = CFDictionaryCreate(kCFAllocatorDefault,nullptr,nullptr,0,
        &kCFTypeDictionaryKeyCallBacks,&kCFTypeDictionaryValueCallBacks);
    CFMutableDictionaryRef a = CFDictionaryCreateMutable(kCFAllocatorDefault,1,
        &kCFTypeDictionaryKeyCallBacks,&kCFTypeDictionaryValueCallBacks);
    CFDictionarySetValue(a, kCVPixelBufferIOSurfacePropertiesKey, io);
    CVPixelBufferRef pb=nullptr;
    CVPixelBufferCreate(kCFAllocatorDefault,W,H,kCVPixelFormatType_420YpCbCr8Planar,a,&pb);
    CFRelease(a); CFRelease(io);
    if (!pb) { printf("no pixel buffer\n"); return 1; }

    // Fill plane 1 (U) with 0xAA and plane 2 (V) with 0xBB.
    CVPixelBufferLockBaseAddress(pb, 0);
    for (int p = 1; p <= 2; p++) {
        auto *base = (uint8_t*)CVPixelBufferGetBaseAddressOfPlane(pb,p);
        size_t bpr = CVPixelBufferGetBytesPerRowOfPlane(pb,p);
        size_t h   = CVPixelBufferGetHeightOfPlane(pb,p);
        memset(base, p==1 ? 0xAA : 0xBB, bpr*h);
    }
    CVPixelBufferUnlockBaseAddress(pb, 0);

    IOSurfaceRef s = CVPixelBufferGetIOSurface(pb);
    const int cw = W/2, ch = H/2;

    auto *d = MTL::TextureDescriptor::alloc()->init();
    d->setTextureType(MTL::TextureType2DArray);
    d->setPixelFormat(MTL::PixelFormatR8Unorm);
    d->setWidth(cw); d->setHeight(ch); d->setArrayLength(2);
    d->setUsage(MTL::TextureUsageShaderRead);
    auto *tex = dev->newTexture(d, s, 1);
    d->release();
    if (!tex) { printf("array texture creation FAILED\n"); return 1; }
    printf("array texture created: OK (storageMode=%d)\n", (int)tex->storageMode());

    std::vector<uint8_t> buf(size_t(cw)*ch);
    for (int slice = 0; slice < 2; slice++) {
        memset(buf.data(), 0, buf.size());
        tex->getBytes(buf.data(), cw, cw*ch, MTL::Region(0,0,cw,ch), 0, slice);
        // Summarise what came back.
        bool allA = true, allB = true, allZero = true;
        for (auto v : buf) { if (v!=0xAA) allA=false; if (v!=0xBB) allB=false; if (v!=0) allZero=false; }
        printf("slice %d first bytes: %02x %02x %02x -> %s\n", slice,
               buf[0], buf[1], buf[2],
               allA ? "ALL 0xAA (plane 1 / U)" :
               allB ? "ALL 0xBB (plane 2 / V)" :
               allZero ? "ALL ZERO" : "MIXED/garbage");
    }
    printf("\nExpected for correct behaviour: slice 0 = 0xAA, slice 1 = 0xBB\n");
    CVPixelBufferRelease(pb);
    return 0;
}
