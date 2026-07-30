#include <CoreVideo/CoreVideo.h>
#include <IOSurface/IOSurfaceRef.h>
#include <stdio.h>

static void num(CFMutableDictionaryRef d, CFStringRef k, int v) {
    CFNumberRef n = CFNumberCreate(kCFAllocatorDefault, kCFNumberIntType, &v);
    CFDictionarySetValue(d, k, n); CFRelease(n);
}

int main() {
    const int W=640,H=480;
    // Hand-built 3-plane 4:4:4 surface: Y, Cb, Cr each full resolution R8.
    CFMutableArrayRef planes = CFArrayCreateMutable(kCFAllocatorDefault,3,&kCFTypeArrayCallBacks);
    for (int i=0;i<3;i++) {
        CFMutableDictionaryRef p = CFDictionaryCreateMutable(kCFAllocatorDefault,4,
            &kCFTypeDictionaryKeyCallBacks,&kCFTypeDictionaryValueCallBacks);
        num(p, kIOSurfacePlaneWidth, W);
        num(p, kIOSurfacePlaneHeight, H);
        num(p, kIOSurfacePlaneBytesPerElement, 1);
        num(p, kIOSurfacePlaneBytesPerRow, W);
        CFArrayAppendValue(planes, p); CFRelease(p);
    }
    CFMutableDictionaryRef props = CFDictionaryCreateMutable(kCFAllocatorDefault,5,
        &kCFTypeDictionaryKeyCallBacks,&kCFTypeDictionaryValueCallBacks);
    num(props, kIOSurfaceWidth, W);
    num(props, kIOSurfaceHeight, H);
    num(props, kIOSurfacePixelFormat, 'v444');
    CFDictionarySetValue(props, kIOSurfacePlaneInfo, planes);
    IOSurfaceRef s = IOSurfaceCreate(props);
    printf("IOSurfaceCreate 3-plane 444: %s\n", s?"OK":"FAILED");
    if (s) {
        printf("  planes=%zu\n", IOSurfaceGetPlaneCount(s));
        for (size_t i=0;i<IOSurfaceGetPlaneCount(s);i++)
            printf("    plane %zu: %zux%zu bpr=%zu offset=%zu\n", i,
                   IOSurfaceGetWidthOfPlane(s,i), IOSurfaceGetHeightOfPlane(s,i),
                   IOSurfaceGetBytesPerRowOfPlane(s,i),
                   (size_t)((char*)IOSurfaceGetBaseAddressOfPlane(s,i)-(char*)IOSurfaceGetBaseAddress(s)));
        CVPixelBufferRef pb=nullptr;
        CVReturn r = CVPixelBufferCreateWithIOSurface(kCFAllocatorDefault, s, nullptr, &pb);
        printf("  CVPixelBufferCreateWithIOSurface: %s (%d)\n", r==kCVReturnSuccess?"OK":"FAILED", r);
        if (pb) { OSType f = CVPixelBufferGetPixelFormatType(pb);
                  printf("  CV planes=%zu fmt=%.4s\n", CVPixelBufferGetPlaneCount(pb), (const char*)&f);
                  CVPixelBufferRelease(pb); }
    }
    CFRelease(props); CFRelease(planes);
}
