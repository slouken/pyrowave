// Renders a solid YUV colour through the per-plane IOSurface path and checks RGB.
// Y=128 U=200 V=60 under BT.709 limited gives roughly R=9 G=168 B=255 (blue).
// If U and V are swapped it comes out red instead, which is unmistakable.
#include <SDL3/SDL.h>
#include <IOSurface/IOSurfaceRef.h>
#include <stdio.h>
#include <string.h>

static void num(CFMutableDictionaryRef d, CFStringRef k, int v) {
    CFNumberRef n = CFNumberCreate(kCFAllocatorDefault, kCFNumberIntType, &v);
    CFDictionarySetValue(d, k, n); CFRelease(n);
}
static IOSurfaceRef mk(int w, int h, unsigned char fill) {
    CFMutableDictionaryRef p = CFDictionaryCreateMutable(kCFAllocatorDefault,5,
        &kCFTypeDictionaryKeyCallBacks,&kCFTypeDictionaryValueCallBacks);
    num(p,kIOSurfaceWidth,w); num(p,kIOSurfaceHeight,h);
    num(p,kIOSurfaceBytesPerElement,1); num(p,kIOSurfacePixelFormat,'L008');
    IOSurfaceRef s = IOSurfaceCreate(p); CFRelease(p);
    IOSurfaceLock(s,0,NULL);
    unsigned char *base = IOSurfaceGetBaseAddress(s);
    size_t bpr = IOSurfaceGetBytesPerRow(s);
    for (int y=0;y<h;y++) memset(base+y*bpr, fill, w);
    IOSurfaceUnlock(s,0,NULL);
    return s;
}

static int test_format(SDL_Renderer *r, SDL_PixelFormat fmt, const char *name,
                       IOSurfaceRef sy, IOSurfaceRef s1, IOSurfaceRef s2, int W, int H)
{
    SDL_PropertiesID props = SDL_CreateProperties();
    SDL_SetNumberProperty(props, SDL_PROP_TEXTURE_CREATE_FORMAT_NUMBER, fmt);
    SDL_SetNumberProperty(props, SDL_PROP_TEXTURE_CREATE_ACCESS_NUMBER, SDL_TEXTUREACCESS_STATIC);
    SDL_SetNumberProperty(props, SDL_PROP_TEXTURE_CREATE_WIDTH_NUMBER, W);
    SDL_SetNumberProperty(props, SDL_PROP_TEXTURE_CREATE_HEIGHT_NUMBER, H);
    SDL_SetNumberProperty(props, SDL_PROP_TEXTURE_CREATE_COLORSPACE_NUMBER, SDL_COLORSPACE_BT709_LIMITED);
    SDL_SetPointerProperty(props, SDL_PROP_TEXTURE_CREATE_METAL_IOSURFACE_POINTER, sy);
    SDL_SetPointerProperty(props, SDL_PROP_TEXTURE_CREATE_METAL_IOSURFACE_U_POINTER, s1);
    SDL_SetPointerProperty(props, SDL_PROP_TEXTURE_CREATE_METAL_IOSURFACE_V_POINTER, s2);
    SDL_Texture *t = SDL_CreateTextureWithProperties(r, props);
    SDL_DestroyProperties(props);
    if (!t) { printf("%-6s CREATE FAILED: %s\n", name, SDL_GetError()); return 1; }

    SDL_SetRenderDrawColor(r,0,0,0,255); SDL_RenderClear(r);
    SDL_RenderTexture(r, t, NULL, NULL);
    SDL_Rect rect = {0,0,8,8};
    SDL_Surface *out = SDL_RenderReadPixels(r, &rect);
    if (!out) { printf("%-6s READPIXELS FAILED\n", name); return 1; }
    Uint8 rr,gg,bb;
    SDL_ReadSurfacePixel(out, 4, 4, &rr, &gg, &bb, NULL);
    SDL_DestroySurface(out);
    SDL_DestroyTexture(t);

    const char *verdict;
    if (rr < 60 && bb > 180)       verdict = "OK (blue, U/V correct)";
    else if (rr > 180 && bb < 60)  verdict = "SWAPPED (red -> U/V reversed)";
    else                           verdict = "UNEXPECTED";
    printf("%-6s R=%3d G=%3d B=%3d  %s\n", name, rr, gg, bb, verdict);
    return strncmp(verdict,"OK",2) != 0;
}

int main(void) {
    const int W=64,H=64;
    SDL_SetHint(SDL_HINT_RENDER_DRIVER,"metal");
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window *w; SDL_Renderer *r;
    SDL_CreateWindowAndRenderer("planes",W,H,SDL_WINDOW_HIDDEN,&w,&r);
    printf("renderer: %s\n\n", SDL_GetRendererName(r));

    IOSurfaceRef sy = mk(W,H,128);
    IOSurfaceRef su = mk(W/2,H/2,200);   // U
    IOSurfaceRef sv = mk(W/2,H/2,60);    // V
    IOSurfaceRef sy4 = mk(W,H,128);
    IOSurfaceRef su4 = mk(W,H,200);
    IOSurfaceRef sv4 = mk(W,H,60);

    int bad = 0;
    printf("Passing plane1=U, plane2=V for every format:\n");
    bad |= test_format(r, SDL_PIXELFORMAT_IYUV, "IYUV", sy, su, sv, W, H);
    bad |= test_format(r, SDL_PIXELFORMAT_YV12, "YV12", sy, su, sv, W, H);
    bad |= test_format(r, SDL_PIXELFORMAT_P408, "P408", sy4, su4, sv4, W, H);

    SDL_Quit();
    return bad;
}
