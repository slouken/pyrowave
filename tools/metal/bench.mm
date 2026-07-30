#import <Metal/Metal.h>
#include "pyrowave_metal.h"
#include "pyrowave_bitstream.hpp"
#include <algorithm>
#include <stdio.h>
#include <vector>
using namespace PyroWave;

static std::vector<uint8_t> slurp(const char *p) {
    FILE *f=fopen(p,"rb"); fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    std::vector<uint8_t> v(n); size_t rd = fread(v.data(),1,n,f); (void)rd; fclose(f); return v;
}
static id<MTLTexture> plane(id<MTLDevice> d,int w,int h){
    auto *desc=[MTLTextureDescriptor new];
    desc.textureType = MTLTextureType2D; desc.pixelFormat = MTLPixelFormatR8Unorm;
    desc.width = w; desc.height = h;
    desc.usage = MTLTextureUsageShaderWrite|MTLTextureUsageShaderRead;
    desc.storageMode = MTLStorageModePrivate;
    auto *t=[d newTextureWithDescriptor:desc]; return t;
}

int main(int argc,char**argv){
    auto data=slurp(argv[1]);
    int32_t p[8]; memcpy(p,data.data()+8,sizeof(p));
    int W=p[0],H=p[1]; bool is420 = p[3]==0;
    // first frame blob
    uint32_t fsz; memcpy(&fsz,data.data()+40,4);
    const uint8_t *frame=data.data()+44;

    auto *mtl=MTLCreateSystemDefaultDevice();
    pyrowave_device_create_info di={}; di.mtl_device = (__bridge void *)mtl;
    pyrowave_device dev=nullptr; pyrowave_device_create(&di,&dev);
    pyrowave_decoder_create_info ci={}; ci.device=dev; ci.width=W; ci.height=H;
    ci.chroma= is420?PYROWAVE_CHROMA_SUBSAMPLING_420:PYROWAVE_CHROMA_SUBSAMPLING_444;
    pyrowave_decoder dec=nullptr; pyrowave_decoder_create(&ci,&dec);

    int cw=is420?W/2:W, ch=is420?H/2:H;
    id<MTLTexture> pl[3]={plane(mtl,W,H),plane(mtl,cw,ch),plane(mtl,cw,ch)};
    pyrowave_gpu_buffers bufs={}; for(int i=0;i<3;i++) bufs.planes[i] = (__bridge void *)pl[i];
    auto *q=[mtl newCommandQueue];

    std::vector<double> ms;
    for(int i=0;i<400;i++){
        pyrowave_decoder_clear(dec);
        pyrowave_decoder_push_packet(dec,frame,fsz);
        id<MTLCommandBuffer> cb=[q commandBuffer];
        pyrowave_decoder_decode(dec,(__bridge void *)cb,&bufs);
        [cb commit]; [cb waitUntilCompleted];
        if(i>=40) ms.push_back((cb.GPUEndTime-cb.GPUStartTime)*1000.0);
    }
    std::sort(ms.begin(),ms.end());
    double p10 = ms[(size_t)(ms.size()*0.10)];
    printf("min %.3f  p10 %.3f  median %.3f\n", ms.front(), p10, ms[ms.size()/2]);
    return 0;
}
