#include <Metal/Metal.hpp>
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
static MTL::Texture *plane(MTL::Device *d,int w,int h){
    auto *desc=MTL::TextureDescriptor::alloc()->init();
    desc->setTextureType(MTL::TextureType2D); desc->setPixelFormat(MTL::PixelFormatR8Unorm);
    desc->setWidth(w); desc->setHeight(h);
    desc->setUsage(MTL::TextureUsageShaderWrite|MTL::TextureUsageShaderRead);
    desc->setStorageMode(MTL::StorageModePrivate);
    auto *t=d->newTexture(desc); desc->release(); return t;
}

int main(int argc,char**argv){
    auto data=slurp(argv[1]);
    int32_t p[8]; memcpy(p,data.data()+8,sizeof(p));
    int W=p[0],H=p[1]; bool is420 = p[3]==0;
    // first frame blob
    uint32_t fsz; memcpy(&fsz,data.data()+40,4);
    const uint8_t *frame=data.data()+44;

    auto *mtl=MTL::CreateSystemDefaultDevice();
    pyrowave_device_create_info di={}; di.mtl_device=mtl;
    pyrowave_device dev=nullptr; pyrowave_device_create(&di,&dev);
    pyrowave_decoder_create_info ci={}; ci.device=dev; ci.width=W; ci.height=H;
    ci.chroma= is420?PYROWAVE_CHROMA_SUBSAMPLING_420:PYROWAVE_CHROMA_SUBSAMPLING_444;
    pyrowave_decoder dec=nullptr; pyrowave_decoder_create(&ci,&dec);

    int cw=is420?W/2:W, ch=is420?H/2:H;
    MTL::Texture *pl[3]={plane(mtl,W,H),plane(mtl,cw,ch),plane(mtl,cw,ch)};
    pyrowave_gpu_buffers bufs={}; for(int i=0;i<3;i++) bufs.planes[i]=pl[i];
    auto *q=mtl->newCommandQueue();

    std::vector<double> ms;
    for(int i=0;i<400;i++){
        auto *pool=NS::AutoreleasePool::alloc()->init();
        pyrowave_decoder_clear(dec);
        pyrowave_decoder_push_packet(dec,frame,fsz);
        auto *cb=q->commandBuffer();
        pyrowave_decoder_decode(dec,cb,&bufs);
        cb->commit(); cb->waitUntilCompleted();
        if(i>=40) ms.push_back((cb->GPUEndTime()-cb->GPUStartTime())*1000.0);
        pool->release();
    }
    std::sort(ms.begin(),ms.end());
    double p10 = ms[(size_t)(ms.size()*0.10)];
    printf("min %.3f  p10 %.3f  median %.3f\n", ms.front(), p10, ms[ms.size()/2]);
    return 0;
}
