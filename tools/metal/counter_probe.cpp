#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Metal/Metal.hpp>
#include <stdio.h>
int main(){
    auto *d = MTL::CreateSystemDefaultDevice();
    printf("device: %s\n", d->name()->utf8String());
    const char *names[] = {"StageBoundary","DrawBoundary","DispatchBoundary","TileDispatchBoundary","BlitBoundary"};
    for (int i=0;i<5;i++)
        printf("  sampling at %-22s : %s\n", names[i],
               d->supportsCounterSampling((MTL::CounterSamplingPoint)i) ? "YES" : "no");
    auto *sets = d->counterSets();
    printf("counter sets: %lu\n", (unsigned long)(sets?sets->count():0));
    for (NS::UInteger i=0; sets && i<sets->count(); i++) {
        auto *cs = (MTL::CounterSet*)sets->object(i);
        printf("  %s\n", cs->name()->utf8String());
    }
    return 0;
}
