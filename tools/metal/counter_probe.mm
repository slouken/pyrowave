#import <Metal/Metal.h>
#include <stdio.h>
int main(){
    id<MTLDevice> d = MTLCreateSystemDefaultDevice();
    printf("device: %s\n", d.name.UTF8String);
    const char *names[] = {"StageBoundary","DrawBoundary","DispatchBoundary","TileDispatchBoundary","BlitBoundary"};
    for (int i=0;i<5;i++)
        printf("  sampling at %-22s : %s\n", names[i],
               [d supportsCounterSampling:(MTLCounterSamplingPoint)i] ? "YES" : "no");
    NSArray<id<MTLCounterSet>> *sets = d.counterSets;
    printf("counter sets: %lu\n", (unsigned long)(sets ? sets.count : 0));
    for (id<MTLCounterSet> cs in sets)
        printf("  %s\n", cs.name.UTF8String);
    return 0;
}
