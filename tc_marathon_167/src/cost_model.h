// Deterministic effort accounting (P2-03).
#ifndef MM167_COST_MODEL_H
#define MM167_COST_MODEL_H

#include "generator.h"

namespace mm167 {
struct LaneModel;

// P2-03 deterministic cost model: predicted nanoseconds on the reference machine
// (Apple M4, clang -O3), least-squares fit on profiled training turns at pool
// sizes 128/512/2048. Every decision that depends on spent effort reads this
// count instead of the wall clock, so a game is reproducible wherever the
// wall-clock safety guard stays inactive.
namespace cost {
constexpr double NODE=10,CROSS=87,RUN=3400,BOARD=6400;                    // second-ply generator run
constexpr double ROOT_NODE=23,ROOT_CROSS=315,RETAIN=510,RETAIN_SCAN=6;     // root generator keeping a pool
constexpr double SCAN=5.2,BUILD=13.4;                                      // lane word lists
}
inline double rootCost(const Generator& gen) {
    return cost::ROOT_NODE*double(gen.nodes)+cost::ROOT_CROSS*double(gen.crossCells)
        +cost::RETAIN*double(gen.retained)+cost::RETAIN_SCAN*double(gen.retainScan);
}
double prefilterCost(const LaneModel& lane,uint64_t scan0,uint64_t build0);
}
#endif
