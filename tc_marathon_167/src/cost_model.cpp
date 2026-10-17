#include "cost_model.h"

#include "lane_model.h"

namespace mm167 {
double prefilterCost(const LaneModel& lane,uint64_t scan0,uint64_t build0) {
    return cost::SCAN*double(lane.scanWords-scan0)+cost::BUILD*double(lane.buildWords-build0);
}
}
