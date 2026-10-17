// Policy-4 fishing: target-specific keeps valued by lane refill chances (O-01A).
#ifndef MM167_FISHING_H
#define MM167_FISHING_H

#include <string>
#include <unordered_map>
#include <vector>

#include "common.h"
#include "cost_model.h"
#include "evaluate.h"
#include "game.h"
#include "generator.h"
#include "lane_model.h"

namespace mm167 {
struct DeepContext;
struct DiscardConfig;

// P2-01: optional, observation-limited continuation evaluation. The live game
// contains no hidden deck. Only the driver owns a hypothetical rollout tape;
// the policy sees a Game with precisely the current rack and X preview tiles.
struct Policy4Config {
    int targets=3,lanes=4;
    double margin=0;
    double maxRate=1e9;     // skip fishing when the greedy move already earns this many points per tile (dev2)
};
// P1-03 cheap keep-plan value: expected best lane score next turn for the kept
// part of the rack plus the visible refill prefix, on the unchanged board.
// Same event merge as LaneModel::moveValue, but a discard touches no line, so
// only the root windows (prepared by the caller) are consulted.
double leaveValue(const Game& g,const Generator& gen,LaneModel& lane,const EvalConfig& ev,
                  const int* kept,int draws);
// O-01A: bounded target-specific keeps. A target retains only the multiplicities
// that one word needs; taking the union over words would retain almost every rack.
struct FishPlans {
    std::vector<std::string> discards;
    double targetValue=0;
};
struct FishMeter {
    LaneModel& lane; double& work; double cap; Clock::time_point deadline;
    uint64_t scan,build;
    FishMeter(LaneModel& l,double& w,double c,Clock::time_point d)
        :lane(l),work(w),cap(c),deadline(d),scan(l.scanWords),build(l.buildWords){}
    void flush(){
        work+=cost::SCAN*(lane.scanWords-scan)+cost::BUILD*(lane.buildWords-build);
        scan=lane.scanWords;build=lane.buildWords;
    }
    bool expired(){flush();return work>=cap||Clock::now()>=deadline;}
    ~FishMeter(){flush();}
};
// Fill only the known prefix into counts; unknown future letters are integrated
// by leaveValue, never supplied from the rollout tape to the policy.
int fishRefill(const Game& g,int removed,int* kept);
bool fishPlans(const Game& g,const Generator& gen,LaneModel& lane,const EvalConfig& base,
               const Policy4Config& cfg,double& work,double cap,Clock::time_point deadline,FishPlans& out);
bool fishingPolicy(const Game& g,const Generator& gen,DeepContext& ctx,Action& action);
bool fishingRoot(const Game& g,const Generator& gen,const Move& move,const DiscardConfig& dc,
                 const LookaheadConfig& la,double cap,Clock::time_point deadline,double& work,FishPlans& plans);
}
#endif
