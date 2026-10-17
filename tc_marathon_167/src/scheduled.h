// Effort-scheduled two-ply evaluation (P2-03).
#ifndef MM167_SCHEDULED_H
#define MM167_SCHEDULED_H

#include <random>
#include <vector>

#include "common.h"
#include "evaluate.h"
#include "game.h"
#include "generator.h"

namespace mm167 {
struct SchedConfig {
    bool on=false;
    double workMs=800;      // effort budget of the whole game, reference-machine ms
    double front=6;         // a turn may spend this multiple of the even share of what is left
    int kMin=8,kMax=32;     // bounds on the number of candidates given the first common sample
    double keep=0.5;        // share of the first-round candidates kept per further sample
    int keepMin=4;
    double r0=0;            // share of the turn allowance for the first round; 0 derives it from keep
    double pre=0.6;         // share of the turn allowance (after generation) for pre-selection keys
    // Guard v2 (0 = INT-01 behaviour): when the time guard squeezes the game, the
    // plan and the decision pool share the safe effort in proportion to what each
    // has left, the pool weighted by this factor, instead of the plan taking all of
    // it and every decision being starved on a machine about twice as slow.
    double guardShare=0;
};
struct RankedMove { Move move; double q; };
// Scheduled two-ply evaluation. Candidates are taken in static-key order and all
// get the first common sample (at least kMin, more while the turn allowance
// lasts); each further common sample goes only to the best `keep` share. The
// final comparison is among candidates that received every completed round.
Move runScheduled(const Game& g,const Generator& gen,const LookaheadConfig& cfg,const SchedConfig& sc,
                  double cap,std::mt19937_64& rng,Clock::time_point deadline,LookaheadReport& rep,
                  std::vector<RankedMove>* ranked=nullptr);
}
#endif
