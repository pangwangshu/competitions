// Selective discard: pre-screened keep plans and the unified root decision
// among PLACE / DISCARD / STOP (P1-03).
#ifndef MM167_DISCARD_H
#define MM167_DISCARD_H

#include <random>
#include <string>
#include <vector>

#include "common.h"
#include "deep.h"
#include "evaluate.h"
#include "fishing.h"
#include "game.h"
#include "generator.h"

namespace mm167 {
// P1-03: selective discard, active discard and stop. mode=0 (default) is fully
// off: runDiscardDecision is never called and no RNG is consumed, so the parent
// behaviour and every RNG stream stay byte-identical.
struct DiscardConfig {
    int mode=0;             // 0 off; 1 selective discard (no-word only); 2 + active low-value discard
    bool stop=false;        // admit STOP (value exactly 0) as a compared action
    int keep=8;             // pre-screened discard candidates incl. the full hand (4/8/16; 1 = full hand only)
    int samples=4;          // common-future rounds (tapes shared across all action types)
    int depth=4;            // continuation steps after the root action, same for every action
    int policy=1;           // rollout policy tier, same meaning as DeepConfig::policy
    double tail=1;          // board-potential leaf weight in continuations (P2-01 semantics)
    double trigger=15;      // mode 2: evaluate only when the chosen placement Q is below this
    double margin=0;        // required mean advantage (points) of a non-anchor action
    // Effort is separate from the two-ply/deep plan (reference-machine ms), so a
    // late decision is not starved by a front-loaded plan and the plan is not
    // shortened by decisions. capMs bounds one decision, workMs the whole game.
    double capMs=10,workMs=300;
    // Deck-rate valuation: a rollout that draws fewer tiles leaves more of the
    // finite deck for later, which a fixed horizon cannot see. With rate on,
    // every action is charged rho per tile drawn within the horizon, rho being
    // the anchor's own gain per tile (floored at 0), so actions are ranked by
    // gain per tile whenever continuing is worth anything at all.
    bool rate=false;
    double confidence=0;    // paired standard errors a non-anchor action must clear on top of margin
    Policy4Config fishing;
    bool fishRoot=false;
    double fishMaxRate=10;
    double front=0;         // dev2: pace the decision pool like the main plan (0 = INT-01 behaviour)
    // O-01C: STOP is dominated by a legal placement followed by STOP, so it is not
    // compared against a placement anchor. A fishing continuation that only pays
    // penalties inside the horizon otherwise ends a game with most of the deck left.
    bool stopDominated=false;
    bool floor=false;       // O-01C: continuation values are best-prefix (optional stopping)
};
struct DiscardReport {
    int attempted=0,subsets=0,cands=0,rounds=0,evals=0,steps=0,discards=0,invalid=0;
    bool cut=false,changed=false,stopChosen=false;
    double work=0,qAnchor=0,qBest=0;
    std::string letters;
    int fishCalls=0,fishTargets=0,fishSwaps=0;
    double fishWork=0;
};
// P1-03: deduped non-empty hand subsets ranked by the cheap keep-plan value.
// Returns the full-hand discard (the parent action) plus the best keep-1 other
// multisets, in deterministic order. An empty return means the enumeration hit
// the deadline: the caller then keeps the parent action (no partial prescreen).
std::vector<std::string> discardCandidates(const Game& g,const Generator& gen,LaneModel* lane,
                                           const EvalConfig& ev,int keep,double& work,int& subsets,
                                           Clock::time_point deadline);
// P1-03: unified root decision among the parent (anchor) action, pre-screened
// discard subsets and, optionally, STOP. Every candidate is valued by the same
// common-random-tape continuation estimator, so action types are compared
// without scale bias; the parent two-ply Q never enters this comparison. Any
// truncation (cut round, expired, no completed round) returns the anchor.
Action runDiscardDecision(const Game& g,const Generator& gen,const Move& placeMove,double /*placeQ*/,
                          const DiscardConfig& dc,const LookaheadConfig& la,double cap,double firstCap,
                          std::mt19937_64& rng,Clock::time_point deadline,bool previousDiscard,
                          LaneModel* lane,DiscardReport& rep,const FishPlans* targets=nullptr);
}
#endif
