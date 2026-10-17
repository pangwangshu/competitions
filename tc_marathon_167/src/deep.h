// Continuation search on shared random tapes (P2-01): rollouts and a bounded
// chance tree, plus the root comparison driver runDeep.
#ifndef MM167_DEEP_H
#define MM167_DEEP_H

#include <algorithm>
#include <random>
#include <string>
#include <vector>

#include "cost_model.h"
#include "evaluate.h"
#include "fishing.h"
#include "game.h"
#include "generator.h"
#include "scheduled.h"

namespace mm167 {
struct DeepConfig {
    int mode=0;                 // 0 off, 1 rollout, 2 bounded chance tree
    int width=4,depth=4,samples=4,branch=4,chance=2,policy=1;
    double share=0.65,tail=1,mix=1,confidence=0;
    double first=0;             // share of the remaining plan the first fair round may use (0: turn cap only)
    Policy4Config fishing;
    bool floor=false;           // O-01C: a continuation may stop after any step (value = best prefix)
};
struct DeepReport {
    int attempted=0,rounds=0,width=0,evals=0,steps=0,discards=0,invalid=0;
    bool cut=false,changed=false;
    double work=0,q=0;
    int fishCalls=0,fishTargets=0,fishSwaps=0;
    double fishWork=0;
};
struct DeepContext {
    const DeepConfig& cfg;
    const LookaheadConfig& la;
    Clock::time_point deadline;
    double cap;
    DeepReport& rep;
    bool expired(){return rep.work>=cap||Clock::now()>=deadline;}
    bool generate(Generator& gen){
        if(expired())return false;
        gen.run(deadline);
        rep.work+=gen.capacity?rootCost(gen):cost::RUN+cost::NODE*gen.nodes+cost::CROSS*gen.crossCells;
        return !gen.stopped;
    }
    double potential(const Game& g,const Generator& gen){
        if(g.terminal||!la.ev.on()||cfg.tail==0)return 0;
        rep.work+=cost::BOARD;
        double phase=std::min(1.0,double(g.rules->n*g.rules->n-g.drawn)/(2*g.rules->t));
        return cfg.tail*phase*la.ev.alpha*boardPotential(g,gen,la.ev);
    }
};
Action fallbackAction(const Game& g,const Move& move);
// No tape or environment RNG argument: policy-internal sampling is independent
// of the hypothetical environment that will subsequently execute this action.
bool rolloutPolicy(const Game& g,uint64_t policySeed,DeepContext& ctx,Action& action,double* leaf=nullptr);
// Absolute draw indexing preserves common random numbers when root actions
// consume different numbers of tiles. The visible prefix is already in Game.
bool applyTape(Game& g,const Action& a,const std::string& tape,int origin,DeepContext& ctx);
// P1-03: continuation value of an arbitrary root action. STOP scores exactly 0
// (no tape step, no cost); every other kind plays the shared tape from the same
// absolute origin, so PLACE and DISCARD are valued on identical futures.
bool rolloutValueAction(const Game& root,const Action& first,const std::string& tape,uint64_t policySeed,
                        DeepContext& ctx,double& value,int* drawn=nullptr);
bool rolloutValue(const Game& root,const Move& first,const std::string& tape,uint64_t policySeed,
                  DeepContext& ctx,double& value);
// Bounded expectimax replaces a determinized beam: each decision maximises an
// average over fresh chance outcomes, never a path selected with a known tail.
// State-equivalent actions are deduplicated before recursive expansion.
bool treeValue(const Game& g,int depth,uint64_t seed,DeepContext& ctx,double& value);
Move runDeep(const Game& g,const Move& parent,const std::vector<RankedMove>& ranked,
             const DeepConfig& cfg,const LookaheadConfig& la,double cap,std::mt19937_64& rng,
             Clock::time_point deadline,DeepReport& rep,double firstCap=0);
}
#endif
