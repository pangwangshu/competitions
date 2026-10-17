// Evaluation layer: lookahead/eval configuration, sampling and board potential.
#ifndef MM167_EVALUATE_H
#define MM167_EVALUATE_H

#include <array>
#include <cstdint>
#include <random>
#include <string>

#include "game.h"
#include "generator.h"

namespace mm167 {
struct LaneModel;

// P1-01 two-ply lookahead: Q(a)=score(a)+gamma*E[best next placement score].
// The continuation value uses only the successor rack produced by the official
// refill rules; unknown draws are sampled from dictionary letter frequencies
// through a per-turn common stream so every candidate sees the same letters.
// P1-02 evaluation on top of the two-ply value:
//   Q(a)=score(a)+gamma*E[next]+phase*alpha*board(a)
// and, with pool>k, a lane-aware pre-selection of the k evaluated candidates.
// alpha=0 and pool=0 disable both and reproduce the P1-01 choice exactly.
struct EvalConfig {
    double alpha=0;             // weight of the board potential
    double rho=0.8;             // per-new-tile realisation discount of a window
    double kappa=0.3;           // weight of a window with no existing letter yet
    double tau=0.5;             // exponent on the cross-check letter mass of a cell
    bool linear=false;          // ablation: sum of (multiplier-1) instead of the product
    int pool=0;                 // >k: generate this many candidates, pre-select k by static key
    double mu=1;                // pre-selection weight of the lane value
    int lanes=24,laneMin=4;     // windows kept per board / minimum multiplier product
    bool on() const { return alpha>0; }
};
struct LookaheadConfig { int k=32,samples=4; double gamma=1.0; bool diversity=false; EvalConfig ev; LaneModel* lane=nullptr; };
struct LookaheadReport {
    int rounds=0,evals=0,deterministic=0,width=0,invalidTransitions=0;
    const char* fallback="";
    double chosenQ=0; int64_t nextValue=0;
    double chosenBoard=0,phase=0,evalMs=0;
    int chosenPlayable=0,chosenSamples=0,chosenRank=0,pool=0;
    bool predicted=false;
    std::string predictedHand,predictedUpcoming;
    // Cost-model counters of this turn (P2-03); timers are filled only for diagnostics.
    uint64_t nodes2=0,cross2=0,runs2=0,boards=0,scanWords=0,buildWords=0,windows=0,preCands=0;
    double preMs=0,plyMs=0;
    // Scheduled evaluation (P2-03): predicted cost spent / allowed, finalists, hard-deadline cut.
    double work=0,cap=0,preWork=0;
    int finalists=0,keyed=0;
    bool hardCut=false;
};
struct LookaheadSampler {
    std::array<uint64_t,26> cumulative{}; uint64_t total=0;
    explicit LookaheadSampler(const std::array<uint64_t,26>& weight){
        uint64_t acc=0;
        for(int i=0;i<26;++i){acc+=weight[i];cumulative[i]=acc;}
        total=acc;
    }
    char draw(std::mt19937_64& rng) const {
        if(!total)return 'A';
        uint64_t u=rng()%total;
        for(int i=0;i<26;++i)if(u<cumulative[i])return static_cast<char>('A'+i);
        return 'Z';
    }
};
// Unused word multipliers are worth most when one placement covers several of
// them. Per line, choose disjoint windows of at most T empty cells maximising
// sum((product-1)*rho^tiles*fit); a window counts fully only if its main word
// would already contain an existing letter. `gen` supplies the cross-checks of
// the evaluated board. This is a potential, not a guaranteed score.
double boardPotential(const Game& g,const Generator& gen,const EvalConfig& ev);
}
#endif
