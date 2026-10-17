// Bounded trie-search move generator with a score-sorted candidate pool.
#ifndef MM167_GENERATOR_H
#define MM167_GENERATOR_H

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "common.h"
#include "game.h"

namespace mm167 {
struct Generator {
    const Game& g;
    struct Cross { uint32_t mask=ALL; int oldScore=0; bool hasWord=false; };
    Cross cross[2][20][20];
    std::array<int,26> rack{};
    Move best;
    uint64_t nodes=0, candidates=0;
    uint64_t crossCells=0, retained=0, retainScan=0;   // cost-model counters (P2-03)
    bool stopped=false;
    bool profile=false;
    double crossMs=0;
    Clock::time_point deadline;
    int startR=0, startC=0, dir=0;
    std::string path;
    struct Candidate { Move move; std::string successor; uint64_t order; };
    size_t capacity=0;
    std::vector<Candidate> top;
    // Optional diversity quotas: best candidate per direction x board quadrant,
    // kept beside the score-sorted top-K for lookahead evaluation.
    struct BucketSlot { bool has=false; Candidate c; };
    BucketSlot bucket[8];
    bool diversity=false;
#ifdef MM167_TEST
    std::vector<Move> all;
#endif
    explicit Generator(const Game& game,size_t keep=0) : g(game),capacity(keep) { path.reserve(20); }
    static bool better(const Candidate& a,const Candidate& b);
    void retain(const Move& m);
    bool expired();
    void buildCross();
    void dfs(int r,int c,int node,int used,bool anchor,int64_t sum,int64_t mul,int64_t extra,uint32_t mask);
    Move run(Clock::time_point limit);
};
}
#endif
