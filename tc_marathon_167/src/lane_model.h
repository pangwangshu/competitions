// Rack-aware lane model: highest-value anchored windows and the exact word
// lists that fit them, with per-window refill expectations.
#ifndef MM167_LANE_MODEL_H
#define MM167_LANE_MODEL_H

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "evaluate.h"
#include "game.h"

namespace mm167 {
// Rack-aware lane model: for the highest-product anchored windows it lists the
// dictionary words that fit exactly (fixed letters + cross-checks) with their
// exact score, then values a kept rack by the chance that the refill supplies
// the missing tiles. Word lists are cached by window signature across turns.
struct LaneModel {
    struct Word {
        uint8_t need[9]; uint8_t tiles; uint32_t set; int32_t score;
#ifdef MM167_TEST
        std::string text;
#endif
    };
    struct Lane { std::vector<Word> words; std::vector<uint32_t> sets; std::vector<int> byScore; };
    std::string flat[21];                    // words of each length, concatenated
    std::vector<int> start[21],ids[21];      // CSR index by (position*26+letter)
    std::unordered_map<uint64_t,Lane> cache;
    size_t cachedWords=0;
    double freq[26];
    bool fixEvents=false; // opt-in until full-game validation approves the estimator change
    bool exactRefill=false;
    std::unordered_map<uint64_t,double> refillMemo;
    // Exact probability that iid draws contain the required multiset (up to
    // three tiles). Counts are capped at their requirements, so at most eight
    // states are needed. Cache by multiset and draw count, not board position.
    double refillProbability(uint64_t key,int draws);
    explicit LaneModel(const Rules& rules);
    struct Cell { char letter; uint32_t mask; int letterMul,wordMul,oldScore; bool cross; };
    const Lane& lane(uint64_t signature,const Cell* cells,int len,int product);
    struct Window { int product,tiles,d,line,first,last; const Lane* lane; };
    static bool order(const Window& a,const Window& b);
    // Windows of one line whose main word would contain an existing letter.
    static void lineWindows(const Game& g,const Generator& gen,const EvalConfig& ev,int d,int line,std::vector<Window>& out);
    const Lane* resolve(const Game& g,const Generator& gen,const Window& w);
    // Per turn: best windows of the root board, ordered; word lists resolved lazily.
    std::vector<Window> root,work;
    double buildMs=0,scanMs=0;   // diagnostics: word-list construction / rack scans
    long builds=0,scans=0;
    uint64_t buildWords=0,scanWords=0,windowsSeen=0;   // cost-model counters (P2-03)
    void prepare(const Game& g,const Generator& gen,const EvalConfig& ev);
    // A way to score in a lane next turn: the tiles still missing (key), the
    // best score among words missing exactly those, and the chance to draw them.
    struct Event { uint64_t key; int32_t score; double p; };
    struct Summary { int count=0; Event e[9]; };
    std::unordered_map<uint64_t,Summary> memo;
    std::vector<Event> events;
    // `kept` = tiles surely in the next rack (leave + visible refill); `draws` =
    // refill tiles not yet visible. Keeps the best complete word and the eight
    // most valuable incomplete ones that would beat it.
    const Summary& summary(const Lane* lane,const int* kept,uint32_t keptSet,uint64_t keptKey,int draws);
    // Expected best lane score next turn on the board after move `m` (`after`
    // has its tiles placed). Only lines the move touches are re-enumerated; the
    // root cross-checks in `gen` are exact for lanes crossing the new word.
    // Ways missing the same tiles are one event; events are combined in score
    // order as if independent.
    double moveValue(const Game& rootGame,const Game& after,const Generator& gen,const EvalConfig& ev,
                     const Move& m,const int* kept,int draws);
    std::vector<Event> merged;
};
}
#endif
