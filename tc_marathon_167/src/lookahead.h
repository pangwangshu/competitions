// Two-ply lookahead with lane-aware pre-selection (P1-01/P1-02).
#ifndef MM167_LOOKAHEAD_H
#define MM167_LOOKAHEAD_H

#include <random>
#include <utility>
#include <vector>

#include "common.h"
#include "evaluate.h"
#include "game.h"
#include "generator.h"

namespace mm167 {
// Static pre-selection key of every pooled candidate (P1-02): immediate score
// plus mu times the expected lane score of the kept rack. Root cross-checks are
// reused: they are exact for lanes crossing the new word. Returns false when
// the deadline expired before every key was computed. With a finite `limit`
// (P2-03) keys are computed in generator order until the predicted cost reaches
// it; `keyed` then holds only the candidates that were reached.
bool prefilterKeys(const Game& g,const Generator& gen,const LookaheadConfig& cfg,
                   const std::vector<const Generator::Candidate*>& cands,int remaining,
                   Clock::time_point deadline,std::vector<std::pair<double,int>>& keyed,LookaheadReport& rep,
                   double limit=-1,size_t floor=0);
Move runLookahead(const Game& g,const Generator& gen,const LookaheadConfig& cfg,
                  std::mt19937_64& rng,Clock::time_point deadline,LookaheadReport& rep);
}
#endif
