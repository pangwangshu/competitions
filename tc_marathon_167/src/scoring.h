// Independent validation and exact scoring of a generated move.
#ifndef MM167_SCORING_H
#define MM167_SCORING_H

#include <cstdint>
#include <string>

#include "game.h"

namespace mm167 {
// Validate and score a generated move independently of any cached Move.score.
// Returns -1 for an invalid move; does not mutate the game.
int64_t checkedScore(const Game& g,const Move& m);
}
#endif
