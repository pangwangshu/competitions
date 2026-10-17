// MM167 WordGame: trie move generation, effort-scheduled two-ply lookahead,
// shared-tape rollouts and a unified PLACE/DISCARD/STOP decision.
// Standalone C++17; stdin/stdout are exclusively the official protocol.
//
// Unity translation unit: the solver is organised in the modules included
// below (one .h/.cpp pair per concern). Tests reach the internals by defining
// MM167_TEST and including this file directly, which drops the CLI/driver
// layer; the built executable compiles exactly this file as the single source,
// and scripts/bundle.py inlines the same includes into one submission file.
#include "game.cpp"
#include "scoring.cpp"
#include "transition.cpp"
#include "generator.cpp"
#include "cost_model.cpp"
#include "evaluate.cpp"
#include "lane_model.cpp"
#include "lookahead.cpp"
#include "scheduled.cpp"
#include "deep.cpp"
#include "fishing.cpp"
#include "discard.cpp"
#ifndef MM167_TEST
#include "options.cpp"
#include "driver.cpp"
#include "main.cpp"
#endif
