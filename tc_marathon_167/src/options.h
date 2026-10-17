// Command-line configuration of the solver.
#ifndef MM167_OPTIONS_H
#define MM167_OPTIONS_H

#include <cstdint>

#include "deep.h"
#include "discard.h"
#include "evaluate.h"
#include "scheduled.h"

namespace mm167 {
// Every tunable the CLI can set; defaults are the validated submission values.
struct Options {
    double budgetMs=7500, turnMs=30;
    bool stats=false, diagnostics=false, profile=false, exactRefill=false, fixEvents=false;
    // Validated P1-01 configuration is the default; --no-lookahead restores
    // the pure greedy parent behaviour for local A/B experiments.
    bool lookahead=true;
    LookaheadConfig la;
    // Validated P1-02 configuration; --ev-pool 0 --ev-alpha 0 restores P1-01.
    // Validated P2-03 effort scheduler with its wider pool; --no-sched restores
    // the fixed per-turn slice and the pool of P1-02.
    SchedConfig sched;
    DeepConfig deep;
    DiscardConfig dc;
    bool poolGiven=false;
    // O-01C: time-adaptive mechanics. 0 off; 1 a turn the wall-clock guard cannot
    // fully fund drops root fishing; 2 such a turn runs the INT-01 mechanics.
    int adaptive=0;
    double adaptMargin=1;
    int leanX=-1;           // O-01C: instances with X<=leanX run the INT-01 mechanics throughout
    uint64_t laSeed=167;
};
// Parses argv into `opts`. `--preset p2-03|int-01|o-01a|o-01c` (default o-01c,
// the final submission) is applied before the other flags. Returns 0 on
// success, 2 on any invalid argument (the original main's usage-error code).
int parseOptions(int argc,char** argv,Options& opts);
}
#endif
