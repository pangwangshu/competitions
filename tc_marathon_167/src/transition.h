// The state machine: apply an action to a Game under the official refill rules.
#ifndef MM167_TRANSITION_H
#define MM167_TRANSITION_H

#include <string>

#include "common.h"
#include "game.h"

namespace mm167 {
// sampledUnknown contains hypothetical letters ONLY for unknown positions after/in
// the visible preview. It is never obtained from the tester's hidden sequence.
// A '?' preview slot is unobserved, not an empty tile. Missing rack samples cause
// NeedSamples with no mutation; missing future preview samples remain '?'.
TransitionResult transition(Game& g,const Action& a,const std::string& sampledUnknown,
                            Clock::time_point deadline=Clock::time_point::max(),
                            Clock::time_point (*clockNow)()=Clock::now);
}
#endif
