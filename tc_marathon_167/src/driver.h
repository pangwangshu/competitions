// Entry point of the interactive game loop.
#ifndef MM167_DRIVER_H
#define MM167_DRIVER_H

#include "options.h"

namespace mm167 {
// Runs the whole interactive game on stdin/stdout; returns the process exit code.
int runGame(const Options& opts);
}
#endif
