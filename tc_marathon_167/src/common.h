// Shared constants and small utilities of the MM167 solver.
#ifndef MM167_COMMON_H
#define MM167_COMMON_H

#include <chrono>
#include <cstdint>

namespace mm167 {
using Clock = std::chrono::steady_clock;
constexpr uint32_t ALL = (1u << 26) - 1;
inline constexpr int VALUES[26] = {1,3,3,2,1,4,2,4,1,8,5,1,3,1,1,3,10,1,1,1,1,4,4,8,4,10};
// >1 stretches every generator run by busy-waiting: a slower machine for guard tests.
inline double simSlow = 1;
// >1 stretches every whole turn by busy-waiting: a uniformly slower machine for guard tests.
inline double simSlowTurn = 1;
inline uint64_t mixSeed(uint64_t x){x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);}
}
#endif
