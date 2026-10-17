// Process entry: stdin/stdout setup, CLI parsing, then the game loop.
#include <iostream>

#include "driver.h"

int main(int argc,char** argv) {
    using namespace mm167;
    std::ios::sync_with_stdio(false); std::cin.tie(nullptr);
    Options opts;
    if (parseOptions(argc,argv,opts)!=0) return 2;
    return runGame(opts);
}
