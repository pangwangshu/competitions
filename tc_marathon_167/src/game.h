// Game rules, mutable state and actions: Move, Rules, State, Game, StateGuard.
#ifndef MM167_GAME_H
#define MM167_GAME_H

#include <array>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "trie.h"

namespace mm167 {
struct Move {
    int r = 0, c = 0, dir = 0, used = 0;
    int64_t score = -1;
    std::string word;
};
inline bool sameMove(const Move& a,const Move& b){return a.r==b.r&&a.c==b.c&&a.dir==b.dir&&a.word==b.word;}

// Rules are published read-only after initialization and shared across search branches.
struct Rules {
    int n=0,t=0,x=0,penalty=0,sr=0,sc=0;
    Trie dict;
    std::array<std::array<char,20>,20> bonus{};
    // Letter multiplicities of the input dictionary; used only to sample unknown draws.
    std::array<uint64_t,26> letterWeight{};
    // Upper-case input words; only the lane model (P1-02) scans them.
    std::vector<std::string> words;
};
struct State {
    int drawn=0,placed=0;
    int64_t score=0;
    bool terminal=false;
    std::array<std::array<char,20>,20> board{};
    std::string hand,upcoming;
    State() { for(auto& row:board) row.fill('.'); }
    bool operator==(const State& b) const {
        return drawn==b.drawn && placed==b.placed && score==b.score && terminal==b.terminal
            && board==b.board && hand==b.hand && upcoming==b.upcoming;
    }
    uint64_t hash() const {
        uint64_t h=14695981039346656037ULL;
        auto add=[&](uint64_t v){ for(int j=0;j<8;++j){ h^=(v>>(8*j))&255; h*=1099511628211ULL; } };
        add(drawn); add(placed); add(static_cast<uint64_t>(score)); add(terminal);
        for(const auto& row:board) for(char c:row) add(static_cast<unsigned char>(c));
        add(hand.size()); for(char c:hand)add(static_cast<unsigned char>(c));
        add(upcoming.size()); for(char c:upcoming)add(static_cast<unsigned char>(c));
        return h;
    }
};
struct Game : State {
    std::shared_ptr<const Rules> rules;
    Game():rules(std::make_shared<Rules>()){}
    explicit Game(std::shared_ptr<const Rules> r):rules(std::move(r)){}
    bool inside(int r,int c) const { return r>=0 && r<rules->n && c>=0 && c<rules->n; }
    bool occupied(int r,int c) const { return inside(r,c) && board[r][c]!='.'; }
    int letterMul(int r,int c) const { return rules->bonus[r][c]=='d'?2:rules->bonus[r][c]=='t'?3:1; }
    int wordMul(int r,int c) const { return rules->bonus[r][c]=='D'?2:rules->bonus[r][c]=='T'?3:1; }
    bool readTiles(std::istream& in=std::cin);
    bool readInitial(std::istream& in=std::cin);
    // Live protocol update: official feedback supplies the actual rack and preview.
    void apply(const Move& m);
};

enum class ActionKind { Place, Discard, Stop };
struct Action {
    ActionKind kind=ActionKind::Stop;
    Move move;
    std::string letters;
};
enum class TransitionResult { Applied, Invalid, NeedSamples, Expired };
// Restores only mutable state; the large dictionary is never copied.
struct StateGuard {
    Game& game;
    State saved;
    bool committed=false;
    explicit StateGuard(Game& g):game(g),saved(g){}
    ~StateGuard(){ if(!committed)static_cast<State&>(game)=std::move(saved); }
    void commit(){committed=true;}
};
}
#endif
