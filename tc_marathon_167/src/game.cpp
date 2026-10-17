// Game protocol readers and the live board/rack update.
#include "game.h"

#include <algorithm>

namespace mm167 {
bool Game::readTiles(std::istream& in) {
    hand.resize(rules->t); upcoming.resize(rules->x);
    for(char& c:hand) if(!(in>>c))return false;
    for(char& c:upcoming) if(!(in>>c))return false;
    return true;
}
bool Game::readInitial(std::istream& in) {
    auto data=std::make_shared<Rules>();
    int count;
    if(!(in>>data->n>>data->t>>data->x>>data->penalty>>count))return false;
    data->dict.nodes.reserve(500000);
    std::string word;
    for(int i=0;i<count;++i) {
        if(!(in>>word))return false;
        data->dict.add(word);
        for(char& ch:word){int k=(ch>='a'&&ch<='z')?ch-'a':ch-'A';if(k>=0&&k<26)++data->letterWeight[k];ch=static_cast<char>('A'+k);}
        data->words.push_back(word);
    }
    rules=data;
    if(!readTiles(in))return false;
    for(int r=0;r<data->n;++r)for(int c=0;c<data->n;++c) {
        if(!(in>>data->bonus[r][c]))return false;
        if(data->bonus[r][c]=='S'){data->sr=r;data->sc=c;}
    }
    drawn=data->t;
    return true;
}
void Game::apply(const Move& m) {
    int r=m.r,c=m.c;
    for(char ch:m.word){ if(board[r][c]=='.')board[r][c]=ch; r+=m.dir;c+=1-m.dir; }
    drawn=std::min(rules->n*rules->n,drawn+m.used);
    ++placed; score+=m.score;
}
}
