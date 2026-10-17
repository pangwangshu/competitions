#include "transition.h"

#include <algorithm>

#include "scoring.h"

namespace mm167 {
TransitionResult transition(Game& g,const Action& a,const std::string& sampledUnknown,
                            Clock::time_point deadline,Clock::time_point (*clockNow)()) {
    if(clockNow()>=deadline)return TransitionResult::Expired;
    if(g.terminal)return TransitionResult::Invalid;
    StateGuard guard(g);
    if(a.kind==ActionKind::Stop){g.terminal=true;guard.commit();return TransitionResult::Applied;}
    int remaining=g.rules->n*g.rules->n-g.drawn;
    if(a.kind==ActionKind::Place) {
        auto score=checkedScore(g,a.move);if(score<0)return TransitionResult::Invalid;
        int r=a.move.r,c=a.move.c;
        for(char ch:a.move.word){
            if(!g.occupied(r,c)){auto p=g.hand.find(ch);g.hand[p]='.';g.board[r][c]=ch;}
            r+=a.move.dir;c+=1-a.move.dir;
        }
        ++g.placed;g.score+=score;
    } else {
        if(a.letters.empty() || a.letters.size()>static_cast<size_t>(g.rules->t))return TransitionResult::Invalid;
        for(char ch:a.letters){auto p=g.hand.find(ch);if(p==std::string::npos)return TransitionResult::Invalid;g.hand[p]='.';}
        g.score-=g.rules->penalty;
    }
    std::string stream;size_t sample=0;
    int holes=static_cast<int>(std::count(g.hand.begin(),g.hand.end(),'.'));
    int refill=std::min(remaining,holes);
    int needed=std::min(remaining,refill+g.rules->x);
    for(int i=0;i<needed;++i) {
        char ch=i<int(g.upcoming.size())?g.upcoming[i]:'?';
        if(ch=='?' && sample<sampledUnknown.size())ch=sampledUnknown[sample++];
        if(ch!='?' && (ch<'A'||ch>'Z'))return TransitionResult::Invalid;
        if(ch=='?' && i<refill)return TransitionResult::NeedSamples;
        stream+=ch;
    }
    int next=0;
    for(char& ch:g.hand)if(ch=='.' && next<refill)ch=stream[next++];
    g.drawn+=refill;
    g.upcoming.assign(g.rules->x,'.');
    for(int i=0;i<g.rules->x && refill+i<remaining;++i)g.upcoming[i]=stream[refill+i];
    g.terminal=g.drawn==g.rules->n*g.rules->n && std::all_of(g.hand.begin(),g.hand.end(),[](char ch){return ch=='.';});
    if(clockNow()>=deadline)return TransitionResult::Expired;
    guard.commit(); return TransitionResult::Applied;
}
}
