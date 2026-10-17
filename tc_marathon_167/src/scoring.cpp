#include "scoring.h"

#include <algorithm>

#include "common.h"

namespace mm167 {
int64_t checkedScore(const Game& g,const Move& m) {
    if(m.dir<0 || m.dir>1 || m.word.empty() || !g.inside(m.r,m.c))return -1;
    for(char c:m.word)if(c<'A'||c>'Z')return -1;
    if(!g.rules->dict.contains(m.word) || g.occupied(m.r-m.dir,m.c-(1-m.dir)))return -1;
    int r=m.r,c=m.c; bool anchor=false; int used=0;
    auto hand=g.hand; int64_t sum=0,mul=1,extra=0;
    for(char ch:m.word) {
        if(!g.inside(r,c))return -1;
        if(g.occupied(r,c)) {
            if(g.board[r][c]!=ch)return -1;
            anchor=true;sum+=VALUES[ch-'A'];
        } else {
            auto pos=hand.find(ch);if(pos==std::string::npos)return -1;hand[pos]='.';++used;
            if(!g.placed && r==g.rules->sr && c==g.rules->sc)anchor=true;
            int value=VALUES[ch-'A']*g.letterMul(r,c),wm=g.wordMul(r,c);
            sum+=value;mul*=wm;
            int dr=1-m.dir,dc=m.dir,rr=r-dr,cc=c-dc,old=0;
            std::string pre,post;
            while(g.occupied(rr,cc)){pre+=g.board[rr][cc];old+=VALUES[g.board[rr][cc]-'A'];rr-=dr;cc-=dc;}
            std::reverse(pre.begin(),pre.end());rr=r+dr;cc=c+dc;
            while(g.occupied(rr,cc)){post+=g.board[rr][cc];old+=VALUES[g.board[rr][cc]-'A'];rr+=dr;cc+=dc;}
            if(!pre.empty() || !post.empty()) {
                if(!g.rules->dict.contains(pre+ch+post))return -1;
                extra+=(old+value)*wm;
            }
        }
        r+=m.dir;c+=1-m.dir;
    }
    if(!used || !anchor || g.occupied(r,c))return -1;
    return sum*mul+extra;
}
}
