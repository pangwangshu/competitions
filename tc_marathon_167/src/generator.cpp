#include "generator.h"

#include <algorithm>
#include <chrono>

namespace mm167 {
bool Generator::better(const Candidate& a,const Candidate& b) {
    if(a.move.score!=b.move.score)return a.move.score>b.move.score;
    if(a.move.used!=b.move.used)return a.move.used>b.move.used;
    return a.order<b.order;
}
void Generator::retain(const Move& m) {
    if(!capacity)return;
    Candidate entry{m,{},candidates};
    if(top.size()==capacity && !better(entry,top.back()))return;
    ++retained;retainScan+=top.size();
    // All entries share rules and parent state. Exact board/rack successor
    // equality implies identical refill position/preview; score is dominated.
    auto board=g.board;auto hand=g.hand;
    int r=m.r,c=m.c;
    for(char ch:m.word){
        if(board[r][c]=='.'){board[r][c]=ch;hand[hand.find(ch)]='.';}
        r+=m.dir;c+=1-m.dir;
    }
    for(int rr=0;rr<g.rules->n;++rr)for(int cc=0;cc<g.rules->n;++cc)entry.successor+=board[rr][cc];
    entry.successor+='|';entry.successor+=hand;
    entry.successor+='|';entry.successor+=std::to_string(std::min(g.rules->n*g.rules->n,g.drawn+m.used));
    auto duplicate=std::find_if(top.begin(),top.end(),[&](const Candidate& old){return old.successor==entry.successor;});
    if(duplicate!=top.end()){
        if(!better(entry,*duplicate))return;
        top.erase(duplicate);
    }
    if(diversity){
        int b=(m.dir<<2)|((m.r*2>=g.rules->n)<<1)|(m.c*2>=g.rules->n);
        if(!bucket[b].has||better(entry,bucket[b].c))bucket[b].c=entry,bucket[b].has=true;
    }
    auto pos=std::lower_bound(top.begin(),top.end(),entry,better);
    top.insert(pos,std::move(entry));if(top.size()>capacity)top.pop_back();
}
bool Generator::expired() {
    if (Clock::now() >= deadline) stopped=true;
    return stopped;
}
void Generator::buildCross() {
    for (int d=0; d<2; ++d) for (int r=0; r<g.rules->n; ++r) {
        if (expired()) return;
        for (int c=0; c<g.rules->n; ++c) {
            Cross& cr=cross[d][r][c];
            if (g.occupied(r,c)) continue;
            int dr=1-d, dc=d, rr=r-dr, cc=c-dc;
            std::string pre, post;
            while (g.occupied(rr,cc)) { pre+=g.board[rr][cc]; rr-=dr; cc-=dc; }
            std::reverse(pre.begin(),pre.end());
            rr=r+dr; cc=c+dc;
            while (g.occupied(rr,cc)) { post+=g.board[rr][cc]; rr+=dr; cc+=dc; }
            if (pre.empty() && post.empty()) continue;
            cr.hasWord=true; cr.mask=0; ++crossCells;
            for (char ch : pre+post) cr.oldScore+=VALUES[ch-'A'];
            std::string w=pre+'A'+post;
            for (int k=0;k<26;++k) {
                w[pre.size()]=static_cast<char>('A'+k);
                if (g.rules->dict.contains(w)) cr.mask|=1u<<k;
            }
        }
    }
}
void Generator::dfs(int r,int c,int node,int used,bool anchor,int64_t sum,int64_t mul,int64_t extra,uint32_t mask) {
    if (stopped) return;
    if ((++nodes & 255u)==0 && expired()) return;
    bool occ=g.occupied(r,c);
    if (g.rules->dict.nodes[node].terminal && used>0 && anchor && !occ) {
        Move m{startR,startC,dir,used,sum*mul+extra,path};
        ++candidates;
#ifdef MM167_TEST
        all.push_back(m);
#endif
        if (m.score>best.score || (m.score==best.score && m.used>best.used)) best=m;
        retain(m);
    }
    if (!g.inside(r,c)) return;
    if (occ) {
        char ch=g.board[r][c];
        int next=g.rules->dict.nodes[node].next[ch-'A'];
        if (!next) return;
        path+=ch;
        dfs(r+dir,c+1-dir,next,used,true,sum+VALUES[ch-'A'],mul,extra,mask);
        path.pop_back();
        return;
    }
    uint32_t options=g.rules->dict.nodes[node].mask & mask & cross[dir][r][c].mask;
    while (options && !stopped) {
        int k=__builtin_ctz(options); options&=options-1;
        --rack[k];
        uint32_t newMask = rack[k] ? mask : mask & ~(1u<<k);
        int value=VALUES[k]*g.letterMul(r,c), wm=g.wordMul(r,c);
        const Cross& cr=cross[dir][r][c];
        int side=cr.hasWord ? (cr.oldScore+value)*wm : 0;
        path+=static_cast<char>('A'+k);
        bool hit=anchor || (g.placed==0 && r==g.rules->sr && c==g.rules->sc);
        dfs(r+dir,c+1-dir,g.rules->dict.nodes[node].next[k],used+1,hit,sum+value,mul*wm,extra+side,newMask);
        path.pop_back();
        ++rack[k];
    }
}
Move Generator::run(Clock::time_point limit) {
    const auto slowStart=simSlow>1 ? Clock::now() : Clock::time_point{};
    rack.fill(0);best=Move{};nodes=candidates=crossCells=retained=retainScan=0;stopped=false;crossMs=0;
    top.clear();path.clear();
    for(auto& b:bucket)b=BucketSlot{};
    for(auto& direction:cross)for(auto& row:direction)for(auto& cell:row)cell=Cross{};
#ifdef MM167_TEST
    all.clear();
#endif
    deadline=limit;
    uint32_t mask=0; int tiles=0;
    for (char c : g.hand) if (c!='.') { ++rack[c-'A']; mask|=1u<<(c-'A'); ++tiles; }
    auto crossStart=profile ? Clock::now() : Clock::time_point{};
    buildCross();
    if (profile) crossMs=std::chrono::duration<double,std::milli>(Clock::now()-crossStart).count();
    for (dir=0;dir<2 && !stopped;++dir) {
        for (startR=0;startR<g.rules->n && !stopped;++startR) {
            for (startC=0;startC<g.rules->n && !stopped;++startC) {
                if (expired()) break;
                if (g.occupied(startR-dir,startC-1+dir)) continue;
                // A reachable existing letter (or initial S) is mandatory.
                int empty=0; bool reachable=false;
                for (int r=startR,c=startC;g.inside(r,c);r+=dir,c+=1-dir) {
                    if (!g.occupied(r,c)) ++empty;
                    if (empty>tiles) break;
                    if ((g.placed && g.occupied(r,c)) || (!g.placed && r==g.rules->sr && c==g.rules->sc)) {
                        reachable=true; break;
                    }
                }
                if (reachable) dfs(startR,startC,0,0,false,0,1,0,mask);
            }
        }
    }
    if (simSlow>1) {
        const auto now=Clock::now();
        const auto until=now+std::chrono::duration_cast<Clock::duration>((now-slowStart)*(simSlow-1));
        while (Clock::now()<until) {}
    }
    return best;
}
}
