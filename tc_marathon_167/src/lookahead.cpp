#include "lookahead.h"

#include <algorithm>
#include <chrono>

#include "cost_model.h"
#include "lane_model.h"
#include "transition.h"

namespace mm167 {
bool prefilterKeys(const Game& g,const Generator& gen,const LookaheadConfig& cfg,
                   const std::vector<const Generator::Candidate*>& cands,int remaining,
                   Clock::time_point deadline,std::vector<std::pair<double,int>>& keyed,LookaheadReport& rep,
                   double limit,size_t floor) {
    auto t0=Clock::now();
    const EvalConfig& ev=cfg.ev;
    LaneModel& lane=*cfg.lane;
    const uint64_t scan0=lane.scanWords,build0=lane.buildWords,windows0=lane.windowsSeen;
    keyed.assign(cands.size(),{0.0,0});
    bool expired=false;
    lane.prepare(g,gen,ev);
    for(size_t i=0;i<cands.size();++i){
        if((i&15)==0&&Clock::now()>=deadline){expired=true;break;}
        if(limit>=0&&i>=floor&&prefilterCost(lane,scan0,build0)>=limit){keyed.resize(i);break;}
        const Move& m=cands[i]->move;
        Game after(g);
        int r=m.r,c=m.c;
        for(char ch:m.word){
            if(after.board[r][c]=='.'){after.board[r][c]=ch;after.hand[after.hand.find(ch)]='.';}
            r+=m.dir;c+=1-m.dir;
        }
        ++after.placed;
        int keptTiles[26]={};
        for(char ch:after.hand)if(ch!='.')++keptTiles[ch-'A'];
        int holes=static_cast<int>(std::count(after.hand.begin(),after.hand.end(),'.'));
        int refill=std::min(remaining,holes),seen=0;
        for(;seen<refill&&seen<int(g.upcoming.size());++seen){
            char ch=g.upcoming[seen];
            if(ch<'A'||ch>'Z')break;
            ++keptTiles[ch-'A'];
        }
        keyed[i]={double(m.score)+ev.mu*lane.moveValue(g,after,gen,ev,m,keptTiles,refill-seen),int(i)};
        ++rep.preCands;
    }
    rep.scanWords+=lane.scanWords-scan0;rep.buildWords+=lane.buildWords-build0;rep.windows+=lane.windowsSeen-windows0;
    double ms=std::chrono::duration<double,std::milli>(Clock::now()-t0).count();
    rep.evalMs+=ms;rep.preMs+=ms;
    return !expired;
}
Move runLookahead(const Game& g,const Generator& gen,const LookaheadConfig& cfg,
                  std::mt19937_64& rng,Clock::time_point deadline,LookaheadReport& rep) {
    const Move parent=gen.best;
    if(gen.top.empty()||parent.score<0){rep.fallback="no_candidates";return parent;}
    std::vector<const Generator::Candidate*> cands;
    cands.reserve(gen.top.size()+4);
    for(const auto& c:gen.top)cands.push_back(&c);
    if(cfg.diversity){
        for(const auto& b:gen.bucket)if(b.has){
            bool known=false;
            for(const auto* c:cands)if(c->successor==b.c.successor){known=true;break;}
            if(!known&&cands.size()<gen.top.size()+4)cands.push_back(&b.c);
        }
    }
    const int remaining=g.rules->n*g.rules->n-g.drawn;
    rep.pool=int(cands.size());
    // Pre-selection (P1-02): the generator kept a larger pool by immediate score;
    // keep the k with the best static key so that low-scoring placements which
    // open a high-multiplier lane for the kept rack reach the sampled evaluation.
    // Root cross-checks are reused: they are exact for lanes crossing the new word.
    // Off once the bag is empty: the endgame choice is then exactly the parent's.
    if(cfg.ev.pool>cfg.k&&int(cands.size())>cfg.k&&cfg.lane&&remaining>0){
        std::vector<std::pair<double,int>> keyed;
        if(prefilterKeys(g,gen,cfg,cands,remaining,deadline,keyed,rep)){
            // Stable: equal keys keep the generator's (score, used, order) ranking.
            std::stable_sort(keyed.begin()+1,keyed.end(),[](const auto& a,const auto& b){return a.first>b.first;});
            std::vector<int> pick;
            for(int i=0;i<cfg.k;++i)pick.push_back(keyed[i].second);   // keyed[0] is the greedy move
            std::sort(pick.begin(),pick.end());
            std::vector<const Generator::Candidate*> chosen;
            for(int i:pick)chosen.push_back(cands[i]);
            cands.swap(chosen);
        } else cands.resize(cfg.k);
    } else if(cfg.ev.pool>cfg.k&&int(cands.size())>cfg.k) cands.resize(cfg.k);
    const int n=int(cands.size());
    rep.width=n;
    std::vector<char> det(n,0);
    std::vector<int64_t> sum(n,0);
    std::vector<int> cnt(n,0),playable(n,0);
    // Static terms fade out as the bag empties and vanish once it is exhausted.
    const EvalConfig& ev=cfg.ev;
    const double phase=ev.on()?std::min(1.0,remaining/(2.0*g.rules->t)):0.0;
    rep.phase=phase;
    std::vector<double> boardV(n,0);
    std::vector<char> hasBoard(n,0);
    for(int i=0;i<n;++i){
        const Move& m=cands[i]->move;
        auto hand=g.hand;
        int r=m.r,c=m.c;
        for(char ch:m.word){
            if(g.board[r][c]=='.'){auto p=hand.find(ch);hand[p]='.';}
            r+=m.dir;c+=1-m.dir;
        }
        int holes=static_cast<int>(std::count(hand.begin(),hand.end(),'.'));
        // Refill is fully determined by the visible preview prefix.
        det[i]=std::min(remaining,holes)<=g.rules->x;
        if(det[i])++rep.deterministic;
    }
    // Draw the whole sampling schedule up front: the RNG stream never depends
    // on how much time evaluation actually had.
    const int S=cfg.samples;
    const int roundsPlanned=S>0?S:1;
    const int streamLen=g.rules->t+g.rules->x;
    LookaheadSampler sampler(g.rules->letterWeight);
    std::vector<std::string> streams(roundsPlanned);
    if(S>0)for(auto& s:streams){s.resize(streamLen);for(char& ch:s)ch=sampler.draw(rng);}
    const auto plyStart=Clock::now();
    for(int round=0;round<roundsPlanned;++round){
        bool complete=true;
        std::vector<int64_t> add(n,0);
        std::vector<int> got(n,0),hit(n,0);
        for(int i=0;i<n;++i){
            const bool participates=round==0?(S>0||det[i]):!det[i];
            if(!participates)continue;
            if(Clock::now()>=deadline){complete=false;break;}
            Game sim(g);
            Action a{ActionKind::Place,cands[i]->move,{}};
            TransitionResult res=transition(sim,a,S>0?streams[round]:std::string(),deadline);
            if(res!=TransitionResult::Applied){
                if(res==TransitionResult::Expired){complete=false;break;}
                // Generator moves must remain legal; count instead of guessing a value.
                ++rep.invalidTransitions;
                continue;
            }
            int64_t v=0;
            if(!sim.terminal){
                Generator next(sim);
                next.run(deadline);
                rep.nodes2+=next.nodes;rep.cross2+=next.crossCells;++rep.runs2;
                if(next.best.score>0)v=next.best.score;
                // The successor board is the same in every sample: evaluate once.
                if(phase>0&&!hasBoard[i]&&!next.stopped){
                    auto t0=Clock::now();
                    boardV[i]=boardPotential(sim,next,ev);hasBoard[i]=1;++rep.boards;
                    rep.evalMs+=std::chrono::duration<double,std::milli>(Clock::now()-t0).count();
                }
            }
            add[i]=v;got[i]=1;hit[i]=v>0;++rep.evals;
        }
        if(!complete)break;
        for(int i=0;i<n;++i){sum[i]+=add[i];cnt[i]+=got[i];playable[i]+=hit[i];}
        ++rep.rounds;
    }
    rep.plyMs=std::chrono::duration<double,std::milli>(Clock::now()-plyStart).count();
    if(rep.rounds==0){rep.fallback="round0_incomplete";return parent;}
    auto staticOf=[&](int i){return phase>0?phase*ev.alpha*boardV[i]:0.0;};
    auto qOf=[&](int i){return cnt[i]?double(cands[i]->move.score)+cfg.gamma*double(sum[i])/cnt[i]+staticOf(i):-1e18;};
    int bestIdx=0;
    for(int i=1;i<n;++i){
        double qi=qOf(i),qb=qOf(bestIdx);
        if(qi>qb||(qi==qb&&Generator::better(*cands[i],*cands[bestIdx])))bestIdx=i;
    }
    if(!cnt[bestIdx]){rep.fallback="no_information";return parent;}
    rep.chosenQ=qOf(bestIdx);
    rep.nextValue=sum[bestIdx]/cnt[bestIdx];
    rep.chosenBoard=boardV[bestIdx];
    rep.chosenPlayable=playable[bestIdx];rep.chosenSamples=cnt[bestIdx];rep.chosenRank=bestIdx;
    if(det[bestIdx]){
        Game sim(g);
        if(transition(sim,{ActionKind::Place,cands[bestIdx]->move,{}},std::string())==TransitionResult::Applied){
            rep.predicted=true;rep.predictedHand=sim.hand;rep.predictedUpcoming=sim.upcoming;
        }
    }
    return cands[bestIdx]->move;
}
}
