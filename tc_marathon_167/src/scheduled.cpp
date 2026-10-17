#include "scheduled.h"

#include <algorithm>
#include <chrono>
#include <cmath>

#include "cost_model.h"
#include "lane_model.h"
#include "lookahead.h"
#include "transition.h"

namespace mm167 {
Move runScheduled(const Game& g,const Generator& gen,const LookaheadConfig& cfg,const SchedConfig& sc,
                  double cap,std::mt19937_64& rng,Clock::time_point deadline,LookaheadReport& rep,
                  std::vector<RankedMove>* ranked) {
    const Move parent=gen.best;
    rep.cap=cap;
    if(gen.top.empty()||parent.score<0){rep.fallback="no_candidates";return parent;}
    const int remaining=g.rules->n*g.rules->n-g.drawn;
    std::vector<const Generator::Candidate*> cands;
    cands.reserve(gen.top.size());
    for(const auto& c:gen.top)cands.push_back(&c);
    const int total=int(cands.size());
    rep.pool=total;
    std::vector<int> order(total);
    for(int i=0;i<total;++i)order[i]=i;
    // Keys are computed in immediate-score order for as many pooled candidates
    // as the pre-selection share of the allowance covers (at least kMin).
    if(cfg.lane&&remaining>0&&total>sc.kMin){
        std::vector<std::pair<double,int>> keyed;
        const uint64_t scan0=cfg.lane->scanWords,build0=cfg.lane->buildWords;
        if(prefilterKeys(g,gen,cfg,cands,remaining,deadline,keyed,rep,std::max(0.0,cap-rep.work)*sc.pre,sc.kMin)){
            // Stable: equal keys keep the generator's (score, used, order) ranking; the greedy move stays first.
            std::stable_sort(keyed.begin()+1,keyed.end(),[](const auto& a,const auto& b){return a.first>b.first;});
            for(size_t i=0;i<keyed.size();++i)order[i]=keyed[i].second;
            rep.keyed=int(keyed.size());
        }
        rep.preWork=prefilterCost(*cfg.lane,scan0,build0);
        rep.work+=rep.preWork;
    }
    const EvalConfig& ev=cfg.ev;
    const double phase=ev.on()?std::min(1.0,remaining/(2.0*g.rules->t)):0.0;
    rep.phase=phase;
    // The whole sampling schedule is drawn up front, as in runLookahead.
    const int S=std::max(1,cfg.samples);
    LookaheadSampler sampler(g.rules->letterWeight);
    std::vector<std::string> streams(S);
    for(auto& s:streams){s.resize(g.rules->t+g.rules->x);for(char& ch:s)ch=sampler.draw(rng);}
    std::vector<char> det(total,0),hasBoard(total,0);
    std::vector<int64_t> sum(total,0);
    std::vector<int> cnt(total,0),playable(total,0);
    std::vector<double> boardV(total,0);
    const int holes0=static_cast<int>(std::count(g.hand.begin(),g.hand.end(),'.'));
    // Refill fully determined by the visible preview prefix: one exact evaluation.
    for(int i=0;i<total;++i)det[i]=std::min(remaining,holes0+cands[i]->move.used)<=g.rules->x;
    enum class Eval { Ok, Invalid, Cut };
    auto evaluate=[&](int i,int round,int64_t& v)->Eval{
        if(Clock::now()>=deadline)return Eval::Cut;
        Game sim(g);
        TransitionResult res=transition(sim,{ActionKind::Place,cands[i]->move,{}},streams[round],deadline);
        if(res==TransitionResult::Expired)return Eval::Cut;
        rep.work+=cost::RUN;
        if(res!=TransitionResult::Applied){++rep.invalidTransitions;return Eval::Invalid;}
        v=0;
        if(!sim.terminal){
            Generator next(sim);
            next.run(deadline);
            if(next.stopped)return Eval::Cut;
            rep.nodes2+=next.nodes;rep.cross2+=next.crossCells;++rep.runs2;
            rep.work+=cost::NODE*double(next.nodes)+cost::CROSS*double(next.crossCells);
            if(next.best.score>0)v=next.best.score;
            // The successor board is the same in every sample: evaluate once.
            if(phase>0&&!hasBoard[i]){
                auto t0=Clock::now();
                boardV[i]=boardPotential(sim,next,ev);hasBoard[i]=1;++rep.boards;
                rep.work+=cost::BOARD;
                rep.evalMs+=std::chrono::duration<double,std::milli>(Clock::now()-t0).count();
            }
        }
        ++rep.evals;
        return Eval::Ok;
    };
    double series=0,power=1;
    for(int r=0;r<S;++r){series+=power;power*=sc.keep;}
    const double r0Limit=rep.work+std::max(0.0,cap-rep.work)*(sc.r0>0?sc.r0:1.0/series);
    const auto plyStart=Clock::now();
    std::vector<int> alive;
    for(int at=0;at<total;++at){
        if(at>=sc.kMin&&(at>=sc.kMax||rep.work>=r0Limit))break;
        int64_t v=0;
        const Eval e=evaluate(order[at],0,v);
        if(e==Eval::Cut){rep.hardCut=true;break;}
        if(e==Eval::Invalid)continue;
        const int i=order[at];
        sum[i]=v;cnt[i]=1;playable[i]=v>0;alive.push_back(i);
        if(det[i])++rep.deterministic;
    }
    rep.width=int(alive.size());
    if(alive.empty()){rep.plyMs=std::chrono::duration<double,std::milli>(Clock::now()-plyStart).count();rep.fallback="round0_incomplete";return parent;}
    rep.rounds=1;
    auto qOf=[&](int i){return double(cands[i]->move.score)+cfg.gamma*double(sum[i])/cnt[i]+(phase>0?phase*ev.alpha*boardV[i]:0.0);};
    auto ahead=[&](int a,int b){
        const double qa=qOf(a),qb=qOf(b);
        return qa!=qb?qa>qb:Generator::better(*cands[a],*cands[b]);
    };
    double share=1;
    for(int round=1;round<S&&!rep.hardCut;++round){
        share*=sc.keep;
        if(sc.keep<1){
            const size_t keepN=static_cast<size_t>(std::max<double>(sc.keepMin,std::ceil(rep.width*share-1e-9)));
            if(keepN<alive.size()){std::sort(alive.begin(),alive.end(),ahead);alive.resize(keepN);}
        }
        if(rep.work>=cap)break;
        if(std::all_of(alive.begin(),alive.end(),[&](int i){return det[i]!=0;}))break;
        std::vector<int64_t> add(alive.size(),0);
        bool complete=true;
        for(size_t j=0;j<alive.size()&&complete;++j){
            if(det[alive[j]])continue;
            const Eval e=evaluate(alive[j],round,add[j]);
            if(e!=Eval::Ok){complete=false;rep.hardCut=e==Eval::Cut;}
        }
        // Only complete rounds count: every survivor then has the same samples.
        if(!complete)break;
        for(size_t j=0;j<alive.size();++j)if(!det[alive[j]]){
            const int i=alive[j];
            sum[i]+=add[j];++cnt[i];playable[i]+=add[j]>0;
        }
        ++rep.rounds;
    }
    rep.plyMs=std::chrono::duration<double,std::milli>(Clock::now()-plyStart).count();
    rep.finalists=int(alive.size());
    int bestIdx=alive[0];
    for(int i:alive)if(ahead(i,bestIdx))bestIdx=i;
    rep.chosenQ=qOf(bestIdx);
    rep.nextValue=sum[bestIdx]/cnt[bestIdx];
    rep.chosenBoard=boardV[bestIdx];
    rep.chosenPlayable=playable[bestIdx];rep.chosenSamples=cnt[bestIdx];rep.chosenRank=bestIdx;
    if(ranked){
        ranked->clear();
        // Include early-eliminated actions as well: a short-sighted elimination
        // must not irrevocably remove every longer-term setup from deep search.
        for(int i=0;i<total;++i)if(cnt[i])ranked->push_back({cands[i]->move,qOf(i)});
        std::stable_sort(ranked->begin(),ranked->end(),[](const auto& a,const auto& b){return a.q>b.q;});
    }
    if(det[bestIdx]){
        Game sim(g);
        if(transition(sim,{ActionKind::Place,cands[bestIdx]->move,{}},std::string())==TransitionResult::Applied){
            rep.predicted=true;rep.predictedHand=sim.hand;rep.predictedUpcoming=sim.upcoming;
        }
    }
    return cands[bestIdx]->move;
}
}
