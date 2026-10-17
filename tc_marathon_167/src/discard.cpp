#include "discard.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <set>

#include "cost_model.h"
#include "lane_model.h"

namespace mm167 {
std::vector<std::string> discardCandidates(const Game& g,const Generator& gen,LaneModel* lane,
                                           const EvalConfig& ev,int keep,double& work,int& subsets,
                                           Clock::time_point deadline){
    std::string tiles;for(char ch:g.hand)if(ch!='.')tiles+=ch;
    const int m=int(tiles.size()),remaining=g.rules->n*g.rules->n-g.drawn;
    const uint64_t scan0=lane?lane->scanWords:0,build0=lane?lane->buildWords:0;
    if(lane)lane->prepare(g,gen,ev);
    struct Cand{double value;uint32_t mask;};
    std::vector<Cand> found;
    std::set<std::string> seen;
    for(int mask=1;mask<(1<<m);++mask){
        if((mask&15)==0&&Clock::now()>=deadline){subsets=-1;return {};}
        std::string letters,rep;
        for(int idx=0;idx<m;++idx)if(mask>>idx&1){letters+=tiles[idx];rep+=tiles[idx];}
        std::sort(rep.begin(),rep.end());
        if(!seen.insert(rep).second)continue;
        int kept[26]={};
        for(char ch:tiles)++kept[ch-'A'];
        for(char ch:letters)--kept[ch-'A'];
        const int refill=std::min(remaining,int(letters.size()));
        int seenPreview=0;
        for(;seenPreview<refill&&seenPreview<int(g.upcoming.size());++seenPreview){
            char ch=g.upcoming[seenPreview];
            if(ch<'A'||ch>'Z')break;
            ++kept[ch-'A'];
        }
        double value;
        if(lane)value=leaveValue(g,gen,*lane,ev,kept,refill-seenPreview);
        else{value=0;for(int k=0;k<26;++k)value+=kept[k]*VALUES[k];}
        found.push_back({value,uint32_t(mask)});
    }
    subsets=int(found.size());
    if(lane)work+=cost::SCAN*double(lane->scanWords-scan0)+cost::BUILD*double(lane->buildWords-build0);
    work+=cost::RETAIN_SCAN*double(subsets);
    std::stable_sort(found.begin(),found.end(),[](const Cand& a,const Cand& b){return a.value>b.value;});
    std::vector<std::string> out{tiles};
    for(const auto& c:found){
        if(int(out.size())>=std::max(1,keep))break;
        if(c.mask==uint32_t((1<<m)-1))continue;   // full hand already first
        std::string letters;
        for(int idx=0;idx<m;++idx)if(c.mask>>idx&1)letters+=tiles[idx];
        out.push_back(letters);
    }
    return out;
}
Action runDiscardDecision(const Game& g,const Generator& gen,const Move& placeMove,double /*placeQ*/,
                          const DiscardConfig& dc,const LookaheadConfig& la,double cap,double firstCap,
                          std::mt19937_64& rng,Clock::time_point deadline,bool previousDiscard,
                          LaneModel* lane,DiscardReport& rep,const FishPlans* targets){
    const int remaining=g.rules->n*g.rules->n-g.drawn;
    // With no tiles left to draw a pure discard cannot help (P1-03 rule): keep
    // a placement, otherwise stop exactly like the parent no-move path.
    if(remaining<=0)return placeMove.score>=0?Action{ActionKind::Place,placeMove,{}}:Action{};
    Action anchor;
    if(placeMove.score>=0)anchor={ActionKind::Place,placeMove,{}};
    else{
        std::string letters;for(char ch:g.hand)if(ch!='.')letters+=ch;
        anchor={ActionKind::Discard,{},letters};
    }
    if(!dc.mode||cap<=0||Clock::now()>=deadline||gen.stopped)return anchor;
    double preWork=0;int subsets=0;
    const std::vector<std::string> screened=targets?targets->discards:
        discardCandidates(g,gen,lane,la.ev,dc.keep,preWork,subsets,deadline);
    if(targets)subsets=int(screened.size());
    rep.subsets=std::max(0,subsets);
    if(subsets<0||screened.empty())return anchor;
    rep.attempted=1;
    std::vector<Action> cands{anchor};
    for(const auto& s:screened){
        if(anchor.kind==ActionKind::Discard&&s==anchor.letters)continue;
        cands.push_back({ActionKind::Discard,{},s});
    }
    if(dc.stop&&!(dc.stopDominated&&anchor.kind==ActionKind::Place))cands.push_back({});
    rep.cands=int(cands.size());
    DeepConfig dcfg;              // continuation via the validated P2-01 machinery
    dcfg.mode=1;dcfg.depth=dc.depth;dcfg.samples=dc.samples;dcfg.policy=dc.policy;dcfg.tail=dc.tail;
    dcfg.fishing=dc.fishing;dcfg.floor=dc.floor;
    DeepReport deepRep;
    // The first fair round is funded up to firstCap so that a triggered decision
    // is not abandoned after paying for the pre-screen; further rounds need cap.
    DeepContext ctx{dcfg,la,deadline,std::max(0.0,std::max(cap,firstCap)-preWork),deepRep};
    LookaheadSampler sampler(g.rules->letterWeight);
    std::vector<double> sums(cands.size(),0),tiles(cands.size(),0),squares(cands.size(),0);
    std::vector<std::vector<double>> gains(cands.size()),draws(cands.size());
    for(int round=0;round<dc.samples;++round){
        if(round==1)ctx.cap=std::max(0.0,cap-preWork);
        if(ctx.expired())break;
        const uint64_t seed=rng();
        std::mt19937_64 env(seed);
        std::string tape(g.rules->n*g.rules->n-g.drawn+g.rules->t+g.rules->x,'A');
        for(char& ch:tape)ch=sampler.draw(env);
        std::vector<double> add(cands.size(),0);std::vector<int> used(cands.size(),0);bool complete=true;
        for(size_t i=0;i<cands.size();++i){
            if(!rolloutValueAction(g,cands[i],tape,mixSeed(seed),ctx,add[i],&used[i])){complete=false;break;}
            ++rep.evals;
        }
        if(!complete){rep.cut=true;break;}
        ++rep.rounds;
        for(size_t i=0;i<cands.size();++i){sums[i]+=add[i];tiles[i]+=used[i];gains[i].push_back(add[i]);draws[i].push_back(used[i]);}
    }
    if(rep.rounds<dc.samples)rep.cut=true;   // any shortfall of fair rounds is a cut
    rep.work=preWork+deepRep.work;
    rep.steps=deepRep.steps;rep.discards=deepRep.discards;rep.invalid=deepRep.invalid;
    rep.fishCalls=deepRep.fishCalls;rep.fishTargets=deepRep.fishTargets;
    rep.fishSwaps=deepRep.fishSwaps;rep.fishWork=deepRep.fishWork;
    if(rep.rounds==0)return anchor;
    // rho: the anchor's gain per tile drawn, 0 when continuing loses or draws nothing.
    const double rho=dc.rate&&sums[0]>0&&tiles[0]>0?sums[0]/tiles[0]:0;
    const auto mean=[&](size_t i){return (sums[i]-rho*tiles[i])/rep.rounds;};
    // Paired standard error of the advantage over the anchor on the shared tapes.
    const auto error=[&](size_t i){
        if(dc.confidence<=0||rep.rounds<2)return 0.0;
        const double n=rep.rounds,delta=mean(i)-mean(0);double ss=0;
        for(int r=0;r<rep.rounds;++r){
            const double d=(gains[i][r]-rho*draws[i][r])-(gains[0][r]-rho*draws[0][r])-delta;ss+=d*d;
        }
        return dc.confidence*std::sqrt(ss/(n-1)/n);
    };
    rep.qAnchor=mean(0);
    size_t best=0;double bestScore=mean(0)+dc.margin;
    for(size_t i=1;i<cands.size();++i){
        const double score=mean(i)-error(i);
        if(score>bestScore){bestScore=score;best=i;}
    }
    // Thin evidence must not end the game or extend a paid discard streak.
    if(best!=0&&rep.rounds<2&&(cands[best].kind==ActionKind::Stop||(previousDiscard&&g.rules->penalty>0)))
        best=0;
    rep.qBest=mean(best);
    rep.stopChosen=cands[best].kind==ActionKind::Stop;
    rep.letters=cands[best].kind==ActionKind::Discard?cands[best].letters:std::string();
    rep.changed=!(cands[best].kind==anchor.kind&&
                  (cands[best].kind!=ActionKind::Place?cands[best].letters==anchor.letters
                                                      :sameMove(cands[best].move,anchor.move)));
    return cands[best];
}
}
