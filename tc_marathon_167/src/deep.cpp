#include "deep.h"

#include <algorithm>
#include <array>
#include <cmath>

#include "lookahead.h"
#include "lane_model.h"
#include "transition.h"

namespace mm167 {
Action fallbackAction(const Game& g,const Move& move){
    if(move.score>=0)return {ActionKind::Place,move,{}};
    if(g.drawn<g.rules->n*g.rules->n){
        std::string letters;for(char ch:g.hand)if(ch!='.')letters+=ch;
        if(!letters.empty())return {ActionKind::Discard,{},letters};
    }
    return {};
}
bool rolloutPolicy(const Game& g,uint64_t policySeed,DeepContext& ctx,Action& action,double* leaf){
    Generator gen(g,ctx.cfg.policy==3?64:ctx.cfg.policy==2?4:0);
    if(!ctx.generate(gen))return false;
    Move move=gen.best;
    if(leaf)*leaf=ctx.potential(g,gen);
    if(ctx.cfg.policy==4)return fishingPolicy(g,gen,ctx,action);
    if((ctx.cfg.policy==2||ctx.cfg.policy==3)&&move.score>=0){
        LookaheadConfig light;light.k=ctx.cfg.policy==3?8:4;light.samples=2;light.ev=ctx.la.ev;
        light.ev.pool=ctx.cfg.policy==3?64:0;light.lane=ctx.cfg.policy==3?ctx.la.lane:nullptr;
        LookaheadReport r;std::mt19937_64 rng(policySeed);
        move=runLookahead(g,gen,light,rng,ctx.deadline,r);
        ctx.rep.work+=cost::RUN*r.runs2+cost::NODE*r.nodes2+cost::CROSS*r.cross2+cost::BOARD*r.boards;
        ctx.rep.work+=cost::SCAN*r.scanWords+cost::BUILD*r.buildWords;
        if(r.rounds==0)return false;
    }
    action=fallbackAction(g,move);return true;
}
bool applyTape(Game& g,const Action& a,const std::string& tape,int origin,DeepContext& ctx){
    int firstUnknown=g.drawn+std::min(g.rules->x,g.rules->n*g.rules->n-g.drawn);
    int offset=firstUnknown-origin;
    if(offset<0||offset>int(tape.size())){++ctx.rep.invalid;return false;}
    auto result=transition(g,a,tape.substr(offset,g.rules->t+g.rules->x),ctx.deadline);
    if(result!=TransitionResult::Applied){if(result!=TransitionResult::Expired)++ctx.rep.invalid;return false;}
    ++ctx.rep.steps;ctx.rep.discards+=a.kind==ActionKind::Discard;
    ctx.rep.work+=cost::RUN;
    return true;
}
bool rolloutValueAction(const Game& root,const Action& first,const std::string& tape,uint64_t policySeed,
                        DeepContext& ctx,double& value,int* drawn){
    if(drawn)*drawn=0;
    if(first.kind==ActionKind::Stop){value=0;return true;}
    Game sim(root);
    const int origin=root.drawn+std::min(root.rules->x,root.rules->n*root.rules->n-root.drawn);
    if(!applyTape(sim,first,tape,origin,ctx))return false;
    double prefix=double(sim.score-root.score);
    for(int step=0;step<ctx.cfg.depth&&!sim.terminal;++step){
        Action a;
        if(!rolloutPolicy(sim,mixSeed(policySeed+step),ctx,a))return false;
        if(!applyTape(sim,a,tape,origin,ctx))return false;
        prefix=std::max(prefix,double(sim.score-root.score));
    }
    value=double(sim.score-root.score);
    if(drawn)*drawn=sim.drawn-root.drawn;
    if(!sim.terminal&&ctx.cfg.tail>0){
        Generator leaf(sim);if(!ctx.generate(leaf))return false;
        value+=ctx.potential(sim,leaf);
    }
    if(ctx.cfg.floor)value=std::max(value,prefix);
    return true;
}
bool rolloutValue(const Game& root,const Move& first,const std::string& tape,uint64_t policySeed,
                  DeepContext& ctx,double& value){
    return rolloutValueAction(root,{ActionKind::Place,first,{}},tape,policySeed,ctx,value);
}
bool treeValue(const Game& g,int depth,uint64_t seed,DeepContext& ctx,double& value){
    if(g.terminal){value=0;return true;}
    Generator gen(g,depth?ctx.cfg.branch:0);if(!ctx.generate(gen))return false;
    if(!depth){value=ctx.potential(g,gen);return true;}
    std::vector<Action> actions;
    for(const auto& c:gen.top)actions.push_back({ActionKind::Place,c.move,{}});
    if(actions.empty())actions.push_back(fallbackAction(g,gen.best));
    std::vector<State> seen;
    value=-1e100;
    LookaheadSampler sampler(g.rules->letterWeight);
    // At the last action only board potential remains; it does not depend on
    // sampled rack/preview letters. An exhausted visible deck is exact too.
    const int chanceCount=(depth==1||g.rules->n*g.rules->n-g.drawn<=g.rules->x)?1:ctx.cfg.chance;
    for(const Action& a:actions){
        double sum=0;
        std::mt19937_64 chanceRng(seed);
        bool duplicate=false;
        for(int sample=0;sample<chanceCount;++sample){
            if(ctx.expired())return false;
            std::string stream(g.rules->t+g.rules->x,'A');for(char& ch:stream)ch=sampler.draw(chanceRng);
            Game next(g);
            auto result=transition(next,a,stream,ctx.deadline);
            if(result!=TransitionResult::Applied){if(result!=TransitionResult::Expired)++ctx.rep.invalid;return false;}
            ctx.rep.work+=cost::RUN;++ctx.rep.steps;ctx.rep.discards+=a.kind==ActionKind::Discard;
            if(sample==0){
                duplicate=std::find(seen.begin(),seen.end(),static_cast<const State&>(next))!=seen.end();
                if(duplicate)break;
                seen.push_back(next);
            }
            double future=0;
            if(!treeValue(next,depth-1,mixSeed(seed+sample+1),ctx,future))return false;
            sum+=double(next.score-g.score)+future;
        }
        if(!duplicate)value=std::max(value,sum/chanceCount);
    }
    return true;
}
Move runDeep(const Game& g,const Move& parent,const std::vector<RankedMove>& ranked,
             const DeepConfig& cfg,const LookaheadConfig& la,double cap,std::mt19937_64& rng,
             Clock::time_point deadline,DeepReport& rep,double firstCap){
    // firstCap>cap funds the first fair round beyond the turn cap (still inside
    // the game plan): an admitted turn then compares its roots at least once.
    if(!cfg.mode||ranked.size()<2||std::max(cap,firstCap)<=0||Clock::now()>=deadline)return parent;
    std::vector<Move> roots{parent};
    for(const auto& c:ranked){
        if(int(roots.size())>=cfg.width)break;
        if(std::none_of(roots.begin(),roots.end(),[&](const Move& m){return sameMove(m,c.move);}))roots.push_back(c.move);
    }
    if(roots.size()<2)return parent;
    rep.attempted=1;rep.width=int(roots.size());
    DeepContext ctx{cfg,la,deadline,std::max(cap,firstCap),rep};
    LookaheadSampler sampler(g.rules->letterWeight);
    std::vector<double> sums(roots.size(),0);
    std::vector<double> deltaSquares(roots.size(),0),base(roots.size(),0);
    for(size_t i=0;i<roots.size();++i)for(const auto& c:ranked)if(sameMove(roots[i],c.move)){base[i]=c.q;break;}
    Move best=parent;
    for(int round=0;round<cfg.samples;++round){
        if(round==1)ctx.cap=cap;
        if(ctx.expired())break;
        const uint64_t seed=rng();std::mt19937_64 env(seed);
        std::string tape(g.rules->n*g.rules->n-g.drawn+g.rules->t+g.rules->x,'A');
        for(char& ch:tape)ch=sampler.draw(env);
        std::vector<double> add(roots.size(),0);bool complete=true;
        for(size_t i=0;i<roots.size();++i){
            bool ok;
            if(cfg.mode==1)ok=rolloutValue(g,roots[i],tape,mixSeed(seed),ctx,add[i]);
            else{
                Game next(g);
                const int origin=g.drawn+std::min(g.rules->x,g.rules->n*g.rules->n-g.drawn);
                ok=applyTape(next,{ActionKind::Place,roots[i],{}},tape,origin,ctx);
                double future=0;
                if(ok)ok=treeValue(next,cfg.depth-1,mixSeed(seed),ctx,future);
                add[i]=double(next.score-g.score)+future;
            }
            if(!ok){complete=false;rep.cut=true;break;}
            ++rep.evals;
        }
        if(!complete)break;
        ++rep.rounds;
        size_t winner=0;
        double advantage=0;
        for(size_t i=0;i<roots.size();++i){
            sums[i]+=add[i];double delta=add[i]-add[0];deltaSquares[i]+=delta*delta;
        }
        for(size_t i=1;i<roots.size();++i){
            const double n=rep.rounds,delta=(sums[i]-sums[0])/n;
            double margin=cfg.mix*delta+(1-cfg.mix)*(base[i]-base[0]);
            if(cfg.confidence>0){
                if(rep.rounds<2)continue;
                const double variance=std::max(0.0,(deltaSquares[i]-n*delta*delta)/(n-1));
                margin-=cfg.confidence*cfg.mix*std::sqrt(variance/n);
            }
            if(margin>advantage){advantage=margin;winner=i;}
        }
        best=roots[winner];rep.q=sums[winner]/rep.rounds;
    }
    rep.changed=!sameMove(parent,best);
    return best;
}
}
