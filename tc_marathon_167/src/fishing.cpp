#include "fishing.h"

#include <algorithm>
#include <chrono>

#include "deep.h"
#include "discard.h"

namespace mm167 {
double leaveValue(const Game& g,const Generator& gen,LaneModel& lane,const EvalConfig& ev,
                  const int* kept,int draws){
    if(lane.root.empty())return 0;
    if(draws<0)draws=0;
    uint32_t keptSet=0;uint64_t keptKey=1469598103934665603ULL;
    for(int k=0;k<26;++k){if(kept[k])keptSet|=1u<<k;keptKey=(keptKey^uint64_t(kept[k]+1))*1099511628211ULL;}
    int count=0;int32_t sure=0;
    lane.merged.clear();
    for(auto& w:lane.root){
        if(count>=ev.lanes)break;
        if(!w.lane)w.lane=lane.resolve(g,gen,w);
        const LaneModel::Summary& sum=lane.summary(w.lane,kept,keptSet,keptKey,draws);
        if(sum.count&&sum.e[0].key==0&&sum.e[0].score>sure)sure=sum.e[0].score;
        ++count;
        for(int i=0;i<sum.count;++i)if(sum.e[i].key&&sum.e[i].score>sure)lane.merged.push_back(sum.e[i]);
    }
    std::sort(lane.merged.begin(),lane.merged.end(),[](const LaneModel::Event& a,const LaneModel::Event& b){
        return a.score!=b.score?a.score>b.score:a.key<b.key;});
    // A later window may have raised the guaranteed score after an earlier
    // event was collected. Such an event cannot improve on that guarantee.
    if(lane.fixEvents)lane.merged.erase(std::remove_if(lane.merged.begin(),lane.merged.end(),
        [&](const LaneModel::Event& e){return e.score<=sure;}),lane.merged.end());
    double none=1,total=0;
    for(size_t i=0;i<lane.merged.size()&&none>1e-6;++i){
        const LaneModel::Event& e=lane.merged[i];
        bool repeated=false;
        for(size_t j=0;j<i&&!repeated;++j)repeated=lane.merged[j].key==e.key;
        if(repeated)continue;
        total+=none*e.p*e.score;none*=1-e.p;
    }
    return total+none*sure;
}
int fishRefill(const Game& g,int removed,int* kept){
    int refill=std::min(g.rules->n*g.rules->n-g.drawn,
                       removed+int(std::count(g.hand.begin(),g.hand.end(),'.')));
    int known=0;
    for(int i=0;i<refill&&i<int(g.upcoming.size());++i){
        char c=g.upcoming[i];if(c>='A'&&c<='Z'){++kept[c-'A'];++known;}
    }
    return refill-known;
}
bool fishPlans(const Game& g,const Generator& gen,LaneModel& lane,const EvalConfig& base,
               const Policy4Config& cfg,double& work,double cap,Clock::time_point deadline,FishPlans& out){
    FishMeter meter(lane,work,cap,deadline);
    if(meter.expired())return false;
    EvalConfig ev=base;ev.lanes=std::min(ev.lanes,cfg.lanes);
    lane.prepare(g,gen,ev);work+=cost::BOARD+cost::RETAIN_SCAN*lane.root.size();
    if(meter.expired())return false;
    std::string full;for(char c:g.hand)if(c!='.')full+=c;
    std::sort(full.begin(),full.end());
    out={};if(full.empty()||g.drawn>=g.rules->n*g.rules->n)return true;
    out.discards.push_back(full);
    std::unordered_map<std::string,double> keyed;
    std::vector<double> top;   // best `targets` plan values, descending
    uint32_t rackSet=0;for(char c:full)rackSet|=1u<<(c-'A');
    int windows=0;
    for(auto& window:lane.root){
        if(windows++>=ev.lanes)break;
        if(meter.expired())return false;
        window.lane=lane.resolve(g,gen,window);
        if(meter.expired())return false;
        size_t scanned=0;
        const LaneModel::Lane& L=*window.lane;
        for(int idx:L.byScore){
            const LaneModel::Word& word=L.words[idx];
            if((scanned++&127)==0&&meter.expired())return false;
            // value<=score: once `targets` distinct plans beat every remaining score, stop.
            if(int(top.size())>=cfg.targets&&double(word.score)<=top.back())break;
            // Cheap filter first (same charge as summary); full target construction costs more.
            ++lane.scanWords;
            if(__builtin_popcount(word.set&~rackSet)>3)continue;
            lane.scanWords+=3;
            int need[26]={},kept[26]={};
            for(int q=0;q<word.tiles;++q)++need[word.need[q]];
            std::string discard;
            for(char c:full){int k=c-'A';if(kept[k]<need[k])++kept[k];else discard+=c;}
            if(discard.empty())continue;
            int draws=fishRefill(g,int(discard.size()),kept),missing=0;uint64_t key=0;
            double p=1;
            for(int k=0;k<26;++k)for(int u=1;u<=need[k]-kept[k];++u){p*=lane.freq[k]/u;++missing;key=key<<5|uint64_t(k+1);}
            if(missing>std::min(draws,3))continue;
            for(int j=0;j<missing;++j)p*=draws-j;
            const double value=(lane.exactRefill?lane.refillProbability(key,draws):std::min(1.0,p))*word.score;
            if(value<=0)continue;
            out.targetValue=std::max(out.targetValue,value);
            auto it=keyed.find(discard);
            if(it==keyed.end()){
                keyed.emplace(discard,value);
                top.insert(std::lower_bound(top.begin(),top.end(),value,std::greater<double>()),value);
                if(int(top.size())>cfg.targets)top.pop_back();
            } else if(value>it->second){
                // Replace this plan's old value in the bounded top list.
                auto pos=std::find(top.begin(),top.end(),it->second);
                if(pos!=top.end())top.erase(pos);
                it->second=value;
                top.insert(std::lower_bound(top.begin(),top.end(),value,std::greater<double>()),value);
                if(int(top.size())>cfg.targets)top.pop_back();
            }
        }
    }
    std::vector<std::pair<std::string,double>> ranked(keyed.begin(),keyed.end());
    std::sort(ranked.begin(),ranked.end(),[](const auto& a,const auto& b){
        return a.second!=b.second?a.second>b.second:a.first<b.first;
    });
    for(const auto& c:ranked){
        if(int(out.discards.size())>=cfg.targets+1)break;
        if(c.first!=full)out.discards.push_back(c.first);
    }
    return !meter.expired();
}
bool fishingPolicy(const Game& g,const Generator& gen,DeepContext& ctx,Action& action){
    action=fallbackAction(g,gen.best);
    if(!ctx.la.lane||g.drawn>=g.rules->n*g.rules->n)return true;
    // dev2: a move already earning maxRate points per tile is never swapped away.
    if(gen.best.score>=0&&gen.best.used>0&&double(gen.best.score)/gen.best.used>=ctx.cfg.fishing.maxRate)return true;
    ++ctx.rep.fishCalls;
    const double start=ctx.rep.work;
    struct Charge { DeepReport& r;double start;~Charge(){r.fishWork+=r.work-start;} } charge{ctx.rep,start};
    auto& lane=*ctx.la.lane;
    EvalConfig ev=ctx.la.ev;ev.lanes=std::min(ev.lanes,ctx.cfg.fishing.lanes);
    FishPlans plans;
    if(!fishPlans(g,gen,lane,ev,ctx.cfg.fishing,ctx.rep.work,ctx.cap,ctx.deadline,plans))return false;
    if(plans.targetValue<=0)return true; // no reachable lane: exact greedy fallback
    ctx.rep.fishTargets+=int(plans.discards.size());
    FishMeter meter(lane,ctx.rep.work,ctx.cap,ctx.deadline);
    double swap=-1e100;std::string bestDiscard;
    for(const auto& letters:plans.discards){
        int kept[26]={};for(char c:g.hand)if(c!='.')++kept[c-'A'];
        for(char c:letters)--kept[c-'A'];
        int draws=fishRefill(g,int(letters.size()),kept);
        const double value=-g.rules->penalty+leaveValue(g,gen,lane,ev,kept,draws);
        if(value>swap){swap=value;bestDiscard=letters;}
        if(meter.expired())return false;
    }
    double place=double(gen.best.score);
    if(gen.best.score>=0){
        Game after(g);int kept[26]={};for(char c:g.hand)if(c!='.')++kept[c-'A'];
        int r=gen.best.r,c=gen.best.c;
        for(char ch:gen.best.word){if(!g.occupied(r,c))--kept[ch-'A'];r+=gen.best.dir;c+=1-gen.best.dir;}
        int draws=fishRefill(g,gen.best.used,kept);
        after.apply(gen.best); // board only; rack expectation is kept + unknown draws
        // dev2: the P1-02 move valuation reuses the root cross-checks and the root
        // windows prepared by fishPlans; no second cross-check build per step.
        place+=lane.moveValue(g,after,gen,ev,gen.best,kept,draws);
        ctx.rep.work+=cost::BOARD;
        if(meter.expired())return false;
    }
    if((gen.best.score>=0&&swap>place+ctx.cfg.fishing.margin)||
       (gen.best.score<0&&swap>0)){
        action={ActionKind::Discard,{},bestDiscard};
        ctx.rep.fishSwaps+=gen.best.score>=0;
    }
    return true;
}
bool fishingRoot(const Game& g,const Generator& gen,const Move& move,const DiscardConfig& dc,
                 const LookaheadConfig& la,double cap,Clock::time_point deadline,double& work,FishPlans& plans){
    if(!dc.fishRoot||!la.lane||move.score<0||move.used<=0||
       double(move.score)/move.used>=dc.fishMaxRate||cap<=0)return false;
    if(!fishPlans(g,gen,*la.lane,la.ev,dc.fishing,work,cap,deadline,plans))return false;
    return plans.targetValue>move.score+g.rules->penalty+dc.fishing.margin;
}
}
