#define MM167_TEST
#include "../src/WordGame.cpp"
#include <cstdlib>
using namespace mm167;
void require(bool ok,const char* msg){if(!ok){std::cerr<<msg<<'\n';std::exit(1);}}
int main(){
    const auto far=Clock::time_point::max();
    // Independent enumeration of all equiprobable tapes validates repeated
    // letters, overlapping requirements, no draws, and the memoized path.
    {
        Rules rules;for(int i=0;i<4;++i)rules.letterWeight[i]=1;
        LaneModel model(rules);
        for(int a=0;a<=3;++a)for(int b=0;b<=3-a;++b)for(int c=0;c<=3-a-b;++c){
            uint64_t key=0;for(int i=0;i<a;++i)key=key<<5|1;
            for(int i=0;i<b;++i)key=key<<5|2;for(int i=0;i<c;++i)key=key<<5|3;
            for(int draws=0;draws<=6;++draws){
                int hits=0,total=1<<(2*draws);
                for(int tape=0;tape<total;++tape){
                    int counts[4]={},value=tape;
                    for(int j=0;j<draws;++j){++counts[value&3];value>>=2;}
                    hits+=counts[0]>=a&&counts[1]>=b&&counts[2]>=c;
                }
                const double expected=double(hits)/total;
                require(std::abs(model.refillProbability(key,draws)-expected)<1e-12,"exact refill exhaustive oracle");
                require(std::abs(model.refillProbability(key,draws)-expected)<1e-12,"exact refill cache oracle");
            }
        }
        require(std::abs(model.refillProbability(1,9)-(1-std::pow(.75,9)))<1e-12,"nine-draw binomial probability");
    }
    // Distinct words needing the same refill are one event. They must not
    // crowd every other refill out of the eight-event summary.
    {
        Rules rules; rules.letterWeight[0]=rules.letterWeight[1]=1;
        LaneModel model(rules);model.fixEvents=true; LaneModel::Lane words;
        for(int i=0;i<10;++i){
            LaneModel::Word w{};w.tiles=1;w.need[0]=i<9?0:1;
            w.set=1u<<w.need[0];w.score=i<9?100+i:90;
            words.words.push_back(w);words.sets.push_back(w.set);
        }
        int kept[26]={};const auto& summary=model.summary(&words,kept,0,123,1);
        require(summary.count==2,"duplicate refill events must be merged before top-eight truncation");
    }
    // A sure 100-point word dominates an earlier 50-point probabilistic word,
    // regardless of the order in which board windows are visited.
    {
        auto rules=std::make_shared<Rules>();rules->n=7;rules->t=4;
        rules->letterWeight[0]=rules->letterWeight[1]=1;
        Game game(rules);Generator generator(game);LaneModel model(*rules);model.fixEvents=true;
        LaneModel::Lane possible,sure;
        LaneModel::Word a{};a.tiles=1;a.need[0]=0;a.set=1;a.score=50;
        possible.words.push_back(a);possible.sets.push_back(a.set);
        a.need[0]=1;a.set=2;a.score=100;sure.words.push_back(a);sure.sets.push_back(a.set);
        model.root={{9,1,0,0,0,0,&possible},{4,1,0,1,0,0,&sure}};
        int kept[26]={};kept[1]=1;EvalConfig ev;ev.lanes=2;
        const double forward=leaveValue(game,generator,model,ev,kept,1);
        std::reverse(model.root.begin(),model.root.end());
        const double reverse=leaveValue(game,generator,model,ev,kept,1);
        require(forward==100&&reverse==100,"sure score floor and window-order invariance");
    }
    auto r=std::make_shared<Rules>();r->n=7;r->t=4;r->x=4;r->penalty=3;r->sr=r->sc=3;
    for(auto& row:r->bonus)row.fill('.');r->bonus[3][3]='S';r->bonus[3][5]=r->bonus[3][6]='T';
    for(std::string w:{"AB","ABCD"}){r->dict.add(w);r->words.push_back(w);for(char c:w)++r->letterWeight[c-'A'];}
    Game g(r);g.board[3][3]='A';g.placed=1;g.drawn=5;g.hand="BZZZ";g.upcoming="CDEE";
    LaneModel lane(*r);LookaheadConfig la;la.lane=&lane;la.ev.lanes=16;
    DeepConfig dc;dc.policy=4;DeepReport rep;DeepContext ctx{dc,la,far,1e12,rep};
    const State saved=g;
    Action action;require(rolloutPolicy(g,167,ctx,action),"policy completion");
    require(action.kind==ActionKind::Discard&&action.letters=="ZZZ","target keep must retain B and discard redundant Zs");
    require(g==saved&&rep.fishSwaps==1&&rep.fishWork>0,"state isolation and accounted active swap");
    Game next(g);require(transition(next,action,"")==TransitionResult::Applied,"known preview discard transition");
    Generator after(next);after.run(far);require(after.best.score==81,"preserved lane must deliver exact 81 points");
    // Stale root windows must not affect a later policy call.
    lane.prepare(next,after,la.ev);
    DeepReport again;DeepContext ctx2{dc,la,far,1e12,again};Action a2;
    require(rolloutPolicy(g,999,ctx2,a2)&&a2.letters==action.letters,"fresh board windows / independent policy seed");
    dc.fishing.margin=100;DeepReport margin;DeepContext cm{dc,la,far,1e12,margin};
    require(rolloutPolicy(g,167,cm,a2)&&a2.kind==ActionKind::Place,"margin must prevent marginal swaps");dc.fishing.margin=0;
    Generator gen(g);gen.run(far);DiscardConfig root;root.mode=2;root.fishRoot=true;
    FishPlans plans;double work=0;
    require(fishingRoot(g,gen,gen.best,root,la,1e12,far,work,plans),"root trigger must open low-score high-potential state");
    require(plans.discards.front()=="BZZZ"&&plans.discards.size()<=4,"full swap retained and bounded candidates");
    for(const auto& s:plans.discards){Game copy(g);require(transition(copy,{ActionKind::Discard,{},s},"")==TransitionResult::Applied,"every target is a legal nonempty subset");}
    root.fishRoot=false;work=0;require(!fishingRoot(g,gen,gen.best,root,la,1e12,far,work,plans)&&work==0,"off trigger has zero work");
    DeepReport tiny;DeepContext ct{dc,la,far,1,tiny};require(!rolloutPolicy(g,167,ct,a2),"exhausted budget must reject incomplete policy");
    DeepReport expired;DeepContext ce{dc,la,Clock::time_point::min(),1e12,expired};require(!rolloutPolicy(g,167,ce,a2),"deadline must reject policy");
    // R=0 and empty-board states retain the parent action.
    for(int mode=0;mode<2;++mode){
        Game z(g);if(mode==0)z.drawn=49;else {z.placed=0;for(auto& row:z.board)row.fill('.');}
        Generator gz(z);gz.run(far);Action expected=fallbackAction(z,gz.best);
        DeepReport rz;DeepContext cz{dc,la,far,1e12,rz};require(rolloutPolicy(z,167,cz,a2),"fallback completion");
        require(a2.kind==expected.kind&&sameMove(a2.move,expected.move)&&a2.letters==expected.letters,"no target fallback equality");
    }
    int kept[26]={};Game shortBag(g);shortBag.drawn=48;
    require(fishRefill(shortBag,3,kept)==0&&kept['C'-'A']==1&&kept['D'-'A']==0,"partial final refill must not read beyond R");
    shortBag.drawn=45;shortBag.upcoming="?D??";std::fill(kept,kept+26,0);
    require(fishRefill(shortBag,3,kept)==2&&kept['D'-'A']==1,"unknown preview slots must remain unknown");
    // Policy 4 must execute legally on identical public tapes and preserve roots.
    for(int x=0;x<=4;++x){
        r->x=x;Game z(g);z.upcoming=g.upcoming.substr(0,x);
        Generator gz(z);gz.run(far);const State before=z;
        DeepConfig cfg;cfg.policy=4;cfg.depth=4;cfg.tail=0;
        DeepReport ra,rb;DeepContext ca{cfg,la,far,1e12,ra},cb{cfg,la,far,1e12,rb};double va=0,vb=0;
        std::string tape(60,'B');
        require(rolloutValueAction(z,fallbackAction(z,gz.best),tape,167,ca,va),"common tape rollout");
        require(rolloutValueAction(z,fallbackAction(z,gz.best),tape,167,cb,vb)&&va==vb&&ra.invalid==0&&rb.invalid==0&&z==before,"repeatable legal rollout and root isolation");
    }
    std::cout<<"fishing tests passed\n";
}
