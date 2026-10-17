#define MM167_TEST
#include "../src/WordGame.cpp"
#include <cstdlib>
using namespace mm167;
void require(bool value,const char* message){if(!value){std::cerr<<message<<'\n';std::exit(1);}}
std::shared_ptr<Rules> rules(int n=5,int t=4,int x=2){
    auto r=std::make_shared<Rules>();r->n=n;r->t=t;r->x=x;r->penalty=3;r->sr=r->sc=n/2;
    for(auto& row:r->bonus)row.fill('.');r->bonus[r->sr][r->sc]='S';
    for(int len=2;len<=3;++len)for(int bits=0;bits<(1<<len);++bits){
        std::string w;for(int i=0;i<len;++i)w+=((bits>>i)&1)?'A':'B';
        r->dict.add(w);r->words.push_back(w);for(char ch:w)++r->letterWeight[ch-'A'];
    }
    return r;
}
// Exact greedy-policy continuation value on a fully visible future: replicates
// rolloutValueAction (policy 1, tail 0) without any sampling. Valid because
// remaining<=x makes every refill and preview letter observable.
double exactPolicyValue(const Game& root,const Action& first,int depth){
    if(first.kind==ActionKind::Stop)return 0;
    Game sim(root);
    auto apply=[&](const Action& a)->bool{
        return transition(sim,a,"")==TransitionResult::Applied;
    };
    if(!apply(first))return -1e9;
    for(int step=0;step<depth&&!sim.terminal;++step){
        Generator now(sim);now.run(Clock::time_point::max());
        Action a=fallbackAction(sim,now.best);
        if(a.kind==ActionKind::Stop)break;
        if(!apply(a))return -1e9;
    }
    return double(sim.score-root.score);
}
// Exact optimal continuation over every action (placements, deduped discard
// subsets, stop) on a fully visible future.
double exactOptimal(const Game& g,int depth){
    if(depth==0||g.terminal)return 0;
    double best=0;   // stop value
    Generator gen(g,64);gen.run(Clock::time_point::max());
    for(const Move& m:gen.all){
        Game next(g);
        if(transition(next,{ActionKind::Place,m,{}},std::string())!=TransitionResult::Applied)continue;
        best=std::max(best,double(next.score-g.score)+exactOptimal(next,depth-1));
    }
    std::string tiles;for(char ch:g.hand)if(ch!='.')tiles+=ch;
    std::set<std::string> seen;
    for(int mask=1;mask<(1<<tiles.size());++mask){
        std::string letters,rep;
        for(size_t i=0;i<tiles.size();++i)if(mask>>i&1){letters+=tiles[i];rep+=tiles[i];}
        std::sort(rep.begin(),rep.end());
        if(!seen.insert(rep).second)continue;
        Game next(g);
        if(transition(next,{ActionKind::Discard,{},letters},std::string())!=TransitionResult::Applied)continue;
        best=std::max(best,double(next.score-g.score)+exactOptimal(next,depth-1));
    }
    return best;
}
int main(){
    LookaheadConfig la;la.ev.alpha=5;
    const auto far=Clock::time_point::max();
    int noWordCases=0,activeCases=0,oracleCases=0,dedupChecks=0;
    for(int tc=0;tc<200;++tc){
        auto r=rules(5,4,tc%4);Game g(r);
        g.hand=tc%2?"ABAB":"BBAA";g.drawn=4;
        std::string future;for(int i=0;i<25;++i)future+=((i+tc)%3)?'A':'B';
        g.upcoming=future.substr(0,r->x);
        Generator gen(g,32);gen.run(far);
        const State saved=g;
        const std::string fullHand=tc%2?"ABAB":"BBAA";
        Move noMove;noMove.score=-1;   // synthetic no-word root (anchor = full-hand discard)
        DiscardConfig dc;dc.mode=1;dc.keep=8;dc.samples=4;dc.depth=4;dc.stop=(tc%2)==0;
        // 1-2: off / exhausted / no-cap / R=0 return the anchor without RNG use.
        std::mt19937_64 offRng(tc),untouched(tc);
        DiscardConfig off=dc;off.mode=0;DiscardReport offRep;
        Action offAction=runDiscardDecision(g,gen,noMove,0,off,la,1e12,1e12,offRng,far,false,nullptr,offRep);
        require(offAction.kind==ActionKind::Discard&&offAction.letters==fullHand,"off mode returns parent no-word discard");
        require(offRng()==untouched()&&offRep.attempted==0,"off mode must not consume RNG or attempt");
        DiscardReport deadRep;std::mt19937_64 deadRng(tc);
        Action cut=runDiscardDecision(g,gen,noMove,0,dc,la,1,1,deadRng,far,false,nullptr,deadRep);
        require(cut.kind==ActionKind::Discard&&cut.letters==fullHand&&deadRep.rounds==0&&deadRep.cut,"tiny cap cuts before any round");
        Game exhausted(g);exhausted.drawn=r->n*r->n;
        DiscardReport r0Rep;std::mt19937_64 r0Rng(tc);
        Generator genEx(exhausted,32);genEx.run(far);
        Action r0=runDiscardDecision(exhausted,genEx,noMove,0,dc,la,1e12,1e12,r0Rng,far,false,nullptr,r0Rep);
        require(r0.kind!=ActionKind::Discard&&r0Rep.attempted==0&&r0Rep.discards==0,"R=0 never discards");
        // 2b: a funded first round completes exactly one fair round under a tiny cap.
        DiscardReport oneRep;std::mt19937_64 oneRng(tc);
        Action one=runDiscardDecision(g,gen,noMove,0,dc,la,1,1e12,oneRng,far,false,nullptr,oneRep);
        require(oneRep.rounds==1&&oneRep.cut&&oneRep.evals==oneRep.cands,"funded first round is exactly one fair round");
        require(one.kind!=ActionKind::Stop,"one round is too thin to stop");
        // 3: determinism and root isolation on a no-word decision.
        DiscardReport a,b;std::mt19937_64 ra(tc),rb(tc);
        Action pa=runDiscardDecision(g,gen,noMove,0,dc,la,1e12,1e12,ra,far,false,nullptr,a);
        Action pb=runDiscardDecision(g,gen,noMove,0,dc,la,1e12,1e12,rb,far,false,nullptr,b);
        require(pa.kind==pb.kind&&pa.letters==pb.letters&&a.rounds==b.rounds&&a.qAnchor==b.qAnchor,"same-seed determinism");
        require(g==saved,"decision must not mutate the root");
        if(pa.kind==ActionKind::Discard){
            int count[26]={};for(char ch:g.hand)if(ch!='.')++count[ch-'A'];
            for(char ch:pa.letters)require(--count[ch-'A']>=0,"discard letters must be a hand sub-multiset");
        }
        require(a.evals==a.cands*a.rounds,"every candidate has every accepted round");
        ++noWordCases;
        // 4: the hidden tail is not an input: the decision sees only `upcoming`.
        Game projected(g);projected.upcoming=g.upcoming;
        DiscardReport vis;std::mt19937_64 rv(tc);
        Action pv=runDiscardDecision(projected,gen,noMove,0,dc,la,1e12,1e12,rv,far,false,nullptr,vis);
        require(pv.kind==pa.kind&&pv.letters==pa.letters&&vis.qAnchor==a.qAnchor,"hidden tail must not change the decision");
        // 5: subset enumeration dedup on duplicate-heavy hands (t=9).
        auto r9=rules(5,9,2);Game dup(r9);dup.hand="AABABABBB";dup.drawn=12;
        dup.upcoming=future.substr(0,r9->x);
        Generator dupGen(dup,32);dupGen.run(far);
        double work=0;int subsets=0;
        auto cands=discardCandidates(dup,dupGen,nullptr,la.ev,8,work,subsets,far);
        require(int(cands.size())<=8&&subsets==29,"distinct multiset count on AABABABBB");
        std::set<std::string> multisets;
        require(!cands.empty()&&cands[0]=="AABABABBB","full hand is always the first candidate");
        for(const auto& s:cands){std::string rep=s;std::sort(rep.begin(),rep.end());multisets.insert(rep);}
        require(multisets.size()==cands.size(),"candidates are distinct multisets");
        double work2=0;int subsets2=0;
        auto again=discardCandidates(dup,dupGen,nullptr,la.ev,8,work2,subsets2,far);
        require(again==cands&&subsets2==subsets,"enumeration is deterministic");
        ++dedupChecks;
        // 6: mode 2 with a placement anchor keeps the place on truncation.
        DiscardConfig act=dc;act.mode=2;
        DiscardReport actRep;std::mt19937_64 actRng(tc);
        Action kept=runDiscardDecision(g,gen,gen.best,0,act,la,1,1,actRng,far,false,nullptr,actRep);
        require(kept.kind==ActionKind::Place&&sameMove(kept.move,gen.best)&&actRep.rounds==0,"active mode truncation keeps the place");
        ++activeCases;
    }
    // 7-8: exhaustive small-state oracles on fully visible futures.
    for(int tc=0;tc<120;++tc){
        auto r=rules(3,4,5);Game g(r);           // remaining=5<=x: whole future visible
        g.hand=tc%2?"AABB":"ABBA";g.drawn=4;
        std::string future;for(int i=0;i<5;++i)future+=((i+tc)%2)?'A':'B';
        g.upcoming=future;
        if(tc%2){Generator pre(g,64);pre.run(far);
            if(!pre.all.empty()){require(transition(g,{ActionKind::Place,pre.all[tc%pre.all.size()],{}},std::string())==TransitionResult::Applied,"endgame prefix");}}
        Generator gen(g,64);gen.run(far);
        // Two configurations: the plain estimator, and the frozen P1-03 candidate
        // (8 samples, keep 8, two paired standard errors). A fully visible future
        // has zero sampling variance, so the confidence margin must not hide a
        // better action.
        for(int variant=0;variant<2;++variant){
        DiscardConfig dc;dc.mode=gen.best.score>=0?2:1;dc.keep=variant?8:16;dc.samples=variant?8:4;dc.depth=4;dc.stop=true;dc.margin=0;dc.tail=0;
        dc.confidence=variant?2:0;
        DiscardReport rep;std::mt19937_64 rng(tc);
        Action chosen=runDiscardDecision(g,gen,gen.best,0,dc,la,1e12,1e12,rng,far,false,nullptr,rep);
        require(rep.rounds==dc.samples&&rep.invalid==0,"visible futures complete every round");
        // O1: never stop while a confirmed positive continuation exists.
        const double optimal=exactOptimal(g,3);
        if(optimal>0)require(chosen.kind!=ActionKind::Stop,"stop must not cut a confirmed positive continuation");
        // O2: the chosen action attains the best exact greedy continuation among
        // the compared candidates (anchor + prescreened subsets + stop).
        std::vector<Action> compared;
        if(gen.best.score>=0)compared.push_back({ActionKind::Place,gen.best,{}});
        else{std::string letters;for(char ch:g.hand)if(ch!='.')letters+=ch;compared.push_back({ActionKind::Discard,{},letters});}
        double work=0;int subsets=0;
        for(const auto& s:discardCandidates(g,gen,nullptr,la.ev,dc.keep,work,subsets,far))compared.push_back({ActionKind::Discard,{},s});
        compared.push_back({});
        double bestValue=exactPolicyValue(g,compared[0],dc.depth);
        for(const Action& c:compared)bestValue=std::max(bestValue,exactPolicyValue(g,c,dc.depth));
        require(std::abs(exactPolicyValue(g,chosen,dc.depth)-bestValue)<1e-9,"chosen action attains the best greedy continuation");
        ++oracleCases;
        }
    }
    std::cout<<"Passed: "<<noWordCases<<" no-word cases, "<<activeCases<<" active cases, "
             <<dedupChecks<<" dedup checks, "<<oracleCases<<" exhaustive small-state oracles.\n";
}
