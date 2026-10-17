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
double exactEndgame(const Game& g,int depth){
    if(!depth||g.terminal)return 0;
    Generator gen(g);gen.run(Clock::time_point::max());
    double best=0;
    for(const Move& m:gen.all){
        Game next(g);require(transition(next,{ActionKind::Place,m,{}},"")==TransitionResult::Applied,"exact transition");
        best=std::max(best,double(next.score-g.score)+exactEndgame(next,depth-1));
    }
    return best;
}
int main(){
    LookaheadConfig la;la.ev.alpha=5;
    DeepConfig cfg;cfg.mode=1;cfg.width=4;cfg.samples=3;cfg.depth=4;
    const auto far=Clock::time_point::max();
    int observations=0,tapeTransitions=0;
    for(int tc=0;tc<100;++tc){
        auto r=rules(5,4,tc%4);Game g(r);g.hand=tc%2?"ABAB":"BBAA";g.drawn=4;
        std::string future;for(int i=0;i<25;++i)future+=((i+tc)%3)?'A':'B';
        g.upcoming=future.substr(0,r->x);
        Generator gen(g,32);gen.run(far);require(!gen.top.empty(),"root candidates");
        std::vector<RankedMove> ranked;for(const auto& c:gen.top)ranked.push_back({c.move,double(c.move.score)});
        const State saved=g;
        DeepReport a,b;std::mt19937_64 ra(tc),rb(tc);
        Move ma=runDeep(g,gen.best,ranked,cfg,la,1e12,ra,far,a);
        Move mb=runDeep(g,gen.best,ranked,cfg,la,1e12,rb,far,b);
        require(sameMove(ma,mb)&&a.q==b.q&&a.work==b.work&&a.rounds==3,"repeatability and full fair rounds");
        require(g==saved&&checkedScore(g,ma)==ma.score&&a.invalid==0,"root isolation and legal action");
        require(a.evals==a.width*a.rounds,"every root has every accepted sample");
        if(tc%10==0){
            LaneModel lane(*r);LookaheadConfig strong=la;strong.lane=&lane;strong.ev.lanes=8;strong.ev.laneMin=2;
            DeepConfig policy=cfg;policy.policy=3;policy.depth=2;
            DeepReport strongRep;std::mt19937_64 strongRng(tc);
            Move strongMove=runDeep(g,gen.best,ranked,policy,strong,1e12,strongRng,far,strongRep);
            require(strongRep.rounds==policy.samples&&strongRep.invalid==0&&checkedScore(g,strongMove)==strongMove.score,"lane-aware continuation must complete legally");
            require(g==saved,"lane-aware continuation mutated the root");
            LaneModel freshLane(*r);strong.lane=&freshLane;DeepReport repeated;std::mt19937_64 again(tc);
            Move repeatedMove=runDeep(g,gen.best,ranked,policy,strong,1e12,again,far,repeated);
            require(sameMove(strongMove,repeatedMove)&&strongRep.q==repeated.q,"lane cache must not change continuation values");
        }
        DeepReport cut;std::mt19937_64 rc(tc);
        Move mc=runDeep(g,gen.best,ranked,cfg,la,1,rc,far,cut);
        require(cut.rounds==0&&cut.cut&&sameMove(mc,gen.best),"incomplete first round retains parent");
        DeepReport expired;std::mt19937_64 rd(tc);
        require(sameMove(runDeep(g,gen.best,ranked,cfg,la,1e12,rd,Clock::time_point::min(),expired),gen.best),"expired search returns parent");
        // A real environment's hidden suffix is not an input to runDeep. Project
        // two decks with the same visible prefix and opposite hidden suffixes.
        Game projected(g);std::string other=future;for(size_t i=r->x;i<other.size();++i)other[i]=other[i]=='A'?'B':'A';
        projected.upcoming=other.substr(0,r->x);
        DeepReport visibility;std::mt19937_64 rv(tc);
        Move mv=runDeep(projected,gen.best,ranked,cfg,la,1e12,rv,far,visibility);
        require(sameMove(ma,mv)&&a.q==visibility.q,"changing true hidden tail changes decision");
        if(tc%10==0){
            DeepConfig tree=cfg;tree.mode=2;tree.depth=3;tree.branch=4;tree.chance=2;
            DeepReport ta,tb;std::mt19937_64 tr1(tc),tr2(tc);
            Move t1=runDeep(g,gen.best,ranked,tree,la,1e12,tr1,far,ta);
            Move t2=runDeep(projected,gen.best,ranked,tree,la,1e12,tr2,far,tb);
            require(sameMove(t1,t2)&&ta.q==tb.q&&ta.rounds==tree.samples&&ta.invalid==0,"chance-tree visibility and fair rounds");
        }
        DeepConfig off=cfg;off.mode=0;DeepReport offReport;std::mt19937_64 offRng(tc),unchanged(tc);
        require(sameMove(runDeep(g,gen.best,ranked,off,la,1e12,offRng,far,offReport),gen.best)&&offRng()==unchanged(),"disabled deep search must preserve action and RNG");
        // Repeated steps exercise absolute indexing after different consumption,
        // preview overlap, partial final refills, and DISCARD transitions.
        for(int branch=0;branch<2;++branch){
            Game sim(g);DeepReport report;DeepContext ctx{cfg,la,far,1e12,report};
            const int origin=g.drawn+r->x;std::string tape=future.substr(r->x);
            for(int step=0;step<20&&!sim.terminal;++step){
                Generator now(sim);now.run(far);Action action=fallbackAction(sim,now.best);
                if(step%3==branch&&sim.drawn<25){action.kind=ActionKind::Discard;action.letters.clear();for(char ch:sim.hand)if(ch!='.')action.letters+=ch;}
                std::string expected=sim.hand;
                if(action.kind==ActionKind::Place){int rr=action.move.r,cc=action.move.c;for(char ch:action.move.word){if(sim.board[rr][cc]=='.')expected[expected.find(ch)]='.';rr+=action.move.dir;cc+=1-action.move.dir;}}
                else if(action.kind==ActionKind::Discard)for(char ch:action.letters)expected[expected.find(ch)]='.';
                int nextDrawn=sim.drawn;
                if(action.kind!=ActionKind::Stop)for(char& ch:expected)if(ch=='.'&&nextDrawn<25)ch=future[nextDrawn++-g.drawn];
                require(applyTape(sim,action,tape,origin,ctx),"tape application");
                require(sim.hand==expected&&sim.drawn==nextDrawn,"absolute stream refill mismatch");
                if(action.kind!=ActionKind::Stop){std::string preview;for(int i=0;i<r->x;++i)preview+=nextDrawn+i<25?future[nextDrawn+i-g.drawn]:'.';require(sim.upcoming==preview,"absolute stream preview mismatch");}
                ++tapeTransitions;
                // The policy sees only this projection, and its own independent RNG.
                if(!sim.terminal){Action p1,p2;DeepReport d1,d2;DeepContext c1{cfg,la,far,1e12,d1},c2{cfg,la,far,1e12,d2};
                    require(rolloutPolicy(sim,167,c1,p1)&&rolloutPolicy(sim,167,c2,p2),"visible policy");
                    require(p1.kind==p2.kind&&sameMove(p1.move,p2.move)&&p1.letters==p2.letters,"policy determinism");++observations;}
            }
        }
    }
    // Exact deterministic reference: no hidden letters, all actions retained.
    for(int tc=0;tc<100;++tc){
        auto r=rules(3,4,0);Game g(r);g.hand=tc%2?"AABB":"ABBA";g.drawn=9;
        if(tc%3){Generator gen(g,64);gen.run(far);require(transition(g,{ActionKind::Place,gen.top[tc%gen.top.size()].move,{}},"")==TransitionResult::Applied,"endgame prefix");}
        DeepConfig tree;tree.mode=2;tree.branch=64;tree.chance=2;tree.tail=0;
        DeepReport rep;DeepContext ctx{tree,la,far,1e12,rep};double actual=0;
        require(treeValue(g,2,167,ctx,actual),"complete exact tree");
        require(actual==exactEndgame(g,2),"chance tree differs from exhaustive deterministic reference");
    }
    std::cout<<"Passed: 100 repeatability/visibility/fairness cases, "<<tapeTransitions<<" absolute tape transitions, "<<observations<<" policy observations, 100 exhaustive tree comparisons.\n";
}
