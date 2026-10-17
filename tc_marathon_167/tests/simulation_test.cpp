#define MM167_TEST
#include "../src/WordGame.cpp"
#include <fstream>
#include <random>
#include <cstdlib>
using namespace mm167;
void require(bool b,const char* msg){if(!b){std::cerr<<msg<<'\n';std::exit(1);}}
std::string letters(std::mt19937& rng,int count){std::string s;while(count--)s+='A'+rng()%2;return s;}
std::string flat(const Game& g,bool bonus=false){std::string s;for(int r=0;r<g.rules->n;++r)for(int c=0;c<g.rules->n;++c)s+=bonus?g.rules->bonus[r][c]:g.board[r][c];return s;}
std::string token(const std::string& s){return s.empty()?"-":s;}
int clockCalls=0;
Clock::time_point fakeClock(){return ++clockCalls==1?Clock::time_point{}:Clock::time_point::max();}

int main(int argc,char** argv){
    std::ofstream fixture;if(argc>1)fixture.open(argv[1]);
    std::mt19937 rng(167);int discards=0,places=0,stops=0,undos=0;
    std::vector<std::string> words;
    for(int len=2;len<=4;++len)for(int mask=0;mask<(1<<len);++mask){std::string w;for(int j=0;j<len;++j)w+=(mask>>j&1)?'B':'A';words.push_back(w);}
    for(int tc=0;tc<1500;++tc){
        auto rules=std::make_shared<Rules>();rules->n=tc%2?10:5;rules->t=tc%3?5:9;rules->x=tc%2?5:0;rules->penalty=tc%6;
        rules->sr=rules->sc=rules->n/2;
        for(int r=0;r<rules->n;++r)for(int c=0;c<rules->n;++c)rules->bonus[r][c]=".dtDT"[rng()%5];
        rules->bonus[rules->sr][rules->sc]='S';for(const auto& w:words)rules->dict.add(w);
        Game g(rules);g.hand=letters(rng,rules->t);
        int remaining=tc%20;g.drawn=rules->n*rules->n-remaining;
        g.upcoming=letters(rng,std::min(remaining,rules->x));g.upcoming.resize(rules->x,'.');
        // Reach several legal board states before the tested action.
        for(int j=0;j<tc%4;++j){
            Generator gen(g,8);gen.run(Clock::now()+std::chrono::seconds(2));
            if(gen.top.empty() || g.terminal)break;
            Action a{ActionKind::Place,gen.top[rng()%gen.top.size()].move,{}};
            require(transition(g,a,letters(rng,100))==TransitionResult::Applied,"Prefix transition");
        }
        if(g.terminal){g.terminal=false;g.hand=letters(rng,rules->t);}
        // Sparse end racks are legal when the bag has run out.
        if(g.drawn==rules->n*rules->n && tc%3==0)g.hand.back()='.';
        Generator gen(g,16);gen.run(Clock::now()+std::chrono::seconds(2));
        Action a;
        if(tc%7==0){a.kind=ActionKind::Stop;++stops;}
        else if(tc%3==0 || gen.top.empty()){
            a.kind=ActionKind::Discard;
            for(char c:g.hand)if(c!='.' && (rng()%2 || a.letters.empty()))a.letters+=c;
            if(a.letters.empty()){a.kind=ActionKind::Stop;++stops;}else ++discards;
        }else{a.kind=ActionKind::Place;a.move=gen.top[rng()%gen.top.size()].move;++places;}
        Game before=g;std::string samples=letters(rng,100);
        require(before.rules.get()==g.rules.get() && &before.rules->dict==&g.rules->dict,"Dictionary copied");
        auto hash=g.hash();
        for(int j=0;j<8;++j){
            {StateGuard rollback(g);require(transition(g,a,samples)==TransitionResult::Applied,"Valid action rejected");}
            require(static_cast<State&>(g)==static_cast<State&>(before) && g.hash()==hash,"Rollback drift");++undos;
        }
        require(transition(g,a,samples,Clock::now()-std::chrono::seconds(1))==TransitionResult::Expired,"Expired transition");
        require(static_cast<State&>(g)==static_cast<State&>(before),"Expired transition changed state");
        require(transition(g,a,samples)==TransitionResult::Applied,"Transition commit failed");
        if(fixture){
            fixture<<rules->n<<' '<<rules->t<<' '<<rules->x<<' '<<rules->penalty<<' '<<words.size();
            for(const auto& w:words)fixture<<' '<<w;
            fixture<<' '<<flat(before,true)<<' '<<flat(before)<<' '<<before.hand<<' '<<token(before.upcoming)
                <<' '<<before.drawn<<' '<<before.placed<<' '<<before.score<<' '<<int(a.kind)
                <<' '<<a.move.r<<' '<<a.move.c<<' '<<a.move.dir<<' '<<token(a.move.word)<<' '<<token(a.letters)<<' '<<samples
                <<' '<<flat(g)<<' '<<g.hand<<' '<<token(g.upcoming)<<' '<<g.drawn<<' '<<g.placed<<' '<<g.score<<' '<<g.terminal<<'\n';
        }
    }
    // Unknown draws are explicit and atomic, including preview expansion.
    auto rules=std::make_shared<Rules>();rules->n=10;rules->t=5;rules->x=2;rules->penalty=3;
    Game g(rules);g.drawn=5;g.hand="AABBC";g.upcoming="DE";State before=g;
    Action discard{ActionKind::Discard,{},"AAB"};
    require(transition(g,discard,"FGH",Clock::time_point{}+std::chrono::seconds(1),fakeClock)==TransitionResult::Expired,
            "Cancellation after mutation");
    require(clockCalls==2 && static_cast<State&>(g)==before,"Mid-transition cancellation must restore all fields");
    require(transition(g,discard,"")==TransitionResult::NeedSamples && static_cast<State&>(g)==before,"Unknown refill must not mutate");
    require(transition(g,discard,"F")==TransitionResult::Applied,"Partial preview transition");
    require(g.hand=="DEFBC" && g.upcoming=="??" && g.drawn==8 && g.score==-3,"Partial preview representation");
    before=g;require(transition(g,{ActionKind::Discard,{},"Z"},"AB")==TransitionResult::Invalid && static_cast<State&>(g)==before,"Invalid discard atomicity");
    require(transition(g,{ActionKind::Discard,{},"DZ"},"AB")==TransitionResult::Invalid && static_cast<State&>(g)==before,"Partially consumed invalid discard rollback");
    require(transition(g,{ActionKind::Discard,{},"DE"},"GHIJKL")==TransitionResult::Applied,"Consecutive discard with unknown preview");
    require(g.hand=="GHFBC" && g.upcoming=="IJ" && g.drawn==10 && g.score==-6,"Consecutive discard sampling order");
    require(transition(g,{ActionKind::Discard,{},"G"},"K")==TransitionResult::Applied,"Consecutive discard with known prefix");
    require(g.hand=="IHFBC" && g.upcoming=="JK" && g.drawn==11 && g.score==-9,"Known prefix must precede sampled tail");
    g.hand="AA...";g.upcoming="DE";g.drawn=99;
    before=g;
    require(transition(g,{ActionKind::Discard,{},"A"},"")==TransitionResult::Applied,"Short bag refill");
    require(g.hand=="DA..." && g.upcoming==".." && g.drawn==100,"Short bag handling");
    g.hand="AA...";g.upcoming="..";
    require(transition(g,{ActionKind::Discard,{},"AA"},"")==TransitionResult::Applied && g.terminal,"Automatic exhausted termination");
    before=g;require(transition(g,{ActionKind::Stop,{},{}},"")==TransitionResult::Invalid && static_cast<State&>(g)==before,"Terminal state immutable");
    // P1-01 lookahead: reproducible choice, complete rounds, expired fallback.
    {
        auto rules2=std::make_shared<Rules>();rules2->n=5;rules2->t=5;rules2->x=0;rules2->penalty=3;rules2->sr=2;rules2->sc=2;
        for(const auto& w:words)rules2->dict.add(w);
        rules2->letterWeight['A'-'A']=7;rules2->letterWeight['B'-'A']=5;rules2->letterWeight['C'-'A']=2;
        Game h(rules2);h.hand="AABBC";
        Generator gen(h,8);gen.run(Clock::now()+std::chrono::seconds(2));
        require(!gen.top.empty(),"Lookahead test needs candidates");
        LookaheadConfig cfg;cfg.k=8;cfg.samples=2;cfg.gamma=0.8;
        LookaheadReport r1,r2,r3;
        std::mt19937_64 rngA(167),rngB(167);
        Move m1=runLookahead(h,gen,cfg,rngA,Clock::now()+std::chrono::seconds(5),r1);
        Move m2=runLookahead(h,gen,cfg,rngB,Clock::now()+std::chrono::seconds(5),r2);
        require(m1.r==m2.r&&m1.c==m2.c&&m1.dir==m2.dir&&m1.word==m2.word,"Same lookahead seed must reproduce the choice");
        require(r1.rounds>0&&r2.rounds==r1.rounds,"Lookahead must complete equal rounds with ample time");
        bool inTop=false;
        for(const auto& c:gen.top)if(c.move.word==m1.word&&c.move.r==m1.r&&c.move.c==m1.c&&c.move.dir==m1.dir)inTop=true;
        require(inTop||m1.word==gen.best.word,"Lookahead must return a retained candidate or the greedy move");
        Move m3=runLookahead(h,gen,cfg,rngA,Clock::time_point::min(),r3);
        require(r3.rounds==0&&m3.word==gen.best.word&&m3.r==gen.best.r&&m3.c==gen.best.c,"Expired lookahead must return the greedy move");
        require(h.hand=="AABBC"&&h.placed==0,"Lookahead must not mutate the root state");
    }
    // P1-02 lane model: every listed word is a legal placement with exactly the
    // listed score, and every generated placement filling a listed window is found.
    int laneWords=0,laneMoves=0,endgames=0,schedEqual=0,schedNarrow=0,schedRaced=0,schedPartial=0;
    for(int tc=0;tc<160;++tc){
        auto rules3=std::make_shared<Rules>();rules3->n=tc%2?8:6;rules3->t=tc%3?5:7;rules3->x=tc%4;rules3->penalty=1;
        rules3->sr=rules3->sc=rules3->n/2;
        for(int r=0;r<rules3->n;++r)for(int c=0;c<rules3->n;++c)rules3->bonus[r][c]=".dtDTDT"[rng()%7];
        rules3->bonus[rules3->sr][rules3->sc]='S';
        for(const auto& w:words){rules3->dict.add(w);rules3->words.push_back(w);for(char ch:w)++rules3->letterWeight[ch-'A'];}
        Game h(rules3);h.hand=letters(rng,rules3->t);h.drawn=rules3->t;
        h.upcoming=letters(rng,rules3->x);
        for(int j=0;j<1+tc%5&&!h.terminal;++j){
            Generator step(h,8);step.run(Clock::now()+std::chrono::seconds(2));
            if(step.top.empty())break;
            require(transition(h,{ActionKind::Place,step.top[rng()%step.top.size()].move,{}},letters(rng,100))==TransitionResult::Applied,"Lane prefix transition");
        }
        if(h.terminal)continue;
        const bool endgame=tc%4==0;
        if(endgame){h.drawn=rules3->n*rules3->n;h.upcoming.assign(rules3->x,'.');}
        Generator gen(h,64);gen.run(Clock::now()+std::chrono::seconds(2));
        if(gen.top.empty())continue;
        LaneModel lanes(*rules3);EvalConfig ev;ev.laneMin=2;ev.lanes=16;ev.pool=64;ev.alpha=5;
        lanes.prepare(h,gen,ev);
        for(auto& w:lanes.root){
            w.lane=lanes.resolve(h,gen,w);
            int a=w.first;
            while(a>0&&h.board[w.d?a-1:w.line][w.d?w.line:a-1]!='.')--a;
            for(const auto& word:w.lane->words){
                Game probe(h);probe.hand.clear();
                for(int q=0;q<word.tiles;++q)probe.hand+=static_cast<char>('A'+word.need[q]);
                Move m{w.d?a:w.line,w.d?w.line:a,w.d,word.tiles,word.score,word.text};
                require(checkedScore(probe,m)==word.score,"Lane word must be legal with the listed score");
                ++laneWords;
            }
        }
        for(const Move& m:gen.all){
            int first=-1,last=-1,product=1,r=m.r,c=m.c;
            for(size_t j=0;j<m.word.size();++j){
                if(h.board[r][c]=='.'){int k=m.dir?r:c;if(first<0)first=k;last=k;product*=h.wordMul(r,c);}
                r+=m.dir;c+=1-m.dir;
            }
            if(product<ev.laneMin)continue;
            for(const auto& w:lanes.root)if(w.d==m.dir&&w.line==(m.dir?m.c:m.r)&&w.first==first&&w.last==last){
                bool found=false;
                for(const auto& word:w.lane->words)if(word.text==m.word){found=true;require(word.score==m.score,"Lane score differs from generator");}
                require(found,"Generated placement missing from its lane");
                ++laneMoves;
            }
        }
        // Same seed reproduces the pre-selected choice; an empty bag disables P1-02.
        LookaheadConfig on;on.k=8;on.samples=2;on.ev=ev;on.lane=&lanes;
        LookaheadConfig off;off.k=8;off.samples=2;
        Generator narrow(h,8);narrow.run(Clock::now()+std::chrono::seconds(2));
        LookaheadReport ra,rb,rc;
        std::mt19937_64 r1(tc),r2(tc),r3(tc);
        Move ma=runLookahead(h,gen,on,r1,Clock::now()+std::chrono::seconds(5),ra);
        Move mb=runLookahead(h,gen,on,r2,Clock::now()+std::chrono::seconds(5),rb);
        Move mc=runLookahead(h,narrow,off,r3,Clock::now()+std::chrono::seconds(5),rc);
        require(ma.r==mb.r&&ma.c==mb.c&&ma.dir==mb.dir&&ma.word==mb.word,"Pre-selection must be reproducible");
        require(ra.width<=on.k&&checkedScore(h,ma)==ma.score,"Pre-selected choice must be a legal retained candidate");
        if(endgame){
            require(ma.r==mc.r&&ma.c==mc.c&&ma.dir==mc.dir&&ma.word==mc.word,"Empty bag must reproduce the parent choice");
            ++endgames;
        }
        // P2-03 scheduled evaluation. Same width, every sample for every candidate
        // and ample effort: exactly the fixed-width choice and value.
        const auto far=Clock::now()+std::chrono::seconds(5);
        SchedConfig same;same.kMin=same.kMax=on.k;same.keep=1;
        LookaheadReport sa,sb,sz,sw,sh,sx;
        std::mt19937_64 q1(tc),q2(tc),q3(tc),q4(tc),q5(tc),q6(tc);
        Move na=runScheduled(h,gen,on,same,1e18,q1,far,sa);
        require(na.r==ma.r&&na.c==ma.c&&na.dir==ma.dir&&na.word==ma.word&&sa.chosenQ==ra.chosenQ,"Scheduled evaluation at fixed width must equal the fixed-width choice");
        // A fully visible refill needs no second sample, so only then may the round count differ.
        require(sa.width==ra.width&&sa.evals==ra.evals&&(sa.rounds==ra.rounds||sa.deterministic==sa.width)&&!sa.hardCut,"Scheduled evaluation at fixed width must do the same work");
        require(q1()==r1(),"Scheduled evaluation must consume the same random stream");
        ++schedEqual;
        // Effort is a pure function of the position: identical count on repetition.
        Move nb=runScheduled(h,gen,on,same,1e18,q2,far,sb);
        require(nb.word==na.word&&sb.work==sa.work&&sa.work>0,"Effort count must be reproducible");
        // No allowance: exactly kMin candidates get the one mandatory common sample.
        SchedConfig racing;racing.kMin=4;racing.kMax=64;racing.keep=0.5;racing.keepMin=2;
        Move nz=runScheduled(h,gen,on,racing,0,q3,far,sz);
        require(sz.width==std::min(4,sz.pool)&&sz.rounds==1,"Zero allowance must evaluate exactly the minimum width once");
        require(checkedScore(h,nz)==nz.score,"Minimum-width choice must be legal");
        if(sz.pool>4)++schedNarrow;
        // Ample allowance: the whole pool is sampled, later samples go to a shrinking share.
        Move nw=runScheduled(h,gen,on,racing,1e18,q4,far,sw);
        require(sw.width==std::min(64,sw.pool)&&sw.width>=sz.width&&checkedScore(h,nw)==nw.score,"Ample allowance must widen to the pool");
        if(sw.rounds>1){
            int bound=std::max(2,(sw.width+(1<<(sw.rounds-1))-1)>>(sw.rounds-1));
            require(sw.finalists<=bound&&sw.finalists>=1&&sw.chosenSamples<=sw.rounds,"Finalists must follow the keep schedule");
            require(sw.evals<sw.width*sw.rounds||sw.width<=2||sw.deterministic==sw.width,"Racing must spend fewer evaluations than full rounds");
            ++schedRaced;
        }
        // Expired safety deadline: nothing evaluated, the complete greedy move is returned.
        Move nh=runScheduled(h,gen,on,racing,1e18,q5,Clock::time_point::min(),sh);
        require(sh.rounds==0&&sh.width==0&&sh.hardCut&&nh.word==gen.best.word&&nh.r==gen.best.r&&nh.c==gen.best.c&&nh.dir==gen.best.dir,"Expired scheduled evaluation must return the greedy move");
        // The first-round share is honoured: a tiny share cannot widen beyond kMin.
        SchedConfig tight=racing;tight.r0=1e-9;
        runScheduled(h,gen,on,tight,1e9,q6,far,sx);
        require(sx.width==std::min(4,sx.pool),"First-round share must bound the width");
        // Pre-selection is anytime: a small share keys only a prefix of the pool (never
        // fewer than kMin), the choice stays legal, and the same allowance repeats exactly.
        SchedConfig lean=racing;lean.pre=1e-9;
        LookaheadReport sp,sq;std::mt19937_64 q7(tc),q8(tc);
        Move np=runScheduled(h,gen,on,lean,1e9,q7,far,sp);
        Move nq=runScheduled(h,gen,on,lean,1e9,q8,far,sq);
        if(!endgame&&sp.pool>4){
            require(sp.keyed>=4&&sp.keyed<=sp.pool&&sw.keyed==sw.pool,"Pre-selection must key between kMin and the whole pool");
            if(sp.keyed<sp.pool)++schedPartial;
        }
        require(checkedScore(h,np)==np.score&&np.word==nq.word&&np.r==nq.r&&np.c==nq.c&&sp.keyed==sq.keyed&&sp.work==sq.work,"Partial pre-selection must be legal and reproducible");
        require(h.placed>0&&checkedScore(h,gen.best)==gen.best.score,"Scheduled evaluation must not mutate the root state");
    }
    require(schedEqual>=100&&schedNarrow>=50&&schedRaced>=50&&schedPartial>=20,"Scheduler tests need coverage");
    require(laneWords>1000&&laneMoves>100&&endgames>=30,"Lane tests need coverage");
    // At least 100 exhausted-bag endgames: P1-02 must not change the parent's decision.
    for(int tc=0;endgames<140&&tc<2000;++tc){
        auto rules4=std::make_shared<Rules>();rules4->n=7;rules4->t=5+tc%3;rules4->x=tc%3;rules4->penalty=2;
        rules4->sr=rules4->sc=3;
        for(int r=0;r<7;++r)for(int c=0;c<7;++c)rules4->bonus[r][c]=".dtDT"[rng()%5];
        rules4->bonus[3][3]='S';
        for(const auto& w:words){rules4->dict.add(w);rules4->words.push_back(w);for(char ch:w)++rules4->letterWeight[ch-'A'];}
        Game h(rules4);h.hand=letters(rng,rules4->t);h.drawn=rules4->t;h.upcoming=letters(rng,rules4->x);
        for(int j=0;j<2+tc%4&&!h.terminal;++j){
            Generator step(h,8);step.run(Clock::now()+std::chrono::seconds(2));
            if(step.top.empty())break;
            require(transition(h,{ActionKind::Place,step.top[rng()%step.top.size()].move,{}},letters(rng,100))==TransitionResult::Applied,"Endgame prefix transition");
        }
        if(h.terminal)continue;
        h.drawn=49;h.upcoming.assign(rules4->x,'.');
        Generator wide(h,64),narrow(h,8);
        wide.run(Clock::now()+std::chrono::seconds(2));narrow.run(Clock::now()+std::chrono::seconds(2));
        if(wide.top.empty())continue;
        LaneModel lanes(*rules4);
        LookaheadConfig on;on.k=8;on.samples=2;on.ev.pool=64;on.ev.alpha=5;on.ev.laneMin=2;on.lane=&lanes;
        LookaheadConfig off;off.k=8;off.samples=2;
        LookaheadReport ra,rb;std::mt19937_64 r1(tc),r2(tc);
        Move ma=runLookahead(h,wide,on,r1,Clock::now()+std::chrono::seconds(5),ra);
        Move mb=runLookahead(h,narrow,off,r2,Clock::now()+std::chrono::seconds(5),rb);
        require(ma.r==mb.r&&ma.c==mb.c&&ma.dir==mb.dir&&ma.word==mb.word&&ra.chosenQ==rb.chosenQ,"Endgame choice must equal the parent");
        ++endgames;
    }
    require(endgames>=100,"Need at least 100 endgame comparisons");
    std::cout<<"Passed: scheduler "<<schedEqual<<" fixed-width equivalences, "<<schedNarrow<<" minimum-width, "<<schedRaced<<" racing and "<<schedPartial<<" partial pre-selection cases.\n";
    std::cout<<"Passed: lane model "<<laneWords<<" words, "<<laneMoves<<" generated placements matched, "<<endgames<<" empty-bag endgames equal to parent.\n";
    std::cout<<"Passed: 1500 transition fixtures; PLACE="<<places<<" DISCARD="<<discards<<" STOP="<<stops
             <<"; "<<undos<<" apply/undo, atomic invalid/unknown/expired cases.\n";
}
