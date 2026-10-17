#define MM167_TEST
#include "../src/WordGame.cpp"
#include <cstdlib>
#include <map>
#include <random>
#include <set>
#include <sstream>

using namespace mm167;
void check(bool ok,const char* message) { if (!ok) { std::cerr<<message<<'\n'; std::exit(1); } }
std::string key(const Move& m) { return std::to_string(m.r)+","+std::to_string(m.c)+","+std::to_string(m.dir)+","+m.word; }

// Independent dictionary-enumeration oracle: materialize each candidate board,
// then scan complete words. This intentionally does not use Trie/cross masks.
int64_t evaluate(const Game& g,const Move& m,const std::set<std::string>& dictionary) {
    int dr=m.dir,dc=1-dr,r=m.r,c=m.c;
    if (g.occupied(r-dr,c-dc)) return -1;
    auto board=g.board;
    std::array<int,26> hand{};
    for(char ch:g.hand) if(ch!='.') ++hand[ch-'A'];
    std::vector<std::pair<int,int>> fresh;
    bool anchor=false;
    int64_t sum=0,mul=1;
    for(char ch:m.word) {
        if(!g.inside(r,c)) return -1;
        if(g.occupied(r,c)) {
            if(board[r][c]!=ch) return -1;
            anchor=true; sum+=VALUES[ch-'A'];
        } else {
            if(--hand[ch-'A']<0) return -1;
            board[r][c]=ch; fresh.emplace_back(r,c);
            if(!g.placed && r==g.rules->sr && c==g.rules->sc) anchor=true;
            int lm=g.rules->bonus[r][c]=='d'?2:g.rules->bonus[r][c]=='t'?3:1;
            int wm=g.rules->bonus[r][c]=='D'?2:g.rules->bonus[r][c]=='T'?3:1;
            sum+=VALUES[ch-'A']*lm; mul*=wm;
        }
        r+=dr; c+=dc;
    }
    if(fresh.empty() || !anchor || g.occupied(r,c)) return -1;
    int64_t score=sum*mul;
    for(auto [rr,cc]:fresh) {
        r=rr;c=cc;
        while(g.inside(r-dc,c-dr) && board[r-dc][c-dr]!='.') { r-=dc; c-=dr; }
        std::string word; int side=0;
        while(g.inside(r,c) && board[r][c]!='.') {
            char ch=board[r][c]; word+=ch;
            int lm=1;
            if(r==rr && c==cc) lm=g.rules->bonus[r][c]=='d'?2:g.rules->bonus[r][c]=='t'?3:1;
            side+=VALUES[ch-'A']*lm; r+=dc;c+=dr;
        }
        if(word.size()>1) {
            if(!dictionary.count(word)) return -1;
            int wm=g.rules->bonus[rr][cc]=='D'?2:g.rules->bonus[rr][cc]=='T'?3:1;
            score+=side*wm;
        }
    }
    return score;
}

void compare(const Game& g,const std::set<std::string>& words) {
    std::map<std::string,int64_t> expected,actual;
    for(const auto& word:words) for(int d=0;d<2;++d) for(int r=0;r<g.rules->n;++r) for(int c=0;c<g.rules->n;++c) {
        Move m{r,c,d,0,0,word};
        auto value=evaluate(g,m,words);
        if(value>=0) expected[key(m)]=value;
    }
    Generator gen(g,16); auto best=gen.run(Clock::now()+std::chrono::seconds(5));
    check(!gen.stopped,"Oracle test unexpectedly exhausted time");
    for(const Move& m:gen.all) {
        actual[key(m)]=m.score;
        check(checkedScore(g,m)==m.score,"Simulation scoring disagrees with oracle");
    }
    check(expected==actual,"Candidate set or score disagrees with independent oracle");
    int64_t maxScore=-1; for(auto& kv:expected) maxScore=std::max(maxScore,kv.second);
    check(best.score==maxScore,"Greedy best score mismatch");
    std::map<std::string,Generator::Candidate> unique;
    uint64_t order=0;
    for(const auto& m:gen.all) {
        auto board=g.board;auto hand=g.hand;
        for(size_t j=0;j<m.word.size();++j) {
            int r=m.r+m.dir*j,c=m.c+(1-m.dir)*j;
            if(board[r][c]=='.'){board[r][c]=m.word[j];hand[hand.find(m.word[j])]='.';}
        }
        std::string state;
        for(auto& row:board)state.append(row.begin(),row.end());
        state+=hand;
        Generator::Candidate candidate{m,state,++order};
        auto it=unique.find(state);
        if(it==unique.end() || Generator::better(candidate,it->second))unique[state]=candidate;
    }
    std::vector<Generator::Candidate> ranked;
    for(auto& entry:unique)ranked.push_back(entry.second);
    std::sort(ranked.begin(),ranked.end(),Generator::better);
    check(gen.top.size()==std::min<size_t>(16,ranked.size()),"Candidate capacity/dedup count");
    for(size_t i=0;i<gen.top.size();++i)check(key(gen.top[i].move)==key(ranked[i].move),"Top K or tie ordering mismatch");
    if(!gen.top.empty())check(key(gen.top.front().move)==key(best),"Top candidate must preserve greedy choice");
}

int main() {
    std::set<std::string> words={"AA","AB","BA","BB","ABA","BAB","AAB","ABB","ABBA","BAAB"};
    auto rules=std::make_shared<Rules>();rules->n=5;rules->t=5;rules->sr=2;rules->sc=2;Game g(rules);g.hand="AABB.";
    for(auto& row:rules->bonus) row.fill('.');
    rules->bonus[2][2]='S';
    for(const auto& word:words) rules->dict.add(word);
    compare(g,words);
    std::mt19937 rng(167);
    for(int test=0;test<1200;++test) {
        auto br=std::make_shared<Rules>(*rules); Game b=g;b.rules=br;
        for(int r=0;r<b.rules->n;++r) for(int c=0;c<b.rules->n;++c) {
            br->bonus[r][c]=".dtDT"[rng()%5];
            if(rng()%4==0) b.board[r][c]="AB"[rng()%2];
        }
        b.placed=1;
        b.hand=test%2?"AAAB.":"ABBB.";
        compare(b,words);
    }
    // Explicit multiplier stacking, and reusing an old bonus only at base value.
    auto br=std::make_shared<Rules>(*rules);Game b=g;b.rules=br;b.hand="AB...";br->sr=0;br->sc=0;br->bonus[0][0]='D';br->bonus[0][1]='T';
    Move m{0,0,0,2,0,"AB"};
    check(evaluate(b,m,words)==24,"Stacked word bonuses");
    b.board[0][0]='A';b.placed=1;b.hand="B....";
    check(evaluate(b,m,words)==12,"Old bonus must not reactivate");
    compare(b,words);
    Generator expired(g);auto none=expired.run(Clock::now()-std::chrono::milliseconds(1));
    check(none.score<0 && expired.stopped,"Expired search must stop without a partial move");
    auto again=expired.run(Clock::now()+std::chrono::seconds(5));
    Generator fresh(g);auto once=fresh.run(Clock::now()+std::chrono::seconds(5));
    check(!expired.stopped && key(again)==key(once) && again.score==once.score,"Generator reuse leaked prior state");
    std::cout<<"Passed: 1202 candidate-set/score oracle cases, multipliers and deadline.\n";
}
