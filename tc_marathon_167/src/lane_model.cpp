#include "lane_model.h"

#include <algorithm>
#include <chrono>

#include "common.h"
#include "generator.h"

namespace mm167 {
double LaneModel::refillProbability(uint64_t key,int draws) {
    if(!key)return 1;
    if(draws<=0)return 0;
    const uint64_t id=(key<<4)|uint64_t(draws);
    auto found=refillMemo.find(id);if(found!=refillMemo.end())return found->second;
    int letters[3]={},need[3]={},stride[3]={},kinds=0,missing=0;
    while(key){
        const int letter=int(key&31)-1;key>>=5;++missing;
        int i=0;while(i<kinds&&letters[i]!=letter)++i;
        if(i==kinds){letters[kinds]=letter;++kinds;}
        ++need[i];
    }
    if(missing>draws)return 0;
    int states=1;double other=1;
    for(int i=0;i<kinds;++i){stride[i]=states;states*=need[i]+1;other-=freq[letters[i]];}
    scanWords+=uint64_t(draws)*states*(kinds+1); // charge cache-miss DP work
    double dp[8]={1},next[8]={};
    for(int d=0;d<draws;++d){
        std::fill(next,next+states,0);
        for(int state=0;state<states;++state){
            next[state]+=dp[state]*std::max(0.0,other);
            for(int i=0;i<kinds;++i){
                const int count=state/stride[i]%(need[i]+1);
                next[state+(count<need[i]?stride[i]:0)]+=dp[state]*freq[letters[i]];
            }
        }
        std::copy(next,next+states,dp);
    }
    return refillMemo[id]=std::clamp(dp[states-1],0.0,1.0);
}
LaneModel::LaneModel(const Rules& rules) {
    uint64_t total=0;
    for(uint64_t w:rules.letterWeight)total+=w;
    for(int k=0;k<26;++k)freq[k]=total?double(rules.letterWeight[k])/double(total):1.0/26;
    for(const auto& w:rules.words)if(w.size()>=2&&w.size()<=20)flat[w.size()]+=w;
    for(int len=2;len<=20;++len){
        const int count=int(flat[len].size())/len;
        start[len].assign(len*26+1,0);
        for(int i=0;i<count;++i)for(int q=0;q<len;++q)++start[len][q*26+(flat[len][i*len+q]-'A')+1];
        for(int k=0;k<len*26;++k)start[len][k+1]+=start[len][k];
        ids[len].resize(start[len][len*26]);
        std::vector<int> fill(start[len].begin(),start[len].end()-1);
        for(int i=0;i<count;++i)for(int q=0;q<len;++q)ids[len][fill[q*26+(flat[len][i*len+q]-'A')]++]=i;
    }
}
const LaneModel::Lane& LaneModel::lane(uint64_t signature,const Cell* cells,int len,int product) {
    auto found=cache.find(signature);
    if(found!=cache.end())return found->second;
    Lane& out=cache[signature];
    ++builds;auto t0=Clock::now();
    struct Timer { double& ms; Clock::time_point t; ~Timer(){ms+=std::chrono::duration<double,std::milli>(Clock::now()-t).count();} } timer{buildMs,t0};
    int pivot=-1,pivotSize=0;
    for(int q=0;q<len;++q)if(cells[q].letter){
        int key=q*26+(cells[q].letter-'A'),size=start[len][key+1]-start[len][key];
        if(pivot<0||size<pivotSize){pivot=q;pivotSize=size;}
    }
    if(pivot<0)return out;
    buildWords+=uint64_t(pivotSize);
    const int key=pivot*26+(cells[pivot].letter-'A');
    const char* base=flat[len].data();
    for(int at=start[len][key];at<start[len][key+1];++at){
        const char* w=base+size_t(ids[len][at])*len;
        Word word{};int64_t sum=0,extra=0;bool ok=true;
        for(int q=0;q<len&&ok;++q){
            const Cell& cell=cells[q];int k=w[q]-'A';
            if(cell.letter){ok=cell.letter==w[q];sum+=VALUES[k];continue;}
            if(!(cell.mask>>k&1)){ok=false;break;}
            int value=VALUES[k]*cell.letterMul;
            sum+=value;if(cell.cross)extra+=int64_t(cell.oldScore+value)*cell.wordMul;
            word.need[word.tiles++]=static_cast<uint8_t>(k);word.set|=1u<<k;
        }
        if(!ok)continue;
        std::sort(word.need,word.need+word.tiles);
#ifdef MM167_TEST
        word.text.assign(w,len);
#endif
        word.score=static_cast<int32_t>(std::min<int64_t>(sum*product+extra,INT32_MAX));
        out.words.push_back(word);out.sets.push_back(word.set);
    }
    cachedWords+=out.words.size();
    // O-01A: score-descending order lets target scans stop early (value<=score).
    out.byScore.resize(out.words.size());
    for(size_t i=0;i<out.byScore.size();++i)out.byScore[i]=int(i);
    std::stable_sort(out.byScore.begin(),out.byScore.end(),[&](int a,int b){return out.words[a].score>out.words[b].score;});
    return out;
}
bool LaneModel::order(const Window& a,const Window& b) {
    if(a.product!=b.product)return a.product>b.product;
    if(a.tiles!=b.tiles)return a.tiles<b.tiles;
    if(a.d!=b.d)return a.d<b.d;
    if(a.line!=b.line)return a.line<b.line;
    return a.first<b.first;
}
void LaneModel::lineWindows(const Game& g,const Generator& gen,const EvalConfig& ev,int d,int line,std::vector<Window>& out) {
    const int n=g.rules->n,maxTiles=std::min(g.rules->t,9);
    int pos[20],wm[20],count=0;bool dead[20];
    for(int k=0;k<n;++k){
        int r=d?k:line,c=d?line:k;
        if(g.board[r][c]!='.')continue;
        pos[count]=k;wm[count]=g.wordMul(r,c);dead[count]=gen.cross[d][r][c].mask==0;++count;
    }
    if(count==n)return;
    auto filled=[&](int k){return k>=0&&k<n&&g.board[d?k:line][d?line:k]!='.';};
    for(int j=0;j<count;++j){
        int product=1;
        for(int i=j;i>=0&&j-i<maxTiles&&!dead[i];--i){
            product*=wm[i];
            if(product<ev.laneMin)continue;
            const int tiles=j-i+1;
            if(pos[j]-pos[i]+1>tiles||filled(pos[i]-1)||filled(pos[j]+1))
                out.push_back({product,tiles,d,line,pos[i],pos[j],nullptr});
        }
    }
}
const LaneModel::Lane* LaneModel::resolve(const Game& g,const Generator& gen,const Window& w) {
    const int n=g.rules->n;
    int a=w.first,b=w.last;
    auto at=[&](int k)->char{return g.board[w.d?k:w.line][w.d?w.line:k];};
    while(a>0&&at(a-1)!='.')--a;
    while(b+1<n&&at(b+1)!='.')++b;
    const int len=b-a+1;
    Cell cells[20];uint64_t h=1469598103934665603ULL;
    auto mix=[&](uint64_t v){h^=v;h*=1099511628211ULL;};
    mix(w.d);mix(w.line);mix(a);mix(len);
    for(int q=0;q<len;++q){
        int r=w.d?a+q:w.line,c=w.d?w.line:a+q;
        Cell& cell=cells[q];
        if(g.board[r][c]!='.'){cell=Cell{g.board[r][c],0,1,1,0,false};mix(uint64_t(cell.letter));continue;}
        const auto& cr=gen.cross[w.d][r][c];
        cell=Cell{0,cr.mask,g.letterMul(r,c),g.wordMul(r,c),cr.oldScore,cr.hasWord};
        mix(0x100000000ULL|cr.mask);mix(uint64_t(cr.oldScore)*2+cr.hasWord);
    }
    return &lane(h,cells,len,w.product);
}
void LaneModel::prepare(const Game& g,const Generator& gen,const EvalConfig& ev) {
    root.clear();memo.clear();
    if(cachedWords>3000000){cache.clear();cachedWords=0;}
    if(!g.placed)return;
    for(int d=0;d<2;++d)for(int line=0;line<g.rules->n;++line)lineWindows(g,gen,ev,d,line,root);
    std::sort(root.begin(),root.end(),order);
}
const LaneModel::Summary& LaneModel::summary(const Lane* lane,const int* kept,uint32_t keptSet,uint64_t keptKey,int draws) {
    uint64_t id=(keptKey^(reinterpret_cast<uintptr_t>(lane)*0x9E3779B97F4A7C15ULL))+uint64_t(draws);
    auto found=memo.find(id);
    if(found!=memo.end())return found->second;
    Summary& out=memo[id];
    ++scans;scanWords+=lane->words.size();auto t0=Clock::now();
    double perm[10];perm[0]=1;
    for(int i=1;i<10;++i)perm[i]=i<=draws?perm[i-1]*(draws-i+1):0;
    events.clear();
    const uint32_t* sets=lane->sets.data();
    const uint32_t lacking=~keptSet;
    // Ways missing more than three tiles are too unlikely to matter.
    const int reach=std::min(draws,3);
    for(size_t at=0,size=lane->words.size();at<size;++at){
        if(__builtin_popcount(sets[at]&lacking)>reach)continue;
        const Word& word=lane->words[at];
        int missing=0;double p=1;uint64_t key=0;
        for(int q=0;q<word.tiles;){
            int k=word.need[q],have=0;
            while(q<word.tiles&&word.need[q]==k){++q;++have;}
            int lack=have-kept[k];
            for(int u=1;u<=lack;++u){p*=freq[k]/u;key=key<<5|uint64_t(k+1);}
            if(lack>0)missing+=lack;
        }
        if(missing>reach)continue;
        events.push_back({key,word.score,exactRefill?refillProbability(key,draws):std::min(1.0,p*perm[missing])});
    }
    // All words with the same missing multiset happen on the same refill.
    // Keep only the best score before the bounded summary, otherwise eight
    // spellings of one event can crowd out every alternative refill.
    if(fixEvents){
        std::sort(events.begin(),events.end(),[](const Event& a,const Event& b){
            return a.key!=b.key?a.key<b.key:a.score>b.score;});
        events.erase(std::unique(events.begin(),events.end(),[](const Event& a,const Event& b){
            return a.key==b.key;}),events.end());
    }
    std::sort(events.begin(),events.end(),[](const Event& a,const Event& b){
        return a.score!=b.score?a.score>b.score:a.key<b.key;});
    size_t end=0;bool complete=false;
    for(;end<events.size()&&!complete;++end)complete=events[end].key==0;
    if(complete)out.e[out.count++]=events[end-1];
    size_t incomplete=end-(complete?1:0);
    if(incomplete>8){
        std::partial_sort(events.begin(),events.begin()+8,events.begin()+incomplete,[](const Event& a,const Event& b){
            double va=a.p*a.score,vb=b.p*b.score;
            if(va!=vb)return va>vb;
            return a.score!=b.score?a.score>b.score:a.key<b.key;});
        incomplete=8;
    }
    for(size_t i=0;i<incomplete;++i)out.e[out.count++]=events[i];
    scanMs+=std::chrono::duration<double,std::milli>(Clock::now()-t0).count();
    return out;
}
double LaneModel::moveValue(const Game& rootGame,const Game& after,const Generator& gen,const EvalConfig& ev,
                            const Move& m,const int* kept,int draws) {
    bool touched[2][20]={};
    touched[m.dir][m.dir?m.c:m.r]=true;
    int r=m.r,c=m.c;
    for(size_t j=0;j<m.word.size();++j){
        if(rootGame.board[r][c]=='.')touched[1-m.dir][m.dir?r:c]=true;
        r+=m.dir;c+=1-m.dir;
    }
    work.clear();
    int kept0=0;
    for(size_t i=0;i<root.size()&&kept0<ev.lanes;++i)if(!touched[root[i].d][root[i].line]){
        // Untouched lines keep their root pattern, so the list is shared.
        if(!root[i].lane)root[i].lane=resolve(after,gen,root[i]);
        work.push_back(root[i]);++kept0;
    }
    for(int d=0;d<2;++d)for(int line=0;line<after.rules->n;++line)if(touched[d][line])lineWindows(after,gen,ev,d,line,work);
    windowsSeen+=work.size();
    if(work.empty())return 0;
    if(int(work.size())>ev.lanes){
        std::partial_sort(work.begin(),work.begin()+ev.lanes,work.end(),order);
        work.resize(ev.lanes);
    }
    if(draws<0)draws=0;
    uint32_t keptSet=0;uint64_t keptKey=1469598103934665603ULL;
    for(int k=0;k<26;++k){if(kept[k])keptSet|=1u<<k;keptKey=(keptKey^uint64_t(kept[k]+1))*1099511628211ULL;}
    // A complete word is a sure score; only better incomplete ways add to it.
    merged.clear();
    const Summary* sums[64];int count=0;int32_t sure=0;
    for(Window& w:work){
        if(!w.lane)w.lane=resolve(after,gen,w);
        const Summary& sum=summary(w.lane,kept,keptSet,keptKey,draws);
        if(sum.count&&sum.e[0].key==0&&sum.e[0].score>sure)sure=sum.e[0].score;
        sums[count++]=&sum;
    }
    for(int j=0;j<count;++j)for(int i=0;i<sums[j]->count;++i)
        if(sums[j]->e[i].key&&sums[j]->e[i].score>sure)merged.push_back(sums[j]->e[i]);
    std::sort(merged.begin(),merged.end(),[](const Event& a,const Event& b){
        return a.score!=b.score?a.score>b.score:a.key<b.key;});
    double none=1,total=0;
    for(size_t i=0;i<merged.size()&&none>1e-6;++i){
        const Event& e=merged[i];
        bool repeated=false;
        for(size_t j=0;j<i&&!repeated;++j)repeated=merged[j].key==e.key;
        if(repeated)continue;
        total+=none*e.p*e.score;none*=1-e.p;
    }
    return total+none*sure;
}
}
