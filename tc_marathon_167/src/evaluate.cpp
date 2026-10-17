#include "evaluate.h"

#include <algorithm>
#include <cmath>

namespace mm167 {
double boardPotential(const Game& g,const Generator& gen,const EvalConfig& ev) {
    const Rules& rules=*g.rules;
    const int n=rules.n,maxTiles=std::min(rules.t,9);
    double freq[26];uint64_t total=0;
    for(uint64_t w:rules.letterWeight)total+=w;
    for(int k=0;k<26;++k)freq[k]=total?double(rules.letterWeight[k])/double(total):1.0/26;
    double rhoPow[10];rhoPow[0]=1;
    for(int i=1;i<10;++i)rhoPow[i]=rhoPow[i-1]*ev.rho;
    double value=0;
    for(int d=0;d<2;++d)for(int line=0;line<n;++line){
        int pos[20],wm[20],count=0;double fit[20];
        for(int k=0;k<n;++k){
            int r=d?k:line,c=d?line:k;
            if(g.board[r][c]!='.')continue;
            const auto& cr=gen.cross[d][r][c];
            double f=1;
            if(cr.hasWord){
                if(!cr.mask)f=0;
                else if(ev.tau>0){
                    double mass=0;
                    for(uint32_t b=cr.mask;b;b&=b-1)mass+=freq[__builtin_ctz(b)];
                    f=std::pow(mass,ev.tau);
                }
            }
            pos[count]=k;wm[count]=g.wordMul(r,c);fit[count]=f;++count;
        }
        const bool startLine=!g.placed&&line==(d?rules.sc:rules.sr);
        const int startPos=d?rules.sr:rules.sc;
        auto filled=[&](int k){return k>=0&&k<n&&g.board[d?k:line][d?line:k]!='.';};
        double best[21];best[0]=0;
        for(int j=0;j<count;++j){
            best[j+1]=best[j];
            double product=1,sum=0,f=1;
            for(int i=j;i>=0&&j-i<maxTiles;--i){
                f*=fit[i];if(f<=0)break;
                product*=wm[i];sum+=wm[i]-1;
                if(product<=1)continue;
                const int tiles=j-i+1;
                bool anchored=g.placed?(pos[j]-pos[i]+1>tiles||filled(pos[i]-1)||filled(pos[j]+1))
                                      :(startLine&&pos[i]<=startPos&&startPos<=pos[j]);
                double phi=(ev.linear?sum:product-1)*rhoPow[tiles]*f*(anchored?1.0:ev.kappa);
                if(best[i]+phi>best[j+1])best[j+1]=best[i]+phi;
            }
        }
        value+=best[count];
    }
    return value;
}
}
