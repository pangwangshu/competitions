#include "options.h"

#include <cmath>
#include <cstdlib>
#include <string>

#include "common.h"

namespace mm167 {
// Named configurations of the submitted builds. Each one reproduces the
// defaults that the corresponding package compiled in; flags given after it
// still override single values. "p2-03" is the plain struct defaults.
static int applyPreset(const std::string& name,Options& o) {
    if (name=="p2-03") return 0;
    if (name!="int-01" && name!="o-01a" && name!="o-01c") return 2;
    // INT-01: P2-01 rollout, effort revision, P1-03 decision.
    o.sched.workMs=3000; o.sched.front=3;
    o.deep.mode=1; o.deep.width=4; o.deep.depth=4; o.deep.samples=32; o.deep.share=0.5; o.deep.mix=0.5;
    o.deep.policy=1; o.deep.tail=1; o.deep.confidence=0; o.deep.first=0.25;
    o.dc.mode=2; o.dc.stop=true; o.dc.keep=8; o.dc.samples=8; o.dc.depth=4; o.dc.policy=1; o.dc.tail=0;
    o.dc.trigger=15; o.dc.margin=0; o.dc.confidence=2; o.dc.capMs=20; o.dc.workMs=1000;
    if (name=="int-01") return 0;
    // O-01A: fishing continuations, root fishing trigger, paced decision pool.
    o.deep.policy=4; o.dc.policy=4; o.dc.fishRoot=true; o.dc.front=2;
    if (name=="o-01a") return 0;
    // O-01C: lane-estimator event fixes, exact refill probability, dominated STOP.
    o.exactRefill=true; o.fixEvents=true; o.dc.stopDominated=true;
    return 0;
}

int parseOptions(int argc,char** argv,Options& o) {
    // Validated P1-02 values on top of the struct defaults.
    o.la.ev.pool=128; o.la.ev.alpha=5; o.la.ev.lanes=16;
    o.sched.on=true;
    // The preset is applied first wherever it appears; the final submission's
    // configuration (O-01C) is the no-argument behaviour.
    std::string preset="o-01c";
    for (int i=1;i+1<argc;++i) if (std::string(argv[i])=="--preset") preset=argv[i+1];
    if (applyPreset(preset,o)!=0) return 2;
    for (int i=1;i<argc;++i) {
        std::string arg=argv[i];
        if (arg=="--preset" && i+1<argc) ++i;
        else if (arg=="--no-fish-root") o.dc.fishRoot=false;
        else if (arg=="--stats") o.stats=true;
        else if (arg=="--diagnostics") o.diagnostics=true;
        else if (arg=="--profile") { o.profile=true; o.diagnostics=true; }
        else if (arg=="--lane-exact") o.exactRefill=true;
        else if (arg=="--lane-fix-events") o.fixEvents=true;
        else if (arg=="--stop-dominated") o.dc.stopDominated=true;
        else if (arg=="--stop-floor") o.dc.floor=true;
        else if (arg=="--lean-x"&&i+1<argc){
            try{
                std::string value=argv[++i];size_t end=0;int v=std::stoi(value,&end);
                if(end!=value.size()||v<0||v>5)return 2;
                o.leanX=v;
            }catch(...){return 2;}
        }
        else if ((arg=="--adaptive"||arg=="--adaptive-margin")&&i+1<argc){
            try{
                std::string value=argv[++i];size_t end=0;double v=std::stod(value,&end);
                if(end!=value.size()||!std::isfinite(v)||v<0||v>10)return 2;
                if(arg=="--adaptive"){if(v!=0&&v!=1&&v!=2)return 2;o.adaptive=int(v);}
                else o.adaptMargin=v;
            }catch(...){return 2;}
        }
        else if (arg=="--lookahead") o.lookahead=true;
        else if (arg=="--no-lookahead") o.lookahead=false;
        else if (arg=="--la-diversity") o.la.diversity=true;
        else if (arg=="--ev-linear") o.la.ev.linear=true;
        else if (arg=="--sched") o.sched.on=true;
        else if (arg=="--no-sched") o.sched.on=false;
        else if (arg=="--rollout") o.deep.mode=1;
        else if (arg=="--chance-tree") o.deep.mode=2;
        else if (arg=="--no-deep") o.deep.mode=0;
        else if ((arg=="--deep-width"||arg=="--deep-depth"||arg=="--deep-samples"||arg=="--deep-branch"||arg=="--deep-chance"||arg=="--deep-policy")&&i+1<argc){
            try{
                std::string value=argv[++i];size_t end=0;long v=std::stol(value,&end);
                if(end!=value.size()||v<1||v>64)return 2;
                if(arg=="--deep-width")o.deep.width=int(v);
                else if(arg=="--deep-depth")o.deep.depth=int(v);
                else if(arg=="--deep-samples")o.deep.samples=int(v);
                else if(arg=="--deep-branch")o.deep.branch=int(v);
                else if(arg=="--deep-chance")o.deep.chance=int(v);
                else {if(v>4)return 2;o.deep.policy=int(v);}
            }catch(...){return 2;}
        }
        else if ((arg=="--deep-share"||arg=="--deep-tail"||arg=="--deep-mix"||arg=="--deep-confidence"||arg=="--deep-first")&&i+1<argc){
            try{
                std::string value=argv[++i];size_t end=0;double v=std::stod(value,&end);
                if(end!=value.size()||!std::isfinite(v)||v<0||v>10)return 2;
                if(arg=="--deep-share"){if(v<=0||v>=1)return 2;o.deep.share=v;}
                else if(arg=="--deep-mix"){if(v>1)return 2;o.deep.mix=v;}
                else if(arg=="--deep-confidence")o.deep.confidence=v;
                else if(arg=="--deep-first"){if(v>=1)return 2;o.deep.first=v;}
                else o.deep.tail=v;
            }catch(...){return 2;}
        }
        else if ((arg=="--work-ms" || arg=="--guard-share" || arg=="--sched-front" || arg=="--sched-keep" || arg=="--sched-r0" || arg=="--sched-pre" || arg=="--sim-slow" || arg=="--sim-slow-turn") && i+1<argc) {
            try {
                std::string value=argv[++i]; size_t end=0;
                double v=std::stod(value,&end);
                if (end!=value.size() || !(v>0)) return 2;
                if (arg=="--work-ms") { if (v>9000) return 2; o.sched.workMs=v; }
                else if (arg=="--guard-share") { if (v>10) return 2; o.sched.guardShare=v; }
                else if (arg=="--sched-front") { if (v<1 || v>100) return 2; o.sched.front=v; }
                else if (arg=="--sched-keep") { if (v>1) return 2; o.sched.keep=v; }
                else if (arg=="--sched-r0") { if (v>1) return 2; o.sched.r0=v; }
                else if (arg=="--sched-pre") { if (v>1) return 2; o.sched.pre=v; }
                else if (arg=="--sim-slow-turn") { if (v<1 || v>100) return 2; simSlowTurn=v; }
                else { if (v<1 || v>100) return 2; simSlow=v; }
            } catch (...) { return 2; }
        } else if ((arg=="--sched-kmin" || arg=="--sched-kmax" || arg=="--sched-keep-min") && i+1<argc) {
            try {
                std::string value=argv[++i]; size_t end=0;
                long v=std::stol(value,&end);
                if (end!=value.size() || v<1 || v>512) return 2;
                if (arg=="--sched-kmin") o.sched.kMin=static_cast<int>(v);
                else if (arg=="--sched-kmax") o.sched.kMax=static_cast<int>(v);
                else o.sched.keepMin=static_cast<int>(v);
            } catch (...) { return 2; }
        }
        else if ((arg=="--budget-ms" || arg=="--turn-ms") && i+1<argc) {
            try {
                std::string value=argv[++i]; size_t end=0;
                double v=std::stod(value,&end);
                if (end!=value.size() || !(v>0 && v<=9000)) return 2;
                if (arg=="--budget-ms") o.budgetMs=v; else o.turnMs=v;
            } catch (...) { return 2; }
        } else if ((arg=="--la-k" || arg=="--la-samples" || arg=="--ev-lanes" || arg=="--ev-lane-min" || arg=="--ev-pool") && i+1<argc) {
            try {
                std::string value=argv[++i]; size_t end=0;
                long v=std::stol(value,&end);
                if (end!=value.size() || v<0 || v>(arg=="--ev-pool"?4096:512)) return 2;
                if (arg=="--la-k") { if (v<1) return 2; o.la.k=static_cast<int>(v); }
                else if (arg=="--ev-lanes") { if (v<1 || v>64) return 2; o.la.ev.lanes=static_cast<int>(v); }
                else if (arg=="--ev-pool") { o.la.ev.pool=static_cast<int>(v); o.poolGiven=true; }
                else if (arg=="--ev-lane-min") { if (v<2) return 2; o.la.ev.laneMin=static_cast<int>(v); }
                else o.la.samples=static_cast<int>(v);
            } catch (...) { return 2; }
        } else if (arg=="--la-gamma" && i+1<argc) {
            try {
                std::string value=argv[++i]; size_t end=0;
                double v=std::stod(value,&end);
                if (end!=value.size() || !(v>0 && v<=2)) return 2;
                o.la.gamma=v;
            } catch (...) { return 2; }
        } else if ((arg=="--ev-alpha" || arg=="--ev-rho" || arg=="--ev-kappa" || arg=="--ev-tau" || arg=="--ev-mu") && i+1<argc) {
            try {
                std::string value=argv[++i]; size_t end=0;
                double v=std::stod(value,&end);
                if (end!=value.size() || !(v>=0 && v<=1000)) return 2;
                if (arg=="--ev-alpha") o.la.ev.alpha=v;
                else if (arg=="--ev-rho") { if (v>1) return 2; o.la.ev.rho=v; }
                else if (arg=="--ev-kappa") { if (v>1) return 2; o.la.ev.kappa=v; }
                else if (arg=="--ev-mu") o.la.ev.mu=v;
                else o.la.ev.tau=v;
            } catch (...) { return 2; }
        } else if (arg=="--la-seed" && i+1<argc) {
            try {
                std::string value=argv[++i]; size_t end=0;
                unsigned long long v=std::stoull(value,&end);
                if (end!=value.size()) return 2;
                o.laSeed=v;
            } catch (...) { return 2; }
        } else if (arg=="--discard" && i+1<argc) {
            try {
                std::string value=argv[++i]; size_t end=0;
                long v=std::stol(value,&end);
                if (end!=value.size() || v<0 || v>2) return 2;
                o.dc.mode=static_cast<int>(v);
            } catch (...) { return 2; }
        } else if (arg=="--fish-root") o.dc.fishRoot=true;
        else if ((arg=="--policy4-margin"||arg=="--fish-max-rate")&&i+1<argc){
            try{
                std::string value=argv[++i];size_t end=0;double v=std::stod(value,&end);
                if(end!=value.size()||!std::isfinite(v)||v<0||v>100)return 2;
                if(arg=="--policy4-margin")o.deep.fishing.margin=o.dc.fishing.margin=v;
                else o.dc.fishMaxRate=v;
            }catch(...){return 2;}
        }
        else if ((arg=="--policy4-max-rate"||arg=="--discard-front")&&i+1<argc){
            try{
                std::string value=argv[++i];size_t end=0;double v=std::stod(value,&end);
                if(end!=value.size()||!std::isfinite(v)||v<0||v>1e9)return 2;
                if(arg=="--policy4-max-rate")o.deep.fishing.maxRate=o.dc.fishing.maxRate=v;
                else o.dc.front=v;
            }catch(...){return 2;}
        }
        else if ((arg=="--policy4-targets"||arg=="--policy4-lanes")&&i+1<argc){
            try{
                std::string value=argv[++i];size_t end=0;int v=std::stoi(value,&end);
                if(end!=value.size()||v<1||v>16)return 2;
                if(arg=="--policy4-targets")o.deep.fishing.targets=o.dc.fishing.targets=v;
                else o.deep.fishing.lanes=o.dc.fishing.lanes=v;
            }catch(...){return 2;}
        }
        else if (arg=="--no-discard") o.dc.mode=0;
        else if (arg=="--discard-stop") o.dc.stop=true;
        else if (arg=="--no-discard-stop") o.dc.stop=false;
        else if (arg=="--discard-rate") o.dc.rate=true;
        else if (arg=="--no-discard-rate") o.dc.rate=false;
        else if ((arg=="--discard-keep" || arg=="--discard-samples" || arg=="--discard-depth" || arg=="--discard-policy") && i+1<argc) {
            try {
                std::string value=argv[++i]; size_t end=0;
                long v=std::stol(value,&end);
                if (end!=value.size() || v<1 || v>64) return 2;
                if (arg=="--discard-keep") { if (v!=1 && v!=4 && v!=8 && v!=16) return 2; o.dc.keep=static_cast<int>(v); }
                else if (arg=="--discard-samples") o.dc.samples=static_cast<int>(v);
                else if (arg=="--discard-depth") o.dc.depth=static_cast<int>(v);
                else { if (v>4) return 2; o.dc.policy=static_cast<int>(v); }
            } catch (...) { return 2; }
        } else if ((arg=="--discard-cap-ms" || arg=="--discard-work-ms" || arg=="--discard-tail" || arg=="--discard-trigger" || arg=="--discard-margin" || arg=="--discard-confidence") && i+1<argc) {
            try {
                std::string value=argv[++i]; size_t end=0;
                double v=std::stod(value,&end);
                if (end!=value.size() || !std::isfinite(v) || v<0) return 2;
                if (arg=="--discard-cap-ms") { if (!(v>0) || v>1000) return 2; o.dc.capMs=v; }
                else if (arg=="--discard-work-ms") { if (!(v>0) || v>9000) return 2; o.dc.workMs=v; }
                else if (arg=="--discard-tail") { if (v>10) return 2; o.dc.tail=v; }
                else if (arg=="--discard-trigger") { if (v>1000) return 2; o.dc.trigger=v; }
                else if (arg=="--discard-confidence") { if (v>10) return 2; o.dc.confidence=v; }
                else { if (v>100) return 2; o.dc.margin=v; }
            } catch (...) { return 2; }
        } else return 2;
    }
    if (o.sched.kMax<o.sched.kMin) return 2;
    const bool scheduled=o.sched.on&&o.lookahead;
    if(o.deep.mode&&!scheduled)return 2;
    if(o.dc.mode&&!scheduled)return 2;
    if(o.dc.stop&&!o.dc.mode)return 2;
    if(o.dc.fishRoot&&o.dc.mode!=2)return 2;
    if (scheduled && !o.poolGiven) o.la.ev.pool=512;
    return 0;
}
}
