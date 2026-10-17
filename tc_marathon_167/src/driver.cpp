// The interactive game loop: budgeting, per-turn policy stack and diagnostics.
#include "driver.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "cost_model.h"
#include "deep.h"
#include "discard.h"
#include "evaluate.h"
#include "fishing.h"
#include "game.h"
#include "generator.h"
#include "lane_model.h"
#include "lookahead.h"
#include "scheduled.h"
#include "transition.h"

namespace mm167 {
int runGame(const Options& o){
    const double budgetMs=o.budgetMs, turnMs=o.turnMs;
    const bool stats=o.stats, diagnostics=o.diagnostics, profile=o.profile;
    const bool exactRefill=o.exactRefill, fixEvents=o.fixEvents;
    const bool lookahead=o.lookahead;
    LookaheadConfig la=o.la;
    const SchedConfig sched=o.sched;
    const DeepConfig deep=o.deep;
    const DiscardConfig dc=o.dc;
    const int adaptive=o.adaptive;
    const double adaptMargin=o.adaptMargin;
    const int leanX=o.leanX;
    const uint64_t laSeed=o.laSeed;
    const bool scheduled=sched.on&&lookahead;
    auto start=Clock::now();
    Game g; if (!g.readInitial()) return 0;
    std::unique_ptr<LaneModel> laneModel;
    if (lookahead && (la.ev.pool>(scheduled?sched.kMin:la.k)||deep.policy==4||dc.policy==4||dc.fishRoot)) { laneModel=std::make_unique<LaneModel>(*g.rules); la.lane=laneModel.get(); }
    if(laneModel){laneModel->exactRefill=exactRefill;laneModel->fixEvents=fixEvents;}
    auto now=Clock::now();
    // Conservative first-turn accounting includes parsing / trie construction.
    double elapsed=std::chrono::duration<double,std::milli>(now-start).count();
    std::ostringstream diagnosticLog;
    if (diagnostics) {
        std::string bonuses; int count=0;
        for(int r=0;r<g.rules->n;++r) for(int c=0;c<g.rules->n;++c) {
            bonuses+=g.rules->bonus[r][c]; count+=(g.rules->bonus[r][c]!='.' && g.rules->bonus[r][c]!='S');
        }
        diagnosticLog << "DIAG {\"type\":\"initial\",\"N\":" << g.rules->n << ",\"T\":" << g.rules->t
            << ",\"X\":" << g.rules->x << ",\"D\":" << g.rules->penalty << ",\"bonus_density\":" << double(count)/(g.rules->n*g.rules->n)
            << ",\"bonus\":\"" << bonuses << "\",\"initial_ms\":" << elapsed
            << ",\"profile\":" << (profile?"true":"false") << "}\n";
    }
    std::string endReason="empty_hand";
    int64_t diagnosticScore=0;
    int turn=0;
    // Strategy RNG is isolated from the input stream and reproducible per seed.
    std::mt19937_64 laRng(laSeed);
    std::mt19937_64 deepRng(mixSeed(laSeed));
    // P1-03 decision RNG: constructed always, consumed only when dc.mode>0.
    std::mt19937_64 discardRng(mixSeed(laSeed+0x5D15CULL));
    long deepTurns=0,deepComplete=0,deepChanged=0,deepInvalid=0;
    double deepWork=0;
    long dcTurns=0,dcDiscards=0,dcDiscardedTiles=0,dcStops=0,dcCuts=0,dcPartial=0;
    double dcWork=0;
    bool prevDiscard=false;
    long predictChecked=0,predictOk=0,predictMismatch=0,predictUpcomingMismatch=0;
    long laFallbacks=0,laInvalid=0;
    double evalMsTotal=0,searchMsTotal=0;
    // Scheduler state. Effort is counted in predicted reference-machine ns; the
    // measured/predicted ratio only shrinks the plan when this machine is slower.
    const double initialMs=elapsed;
    double workUsed=0,rootWork=0,measuredMs=0,slow=1;
    long tilesConsumed=0,starvedTurns=0,hardCuts=0,guardTurns=0;
    long dcStarved=0;
    DeepConfig deepLean=deep;
    DiscardConfig dcLean=dc;
    dcLean.fishRoot=false;
    if(adaptive!=1){deepLean.policy=1;dcLean.policy=1;dcLean.front=0;}
    const bool leanGame=g.rules->x<=leanX;
    long leanTurns=0;
    while (true) {
        auto turnStart=Clock::now();
        if (elapsed>=budgetMs-5) { endReason="budget"; break; }
        int tiles=0; for (char c : g.hand) tiles+=(c!='.');
        if (!tiles) break;
        int remaining=g.rules->n*g.rules->n-g.drawn;
        // Bound each decision and retain a budget for future turns.
        double slice=std::min({turnMs,budgetMs-elapsed-5,(budgetMs-elapsed)/(remaining+tiles+1)});
        double cap=0,workLeft=0,dcCap=0,dcFirst=0;
        bool lean=false;
        size_t capacity=lookahead?static_cast<size_t>(std::max(scheduled?sched.kMax:la.k,la.ev.pool)):0;
        if (scheduled) {
            // Decisions still to come, from the tiles used per turn so far.
            const double perTurn=(8*(1+0.3*g.rules->t)+tilesConsumed)/(8.0+turn);
            const double decisions=std::max(1.0,(remaining+tiles)/perTurn);
            const double timeLeft=budgetMs-elapsed-5;
            // Machine-speed prior from the input phase until turns have been measured.
            const double prior=std::min(6.0,std::max(1.0,initialMs/120.0));
            slow=std::max(1.0,(measuredMs+prior*20)/((workUsed+dcWork)/1e6+20));
            // The time guard keeps back the move generation of every turn still to come.
            const double reserve=decisions*(rootWork+4*3e5)/(turn+4.0);
            const double plan=sched.workMs*1e6-workUsed,safe=timeLeft*0.85/slow*1e6-reserve;
            const double poolLeft=dc.mode?std::max(0.0,dc.workMs*1e6-dcWork):0;
            double poolAvail=poolLeft;
            lean=leanGame||(adaptive&&safe<adaptMargin*(std::max(0.0,plan)+poolLeft));
            leanTurns+=lean;
            if (sched.guardShare>0) {
                if (safe<plan+poolLeft) {
                    ++guardTurns;
                    const double weight=sched.guardShare*poolLeft;
                    poolAvail=std::min(poolLeft,std::max(0.0,safe)*weight/std::max(1.0,std::max(0.0,plan)+weight));
                    workLeft=std::min(plan,safe-poolAvail);
                } else workLeft=plan;
            } else {
                if (safe<plan) ++guardTurns;
                workLeft=std::min(plan,safe);
            }
            cap=workLeft>0?std::min(0.5*workLeft,sched.front*workLeft/decisions):0;
            // Nothing left to schedule: plain greedy generation, without the candidate pool.
            if (workLeft<=0) capacity=0;
            if (dc.mode) {
                // P1-03 decision effort: own pool, bounded by the same time guard.
                const double pool=sched.guardShare>0?poolAvail:std::min(poolLeft,safe-std::max(0.0,workLeft));
                // dev2 pacing: a share of the remaining pool per expected decision, so a
                // long game cannot exhaust the pool in its first half (O-01A pressure failure).
                const double front=lean?dcLean.front:dc.front;
                const double paced=front>0?std::min(pool,front*poolAvail/decisions):pool;
                dcCap=std::max(0.0,std::min(dc.capMs*1e6,paced));
                dcFirst=std::max(0.0,std::min(4*dc.capMs*1e6,front>0?std::min(pool,4*paced):pool));
            }
            // Wall-clock safety net only: far above the planned effort of this turn.
            slice=std::min(0.5*timeLeft,(8*(cap+dcCap)+2*deep.first*std::max(0.0,workLeft))/1e6*slow+50);
        }
        auto deadline=turnStart+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double,std::milli>(slice));
        const DeepConfig& deepT=lean?deepLean:deep;
        const DiscardConfig& dcT=lean?dcLean:dc;
        // With lookahead the generation phase gets a share of the slice and the
        // ply-2 evaluation uses the rest; both stay inside the same turn bound.
        auto genDeadline=deadline;
        if (lookahead && !scheduled)
            genDeadline=turnStart+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double,std::milli>(0.4*slice));
        Generator gen(g,capacity);
        gen.profile=profile;
        gen.diversity=lookahead&&la.diversity&&!scheduled;
        Move move=gen.run(genDeadline);
        double rootMs=profile ? std::chrono::duration<double,std::milli>(Clock::now()-turnStart).count() : 0;
        LookaheadReport laRep;
        DeepReport deepRep;
        std::vector<RankedMove> ranked;
        if (scheduled) {
            laRep.work=rootCost(gen);
            rootWork+=laRep.work;
            // No effort left beyond generation: the greedy move, already complete, is played.
            if (move.score>=0 && cap<=laRep.work) { ++starvedTurns; laRep.cap=cap; }
            else if (move.score>=0 && !gen.top.empty())
                move=runScheduled(g,gen,la,sched,deep.mode?cap*(1-deep.share):cap,laRng,deadline,laRep,deep.mode?&ranked:nullptr);
            if(deep.mode&&move.score>=0){
                move=runDeep(g,move,ranked,deepT,la,std::max(0.0,cap-laRep.work),deepRng,deadline,deepRep,
                             deep.first*std::max(0.0,workLeft-laRep.work));
                laRep.work+=deepRep.work;
                deepWork+=deepRep.work;deepTurns+=deepRep.attempted;deepComplete+=deepRep.rounds>0;
                deepChanged+=deepRep.changed;deepInvalid+=deepRep.invalid;
                if(deepRep.changed){
                    laRep.predicted=false;Game predicted(g);
                    if(transition(predicted,{ActionKind::Place,move,{}},"")==TransitionResult::Applied){
                        laRep.predicted=true;laRep.predictedHand=predicted.hand;laRep.predictedUpcoming=predicted.upcoming;
                    }
                }
            }
            hardCuts+=laRep.hardCut;
        } else if (lookahead && move.score>=0 && !gen.top.empty())
            move=runLookahead(g,gen,la,laRng,deadline,laRep);
        if (laRep.rounds==0 && laRep.fallback && *laRep.fallback) ++laFallbacks;
        laInvalid+=laRep.invalidTransitions;
        // P1-03: unified PLACE/DISCARD/STOP decision on top of the final move.
        // Without a decision (mode off, not triggered, R=0) `chosen` reproduces
        // the parent action exactly, including the full-hand no-word discard.
        Action chosen;
        if(move.score>=0)chosen={ActionKind::Place,move,{}};
        else if(remaining>0){std::string letters;for(char ch:g.hand)if(ch!='.')letters+=ch;chosen={ActionKind::Discard,{},letters};}
        DiscardReport dcRep;
        int dcTrigger=0;
        double fishRootWork=0;FishPlans rootPlans;
        bool policyStop=false;
        if(dc.mode&&remaining>0&&!gen.stopped){
            // Without a completed two-ply round there is no Q: the immediate score
            // stands in, so a starved turn with a good move does not trigger.
            const double q=laRep.rounds>0?laRep.chosenQ:static_cast<double>(move.score);
            if(move.score<0)dcTrigger=1;
            else if(dc.mode>=2&&(q<dc.trigger||q<g.rules->penalty))dcTrigger=2;
            else if(dc.mode>=2&&fishingRoot(g,gen,move,dcT,la,dcCap,deadline,fishRootWork,rootPlans))dcTrigger=3;
            if(dcTrigger){
                ++dcTurns;
                chosen=runDiscardDecision(g,gen,move,q,dcT,la,std::max(0.0,dcCap-fishRootWork),
                                          std::max(0.0,dcFirst-fishRootWork),discardRng,deadline,
                                          prevDiscard,laneModel.get(),dcRep,dcTrigger==3?&rootPlans:nullptr);
                dcWork+=dcRep.work;dcCuts+=dcRep.cut;dcStarved+=dcRep.rounds==0;
                if(chosen.kind==ActionKind::Discard){
                    if(chosen.letters.size()<static_cast<size_t>(tiles))++dcPartial;
                    // Refill fully determined by the visible preview: predict it.
                    laRep.predicted=false;
                    if(std::min(remaining,static_cast<int>(chosen.letters.size()))<=g.rules->x){
                        Game predicted(g);
                        if(transition(predicted,chosen,std::string())==TransitionResult::Applied){
                            laRep.predicted=true;laRep.predictedHand=predicted.hand;
                            laRep.predictedUpcoming=predicted.upcoming;
                        }
                    }
                } else if(chosen.kind==ActionKind::Stop)policyStop=true;
            }
            dcWork+=fishRootWork;
        }
        double searchMs=diagnostics ? std::chrono::duration<double,std::milli>(Clock::now()-turnStart).count() : 0;
        evalMsTotal+=laRep.evalMs;searchMsTotal+=searchMs;
        if (diagnostics) {
            int bonusUsed=0; std::string action="STOP";
            if (chosen.kind==ActionKind::Place && chosen.move.score>=0) {
                const Move& m=chosen.move;
                action="PLACE "+std::to_string(m.r)+" "+std::to_string(m.c)+" "+(m.dir?"V":"H")+" "+m.word;
                for(int j=0;j<int(m.word.size());++j) {
                    int r=m.r+m.dir*j,c=m.c+(1-m.dir)*j;
                    bonusUsed+=!g.occupied(r,c) && g.rules->bonus[r][c]!='.' && g.rules->bonus[r][c]!='S';
                }
                diagnosticScore+=m.score;
            } else if (chosen.kind==ActionKind::Discard) {
                action="DISCARD "+chosen.letters;
                diagnosticScore-=g.rules->penalty;
            }
            diagnosticLog << "DIAG {\"type\":\"turn\",\"turn\":" << turn
                << ",\"hand\":\"" << g.hand << "\",\"upcoming\":\"" << g.upcoming
                << "\",\"remaining\":" << remaining << ",\"nodes\":" << gen.nodes
                << ",\"candidates\":" << gen.candidates << ",\"truncated\":" << (gen.stopped?"true":"false")
                << ",\"score\":" << move.score << ",\"used\":" << (chosen.kind==ActionKind::Place?chosen.move.used:0)
                << ",\"discarded\":" << (chosen.kind==ActionKind::Discard?int(chosen.letters.size()):0)
                << ",\"bonus_used\":" << bonusUsed << ",\"search_ms\":" << searchMs
                << ",\"cross_ms\":" << gen.crossMs << ",\"action\":\"" << action << "\"";
            if (lookahead)
                diagnosticLog << ",\"la_rounds\":" << laRep.rounds
                    << ",\"la_width\":" << laRep.width << ",\"la_det\":" << laRep.deterministic
                    << ",\"la_evals\":" << laRep.evals << ",\"la_q\":" << laRep.chosenQ
                    << ",\"la_next\":" << laRep.nextValue
                    << ",\"la_pool\":" << laRep.pool << ",\"la_rank\":" << laRep.chosenRank << ",\"la_playable\":" << laRep.chosenPlayable << ",\"la_samples\":" << laRep.chosenSamples
                    << ",\"ev_phase\":" << laRep.phase << ",\"ev_board\":" << laRep.chosenBoard
                    << ",\"la_fallback\":\"" << (laRep.fallback?laRep.fallback:"") << "\"";
            if (scheduled)
                diagnosticLog << ",\"s_work\":" << laRep.work/1e6 << ",\"s_root\":" << rootCost(gen)/1e6
                    << ",\"s_pre\":" << laRep.preWork/1e6 << ",\"s_keyed\":" << laRep.keyed
                    << ",\"s_cap\":" << cap/1e6 << ",\"s_left\":" << workLeft/1e6
                    << ",\"s_final\":" << laRep.finalists << ",\"s_cut\":" << (laRep.hardCut?"true":"false");
            if(deep.mode)
                diagnosticLog << ",\"deep_attempt\":" << deepRep.attempted << ",\"deep_rounds\":" << deepRep.rounds
                    << ",\"deep_width\":" << deepRep.width << ",\"deep_evals\":" << deepRep.evals
                    << ",\"deep_steps\":" << deepRep.steps << ",\"deep_discards\":" << deepRep.discards
                    << ",\"deep_cut\":" << (deepRep.cut?"true":"false") << ",\"deep_changed\":" << (deepRep.changed?"true":"false")
                    << ",\"deep_work\":" << deepRep.work/1e6 << ",\"deep_q\":" << deepRep.q;
            if(dc.mode)
                diagnosticLog << ",\"dc_trigger\":" << dcTrigger
                    << ",\"dc_subsets\":" << dcRep.subsets << ",\"dc_cands\":" << dcRep.cands
                    << ",\"dc_rounds\":" << dcRep.rounds << ",\"dc_evals\":" << dcRep.evals
                    << ",\"dc_steps\":" << dcRep.steps << ",\"dc_invalid\":" << dcRep.invalid
                    << ",\"dc_cut\":" << (dcRep.cut?"true":"false")
                    << ",\"dc_changed\":" << (dcRep.changed?"true":"false")
                    << ",\"dc_chosen\":\"" << (chosen.kind==ActionKind::Place?"place":chosen.kind==ActionKind::Discard?"discard":"stop")
                    << "\",\"dc_letters\":\"" << chosen.letters << "\""
                    << ",\"dc_q_anchor\":" << dcRep.qAnchor << ",\"dc_q_best\":" << dcRep.qBest
                    << ",\"dc_work\":" << dcRep.work/1e6;
            if (profile)
                diagnosticLog << ",\"c_root_ms\":" << rootMs << ",\"c_pre_ms\":" << laRep.preMs << ",\"c_ply_ms\":" << laRep.plyMs
                    << ",\"c_cross\":" << gen.crossCells << ",\"c_ret\":" << gen.retained << ",\"c_retscan\":" << gen.retainScan
                    << ",\"c_pre\":" << laRep.preCands << ",\"c_scan\":" << laRep.scanWords << ",\"c_build\":" << laRep.buildWords
                    << ",\"c_win\":" << laRep.windows << ",\"c_nodes2\":" << laRep.nodes2 << ",\"c_cross2\":" << laRep.cross2
                    << ",\"c_runs2\":" << laRep.runs2 << ",\"c_boards\":" << laRep.boards;
            if(deep.policy==4||dc.policy==4||dc.fishRoot)
                diagnosticLog << ",\"fish_calls\":" << deepRep.fishCalls+dcRep.fishCalls
                    << ",\"fish_targets\":" << deepRep.fishTargets+dcRep.fishTargets
                    << ",\"fish_swaps\":" << deepRep.fishSwaps+dcRep.fishSwaps
                    << ",\"fish_work_ms\":" << (deepRep.fishWork+dcRep.fishWork)/1e6
                    << ",\"fish_root_work_ms\":" << fishRootWork/1e6;
            diagnosticLog << "}\n";
        }
        if (stats) std::cerr << "turn=" << turn << " nodes=" << gen.nodes << " candidates=" << gen.candidates
                             << " truncated=" << gen.stopped << " score=" << move.score;
        if (stats && lookahead)
            std::cerr << " la_rounds=" << laRep.rounds << " la_width=" << laRep.width
                      << " la_q=" << laRep.chosenQ << " la_fallback=" << (laRep.fallback?laRep.fallback:"none");
        if (stats) std::cerr << '\n';
        workUsed+=laRep.work;
        tilesConsumed+=chosen.kind==ActionKind::Place?chosen.move.used:
                       chosen.kind==ActionKind::Discard?static_cast<long>(chosen.letters.size()):(remaining>0?tiles:0);
        if (chosen.kind==ActionKind::Place) {
            g.apply(chosen.move);
            std::cout << "PLACE " << chosen.move.r << ' ' << chosen.move.c << ' ' << (chosen.move.dir ? 'V':'H') << ' ' << chosen.move.word << '\n' << std::flush;
            prevDiscard=false;
        } else if (chosen.kind==ActionKind::Discard) {
            for (char ch : chosen.letters) { auto p=g.hand.find(ch); if(p!=std::string::npos) g.hand[p]='.'; }
            g.drawn=std::min(g.rules->n*g.rules->n,g.drawn+static_cast<int>(chosen.letters.size()));
            g.score-=g.rules->penalty;
            ++dcDiscards; dcDiscardedTiles+=static_cast<long>(chosen.letters.size());
            std::cout << "DISCARD " << chosen.letters << '\n' << std::flush;
            prevDiscard=true;
        } else if (policyStop) { ++dcStops; endReason="policy_stop"; break; }
        else { endReason=gen.stopped?"truncated_no_move":"no_move_exhausted"; break; }
        if (simSlowTurn>1) {
            const auto now=Clock::now();
            const auto until=now+std::chrono::duration_cast<Clock::duration>((now-turnStart)*(simSlowTurn-1));
            while (Clock::now()<until) {}
        }
        double local=std::chrono::duration<double,std::milli>(Clock::now()-turnStart).count();
        measuredMs+=local;
        long long reported;
        if (!(std::cin >> reported) || !g.readTiles()) {
            if(diagnostics) std::cerr << diagnosticLog.str() << "DIAG {\"type\":\"end\",\"reason\":\"feedback_eof\"}\n";
            return 0;
        }
        // Deterministic refills must predict the official feedback exactly.
        if (laRep.predicted) {
            ++predictChecked;
            if (g.hand==laRep.predictedHand) ++predictOk; else ++predictMismatch;
            bool upOk=true;
            for (size_t j=0;j<g.upcoming.size()&&j<laRep.predictedUpcoming.size();++j)
                if (laRep.predictedUpcoming[j]!='?' && laRep.predictedUpcoming[j]!=g.upcoming[j]) { upOk=false; break; }
            if (!upOk) ++predictUpcomingMismatch;
        }
        elapsed=std::max(elapsed+local,static_cast<double>(reported));
        ++turn;
    }
    if(diagnostics) {
        std::ostringstream tail;
        if (lookahead)
            tail << ",\"la_on\":true,\"la_k\":" << la.k << ",\"la_samples\":" << la.samples
                 << ",\"la_gamma\":" << la.gamma << ",\"la_seed\":" << laSeed
                 << ",\"la_diversity\":" << (la.diversity?"true":"false")
                 << ",\"ev_alpha\":" << la.ev.alpha
                 << ",\"ev_rho\":" << la.ev.rho << ",\"ev_kappa\":" << la.ev.kappa
                 << ",\"ev_tau\":" << la.ev.tau << ",\"ev_linear\":" << (la.ev.linear?"true":"false")
                 << ",\"ev_lanes\":" << la.ev.lanes << ",\"ev_lane_min\":" << la.ev.laneMin
                 << ",\"ev_pool\":" << la.ev.pool << ",\"ev_mu\":" << la.ev.mu
                 << ",\"ev_ms\":" << evalMsTotal << ",\"search_ms\":" << searchMsTotal
                 << ",\"lane_build_ms\":" << (laneModel?laneModel->buildMs:0) << ",\"lane_builds\":" << (laneModel?laneModel->builds:0)
                 << ",\"lane_scan_ms\":" << (laneModel?laneModel->scanMs:0) << ",\"lane_scans\":" << (laneModel?laneModel->scans:0)
                 << ",\"la_fallbacks\":" << laFallbacks << ",\"la_invalid\":" << laInvalid
                 << ",\"sched_on\":" << (scheduled?"true":"false") << ",\"work_budget_ms\":" << sched.workMs
                 << ",\"sched_front\":" << sched.front << ",\"sched_kmin\":" << sched.kMin << ",\"sched_kmax\":" << sched.kMax
                 << ",\"sched_keep\":" << sched.keep << ",\"sched_keep_min\":" << sched.keepMin << ",\"sched_r0\":" << sched.r0
                 << ",\"sched_pre\":" << sched.pre << ",\"guard_share\":" << sched.guardShare
                 << ",\"lane_fix_events\":" << (fixEvents?"true":"false")
                 << ",\"lane_exact\":" << (exactRefill?"true":"false")
                 << ",\"work_ms\":" << workUsed/1e6 << ",\"measured_ms\":" << measuredMs << ",\"slow\":" << slow
                 << ",\"sched_starved\":" << starvedTurns << ",\"sched_hard_cuts\":" << hardCuts << ",\"sched_guard_turns\":" << guardTurns
                 << ",\"sim_slow\":" << simSlow
                 << ",\"adaptive\":" << adaptive << ",\"adaptive_margin\":" << adaptMargin << ",\"lean_x\":" << leanX << ",\"stop_dominated\":" << (dc.stopDominated?"true":"false") << ",\"stop_floor\":" << (dc.floor?"true":"false") << ",\"lean_turns\":" << leanTurns << ",\"turns\":" << turn
                 << ",\"predict_checked\":" << predictChecked << ",\"predict_ok\":" << predictOk
                 << ",\"predict_mismatch\":" << predictMismatch
                 << ",\"predict_upcoming_mismatch\":" << predictUpcomingMismatch;
        if(deep.mode)
            tail << ",\"deep_mode\":" << deep.mode << ",\"deep_depth\":" << deep.depth
                 << ",\"deep_width\":" << deep.width << ",\"deep_samples\":" << deep.samples
                 << ",\"deep_branch\":" << deep.branch << ",\"deep_chance\":" << deep.chance
                 << ",\"deep_share\":" << deep.share << ",\"deep_tail\":" << deep.tail
                 << ",\"deep_mix\":" << deep.mix << ",\"deep_confidence\":" << deep.confidence
                 << ",\"deep_first\":" << deep.first
                 << ",\"deep_policy\":" << deep.policy << ",\"deep_turns\":" << deepTurns
                 << ",\"deep_complete\":" << deepComplete << ",\"deep_changed\":" << deepChanged
                 << ",\"deep_invalid\":" << deepInvalid << ",\"deep_work_ms\":" << deepWork/1e6;
        if(dc.mode)
            tail << ",\"dc_mode\":" << dc.mode << ",\"dc_stop\":" << (dc.stop?"true":"false")
                 << ",\"dc_keep\":" << dc.keep << ",\"dc_samples\":" << dc.samples
                 << ",\"dc_depth\":" << dc.depth << ",\"dc_policy\":" << dc.policy
                 << ",\"dc_cap_ms\":" << dc.capMs << ",\"dc_work_budget_ms\":" << dc.workMs << ",\"dc_tail\":" << dc.tail
                 << ",\"dc_trigger\":" << dc.trigger << ",\"dc_margin\":" << dc.margin
                 << ",\"dc_rate\":" << (dc.rate?"true":"false") << ",\"dc_confidence\":" << dc.confidence
                 << ",\"dc_turns\":" << dcTurns << ",\"dc_discards\":" << dcDiscards
                 << ",\"dc_partial\":" << dcPartial << ",\"dc_discarded_tiles\":" << dcDiscardedTiles
                 << ",\"dc_stops\":" << dcStops << ",\"dc_cuts\":" << dcCuts << ",\"dc_incomplete\":" << dcStarved
                 << ",\"dc_work_ms\":" << dcWork/1e6;
        if(deep.policy==4||dc.policy==4||dc.fishRoot)
            tail << ",\"fish_root\":" << (dc.fishRoot?"true":"false")
                 << ",\"fish_max_rate\":" << dc.fishMaxRate
                 << ",\"policy4_targets\":" << deep.fishing.targets
                 << ",\"policy4_lanes\":" << deep.fishing.lanes
                 << ",\"policy4_margin\":" << deep.fishing.margin
                 << ",\"policy4_max_rate\":" << deep.fishing.maxRate << ",\"discard_front\":" << dc.front;
        std::cerr << diagnosticLog.str() << "DIAG {\"type\":\"end\",\"reason\":\"" << endReason
            << "\",\"score\":" << diagnosticScore << ",\"drawn\":" << g.drawn << tail.str() << "}\n";
    }
    std::cout << "-1\n" << std::flush;
    return 0;
}
}
