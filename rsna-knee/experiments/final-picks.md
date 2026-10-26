# F13 / H3: final picks (2026-10-21)

Selected on the Kaggle Submissions page by the user on 2026-10-21 (2/2 shown).

| Pick | Ref | Kernel | Config | Public |
|---|---|---|---|---|
| ② rank | 57376196 | wangshu/rsna-knee-submit-blend-s6 v2 | rank(0.7 nartaa S6 ConvNeXt-small + 0.3 C*x10), S6-final-blend-s6w07 | 0.950 |
| ① own reader | 57335130 | wangshu/rsna-knee-submit-own v7 | T4-S6 distilled students x2 (fold 0) + I*2, S5-probe-F-t4s6x2 | 0.938 |

Why: Kaggle reveals private scores for every submission after close, so the picks only set the official
rank. ②′ carries the rank; probe F is the best own reader and the write-up's main finding (C*x10 0.935 would be
the more robust choice for the second slot). s6_weight 1.0 was not submitted: ②′ already equals S6 alone.

Public leaderboard snapshot 2026-10-21 20:45 UTC: 6,343 teams; top-30% cutoff 0.957, median 0.941.

| Public | Rank range | Percentile |
|---|---|---|
| 0.950 | 2385–2502 | 37.6–39.4% |
| 0.944 (baseline) | 2637–2690 | 41.6–42.4% |
| 0.938 (F) | 3482–3503 | 54.9–55.2% |
| 0.935 (C*x10) | 3756–3796 | 59.2–59.8% |
| 0.924 (E) | 3970–4013 | 62.6–63.3% |

## Private leaderboard (recorded 2026-10-25, provisional)

Submissions closed 2026-10-22 23:59 UTC. The private leaderboard is visible (API `get_leaderboard` now returns private
scores) but Kaggle/RSNA have not yet announced official results; winner verification runs into November, so ranks
may still shift slightly. Summary numbers: `experiments/final-leaderboard.json`.

**Official (private) rank: 2,432 / 6,425 (top 37.9%)**, scored by ② 57376196 = 0.936. Public rank at close 2,493.

| Pick | Ref | Public | Private |
|---|---|---|---|
| ② blend-s6 v2 | 57376196 | 0.950 | **0.936** (ties G, best of all 11 submissions) |
| ① probe F | 57335130 | 0.938 | **0.927** (best own reader; C*x10 0.921) |

Both picks were the best available choice on private as well. Private scores of all 11 submissions are recorded in
each experiment's metrics.json (`private_score`) and in experiments/results.csv; the safety baseline 56935305 has no
registry row: public 0.944, private 0.929.

Board at close (6,425 teams): public top 0.970 / 30% 0.957 / median 0.942; private top 0.960 / 30% 0.942 /
median 0.925; private bronze (10%) ≈ 0.942, silver (5%) ≈ 0.943; 1,322 teams tied at private 0.942.
