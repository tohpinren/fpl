# FPL model improvement experiments — 2025/26

Completed 30 October 2026. Eighteen fixed configurations were tested, covering all five proposed steps. Production forecasts and model_config.json are unchanged.

The strongest practical result is fresh availability and set-piece context for defenders. Defensive-contribution rates help midfielders modestly. No forward challenger selected on development improved both later ranking and error. More data and more complex models are not universal improvements.

## Evaluation

- Development: GW16–20 with training frozen before GW16, then GW21–24 with training frozen before GW21.
- Later audit: GW25–29 with training frozen before GW25, then GW30–34 with training frozen before GW30. Every forecast covers five GWs; the last target ends in GW38.
- Five-GW training labels must mature before each training cutoff and must not overlap the validation block. Fixture-level models use only individually completed fixture labels. Their training cases cover the same one-to-five-GW lead times used at inference; inverse multiplicity weights give each player-fixture one unit of total weight. An initial next-match-only prototype was discarded for a horizon mismatch before the final comparison.
- All models use identical eligible cohorts: at least 180 historical minutes, last-five average minutes ≥50, and at least two recent appearances. Later audit: DEF 788, MID 847, FWD 211 player-origins, 1,846 total.
- Top-five score means actual points per selected player over the following five GWs, averaged across forecast origins. Higher is better. MAE covers every eligible player-origin; lower is better. Price-band evaluation selects three players within fixed position-specific price brackets.
- These are individual candidate forecasts, not a full-season team simulation with budgets, transfers, captaincy and club limits.
- **GW25–34 was already inspected in the previous review. This is a reused chronological audit, not an untouched test.** Choices were frozen using development scores before this run's audit. The current run uses five-GW block refits; the previous report refit every origin, so its numerical baseline differs.

## Development-selected recipes on later weeks

Selection maximized development top-five points, subject to no worse development MAE and price-band top-three score than baseline. The rule selected the following recipes before the audit:

| Position | Frozen recipe | Baseline MAE → candidate | Baseline top-five → candidate | Price-band top-three → candidate |
|---|---|---:|---:|---:|
| DEF | snapshots | 6.820 → 6.607 | 21.94 → 23.18 | 19.43 → 20.90 |
| MID | defcon | 6.576 → 6.569 | 19.46 → 20.06 | 20.64 → 20.66 |
| FWD | two_stage_more_minutes | 6.883 → 6.739 | 20.12 → 19.38 | 17.86 → 19.52 |

Combined frozen recipes: MAE 6.716 → 6.605; equal-position top-five score 20.51 → 20.87. The forward recipe is included despite worse top-five ranking, rather than silently replaced with the baseline.

## Findings for each step

1. **Features:** tested defensive-contribution hit rates over five/ten matches; team xG for, upcoming opponents' xG against and player's share of team xG; days since a PL appearance and minutes trend; and all seven together. MID defcon improved later top-five points 19.46 → 20.06, with MAE 6.576 → 6.569. The full feature bundle looked better for DEF in development but fell from 21.94 → 20.94 on later weeks. Additions need separate validation; more features did not reliably help.

2. **Candidate and price alignment:** tested eligible-only training, 3:1 weights favouring eligible players, inverse price-band weights, and feature/weight combinations. Eligible-only training reduced later top-five scores for every position. Price balancing also reduced top-five scores for every position and increased MAE. A MID weighted variant gained ranking points but worsened error. Keep broad training data and retain price-band reporting; do not apply one filtering rule to every position.

3. **More seasons:** downloaded 2023/24 and 2024/25 at the same pinned revision. Extra seasons for the minutes classifier improved DEF relative to the single-season two-stage model (top-five 20.14 → 21.10; MAE 6.946 → 6.704), hurt MID rankings, and left FWD ranking flat while improving its MAE (6.876 → 6.739). Pooling common scoring components improved FWD versus its single-season component control (19.60 → 20.78; MAE 6.841 → 6.772), also beating the current FWD baseline on these later weeks. It reduced DEF error but hurt DEF/MID rankings. The FWD pooled-component variant missed the development price-band guardrail, so it was not the frozen primary choice. Older total_points were never pooled as though scoring rules were identical.

4. **Architecture:** tested LightGBM L2, L1 and LambdaRank, plus a per-fixture minutes classifier (0 / 1–59 / 60+ minutes) with conditional point heads, summed over five GWs. A separate component model predicts the common scoring subtotal plus a 2025/26 residual for assists/bonus/defcon. The horizon-matched MID two-stage model reached 23.90 top-five points with MAE 6.364; the MID single-season component variant reached 25.22 with MAE 6.297, versus baseline 19.46 and 6.576 (+29.6% ranking, −4.2% error). These models were weaker on development top-five selection and were not the frozen primary MID choice. They are promising exploratory findings, not independent confirmation. Simple architecture changes do not reliably improve every position. LambdaRank emits ranking scores, so its MAE is intentionally omitted.

5. **Pre-deadline context:** the main archive's 12 bootstrap revisions were too sparse. Git history from olbauday/FPL-Elo-Insights supplied 31 additional states before GW4–34 cutoffs. Historical status, chance of playing, penalties and corner roles were read only from the latest compatible state before the cutoff, at most eight days old, with explicit missingness. All eligible GW16–34 observations had fresh context; selected snapshots were approximately 1.5–10.1 hours before the deadline proxy. DEF top-five score improved 21.94 → 23.18 (+5.7%), MAE fell 6.820 → 6.607 (−3.1%), and price-band ranking improved. MID MAE improved but ranking worsened; FWD did not improve.

## What to take forward

Prioritize a defender availability/role pilot: it improved error, overall top-five selection and price-band selection after also passing development checks. Keep the small MID defensive-contribution variant as a simple reference. Prospectively compare the MID per-fixture component model and FWD multi-season component model against both these references and the existing baseline. Their attractive later-period results were discovered during this audit. Reject blanket candidate filtering, inverse price weighting and a universal LightGBM/ranker replacement. The website's existing model remains unchanged while these forecasts accumulate outcomes.

## Data validation and scoring changes

All three seasons contain 380 completed fixtures and 38 GWs. 2023/24 has 29,725 player-fixture rows; 2024/25 has 27,605; 2025/26 has 29,747 after removing ten identical duplicate rows. The 2025/26 point totals reconcile for all players. One 2024/25 record differs: Evan Ferguson's bootstrap total is 28 but match rows sum to 27. His records were quarantined from auxiliary training. No labels were repaired by guesswork.

Defensive contributions are absent from the older seasons. They are excluded from the common cross-season models, rather than used as observed zeroes. The common subtotal covers appearance, goals, clean sheets, goal-conceded deductions and disciplinary/penalty deductions; rule-sensitive assists, bonus/BPS and defcon are handled by a current-era residual. The minutes classifier uses scoring-independent features. Goalkeepers and 2024/25 assistant managers are outside this study.

Historical fixtures still use final event assignments and first-kickoff-minus-90-minutes deadline proxies. Exact historical schedule revisions were not reconstructed, so blank/double-week fixture information may be optimistic. Historical player positions are season-level positions. Days since appearance covers Premier League matches, not cup/international workload. Commit time is the archive publication proxy, not a verified API collection timestamp.

## How certain are the gains?

There are only ten later forecast origins and their five-GW outcomes overlap. The defender top-five gain is concentrated in a few origins. Circular block bootstrap intervals below use blocks of five origins and 10,000 resamples; with only about two blocks of information, treat them as descriptive uncertainty, not proof of significance.

| Frozen recipe | Mean top-five change | Approx. 95% block interval | MAE improvement interval |
|---|---:|---:|---:|
| DEF snapshots | +1.24 | [-0.46, +2.94] | [+0.147, +0.281] |
| MID defcon | +0.60 | [-0.80, +2.00] | [-0.027, +0.041] |
| FWD two_stage_more_minutes | -0.74 | [-3.44, +1.96] | [-0.410, +0.665] |

The intervals for top-five gains all include zero. The defender error reduction is the most consistent signal among the development-selected recipes. A new prospective ledger freezes baseline and development-selected forecasts for 2026/27 GW9–13 before the GW9 deadline; it also includes the MID single-season component and FWD multi-season component challengers, explicitly labelled as selected after inspecting the reused audit. Their future outcomes are not yet available. It records current availability, fixture assignments and raw-input hashes. 2026/27 scoring differs, so that is a separate prospective test. The ledger is a one-time snapshot; no new scheduler or live model deployment was added.

## Complete results

Each cell is **top-five points / MAE**. A dash denotes an uncalibrated ranking score.

### Development

| Configuration | DEF | MID | FWD |
|---|---:|---:|---:|
| baseline | 14.91 / 6.421 | 18.60 / 7.722 | 19.13 / 7.801 |
| defcon | 13.89 / 6.415 | 20.76 / 7.686 | 19.13 / 7.841 |
| xg | 16.24 / 6.475 | 19.82 / 7.659 | 19.13 / 7.836 |
| minutes | 15.38 / 6.351 | 19.98 / 7.722 | 18.98 / 7.771 |
| all | 17.04 / 6.387 | 19.36 / 7.721 | 19.13 / 7.824 |
| eligible | 15.60 / 6.427 | 19.84 / 7.633 | 17.93 / 8.059 |
| weighted | 15.33 / 6.433 | 19.07 / 7.697 | 17.96 / 8.186 |
| price_balanced | 14.40 / 6.863 | 17.98 / 7.725 | 19.58 / 9.031 |
| all_eligible | 17.49 / 6.297 | 18.80 / 7.634 | 20.53 / 8.005 |
| all_weighted | 17.16 / 6.343 | 18.49 / 7.706 | 19.36 / 8.229 |
| lgb_l2 | 15.60 / 7.120 | 17.33 / 7.932 | 15.18 / 8.565 |
| lgb_l1 | 15.82 / 6.974 | 18.24 / 7.633 | 16.40 / 8.418 |
| ranker | 12.73 / — | 18.87 / — | 18.60 / — |
| snapshots | 15.84 / 6.148 | 19.80 / 7.711 | 19.00 / 7.708 |
| two_stage | 16.49 / 6.825 | 17.44 / 7.745 | 19.44 / 7.796 |
| two_stage_more_minutes | 18.11 / 6.513 | 17.11 / 7.874 | 19.27 / 7.667 |
| components_single | 18.02 / 6.854 | 18.33 / 7.719 | 18.07 / 8.078 |
| components_multi | 17.91 / 6.427 | 18.89 / 7.780 | 20.18 / 7.485 |

### Audit

| Configuration | DEF | MID | FWD |
|---|---:|---:|---:|
| baseline | 21.94 / 6.820 | 19.46 / 6.576 | 20.12 / 6.883 |
| defcon | 21.84 / 6.804 | 20.06 / 6.569 | 20.12 / 6.805 |
| xg | 20.32 / 6.819 | 19.54 / 6.578 | 19.82 / 6.892 |
| minutes | 21.90 / 6.843 | 19.70 / 6.534 | 19.94 / 6.950 |
| all | 20.94 / 6.812 | 19.00 / 6.543 | 20.14 / 6.863 |
| eligible | 21.28 / 7.210 | 17.44 / 6.629 | 18.84 / 7.086 |
| weighted | 19.86 / 7.045 | 20.10 / 6.593 | 20.02 / 7.134 |
| price_balanced | 17.92 / 8.170 | 18.16 / 6.692 | 16.64 / 8.509 |
| all_eligible | 19.84 / 7.281 | 18.74 / 6.608 | 19.44 / 7.101 |
| all_weighted | 20.10 / 7.057 | 18.88 / 6.558 | 20.20 / 7.139 |
| lgb_l2 | 16.08 / 7.311 | 17.44 / 6.798 | 20.00 / 7.137 |
| lgb_l1 | 17.56 / 7.253 | 20.30 / 6.505 | 20.92 / 6.973 |
| ranker | 14.08 / — | 19.14 / — | 19.86 / — |
| snapshots | 23.18 / 6.607 | 19.06 / 6.480 | 19.94 / 6.927 |
| two_stage | 20.14 / 6.946 | 23.90 / 6.364 | 19.38 / 6.876 |
| two_stage_more_minutes | 21.10 / 6.704 | 20.90 / 6.455 | 19.38 / 6.739 |
| components_single | 20.58 / 6.905 | 25.22 / 6.297 | 19.60 / 6.841 |
| components_multi | 20.18 / 6.623 | 21.62 / 6.386 | 20.78 / 6.772 |

## Reproduction and artifacts

```sh
.venv/bin/python research/model_improvements.py
.venv/bin/python research/report_improvements.py
.venv/bin/python -m unittest discover -s tests
```

The optional `research/freeze_forecast.py` fetches current official context and writes a new ledger with an exclusive timestamped filename. It never edits the website's data or model configuration. Run it before a deadline; it does not evaluate future outcomes.

- [Structured results](model-improvements-2025-26.json): all metrics, development selection, software versions and paired uncertainty.
- [Pinned source manifests](model-improvements-sources.json): archive revisions, snapshot commit IDs, hashes and freshness coverage.
- [Baseline and frozen-recipe predictions](model-improvements-predictions.csv): individual development/audit predictions. All challenger predictions remain reproducible in the ignored experiment cache.
- [FPL archive](https://github.com/vaastav/Fantasy-Premier-League/tree/d7b05b51d137bfefef4b692386d18c610975b9b9/data).
- [Availability archive](https://github.com/olbauday/FPL-Elo-Insights): exact historical commits are recorded in the source manifest.

Fixed model parameters: existing Ridge(alpha=100)/RandomForest(120 trees, depth 8, leaf 20); LightGBM 180 trees, 7 leaves, learning rate .04, leaf minimum 30, L2 regularization 20, seed 42. No audit-driven hyperparameter search was performed.
