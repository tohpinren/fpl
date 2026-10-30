# Availability rollout and prospective midfield test

The defender Ridge forecast now uses the nine availability/role features from
the historical experiment. The model choice is `ridge_availability`, with the
same Ridge regularization and historical
vectors used by the successful defender experiment. MID stays on the live forest;
FWD stays on Ridge.

Historical experiments are preserved in commit `3e2886a`. Their observed gains
are reused historical comparisons, and have not become independent evidence.
The report's statement that production was unchanged describes the experiment
before this authorized rollout.

## Defender context

Features: known context, age, available/doubtful/unavailable status, known playing
probability, chance of playing next round, first-choice penalty and corner roles.
Historical vectors have a pinned source revision, feature schema and source
manifest hash. The production model uses 43 features for DEF and the existing 34
for MID/FWD. Context older than eight days, future context and unknown probability
remain explicitly missing. News dated after the cutoff is rejected.

New-season training uses only archived context for that historical origin. Current
bootstrap availability is only used for the current forecast. Missing earlier
2026/27 snapshots are not filled using today's injury information.

## Prospective protocol

- The first successful pre-deadline run per origin is frozen. All later Friday,
  push and manual runs preserve it, even if the live website changes.
- Eligible players use the same historical minutes criteria as the review. Model
  comparisons keep that cohort fixed, including later injuries and zero returns.
- DEF: previous Ridge versus live availability Ridge. MID: live forest versus the
  frozen `components_single` per-fixture minutes/point-component architecture.
- Five-gameweek forecasts include the known fixture plan, source hashes and raw
  defender context. The researcher selected the midfield architecture after seeing
  the old audit; its new future observations provide the additional test.
- Results are scored only after the full target horizon has finished, every
  relevant fixture has finished and the three-hour settlement buffer has passed.
  Originally scheduled postponed matches also block premature scoring.
- Every run updates paired MAE, actual points per top-five pick and aggregate
  summaries once outcomes mature. Missing player histories fail the evaluation
  instead of dropping difficult examples or assuming zero.
- No automatic model promotion or hyperparameter tuning uses prospective outcomes.
  Overlapping forecast windows require cautious interpretation.

## Persistence and deployment

The existing Friday 08:17 Singapore / manual Pages workflow restores and appends
history to `codex/forecast-ledger`. Snapshots cannot be overwritten, and branch
updates preserve the existing tree without force pushes. The branch contains data
only and does not trigger the main-branch Pages workflow. The Actions token is
never included in the public site.

The earlier one-time research ledger in `reports/forecast-ledger/` remains a
separate experiment record. The automated protocol uses the data branch as its
authoritative history; it starts with fresh inputs on the first deployment.
