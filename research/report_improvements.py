"""Create the human-readable report from the frozen experiment outputs."""

import csv
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from research.model_improvements import BLOCKS, MODEL, OUT, REPORTS, VARIANTS, dump, summarize


def records(phase, position, variant):
    return json.loads((OUT / "predictions" / f"{phase}-{position}-{variant}.json").read_text())


def comparison(position, variant):
    base = records("audit", position, "baseline")
    candidate = records("audit", position, variant)
    top, error = [], []
    for gw in range(25, 35):
        a, b = [r for r in base if r["gw"] == gw], [r for r in candidate if r["gw"] == gw]
        top.append(np.mean([r["target"] for r in sorted(b, key=lambda r: r["prediction"], reverse=True)[:5]]) -
                   np.mean([r["target"] for r in sorted(a, key=lambda r: r["prediction"], reverse=True)[:5]]))
        error.append(np.mean([abs(r["target"]-r["prediction"]) for r in a]) -
                     np.mean([abs(r["target"]-r["prediction"]) for r in b]))
    # Circular moving blocks preserve some dependence from five-GW overlaps.
    # Ten origins are too few for strong significance claims, even with blocks.
    rng, draws = np.random.default_rng(42), []
    for _ in range(10000):
        index = np.concatenate([(rng.integers(0, 10) + np.arange(5)) % 10 for _ in range(2)])
        draws.append([np.mean(np.array(top)[index]), np.mean(np.array(error)[index])])
    limits = np.quantile(draws, [.025, .975], axis=0)
    return dict(per_origin_top5_change=[float(v) for v in top],
                per_origin_mae_improvement=[float(v) for v in error],
                top5_change_block95=limits[:, 0].tolist(), mae_improvement_block95=limits[:, 1].tolist())


def main():
    value = json.loads((OUT / "all-results.json").read_text())
    selected = value["selected"]
    # A model may not improve its apparent score by losing difficult players.
    for phase in BLOCKS:
        for pos in MODEL:
            expected = [(r["gw"], r["id"], r["target"]) for r in records(phase, pos, "baseline")]
            for variant in VARIANTS:
                assert [(r["gw"], r["id"], r["target"]) for r in records(phase, pos, variant)] == expected
    value["paired_audit"] = {pos: comparison(pos, variant) for pos, variant in selected.items()}
    value["software"] = dict(python=platform.python_version(), **{
        name: importlib.metadata.version(name) for name in ("numpy", "scikit-learn", "scipy", "lightgbm")})
    value["protocol"] = dict(blocks=BLOCKS, horizon=5,
        selection="Max development top5 with no worse MAE and price-band top3 than baseline, per position.",
        primary_metric="Actual five-GW points per pick among the forecast top five at each origin.",
        warning="GW25-34 was inspected in a prior review; this is a reused out-of-time audit, not a fresh test.")
    aggregate = {}
    for name in ("baseline", "selected"):
        rows = [r for pos in MODEL for r in records("audit", pos, "baseline" if name == "baseline" else selected[pos])]
        aggregate[name] = dict(mae=float(np.mean([abs(r["target"]-r["prediction"]) for r in rows])),
            samples=len(rows), equal_position_top5=float(np.mean([
                value["results"]["audit"][pos]["baseline" if name == "baseline" else selected[pos]]["top5_points_per_pick"] for pos in MODEL])))
    value["aggregate_audit"] = aggregate
    dump(REPORTS / "model-improvements-2025-26.json", value)
    with (REPORTS / "model-improvements-predictions.csv").open("w", newline="") as stream:
        fields = ("phase", "position", "variant", "id", "gw", "band", "target", "prediction")
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for phase in BLOCKS:
            for pos in MODEL:
                for variant in dict.fromkeys(("baseline", selected[pos])):
                    for r in records(phase, pos, variant):
                        writer.writerow(dict(phase=phase, position=pos, variant=variant,
                                             **{k: r[k] for k in fields[3:]}))
    lines = ["# FPL model improvement experiments — 2025/26", "",
        "Completed 30 October 2026. Eighteen fixed configurations were tested, covering all five proposed steps. Production forecasts and model_config.json are unchanged.", "",
        "The strongest practical result is fresh availability and set-piece context for defenders. Defensive-contribution rates help midfielders modestly. No forward challenger selected on development improved both later ranking and error. More data and more complex models are not universal improvements.", "",
        "## Evaluation", "",
        "- Development: GW16–20 with training frozen before GW16, then GW21–24 with training frozen before GW21.",
        "- Later audit: GW25–29 with training frozen before GW25, then GW30–34 with training frozen before GW30. Every forecast covers five GWs; the last target ends in GW38.",
        "- Five-GW training labels must mature before each training cutoff and must not overlap the validation block. Fixture-level models use only individually completed fixture labels. Their training cases cover the same one-to-five-GW lead times used at inference; inverse multiplicity weights give each player-fixture one unit of total weight. An initial next-match-only prototype was discarded for a horizon mismatch before the final comparison.",
        "- All models use identical eligible cohorts: at least 180 historical minutes, last-five average minutes ≥50, and at least two recent appearances. Later audit: DEF 788, MID 847, FWD 211 player-origins, 1,846 total.",
        "- Top-five score means actual points per selected player over the following five GWs, averaged across forecast origins. Higher is better. MAE covers every eligible player-origin; lower is better. Price-band evaluation selects three players within fixed position-specific price brackets.",
        "- These are individual candidate forecasts, not a full-season team simulation with budgets, transfers, captaincy and club limits.",
        "- **GW25–34 was already inspected in the previous review. This is a reused chronological audit, not an untouched test.** Choices were frozen using development scores before this run's audit. The current run uses five-GW block refits; the previous report refit every origin, so its numerical baseline differs.", "",
        "## Development-selected recipes on later weeks", "",
        "Selection maximized development top-five points, subject to no worse development MAE and price-band top-three score than baseline. The rule selected the following recipes before the audit:", "",
        "| Position | Frozen recipe | Baseline MAE → candidate | Baseline top-five → candidate | Price-band top-three → candidate |",
        "|---|---|---:|---:|---:|"]
    for pos, variant in selected.items():
        a, b = value["results"]["audit"][pos]["baseline"], value["results"]["audit"][pos][variant]
        lines.append(f'| {pos} | {variant} | {a["mae"]:.3f} → {b["mae"]:.3f} | {a["top5_points_per_pick"]:.2f} → {b["top5_points_per_pick"]:.2f} | {a["price_band_top3_points"]:.2f} → {b["price_band_top3_points"]:.2f} |')
    lines += ["", f'Combined frozen recipes: MAE {aggregate["baseline"]["mae"]:.3f} → {aggregate["selected"]["mae"]:.3f}; equal-position top-five score {aggregate["baseline"]["equal_position_top5"]:.2f} → {aggregate["selected"]["equal_position_top5"]:.2f}. The forward recipe is included despite worse top-five ranking, rather than silently replaced with the baseline.', "",
        "## Findings for each step", "",
        "1. **Features:** tested defensive-contribution hit rates over five/ten matches; team xG for, upcoming opponents' xG against and player's share of team xG; days since a PL appearance and minutes trend; and all seven together. MID defcon improved later top-five points 19.46 → 20.06, with MAE 6.576 → 6.569. The full feature bundle looked better for DEF in development but fell from 21.94 → 20.94 on later weeks. Additions need separate validation; more features did not reliably help.", "",
        "2. **Candidate and price alignment:** tested eligible-only training, 3:1 weights favouring eligible players, inverse price-band weights, and feature/weight combinations. Eligible-only training reduced later top-five scores for every position. Price balancing also reduced top-five scores for every position and increased MAE. A MID weighted variant gained ranking points but worsened error. Keep broad training data and retain price-band reporting; do not apply one filtering rule to every position.", "",
        "3. **More seasons:** downloaded 2023/24 and 2024/25 at the same pinned revision. Extra seasons for the minutes classifier improved DEF relative to the single-season two-stage model (top-five 20.14 → 21.10; MAE 6.946 → 6.704), hurt MID rankings, and left FWD ranking flat while improving its MAE (6.876 → 6.739). Pooling common scoring components improved FWD versus its single-season component control (19.60 → 20.78; MAE 6.841 → 6.772), also beating the current FWD baseline on these later weeks. It reduced DEF error but hurt DEF/MID rankings. The FWD pooled-component variant missed the development price-band guardrail, so it was not the frozen primary choice. Older total_points were never pooled as though scoring rules were identical.", "",
        "4. **Architecture:** tested LightGBM L2, L1 and LambdaRank, plus a per-fixture minutes classifier (0 / 1–59 / 60+ minutes) with conditional point heads, summed over five GWs. A separate component model predicts the common scoring subtotal plus a 2025/26 residual for assists/bonus/defcon. The horizon-matched MID two-stage model reached 23.90 top-five points with MAE 6.364; the MID single-season component variant reached 25.22 with MAE 6.297, versus baseline 19.46 and 6.576 (+29.6% ranking, −4.2% error). These models were weaker on development top-five selection and were not the frozen primary MID choice. They are promising exploratory findings, not independent confirmation. Simple architecture changes do not reliably improve every position. LambdaRank emits ranking scores, so its MAE is intentionally omitted.", "",
        "5. **Pre-deadline context:** the main archive's 12 bootstrap revisions were too sparse. Git history from olbauday/FPL-Elo-Insights supplied 31 additional states before GW4–34 cutoffs. Historical status, chance of playing, penalties and corner roles were read only from the latest compatible state before the cutoff, at most eight days old, with explicit missingness. All eligible GW16–34 observations had fresh context; selected snapshots were approximately 1.5–10.1 hours before the deadline proxy. DEF top-five score improved 21.94 → 23.18 (+5.7%), MAE fell 6.820 → 6.607 (−3.1%), and price-band ranking improved. MID MAE improved but ranking worsened; FWD did not improve.", "",
        "## What to take forward", "",
        "Prioritize a defender availability/role pilot: it improved error, overall top-five selection and price-band selection after also passing development checks. Keep the small MID defensive-contribution variant as a simple reference. Prospectively compare the MID per-fixture component model and FWD multi-season component model against both these references and the existing baseline. Their attractive later-period results were discovered during this audit. Reject blanket candidate filtering, inverse price weighting and a universal LightGBM/ranker replacement. The website's existing model remains unchanged while these forecasts accumulate outcomes.", "",
        "## Data validation and scoring changes", "",
        "All three seasons contain 380 completed fixtures and 38 GWs. 2023/24 has 29,725 player-fixture rows; 2024/25 has 27,605; 2025/26 has 29,747 after removing ten identical duplicate rows. The 2025/26 point totals reconcile for all players. One 2024/25 record differs: Evan Ferguson's bootstrap total is 28 but match rows sum to 27. His records were quarantined from auxiliary training. No labels were repaired by guesswork.", "",
        "Defensive contributions are absent from the older seasons. They are excluded from the common cross-season models, rather than used as observed zeroes. The common subtotal covers appearance, goals, clean sheets, goal-conceded deductions and disciplinary/penalty deductions; rule-sensitive assists, bonus/BPS and defcon are handled by a current-era residual. The minutes classifier uses scoring-independent features. Goalkeepers and 2024/25 assistant managers are outside this study.", "",
        "Historical fixtures still use final event assignments and first-kickoff-minus-90-minutes deadline proxies. Exact historical schedule revisions were not reconstructed, so blank/double-week fixture information may be optimistic. Historical player positions are season-level positions. Days since appearance covers Premier League matches, not cup/international workload. Commit time is the archive publication proxy, not a verified API collection timestamp.", "",
        "## How certain are the gains?", "",
        "There are only ten later forecast origins and their five-GW outcomes overlap. The defender top-five gain is concentrated in a few origins. Circular block bootstrap intervals below use blocks of five origins and 10,000 resamples; with only about two blocks of information, treat them as descriptive uncertainty, not proof of significance.", "",
        "| Frozen recipe | Mean top-five change | Approx. 95% block interval | MAE improvement interval |",
        "|---|---:|---:|---:|"]
    for pos, variant in selected.items():
        p = value["paired_audit"][pos]
        lo, hi = p["top5_change_block95"]
        elo, ehi = p["mae_improvement_block95"]
        lines.append(f'| {pos} {variant} | {np.mean(p["per_origin_top5_change"]):+.2f} | [{lo:+.2f}, {hi:+.2f}] | [{elo:+.3f}, {ehi:+.3f}] |')
    lines += ["", "The intervals for top-five gains all include zero. The defender error reduction is the most consistent signal among the development-selected recipes. A new prospective ledger freezes baseline and development-selected forecasts for 2026/27 GW9–13 before the GW9 deadline; it also includes the MID single-season component and FWD multi-season component challengers, explicitly labelled as selected after inspecting the reused audit. Their future outcomes are not yet available. It records current availability, fixture assignments and raw-input hashes. 2026/27 scoring differs, so that is a separate prospective test. The ledger is a one-time snapshot; no new scheduler or live model deployment was added.", "",
        "## Complete results", "", "Each cell is **top-five points / MAE**. A dash denotes an uncalibrated ranking score."]
    for phase in BLOCKS:
        lines += ["", f"### {phase.capitalize()}", "", "| Configuration | DEF | MID | FWD |", "|---|---:|---:|---:|"]
        for variant in VARIANTS:
            cells = []
            for pos in MODEL:
                m = value["results"][phase][pos][variant]
                cells.append(f'{m["top5_points_per_pick"]:.2f} / {m["mae"]:.3f}' if "mae" in m else f'{m["top5_points_per_pick"]:.2f} / —')
            lines.append("| " + " | ".join([variant] + cells) + " |")
    lines += ["", "## Reproduction and artifacts", "", "```sh", ".venv/bin/python research/model_improvements.py", ".venv/bin/python research/report_improvements.py", ".venv/bin/python -m unittest discover -s tests", "```", "",
        "The optional `research/freeze_forecast.py` fetches current official context and writes a new ledger with an exclusive timestamped filename. It never edits the website's data or model configuration. Run it before a deadline; it does not evaluate future outcomes.", "",
        "- [Structured results](model-improvements-2025-26.json): all metrics, development selection, software versions and paired uncertainty.",
        "- [Pinned source manifests](model-improvements-sources.json): archive revisions, snapshot commit IDs, hashes and freshness coverage.",
        "- [Baseline and frozen-recipe predictions](model-improvements-predictions.csv): individual development/audit predictions. All challenger predictions remain reproducible in the ignored experiment cache.",
        "- [FPL archive](https://github.com/vaastav/Fantasy-Premier-League/tree/d7b05b51d137bfefef4b692386d18c610975b9b9/data).",
        "- [Availability archive](https://github.com/olbauday/FPL-Elo-Insights): exact historical commits are recorded in the source manifest.", "",
        "Fixed model parameters: existing Ridge(alpha=100)/RandomForest(120 trees, depth 8, leaf 20); LightGBM 180 trees, 7 leaves, learning rate .04, leaf minimum 30, L2 regularization 20, seed 42. No audit-driven hyperparameter search was performed.", ""]
    path = REPORTS / "model-improvements-2025-26.md"
    path.write_text("\n".join(lines))
    print(path)


if __name__ == "__main__":
    main()
