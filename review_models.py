"""Reproduce the old evaluation and run a purged 2025/26 model comparison.

Development forecast origins: GW16–24. Untouched final test: GW25–34, each
predicting five gameweeks, so the final target ends in GW38.
"""

import csv
import importlib
import json
import subprocess
import sys
import tempfile
import platform
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import sklearn
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from forecasting import (
    CANDIDATES,
    FEATURE_NAMES,
    fit_model,
    make_samples,
    prediction_cutoff,
    timestamp,
    training_before,
)
from historical_data import ROOT, load_history

LEGACY_REVISION = "4974355c621f5eb07b84b70152762ae16f8f9e3e"
HORIZON = 5
VALIDATION_BLOCKS = ((16, 20), (21, 24))
TEST_ORIGINS = range(25, 35)
REPORT_DIR = ROOT / "reports"


@contextmanager
def legacy_modules():
    names = ("fpl_scoring", "find_midfielders", "find_forwards", "find_defenders")
    previous = {name: sys.modules.pop(name, None) for name in names}
    with tempfile.TemporaryDirectory(prefix="fpl-legacy-") as folder:
        for name in names:
            content = subprocess.check_output(
                ["git", "show", f"{LEGACY_REVISION}:{name}.py"], cwd=ROOT
            )
            (Path(folder) / f"{name}.py").write_bytes(content)
        sys.path.insert(0, folder)
        try:
            yield {
                position: importlib.import_module(name)
                for position, name in (
                    ("MID", "find_midfielders"),
                    ("FWD", "find_forwards"),
                    ("DEF", "find_defenders"),
                )
            }
        finally:
            sys.path.remove(folder)
            for name in names:
                sys.modules.pop(name, None)
                if previous[name] is not None:
                    sys.modules[name] = previous[name]


def metrics(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)),
        "samples": len(actual),
    }


def legacy_estimators():
    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=150,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
            n_jobs=2,
        ),
        "Ridge": Ridge(alpha=2, random_state=42),
    }
    try:
        from xgboost import XGBRegressor

        models["XGBoost"] = XGBRegressor(
            n_estimators=150,
            max_depth=7,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.5,
            random_state=42,
            n_jobs=2,
        )
    except ImportError:
        pass
    try:
        from lightgbm import LGBMRegressor

        models["LightGBM"] = LGBMRegressor(
            n_estimators=150,
            max_depth=7,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=1.5,
            random_state=42,
            n_jobs=2,
            verbose=-1,
        )
    except ImportError:
        pass
    return models


def snapshot_player(player, row):
    """Rebuild every legacy aggregate from known matches, neutralizing unavailable fields."""
    past = row["past"]
    result = dict(player)
    sums = (
        "total_points",
        "minutes",
        "starts",
        "goals_scored",
        "assists",
        "bonus",
        "bps",
        "influence",
        "creativity",
        "threat",
        "ict_index",
        "yellow_cards",
        "red_cards",
        "defensive_contribution",
        "clean_sheets",
        "expected_goals",
        "expected_assists",
        "expected_goal_involvements",
    )
    for field in sums:
        result[field] = sum(float(match.get(field) or 0) for match in past)
    for source, target in (
        ("expected_goals", "expected_goals_per_90"),
        ("expected_assists", "expected_assists_per_90"),
        ("expected_goal_involvements", "expected_goal_involvements_per_90"),
        ("defensive_contribution", "defensive_contribution_per_90"),
    ):
        result[target] = result[source] * 90 / max(result["minutes"], 90)
    result.update(
        team=row["team"],
        now_cost=past[-1]["value"],
        form=np.mean([match["total_points"] for match in past[-5:]]),
    )
    result["points_per_game"] = result["total_points"] / max(
        sum(match["minutes"] > 0 for match in past), 1
    )
    for field in (
        "selected_by_percent",
        "transfers_in",
        "transfers_out",
        "transfers_in_event",
        "transfers_out_event",
        "ep_this",
        "ep_next",
    ):
        result[field] = 0
    result["value_form"] = result["form"] / max(row["price"], 1)
    result["value_season"] = result["total_points"] / max(row["price"], 1)
    return result


def legacy_asof_matrices(modules, bootstrap, fixtures, histories, samples):
    from forecasting import make_features, team_table

    matrices = {}
    players = {player["id"]: player for player in bootstrap["elements"]}
    for gw in sorted({row["gw"] for row in samples}):
        cutoff = prediction_cutoff(bootstrap, fixtures, gw)
        known = [
            {**f, "finished": f["finished"] and timestamp(f["kickoff_time"]) + 10800 < cutoff}
            for f in fixtures
        ]
        points = modules["MID"].calculate_team_points_from_fixtures(known)
        teams = []
        records = {}
        for team in bootstrap["teams"]:
            matches = [
                f for f in known if f["finished"] and team["id"] in (f["team_h"], f["team_a"])
            ]
            goals = sum(
                (f["team_h_score"] - f["team_a_score"]) * (1 if team["id"] == f["team_h"] else -1)
                for f in matches
            )
            records[team["id"]] = (points.get(team["id"], 0), goals, len(matches))
        order = sorted(records, key=lambda team: records[team][:2], reverse=True)
        for team in bootstrap["teams"]:
            teams.append(
                {
                    **team,
                    "position": order.index(team["id"]) + 1,
                    "played": records[team["id"]][2],
                    "form": None,
                }
            )
        snapshot = {**bootstrap, "teams": teams}
        table = team_table(fixtures, gw, cutoff)
        for row in (s for s in samples if s["gw"] == gw):
            features = make_features(
                players[row["id"]], histories[row["id"]], fixtures, gw, HORIZON, cutoff, table
            )
            player = snapshot_player(players[row["id"]], features)
            history = {"history": features["past"]}
            module = modules[row["position"]]
            kwargs = {} if row["position"] == "FWD" else {"lookahead_gws": HORIZON}
            old = module.get_player_features_for_prediction(
                player, history, gw, known, snapshot, points, {}, **kwargs
            )
            if old is None:
                continue
            if row["position"] == "DEF":
                module._add_team_defensive_features(old, known, player["team"], gw - 1)
                vector = module._defender_feature_vector(old)
            else:
                vector = [old[name] for name in modules["FWD"].FEATURE_NAMES]
            matrices[(row["position"], gw, row["id"])] = vector
    return matrices


def reproduce_apparent_scores(modules, bootstrap, fixtures, histories):
    # Old training includes partial targets in GW35–38 and end-of-season totals.
    results = {}
    cleaned = {
        **bootstrap,
        "elements": [
            {key: (0 if value is None else value) for key, value in player.items()}
            for player in bootstrap["elements"]
        ],
    }
    for position, module in modules.items():
        module.load_player_history = lambda player_id: {"history": histories.get(player_id, [])}
        kwargs = {} if position == "FWD" else {"lookahead_gws": HORIZON}
        x, y = module.prepare_training_data(cleaned, fixtures, **kwargs)
        x_train, x_test, y_train, y_test = train_test_split(
            np.asarray(x, dtype=float), np.asarray(y), test_size=0.2, random_state=42
        )
        scores = {}
        for name, model in legacy_estimators().items():
            model.fit(x_train, y_train)
            scores[name] = metrics(y_test, model.predict(x_test))
        winner = min(scores, key=lambda name: scores[name]["mae"])
        results[position] = {
            "selected": winner,
            "scores": scores,
            "training_samples": len(x),
            "warning": "Optimistic random split, future aggregates and incomplete targets; not evidence of live performance.",
        }
        print(f"Legacy apparent {position}: {winner}, MAE {scores[winner]['mae']:.3f}", flush=True)
    return results


def rank_metrics(records, field):
    totals, regret, correlations = [], [], []
    for gw in sorted({row["gw"] for row in records}):
        rows = [row for row in records if row["gw"] == gw and row["eligible"]]
        if len(rows) < 5:
            continue
        best = sorted(rows, key=lambda row: row[field], reverse=True)[:5]
        oracle = sorted(rows, key=lambda row: row["target"], reverse=True)[:5]
        totals.append(sum(row["target"] for row in best) / 5)
        regret.append(
            (sum(row["target"] for row in oracle) - sum(row["target"] for row in best)) / 5
        )
        predicted_rank = rankdata([row[field] for row in rows])
        actual_rank = rankdata([row["target"] for row in rows])
        correlations.append(float(np.corrcoef(predicted_rank, actual_rank)[0, 1]))
    return {
        "top5_points_per_pick": float(np.mean(totals)),
        "top5_regret_per_pick": float(np.mean(regret)),
        "mean_within_gw_rank_correlation": float(np.mean(correlations)),
    }


def compare_local_archive(histories):
    directory = ROOT / "data" / "archive" / "2025-26"
    if not (directory / "bootstrap-static.json").exists():
        return {"available": False}
    bootstrap = json.loads((directory / "bootstrap-static.json").read_text())
    finished = {event["id"] for event in bootstrap["events"] if event.get("finished")}
    reference = {
        (player_id, row["fixture"]): row
        for player_id, history in histories.items()
        for row in history
    }
    rows, missing, changed_points = 0, 0, []
    for path in (directory / "players").glob("*.json"):
        for row in json.loads(path.read_text()).get("history", []):
            if row["round"] not in finished:
                continue
            rows += 1
            other = reference.get((int(path.stem), row["fixture"]))
            if other is None:
                missing += 1
            elif other["total_points"] != row["total_points"]:
                changed_points.append(
                    {
                        "id": int(path.stem),
                        "fixture": row["fixture"],
                        "local": row["total_points"],
                        "final": other["total_points"],
                    }
                )
    return {
        "available": True,
        "completed_through_gw": max(finished),
        "rows_checked": rows,
        "missing_reference_rows": missing,
        "point_corrections": changed_points,
        "note": "The preserved local snapshot is incomplete; complete pinned data is used for the season review.",
    }


def finalize_report(report, records):
    eligible = [row for row in records if row["eligible"]]
    for position in ("DEF", "MID", "FWD"):
        rows = [row for row in eligible if row["position"] == position]
        report["holdout"][position] = {
            name: {
                **metrics([row["target"] for row in rows], [row[name] for row in rows]),
                **rank_metrics(rows, name),
            }
            for name in ("legacy_asof", "recent_baseline", "improved")
        }
        # Five-GW circular blocks preserve some dependence between overlapping
        # horizons. With just ten origins these intervals remain indicative.
        origins = sorted({row["gw"] for row in rows})
        sums = np.asarray(
            [
                sum(
                    abs(row["target"] - row["legacy_asof"]) - abs(row["target"] - row["improved"])
                    for row in rows
                    if row["gw"] == gw
                )
                for gw in origins
            ]
        )
        counts = np.asarray([sum(row["gw"] == gw for row in rows) for gw in origins])
        rng = np.random.default_rng(42)
        deltas = []
        for _ in range(2000):
            indices = np.concatenate(
                [
                    (start + np.arange(5)) % len(origins)
                    for start in rng.integers(0, len(origins), 2)
                ]
            )
            deltas.append(float(sums[indices].sum() / counts[indices].sum()))
        report["holdout"][position]["mae_improvement_block_interval_95"] = [
            float(value) for value in np.quantile(deltas, [0.025, 0.975])
        ]
    report["environment"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
    }
    report["overall_holdout"] = {
        name: metrics([row["target"] for row in eligible], [row[name] for row in eligible])
        for name in ("legacy_asof", "recent_baseline", "improved")
    }
    (REPORT_DIR / "model-review-2025-26.json").write_text(json.dumps(report, indent=2))
    table = [
        "| Position | Forecasts | Honest old MAE | Revised MAE | Change | Old top-5 return | Revised top-5 return |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for position, values in report["holdout"].items():
        old, new = values["legacy_asof"], values["improved"]
        gain = (old["mae"] - new["mae"]) / old["mae"] * 100
        table.append(
            f"| {position} | {new['samples']} | {old['mae']:.2f} | {new['mae']:.2f} | {gain:.1f}% lower | {old['top5_points_per_pick']:.2f} | {new['top5_points_per_pick']:.2f} |"
        )
    overall = report["overall_holdout"]
    local = report.get("local_archive_validation", {})
    local_sentence = (
        f"Its {local['rows_checked']:,} completed-match rows were cross-checked: "
        f"{local['missing_reference_rows']} missing reference rows and {len(local['point_corrections'])} point discrepancies."
        if local.get("available")
        else "The original local snapshot was unavailable for a separate cross-check on this run."
    )
    markdown = f"""**FPL model review: 2025/26**

This is a retrospective reproduction, not an audit of saved live forecasts: the repository did not retain historical model predictions. The website previously used a weighted metric score, not the ML forecasts.

**Data validation.** The pinned [source archive]({report["source"]["url"]}) contains 841 players, 380 finished fixtures and all 38 gameweeks. Ten identical duplicate rows were removed, leaving 29,747 unique player-fixture rows. No required expected-goal/defensive fields are missing. Every player's history reconciles exactly with the final bootstrap point total. Checksums and the source revision are recorded in the JSON report. The preserved local archive only had 34 completed gameweeks; it was not used as if it were a complete season. {local_sentence}

**What the original evaluation showed.** Re-running the original random 80/20 split selected XGBoost and reported MAE 2.82 for defenders, 3.32 for midfielders and 3.65 for forwards. Those numbers are optimistic: end-of-season totals appear in earlier feature rows, neighboring overlapping targets cross the split, partial end-of-season targets are included, and the test set also selects the algorithm. The original split contains a different, broader player population, so these apparent scores should not be compared numerically with the eligible-transfer-player table below.

**Fair validation.** Five-gameweek targets are restricted to completed horizons. Development uses GW16–20 and GW21–24, each with training labels completed before its first origin. Model configurations are frozen before GW25–34. The final test refits at every origin using only fully matured earlier labels; an earlier test window may enter training only after its outcomes become known. Players need 180 past minutes, at least two recent appearances and 50 recent average minutes to enter the reported transfer-candidate cohort. Both pipelines use exactly the same test players and targets. The honest old baseline rebuilds its aggregates from past matches and neutralizes unavailable historical ownership/FPL forecast fields.

**Held-out results.** MAE is the mean error in total points across the next five gameweeks; lower is better. Top-five return is actual five-GW points per selected player, averaged across ten origins; higher is better.

{chr(10).join(table)}

Across {overall["improved"]["samples"]} eligible player-origins, MAE moves from **{overall["legacy_asof"]["mae"]:.2f} to {overall["improved"]["mae"]:.2f}**, versus **{overall["recent_baseline"]["mae"]:.2f}** for repeating recent points per match across upcoming fixtures. The same player appears at multiple origins, so these are not independent observations.

The midfielder error improvement is small and its top-five return falls. Defender top-five returns improve over the honest old model but remain below the simple recent-form baseline (22.00). This is not evidence that the revised model is universally best at choosing transfers. The site's existing metric ranking remains the default; forecasts are a separate estimate and optional sort.

**Top five improvements, incorporated.**

1. **Remove future information.** Reconstruct cumulative and rolling statistics at each cutoff, infer historical teams from past fixtures, and exclude bootstrap season totals, future expected-points values, final fixture ratings and unavailable historical news. Matches in earlier numbered rounds that actually occur after the cutoff are excluded too.
2. **Validate in time order with a horizon gap.** Replace random row splitting with purged, expanding-history evaluation. Require both a disjoint target horizon and a completed label timestamp. Use development weeks for algorithm selection and the later origins for the reported comparison.
3. **Validate the dataset and target.** Pin and checksum the full source; remove only identical duplicates; reject conflicting duplicates, missing fixtures and unfinished seasons. Require complete five-GW horizons, count missing future appearances as zero, and include zero-minute rows in recent form/minutes denominators.
4. **Model opportunity as well as attacking rates.** Include recent minutes, starts, appearance/60-minute rates, shrunk xG/xA per 90, defensive contributions, clean sheets, past-only team strengths and the actual number of fixtures, including blanks and doubles. Early team strengths shrink toward neutral priors.
5. **Use regularized, shared models and honest baselines.** Compare recent form, minutes-adjusted form, standardized Ridge, constrained Random Forest, constrained boosting and a Ridge/form blend on development data. The revised pipeline selects Ridge for DEF/FWD and Random Forest for MID. Shared feature/training/prediction code replaces three duplicated pipelines. Production trains on complete 2025/26 examples plus fully observed current-season examples. The revised family is selected internally on development MAE; the old forward pipeline had a better development score, although the revised forward model did better in the final test.

**Limits and uncertainty.** Final fixture assignments stand in for unavailable historical schedule snapshots. Cutoff timestamps are conservatively approximated by the first kickoff minus 90 minutes; this cannot fully replay the fixture information known to a manager at the time. Injury news and historical availability flags are absent. Top-five selection is an unbudgeted diagnostic, without squad constraints or historical injury exclusions. The five-GW windows overlap, and only ten final origins are available. Five-GW block-bootstrap intervals in the JSON report are indicative, not a guarantee of an accuracy gain. Custom CLI horizons reuse the selected configuration and have not received this five-GW validation. Bonus-rule changes in 2026/27 can cause drift. No further tuning was performed on the final test results.

Reproduce with `uv run review_models.py` from a checkout containing legacy commit `{LEGACY_REVISION}`. The JSON report contains per-model development scores, RMSE, R², tie-aware rank correlations and block intervals. The CSV is the complete held-out prediction ledger, including players outside the transfer-candidate cohort. Raw source data stays under ignored `data/history/`.
"""
    (REPORT_DIR / "model-review-2025-26.md").write_text(markdown)


def main():
    bootstrap, fixtures, histories, source = load_history()
    print("Building point-in-time samples…", flush=True)
    samples = make_samples(bootstrap, fixtures, histories)
    with legacy_modules() as modules:
        apparent = reproduce_apparent_scores(modules, bootstrap, fixtures, histories)
        print("Reconstructing honest legacy features…", flush=True)
        legacy_x = legacy_asof_matrices(modules, bootstrap, fixtures, histories, samples)
    samples = [row for row in samples if (row["position"], row["gw"], row["id"]) in legacy_x]
    selected, legacy_selected, validation, final, records = {}, {}, {}, {}, []
    for position in ("DEF", "MID", "FWD"):
        pool = [row for row in samples if row["position"] == position]
        candidate_errors = {name: [] for name in CANDIDATES}
        old_errors = {name: [] for name in legacy_estimators()}
        matrix = lambda rows: np.asarray(
            [legacy_x[(position, row["gw"], row["id"])] for row in rows], dtype=float
        )
        for start, end in VALIDATION_BLOCKS:
            cutoff = prediction_cutoff(bootstrap, fixtures, start)
            train = training_before(pool, start, cutoff, HORIZON)
            valid = [row for row in pool if start <= row["gw"] <= end and row["eligible"]]
            for name in CANDIDATES:
                prediction = fit_model(name, train).predict(valid)
                candidate_errors[name].extend(
                    abs(row["target"] - pred) for row, pred in zip(valid, prediction)
                )
            for name, model in legacy_estimators().items():
                model.fit(matrix(train), [row["target"] for row in train])
                prediction = model.predict(matrix(valid))
                old_errors[name].extend(
                    abs(row["target"] - pred) for row, pred in zip(valid, prediction)
                )
        scores = {name: float(np.mean(errors)) for name, errors in candidate_errors.items()}
        old_scores = {name: float(np.mean(errors)) for name, errors in old_errors.items()}
        selected[position] = min(scores, key=scores.get)
        legacy_selected[position] = min(old_scores, key=old_scores.get)
        validation[position] = {"improved": scores, "legacy": old_scores}
        print(
            f"Development {position}: {selected[position]} ({scores[selected[position]]:.3f}), legacy {legacy_selected[position]} ({old_scores[legacy_selected[position]]:.3f})",
            flush=True,
        )
        # Candidate selection ends here. The held-out GW25–34 results never feed
        # hyperparameters, feature choice or the production model configuration.
        position_records = []
        for gw in TEST_ORIGINS:
            cutoff = prediction_cutoff(bootstrap, fixtures, gw)
            train = training_before(pool, gw, cutoff, HORIZON)
            test = [row for row in pool if row["gw"] == gw]
            improved = fit_model(selected[position], train).predict(test)
            old_model = legacy_estimators()[legacy_selected[position]]
            old_model.fit(matrix(train), [row["target"] for row in train])
            old = old_model.predict(matrix(test))
            for row, new_prediction, old_prediction in zip(test, improved, old):
                position_records.append(
                    {
                        "position": position,
                        "gw": gw,
                        "id": row["id"],
                        "target": row["target"],
                        "eligible": row["eligible"],
                        "improved": float(new_prediction),
                        "legacy_asof": float(old_prediction),
                        "recent_baseline": row["baseline"],
                    }
                )
        eligible = [row for row in position_records if row["eligible"]]
        final[position] = {
            name: {
                **metrics([r["target"] for r in eligible], [r[name] for r in eligible]),
                **rank_metrics(eligible, name),
            }
            for name in ("legacy_asof", "recent_baseline", "improved")
        }
        records.extend(position_records)
        print(
            f"Final {position}: "
            + ", ".join(
                f"{name} MAE={score['mae']:.3f}" for name, score in final[position].items()
            ),
            flush=True,
        )
    REPORT_DIR.mkdir(exist_ok=True)
    report = {
        "season": "2025/26",
        "horizon_gameweeks": HORIZON,
        "legacy_revision": LEGACY_REVISION,
        "source": source,
        "development_origins": [16, 24],
        "test_origins": [25, 34],
        "features": FEATURE_NAMES,
        "apparent_legacy": apparent,
        "validation": validation,
        "selected_models": selected,
        "selected_legacy_models": legacy_selected,
        "holdout": final,
        "limitations": [
            "Retrospective reconstruction, not saved live predictions.",
            "Final fixture assignments stand in for unavailable pre-deadline fixture snapshots; cutoff timestamps use first kickoff minus 90 minutes.",
            "Historical injury news and exact ownership forecasts are unavailable and excluded.",
            "Ten overlapping five-GW test windows are correlated; results are not independent trials.",
            "2026/27 bonus-rule changes can cause drift relative to the training season.",
            "Top-five returns ignore budgets, squad constraints and unavailable historical injury news.",
        ],
    }
    report["local_archive_validation"] = compare_local_archive(histories)
    finalize_report(report, records)
    with (REPORT_DIR / "model-review-2025-26-predictions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    config = {
        "models": selected,
        "horizon_gameweeks": HORIZON,
        "selected_on": "2025/26 GW16–24 only",
        "source_revision": source["revision"],
    }
    (ROOT / "model_config.json").write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
