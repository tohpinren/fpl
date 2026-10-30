"""Research-only ablations; does not alter the production model or configuration.

Run with .venv/bin/python research/model_improvements.py. See the generated
protocol and report for the limits of the reused historical audit.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import csv
import hashlib
import io
import json
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import requests
from lightgbm import LGBMClassifier, LGBMRanker, LGBMRegressor

from forecasting import (
    FEATURE_NAMES, POSITIONS, estimator, fit_model, make_features, make_samples,
    prediction_cutoff, team_table, timestamp, training_before,
)
from historical_data import FILES, SOURCE_REPOSITORY, SOURCE_REVISION, load_history, parse_value
from review_models import metrics, rank_metrics

OUT = ROOT / "data/experiments/2026-10-30"
REPORTS = ROOT / "reports"
SEASONS = ("2023-24", "2024-25", "2025-26")
MODEL = {"DEF": "ridge", "MID": "forest", "FWD": "ridge"}
PARAMS = dict(n_estimators=180, num_leaves=7, learning_rate=.04,
              min_child_samples=30, reg_lambda=20, verbosity=-1,
              random_state=42, n_jobs=2)
EXTRA = ("defcon_hit5", "defcon_hit10", "team_xgf5", "opponent_xga5",
         "player_xg_share5", "days_since_appearance", "minutes_trend3v10")
BUNDLES = {"defcon": [0, 1], "xg": [2, 3, 4], "minutes": [5, 6], "all": list(range(7))}
VARIANTS = ("baseline", "defcon", "xg", "minutes", "all", "eligible", "weighted",
            "price_balanced", "all_eligible", "all_weighted", "lgb_l2", "lgb_l1",
            "ranker", "snapshots", "two_stage", "two_stage_more_minutes",
            "components_single", "components_multi")
BLOCKS = {"development": [(16, 20), (21, 24)], "audit": [(25, 29), (30, 34)]}


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def read_csv(content):
    return [{k: parse_value(v) for k, v in r.items()}
            for r in csv.DictReader(io.StringIO(content.decode("utf-8-sig")))]


def archive(season):
    if season == "2025-26":
        return load_history()
    folder = ROOT / "data/history" / season
    folder.mkdir(parents=True, exist_ok=True)

    def get(name):
        path = folder / name
        if not path.exists():
            url = (f"https://raw.githubusercontent.com/{SOURCE_REPOSITORY}/"
                   f"{SOURCE_REVISION}/data/{season}/{name}")
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(response.content)
        content = path.read_bytes()
        return name, read_csv(content), hashlib.sha256(content).hexdigest()

    with ThreadPoolExecutor(max_workers=4) as pool:
        tables = list(pool.map(get, FILES))
    checksums = {name: sha for name, _, sha in tables}
    manifest_path = folder / "source.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["sha256"] != checksums or manifest["revision"] != SOURCE_REVISION:
            raise ValueError(f"Changed archive: {season}")
    tables = {name: table for name, table, _ in tables}
    fixtures = tables["fixtures.csv"]
    histories, unique = defaultdict(list), {}
    removed = 0
    for r in tables["gws/merged_gw.csv"]:
        key = r["element"], r["fixture"]
        if key in unique:
            if unique[key] != r:
                raise ValueError(f"Conflicting duplicate: {season} {key}")
            removed += 1
            continue
        unique[key] = r
        histories[r["element"]].append(r)
    for history in histories.values():
        history.sort(key=lambda r: (r["kickoff_time"], r["fixture"]))
    assert len(fixtures) == len({f["id"] for f in fixtures}) == 380
    assert all(f["finished"] for f in fixtures)
    events = []
    for gw in range(1, 39):
        games = [f for f in fixtures if f["event"] == gw]
        deadline = datetime.fromisoformat(min(f["kickoff_time"] for f in games).replace("Z", "+00:00")) - timedelta(minutes=90)
        events.append(dict(id=gw, finished=True, deadline_time=deadline.isoformat()))
    players = tables["players_raw.csv"]
    mismatches = [p["id"] for p in players if p["total_points"] != sum(r["total_points"] for r in histories[p["id"]])]
    # A one-point 2024/25 Ferguson discrepancy exists upstream. Quarantine any
    # unreconciled players from auxiliary training rather than inventing a label.
    quarantined = [{"id": p["id"], "name": p["web_name"], "total": p["total_points"],
                    "history": sum(r["total_points"] for r in histories[p["id"]])}
                   for p in players if p["id"] in mismatches]
    players = [p for p in players if p["id"] not in mismatches]
    audit = dict(players=len(players), fixtures=380, player_fixture_rows=len(unique),
                 duplicate_rows_removed=removed, quarantined_players=quarantined,
                 missing_defcon_rows=sum(r.get("defensive_contribution") is None for r in unique.values()))
    source = dict(revision=SOURCE_REVISION, sha256=checksums, audit=audit)
    dump(manifest_path, source)
    return dict(elements=players, teams=tables["teams.csv"], events=events), fixtures, dict(histories), source


def stable_points(row, position):
    """Scoring subtotal shared by these seasons; assists/bonus/defcon excluded."""
    minutes = float(row.get("minutes") or 0)
    appearance = 2 if minutes >= 60 else int(minutes > 0)
    goals = float(row.get("goals_scored") or 0) * {"DEF": 6, "MID": 5, "FWD": 4}[position]
    clean = float(row.get("clean_sheets") or 0) * {"DEF": 4, "MID": 1, "FWD": 0}[position]
    conceded = int(float(row.get("goals_conceded") or 0) // 2) if position == "DEF" else 0
    penalties = sum(float(row.get(k) or 0) * weight for k, weight in (
        ("yellow_cards", 1), ("red_cards", 3), ("own_goals", 2), ("penalties_missed", 2)))
    return appearance + goals + clean - conceded - penalties


def extras(bootstrap, fixtures, histories, samples):
    fixture_map = {f["id"]: f for f in fixtures}
    xg = defaultdict(lambda: defaultdict(float))
    for history in histories.values():
        for r in history:
            f = fixture_map[r["fixture"]]
            xg[f["id"]][f["team_h"] if r["was_home"] else f["team_a"]] += float(r.get("expected_goals") or 0)
    tables = {}
    for gw in {r["gw"] for r in samples}:
        cutoff, table = prediction_cutoff(bootstrap, fixtures, gw), {}
        for t in bootstrap["teams"]:
            team = t["id"]
            games = sorted([f for f in fixtures if f["event"] < gw and
                            timestamp(f["kickoff_time"]) + 10800 < cutoff and
                            team in (f["team_h"], f["team_a"])], key=lambda f: f["kickoff_time"])[-5:]
            gf = sum(xg[f["id"]][team] for f in games)
            ga = sum(xg[f["id"]][f["team_a"] if team == f["team_h"] else f["team_h"]] for f in games)
            table[team] = ((gf + 2.6) / (len(games) + 2), (ga + 2.6) / (len(games) + 2), gf)
        tables[gw] = table
    result = []
    for s in samples:
        past = [r for r in histories[s["id"]] if r["round"] < s["gw"] and timestamp(r["kickoff_time"]) + 10800 < s["cutoff"]]
        threshold = 10 if s["position"] == "DEF" else 12
        hit = lambda rs: sum(float(r.get("defensive_contribution") or 0) >= threshold for r in rs) / max(len(rs), 1)
        upcoming = [f for f in fixtures if s["gw"] <= f["event"] < s["gw"] + 5 and s["team"] in (f["team_h"], f["team_a"])]
        opponent_xga = np.mean([tables[s["gw"]][f["team_a"] if s["team"] == f["team_h"] else f["team_h"]][1] for f in upcoming]) if upcoming else 1.3
        played = [r for r in past if r["minutes"] > 0]
        days = min(60, (s["cutoff"] - timestamp(played[-1]["kickoff_time"])) / 86400) if played else 60
        team = tables[s["gw"]][s["team"]]
        values = [hit(past[-5:]), hit(past[-10:]), team[0], float(opponent_xga),
                  sum(float(r.get("expected_goals") or 0) for r in past[-5:]) / max(team[2], .1),
                  days, float(np.mean([r["minutes"] for r in past[-3:]]) - np.mean([r["minutes"] for r in past[-10:]]))]
        result.append({**s, "extra": values})
    return result


def snapshots(bootstrap, fixtures):
    folder = OUT / "snapshots"
    folder.mkdir(parents=True, exist_ok=True)
    index = folder / "commits.json"
    pins_file = REPORTS / "model-improvements-sources.json"
    pins = json.loads(pins_file.read_text()).get("snapshots", []) if pins_file.exists() else []
    if not index.exists():
        if pins:
            dump(index, [s for s in pins if "origin" not in s])
        else:
            raw = subprocess.check_output(["gh", "api", "--method", "GET",
                f"repos/{SOURCE_REPOSITORY}/commits", "-f", "path=data/2025-26/players_raw.csv", "-F", "per_page=100"])
            commits = json.loads(raw)
            dump(index, [{"sha": c["sha"], "date": c["commit"]["committer"]["date"]} for c in commits])
    commits = json.loads(index.read_text())
    def get(c):
        path = folder / (c["sha"] + ".csv")
        if not path.exists():
            r = requests.get(f'https://raw.githubusercontent.com/{SOURCE_REPOSITORY}/{c["sha"]}/data/2025-26/players_raw.csv', timeout=60)
            r.raise_for_status()
            path.write_bytes(r.content)
        content = path.read_bytes()
        if c.get("sha256") and c["sha256"] != hashlib.sha256(content).hexdigest():
            raise ValueError("Snapshot checksum mismatch: " + c["sha"])
        return {**c, "timestamp": timestamp(c["date"]), "sha256": hashlib.sha256(content).hexdigest(),
                "players": {p["id"]: p for p in read_csv(content)}}
    with ThreadPoolExecutor(max_workers=4) as pool:
        primary = list(pool.map(get, commits))

    def alternate(gw):
        # Fetch the repository state published before this deadline, not a final
        # By Gameweek export whose status fields may have been revised later.
        index = folder / f"alternate-gw{gw}.json"
        repo = "olbauday/FPL-Elo-Insights"
        if not index.exists():
            if pins:
                dump(index, next((s for s in pins if s.get("origin") == gw), {}))
            else:
                cutoff = datetime.fromtimestamp(prediction_cutoff(bootstrap, fixtures, gw), timezone.utc).isoformat()
                raw = subprocess.check_output(["gh", "api", "--method", "GET", f"repos/{repo}/commits",
                    "-f", "path=data/2025-2026/playerstats.csv", "-f", "until=" + cutoff, "-F", "per_page=1"])
                values = json.loads(raw)
                dump(index, {"sha": values[0]["sha"], "date": values[0]["commit"]["committer"]["date"]} if values else {})
        c = json.loads(index.read_text())
        if not c:
            return None
        path = folder / (c["sha"] + "-alternate.csv")
        if not path.exists():
            r = requests.get(f'https://raw.githubusercontent.com/{repo}/{c["sha"]}/data/2025-2026/playerstats.csv', timeout=60)
            r.raise_for_status()
            path.write_bytes(r.content)
        content = path.read_bytes()
        if c.get("sha256") and c["sha256"] != hashlib.sha256(content).hexdigest():
            raise ValueError("Snapshot checksum mismatch: " + c["sha"])
        rows = read_csv(content)
        latest = max(int(p["gw"]) for p in rows)
        if latest >= gw:
            # Can't map chance_of_playing_next_round safely after a rollover.
            return None
        return {**c, "origin": gw, "repository": repo, "snapshot_gw": latest,
                "timestamp": timestamp(c["date"]), "sha256": hashlib.sha256(content).hexdigest(),
                "players": {p["id"]: p for p in rows if int(p["gw"]) == latest}}

    with ThreadPoolExecutor(max_workers=4) as pool:
        extra = list(pool.map(alternate, range(4, 35)))
    return sorted(primary + [s for s in extra if s], key=lambda c: c["timestamp"])


def context(sample, snapshots):
    known = [s for s in snapshots if s["timestamp"] < sample["cutoff"]
             and s.get("origin", sample["gw"]) == sample["gw"]]
    snapshot = known[-1] if known else None
    age = (sample["cutoff"] - snapshot["timestamp"]) / 86400 if snapshot else 999
    p = snapshot["players"].get(sample["id"]) if snapshot and age <= 8 else None
    if p is None:
        return [0] * 9
    if p.get("news_added") and timestamp(p["news_added"]) >= sample["cutoff"]:
        return [0] * 9
    chance = p.get("chance_of_playing_next_round")
    # Missing probability is a missing value, not a claim of perfect fitness.
    return [1, age / 8, int(p.get("status") == "a"), int(p.get("status") == "d"),
            int(p.get("status") in ("i", "s", "u")), int(chance is not None),
            float(chance or 0) / 100, int(p.get("penalties_order") == 1),
            int(p.get("corners_and_indirect_freekicks_order") == 1)]


def price_band(row):
    # Fixed FPL price brackets, never learned from future outcomes.
    limits = {"DEF": (5, 6), "MID": (6.5, 9), "FWD": (7, 9)}[row["position"]]
    return sum(row["price"] > v for v in limits)


def common_scoring_features(x):
    # Assists, bonus/BPS and defensive contributions changed or are unavailable
    # in older eras. Excluding them is preferable to treating missing as zero.
    return [v for i, v in enumerate(x) if i not in (13, 14, 15, 17)]


def direct(variant, train, valid, position):
    name = MODEL[position]
    bundle = variant if variant in BUNDLES else "all" if variant.startswith("all_") else None
    def transform(rows):
        return [{**s, "x": s["x"] + ([s["extra"][i] for i in BUNDLES[bundle]] if bundle else [])
                 + (s["context"] if variant == "snapshots" else [])} for s in rows]
    if variant in ("eligible", "all_eligible", "lgb_l2", "lgb_l1", "ranker"):
        train = [s for s in train if s["eligible"]]
    train, valid = transform(train), transform(valid)
    x, y = np.array([s["x"] for s in train]), np.array([s["target"] for s in train])
    xv = np.array([s["x"] for s in valid])
    if variant.startswith("lgb_"):
        model = LGBMRegressor(objective="regression_l1" if variant == "lgb_l1" else "regression", **PARAMS)
        model.fit(x, y)
        return np.maximum(0, model.predict(xv))
    if variant == "ranker":
        order = np.argsort([s["gw"] for s in train], kind="stable")
        groups = list(Counter(train[i]["gw"] for i in order).values())
        model = LGBMRanker(objective="lambdarank", label_gain=list(range(int(max(y)) + 1)), **PARAMS)
        model.fit(x[order], np.maximum(0, y[order]).astype(int), group=groups)
        return model.predict(xv)
    if variant in ("weighted", "all_weighted", "price_balanced"):
        if variant == "price_balanced":
            counts = Counter(price_band(s) for s in train if s["eligible"])
            weights = np.array([(3 if s["eligible"] else 1) / max(counts[price_band(s)], 1) for s in train])
            weights *= len(weights) / weights.sum()
        else:
            weights = np.array([3 if s["eligible"] else 1 for s in train], dtype=float)
        model = estimator(name)
        kw = {"ridge__sample_weight": weights} if name == "ridge" else {"sample_weight": weights}
        model.fit(x, y, **kw)
        return np.where(xv[:, 23] == 0, 0, np.maximum(0, model.predict(xv)))
    return fit_model(name, train).predict(valid)


def fixture_samples(bootstrap, fixtures, histories, season):
    """Match training to inference: each origin forecasts its next five GWs.

    The same fixture can be forecast at several origins with different known
    histories/lead times. Inverse multiplicity weights retain one unit of total
    training weight per player-fixture after the chronological cutoff is applied.
    """
    rows = []
    players = {p["id"]: p for p in bootstrap["elements"] if p["element_type"] in POSITIONS}
    observed = {(pid, r["fixture"]): r for pid, history in histories.items() for r in history}
    stable_histories = {pid: [{**r, "total_points": stable_points(r, POSITIONS[players[pid]["element_type"]])} for r in history]
                        for pid, history in histories.items() if pid in players}
    for gw in range(4, 39):
        cutoff = prediction_cutoff(bootstrap, fixtures, gw)
        table = team_table(fixtures, gw, cutoff)
        for pid, player in players.items():
            position = POSITIONS[player["element_type"]]
            feat = make_features(player, histories.get(pid, []), fixtures, gw, 5, cutoff, table)
            if feat is None:
                continue
            stable = make_features(player, stable_histories.get(pid, []), fixtures, gw, 5, cutoff, table)
            for f in fixtures:
                if not f.get("finished") or not gw <= (f.get("event") or 0) < gw + 5 or feat["team"] not in (f["team_h"], f["team_a"]):
                    continue
                actual = observed.get((pid, f["id"]), {})
                minutes = float(actual.get("minutes") or 0)
                rows.append(dict(id=pid, position=position, gw=gw, season=season, fixture=f["id"],
                    label_end=timestamp(f["kickoff_time"]) + 10800,
                    x=fixture_x(feat, f, cutoff, table), stable_x=fixture_x(stable, f, cutoff, table),
                    minutes_class=2 if minutes >= 60 else int(minutes > 0),
                    target=float(actual.get("total_points") or 0), stable_target=stable_points(actual, position)))
    return rows


def fixture_x(features, fixture, cutoff, table):
    x = list(features["x"])
    home = features["team"] == fixture["team_h"]
    opponent = table.get(fixture["team_a"] if home else fixture["team_h"], dict(ppg=4/3, ga=4/3, cs=1/3))
    x[23:28] = [1, int(home), opponent["ppg"], opponent["ga"], opponent["cs"]]
    # Divide horizon-scaled baselines down to one fixture.
    for i in (32, 33):
        x[i] /= max(features["fixture_count"], 1)
    return x + [max(0, (timestamp(fixture["kickoff_time"]) - cutoff) / 86400)]


def fixture_weights(rows):
    counts = Counter((s["season"], s["id"], s["fixture"]) for s in rows)
    return np.array([1 / counts[(s["season"], s["id"], s["fixture"])] for s in rows])


def two_stage(variant, recent, older, valid, bootstrap, fixtures, histories, origin,
              current_season="2025-26", live=False):
    cutoff = prediction_cutoff(bootstrap, fixtures, origin)
    train = [s for s in recent if s["label_end"] < cutoff and
             (s["season"] != current_season or s["gw"] < origin)]
    minutes_train = train + older if variant in ("two_stage_more_minutes", "components_multi") else train
    # The minutes head uses only scoring-independent covariates, common to every era.
    indices = [4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 34]
    classifier = LGBMClassifier(**PARAMS)
    classifier.fit(np.array([s["x"] for s in minutes_train])[:, indices],
                   [s["minutes_class"] for s in minutes_train], sample_weight=fixture_weights(minutes_train))
    components = variant.startswith("components_")
    point_models, residual_models = {}, {}
    for cls in (1, 2):
        # Pool the common subtotal only; learn rule-dependent points on 2025/26.
        source = train + older if variant == "components_multi" else train
        source = [s for s in source if s["minutes_class"] == cls]
        model = LGBMRegressor(**PARAMS)
        model.fit([common_scoring_features(s["stable_x"]) if components else s["x"] for s in source],
                  [s["stable_target"] if components else s["target"] for s in source], sample_weight=fixture_weights(source))
        point_models[cls] = model
        if components:
            now = [s for s in train if s["minutes_class"] == cls]
            residual = LGBMRegressor(**PARAMS)
            residual.fit([s["x"] for s in now], [s["target"] - s["stable_target"] for s in now], sample_weight=fixture_weights(now))
            residual_models[cls] = residual
    all_x, all_stable, owners = [], [], []
    players = {p["id"]: p for p in bootstrap["elements"]}
    for i, s in enumerate(valid):
        table = team_table(fixtures, s["gw"], s["cutoff"])
        history = histories[s["id"]]
        stable_history = [{**r, "total_points": stable_points(r, s["position"])} for r in history]
        feat = make_features(players[s["id"]], history, fixtures, s["gw"], 5, s["cutoff"], table, live=live)
        stable = make_features(players[s["id"]], stable_history, fixtures, s["gw"], 5, s["cutoff"], table, live=live)
        for f in fixtures:
            if s["gw"] <= f["event"] < s["gw"] + 5 and s["team"] in (f["team_h"], f["team_a"]):
                all_x.append(fixture_x(feat, f, s["cutoff"], table))
                all_stable.append(fixture_x(stable, f, s["cutoff"], table))
                owners.append(i)
    prediction = np.zeros(len(valid))
    if not all_x:
        return prediction
    probs = classifier.predict_proba(np.array(all_x)[:, indices])
    points = np.zeros(len(all_x))
    for cls in (1, 2):
        conditional = point_models[cls].predict([common_scoring_features(x) for x in all_stable] if components else all_x)
        if components:
            conditional += residual_models[cls].predict(all_x)
        points += probs[:, list(classifier.classes_).index(cls)] * conditional
    np.add.at(prediction, owners, points)
    return np.maximum(0, prediction)


def summarize(records, field="prediction", calibrated=True):
    result = rank_metrics(records, field)
    if calibrated:
        result.update(metrics([r["target"] for r in records], [r[field] for r in records]))
    groups = defaultdict(list)
    for r in records:
        groups[(r["gw"], r["band"])].append(r)
    picks = [r["target"] for group in groups.values() for r in sorted(group, key=lambda r: r[field], reverse=True)[:3]]
    result["price_band_top3_points"] = float(np.mean(picks))
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # Source-code changes must not silently reuse predictions from old logic.
    cache_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    cache_version = OUT / "cache-version.txt"
    if not cache_version.exists() or cache_version.read_text() != cache_digest:
        for path in list((OUT / "predictions").glob("*.json")) + list(OUT.glob("fixture-samples-*.json")):
            path.unlink()
        cache_version.write_text(cache_digest)
    archives = {}
    for season in SEASONS:
        archives[season] = archive(season)
        print("AUDIT", season, json.dumps(archives[season][3]["audit"]), flush=True)
    b, f, h, source = archives["2025-26"]
    snap = snapshots(b, f)
    samples = extras(b, f, h, make_samples(b, f, h))
    for s in samples:
        s["context"] = context(s, snap)
    coverage = {str(gw): dict(eligible=sum(s["eligible"] for s in samples if s["gw"] == gw),
                            fresh=sum(s["eligible"] and bool(s["context"][0]) for s in samples if s["gw"] == gw))
                for gw in range(16, 35)}
    audit = dict(seasons={s: v[3] for s, v in archives.items()},
        snapshots=[{k: s[k] for k in ("sha", "date", "sha256", "repository", "origin", "snapshot_gw") if k in s} for s in snap], coverage=coverage)
    dump(OUT / "data-audit.json", audit)
    dump(REPORTS / "model-improvements-sources.json", audit)
    fixture_rows = {}
    for season, (bs, fs, hs, _) in archives.items():
        cache = OUT / ("fixture-samples-" + season + ".json")
        if cache.exists():
            fixture_rows[season] = json.loads(cache.read_text())
        else:
            fixture_rows[season] = fixture_samples(bs, fs, hs, season)
            dump(cache, fixture_rows[season])
        print("FIXTURE SAMPLES", season, len(fixture_rows[season]), flush=True)
    records, summaries, selected = {}, {}, {}
    for phase, blocks in BLOCKS.items():
        summaries[phase] = {}
        for pos in MODEL:
            summaries[phase][pos] = {}
            pool = [s for s in samples if s["position"] == pos]
            recent = [s for s in fixture_rows["2025-26"] if s["position"] == pos]
            older = [s for season in SEASONS[:2] for s in fixture_rows[season] if s["position"] == pos]
            for variant in VARIANTS:
                key = f"{phase}:{pos}:{variant}"
                cache = OUT / "predictions" / (key.replace(":", "-") + ".json")
                if cache.exists():
                    result = json.loads(cache.read_text())
                else:
                    result = []
                    for start, end in blocks:
                        train = training_before(pool, start, prediction_cutoff(b, f, start), 5)
                        valid = [s for s in pool if start <= s["gw"] <= end and s["eligible"]]
                        pred = (two_stage(variant, recent, older, valid, b, f, h, start)
                                if variant.startswith(("two_stage", "components_")) else direct(variant, train, valid, pos))
                        result += [dict(id=s["id"], gw=s["gw"], eligible=True, target=s["target"],
                                        band=price_band(s), prediction=float(p)) for s, p in zip(valid, pred)]
                    dump(cache, result)
                records[key] = result
                value = summarize(result, calibrated=variant != "ranker")
                summaries[phase][pos][variant] = value
                print(phase, pos, variant, "top5", round(value["top5_points_per_pick"], 3),
                      "mae", round(value.get("mae", 0), 3), flush=True)
            if phase == "development":
                # Predeclared point-valued choice: maximize top-five score while
                # requiring both <= baseline MAE and >= baseline price-band score.
                base = summaries[phase][pos]["baseline"]
                candidates = [v for v in VARIANTS if v != "ranker" and
                    summaries[phase][pos][v]["mae"] <= base["mae"] and
                    summaries[phase][pos][v]["price_band_top3_points"] >= base["price_band_top3_points"]]
                selected[pos] = max(candidates, key=lambda v: summaries[phase][pos][v]["top5_points_per_pick"])
        dump(OUT / (phase + "-results.json"), dict(results=summaries[phase], selected=selected))
        if phase == "development":
            dump(OUT / "frozen-selection.json", dict(selected=selected, rule="Max top5 subject to no worse development MAE and price-band top3 than baseline; fixed before audit.",
                 warning="GW25-34 was inspected in the prior review; it is a reused out-of-time audit."))
            print("FROZEN", selected, flush=True)
    dump(OUT / "all-results.json", dict(results=summaries, selected=selected, code_sha256=cache_digest))


if __name__ == "__main__":
    main()
