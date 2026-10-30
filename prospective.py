"""Record first pre-deadline forecasts and score fully completed horizons.

The live MID model remains the baseline. The frozen per-fixture component model
runs only here, alongside it. GitHub history lives on a separate data branch.
"""

import argparse
import base64
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests

from download_fpl_data import get_season
from forecasting import (fit_model, live_forecasts, load_live_histories,
                         make_features, timestamp, training_samples)

ROOT = Path(__file__).resolve().parent
LEDGER_BRANCH = "codex/forecast-ledger"
CONTEXT_FIELDS = ("status", "chance_of_playing_next_round", "penalties_order",
                  "corners_and_indirect_freekicks_order", "news_added")


def write_json(path, value, exclusive=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


class GitHubLedger:
    """Read existing snapshots; atomically append data without touching main."""

    def __init__(self, repository, token):
        self.base = f"https://api.github.com/repos/{repository}"
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"})
        self.head, self.tree, self.blobs = None, None, {}

    def request(self, method, path, **kwargs):
        response = self.session.request(method, self.base + path, timeout=45, **kwargs)
        response.raise_for_status()
        return response.json()

    def restore(self, directory, season):
        try:
            ref = self.request("GET", f"/git/ref/heads/{LEDGER_BRANCH}")
        except requests.HTTPError as error:
            if error.response.status_code == 404:
                return
            raise
        self.head = ref["object"]["sha"]
        self.tree = self.request("GET", f"/git/commits/{self.head}")["tree"]["sha"]
        tree = self.request("GET", f"/git/trees/{self.tree}?recursive=1")
        if tree.get("truncated"):
            raise RuntimeError("Forecast history tree was truncated.")
        for node in tree["tree"]:
            if node["type"] != "blob":
                continue
            self.blobs[node["path"]] = node["sha"]
            path = Path(node["path"])
            if (path.parent == Path("snapshots") / season and path.name.startswith("gw")
                    and path.suffix == ".json"):
                value = self.request("GET", f'/git/blobs/{node["sha"]}')
                destination = directory / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(base64.b64decode(value["content"]))

    def publish(self, directory, season):
        entries = []
        paths = list((directory / "snapshots" / season).glob("gw*.json"))
        report = directory / "evaluations" / (season + ".json")
        if report.exists():
            paths.append(report)
        for path in paths:
            relative = path.relative_to(directory).as_posix()
            content = path.read_bytes()
            sha = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
            previous = self.blobs.get(relative)
            if previous == sha:
                continue
            if previous and relative.startswith("snapshots/"):
                raise RuntimeError("Refusing to overwrite a frozen forecast: " + relative)
            entries.append(dict(path=relative, mode="100644", type="blob", content=content.decode()))
        if not entries:
            return False
        payload = dict(tree=entries)
        if self.tree:
            payload["base_tree"] = self.tree
        tree = self.request("POST", "/git/trees", json=payload)
        commit = self.request("POST", "/git/commits", json=dict(
            message="data: record and evaluate pre-deadline FPL forecasts", tree=tree["sha"],
            parents=[self.head] if self.head else []))
        if self.head:
            self.request("PATCH", f"/git/refs/heads/{LEDGER_BRANCH}", json=dict(sha=commit["sha"], force=False))
        else:
            self.request("POST", "/git/refs", json=dict(ref=f"refs/heads/{LEDGER_BRANCH}", sha=commit["sha"]))
        return True


def github_ledger(repository):
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        token = subprocess.check_output(["gh", "auth", "token"], text=True).strip()
    return GitHubLedger(repository, token)


def challenger_forecast(bootstrap, fixtures, histories, valid, origin, cutoff):
    # Reuse the frozen research implementation rather than reimplementing its
    # architecture during the prospective experiment.
    from historical_data import load_history
    from research.model_improvements import fixture_samples, two_stage

    name = json.loads((ROOT / "model_config.json").read_text())["midfield_challenger"]
    if name != "components_single":
        raise ValueError("The prospective midfield experiment is frozen to components_single.")

    old_b, old_f, old_h, source = load_history()
    old_b = {**old_b, "elements": [p for p in old_b["elements"] if p["element_type"] == 3]}
    code = hashlib.sha256((ROOT / "research/model_improvements.py").read_bytes()).hexdigest()
    cache = ROOT / "data" / f"midfield-fixtures-{source['revision'][:8]}-{code[:8]}.json"
    if cache.exists():
        old = json.loads(cache.read_text())
    else:
        old = fixture_samples(old_b, old_f, old_h, "2025-26")
        write_json(cache, old)
    season = get_season(bootstrap)
    current_b = {**bootstrap, "elements": [p for p in bootstrap["elements"] if p["element_type"] == 3]}
    current = fixture_samples(current_b, fixtures, histories, season)
    train = [r for r in old + current if r["label_end"] < cutoff]
    return two_stage(name, train, [], valid, bootstrap, fixtures, histories,
                     origin, current_season=season, live=True)


def capture(directory, data_dir, bootstrap, fixtures, now=None):
    now = now or datetime.now(timezone.utc).timestamp()
    season = get_season(bootstrap)
    event = next((e for e in bootstrap["events"] if e.get("is_next")), None)
    if event is None or now >= timestamp(event["deadline_time"]):
        return None
    origin, horizon = event["id"], 5
    path = directory / "snapshots" / season / f"gw{origin:02}.json"
    if path.exists():
        return path  # First forecast is immutable, including manual reruns.
    if origin + horizon - 1 > max(e["id"] for e in bootstrap["events"]):
        return None
    histories = load_live_histories(data_dir, bootstrap)
    forecasts, _ = live_forecasts(bootstrap, fixtures, data_dir, horizon)
    valid = []
    for p in bootstrap["elements"]:
        if p["element_type"] not in (2, 3):
            continue
        feature = make_features(p, histories.get(p["id"], []), fixtures, origin, horizon, now, live=True)
        if feature and feature["eligible"] and p["id"] in forecasts:
            valid.append(dict(id=p["id"], name=p["web_name"], position="DEF" if p["element_type"] == 2 else "MID",
                              gw=origin, cutoff=now, **{k: v for k, v in feature.items() if k != "past"}))
    midfield = [r for r in valid if r["position"] == "MID"]
    if len(midfield) < 5:
        return None
    challenger = challenger_forecast(bootstrap, fixtures, histories, midfield, origin, now)
    baseline_pool = training_samples(bootstrap, fixtures, histories, data_dir, horizon, origin, now, False)
    defender_baseline = fit_model("ridge", [r for r in baseline_pool if r["position"] == "DEF"])
    defenders = [r for r in valid if r["position"] == "DEF"]
    base = defender_baseline.predict(defenders) if defenders else []
    rows = [dict(id=r["id"], name=r["name"], position="MID", price=r["price"],
                 baseline=forecasts[r["id"]]["points"], candidate=float(c)) for r, c in zip(midfield, challenger)]
    rows += [dict(id=r["id"], name=r["name"], position="DEF", price=r["price"],
                  baseline=float(a), candidate=forecasts[r["id"]]["points"]) for r, a in zip(defenders, base)]
    source_hashes = {name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
                     for name in ("bootstrap-static.json", "fixtures.json")}
    source_hashes["histories"] = hashlib.sha256(json.dumps(histories, sort_keys=True).encode()).hexdigest()
    value = dict(schema_version=1, season=season, origin=origin, horizon=horizon, cutoff=now,
        deadline=event["deadline_time"], models=dict(DEF="ridge_availability vs ridge", MID="components_single vs forest"),
        code_sha256=hashlib.sha256((ROOT / "research/model_improvements.py").read_bytes()).hexdigest(),
        config_sha256=hashlib.sha256((ROOT / "model_config.json").read_bytes()).hexdigest(),
        data_sha256=source_hashes, bootstrap_captured_at=(data_dir / "bootstrap-static.json").stat().st_mtime,
        defender_context={str(p["id"]): {k: p.get(k) for k in CONTEXT_FIELDS}
                          for p in bootstrap["elements"] if p["element_type"] == 2},
        predictions=rows, fixtures=[{k: f.get(k) for k in ("id", "event", "team_h", "team_a", "kickoff_time")}
                                   for f in fixtures if origin <= (f.get("event") or 0) < origin + horizon])
    # Also check after training: a long-running build must not cross the deadline.
    if datetime.now(timezone.utc).timestamp() >= timestamp(event["deadline_time"]):
        return None
    write_json(path, value, exclusive=True)
    return path


def evaluate(directory, bootstrap, fixtures, histories, now=None):
    now = now or datetime.now(timezone.utc).timestamp()
    season = get_season(bootstrap)
    completed = {e["id"] for e in bootstrap["events"] if e.get("finished")}
    windows, pending = [], []
    for path in sorted((directory / "snapshots" / season).glob("gw*.json")):
        record = json.loads(path.read_text())
        if record["season"] != season or record["cutoff"] >= timestamp(record["deadline"]):
            raise ValueError("Invalid pre-deadline forecast record: " + str(path))
        rounds = set(range(record["origin"], record["origin"] + record["horizon"]))
        games = [f for f in fixtures if f.get("event") in rounds]
        ready = rounds <= completed and games and all(f.get("finished") and f.get("kickoff_time")
                and timestamp(f["kickoff_time"]) + 10800 < now for f in games)
        # Include rescheduled fixtures from the originally known schedule as
        # another maturity guard, without treating final dates as forecast inputs.
        ids = {f["id"] for f in record["fixtures"]}
        current = {f["id"]: f for f in fixtures}
        ready = ready and all(fid in current and current[fid].get("finished") and current[fid].get("kickoff_time")
                    and timestamp(current[fid]["kickoff_time"]) + 10800 < now for fid in ids)
        if not ready:
            pending.append(record["origin"])
            continue
        for position in ("DEF", "MID"):
            rows = [r for r in record["predictions"] if r["position"] == position]
            if not rows:
                continue
            missing = [r["id"] for r in rows if r["id"] not in histories]
            if missing:
                raise ValueError(f"Cannot score missing player histories: {missing}")
            actual = {r["id"]: sum(float(m.get("total_points") or 0) for m in histories[r["id"]]
                                   if m.get("round") in rounds) for r in rows}
            scores = {}
            for field in ("baseline", "candidate"):
                picks = sorted(rows, key=lambda r: r[field], reverse=True)[:5]
                scores[field] = dict(mae=float(np.mean([abs(actual[r["id"]] - r[field]) for r in rows])),
                    top5_points_per_pick=float(np.mean([actual[r["id"]] for r in picks])),
                    picks=[r["id"] for r in picks])
            windows.append(dict(origin=record["origin"], position=position, players=len(rows), scores=scores))
    summary = {}
    for position in ("DEF", "MID"):
        rows = [r for r in windows if r["position"] == position]
        if rows:
            count = sum(r["players"] for r in rows)
            summary[position] = dict(origins=len(rows), player_origins=count, **{
                field: dict(mae=sum(r["scores"][field]["mae"] * r["players"] for r in rows) / count,
                            top5_points_per_pick=float(np.mean([r["scores"][field]["top5_points_per_pick"] for r in rows])))
                for field in ("baseline", "candidate")})
    report = dict(season=season, pending_origins=pending, completed_windows=windows, summary=summary,
        note="Overlapping five-GW windows are correlated. Challenger results never automatically change the live model.")
    write_json(directory / "evaluations" / (season + ".json"), report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--repository", help="Restore and publish history on the separate GitHub data branch")
    parser.add_argument("--restore-only", action="store_true")
    args = parser.parse_args()
    data_dir = args.data.resolve()
    bootstrap = json.loads((data_dir / "bootstrap-static.json").read_text())
    season = get_season(bootstrap)
    directory = data_dir / "forecast-ledger"
    github = None
    if args.repository:
        github = github_ledger(args.repository)
        github.restore(directory, season)
    if args.restore_only:
        return
    fixtures = json.loads((data_dir / "fixtures.json").read_text())
    path = capture(directory, data_dir, bootstrap, fixtures)
    report = evaluate(directory, bootstrap, fixtures, load_live_histories(data_dir, bootstrap))
    if github:
        github.publish(directory, season)
    print(f"Frozen forecast: {path or 'no eligible pre-deadline origin'}; "
          f"completed comparisons: {len(report['completed_windows'])}; pending origins: {report['pending_origins']}")


if __name__ == "__main__":
    main()
