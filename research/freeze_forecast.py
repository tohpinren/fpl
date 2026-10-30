"""Freeze baseline and development-selected forecasts before real future outcomes.

This is an explicit research snapshot, not a deployment or scheduled job. Raw
inputs remain under ignored data/; the ledger includes their hashes and context.
"""

import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from download_fpl_data import BASE_URL, create_session, get_season
from forecasting import (POSITIONS, load_live_histories, make_features, make_samples,
                         prediction_cutoff, team_table, training_before)
from research.model_improvements import (OUT, SEASONS, archive, context, direct, dump,
    extras, fixture_samples, snapshots, two_stage)


def main():
    session = create_session()
    def get(endpoint):
        response = session.get(BASE_URL + endpoint, timeout=45)
        response.raise_for_status()
        return response.json()
    with ThreadPoolExecutor(max_workers=2) as pool:
        b, f = list(pool.map(get, ("/bootstrap-static/", "/fixtures/")))
    now = datetime.now(timezone.utc)
    season = get_season(b)
    event = next(e for e in b["events"] if e.get("is_next"))
    gw = event["id"]
    cutoff = min(now.timestamp(), prediction_cutoff(b, f, gw))
    if cutoff != now.timestamp():
        raise ValueError("The next deadline has passed; cannot freeze a pre-deadline forecast.")
    h = load_live_histories(ROOT / "data", b)
    local_season = get_season(json.loads((ROOT / "data/bootstrap-static.json").read_text()))
    update = [p["id"] for p in b["elements"] if local_season != season or
              sum(r["total_points"] for r in h.get(p["id"], [])) != p["total_points"] or
              sum(r["minutes"] for r in h.get(p["id"], [])) != p["minutes"]]
    def update_player(pid):
        return pid, get(f"/element-summary/{pid}/")["history"]
    with ThreadPoolExecutor(max_workers=8) as pool:
        h.update(pool.map(update_player, update))
    for p in b["elements"]:
        if sum(r["total_points"] for r in h.get(p["id"], [])) != p["total_points"]:
            raise ValueError(f"Live points do not reconcile: {p['id']}")
    stamp = now.strftime("%Y-%m-%dT%H%M%SZ")
    folder = OUT / "prospective-inputs" / stamp
    hashes = {}
    for name, value in (("bootstrap.json", b), ("fixtures.json", f), ("histories.json", h)):
        path = folder / name
        dump(path, value)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    archives = {s: archive(s) for s in SEASONS}
    old_b, old_f, old_h, _ = archives["2025-26"]
    snap = snapshots(old_b, old_f)
    past = extras(old_b, old_f, old_h, make_samples(old_b, old_f, old_h))
    for s in past:
        s["context"] = context(s, snap)
    dated_fixtures = [x for x in f if x.get("event") is not None and x.get("kickoff_time")]
    current = extras(b, dated_fixtures, h, training_before(make_samples(b, f, h), gw, cutoff, 5))
    for s in current:
        # Earlier 2026/27 availability snapshots have not been collected. Do not
        # backfill them using today's injuries.
        s["context"] = [0] * 9
    table = team_table(f, gw, cutoff)
    valid = []
    live_snapshot = [dict(timestamp=cutoff - .001, players={p["id"]: p for p in b["elements"]})]
    for p in b["elements"]:
        if p["element_type"] not in POSITIONS:
            continue
        feat = make_features(p, h.get(p["id"], []), f, gw, 5, cutoff, table, live=True)
        if feat is None or not feat["eligible"]:
            continue
        s = dict(id=p["id"], name=p["web_name"], position=POSITIONS[p["element_type"]],
                 gw=gw, cutoff=cutoff, **{k: v for k, v in feat.items() if k != "past"})
        s["context"] = context(s, live_snapshot)
        valid.append(s)
    valid = extras(b, dated_fixtures, h, valid)
    selected = json.loads((OUT / "frozen-selection.json").read_text())["selected"]
    recent_fixtures = json.loads((OUT / "fixture-samples-2025-26.json").read_text())
    recent_fixtures += fixture_samples(b, f, h, season)
    older = [r for s in SEASONS[:2] for r in json.loads((OUT / f"fixture-samples-{s}.json").read_text())]
    records = []
    exploratory = {"MID": "components_single", "FWD": "components_multi"}
    for pos, variant in selected.items():
        train = [s for s in past + current if s["position"] == pos]
        rows = [s for s in valid if s["position"] == pos]
        base = direct("baseline", train, rows, pos)
        if variant.startswith(("components_", "two_stage")):
            prediction = two_stage(variant, [s for s in recent_fixtures if s["position"] == pos],
                [s for s in older if s["position"] == pos], rows, b, f, h, gw, current_season=season, live=True)
        else:
            prediction = direct(variant, train, rows, pos)
        extra_prediction = None
        if pos in exploratory:
            extra_prediction = two_stage(exploratory[pos], [s for s in recent_fixtures if s["position"] == pos],
                [s for s in older if s["position"] == pos], rows, b, f, h, gw, current_season=season, live=True)
        records += [dict(id=s["id"], name=s["name"], position=pos, price=s["price"],
                         baseline=float(a), candidate=float(c), model=variant,
                         exploratory_challengers={exploratory[pos]: float(extra_prediction[i])} if extra_prediction is not None else {},
                         availability_context=s["context"])
                    for i, (s, a, c) in enumerate(zip(rows, base, prediction))]
    result = dict(created_at=now.isoformat(), season=season, forecast_gameweeks=list(range(gw, gw+5)),
        deadline=event["deadline_time"], code_sha256=hashlib.sha256((ROOT / "research/model_improvements.py").read_bytes()).hexdigest(),
        input_sha256=hashes, refreshed_player_histories=len(update), selected_on="2025/26 GW16-24 only",
        exploratory_selected_on="Reused GW25-34 audit: MID components_single and FWD components_multi; these require fresh prospective evaluation.",
        note="Research ledger; no outcomes known. Production model unchanged. 2026/27 scoring differs from 2025/26.",
        fixtures=[{k: x.get(k) for k in ("id", "event", "team_h", "team_a", "kickoff_time")}
                  for x in f if gw <= (x.get("event") or 0) < gw + 5], predictions=records)
    target = ROOT / "reports/forecast-ledger" / (stamp + ".json")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(f"Frozen {len(records)} forecasts for {season} GW{gw}-{gw+4}: {target}")


if __name__ == "__main__":
    main()
