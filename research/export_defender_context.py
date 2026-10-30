"""Export the audited context vectors without deploying the large raw archives."""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from availability import FEATURE_NAMES, HISTORICAL_FILE
from forecasting import make_samples
from historical_data import load_history
from research.model_improvements import context, snapshots


def main():
    b, f, h, source = load_history()
    states = snapshots(b, f)
    contexts = {}
    for row in make_samples(b, f, h):
        if row["position"] == "DEF":
            contexts.setdefault(str(row["gw"]), {})[str(row["id"])] = context(row, states)
    manifest = ROOT / "reports/model-improvements-sources.json"
    value = dict(source_revision=source["revision"], feature_names=list(FEATURE_NAMES),
                 source_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(), contexts=contexts)
    HISTORICAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    HISTORICAL_FILE.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")
    print(f"Exported {sum(map(len, contexts.values()))} defender context vectors")


if __name__ == "__main__":
    main()
