"""Resume JSONL capturado da serial: python scripts/summarize_latency.py serial.log."""
import argparse
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("log")
args = parser.parse_args()
rows = []
for line in Path(args.log).read_text(encoding="utf-8", errors="replace").splitlines():
    try:
        row = json.loads(line[line.index("{"):])
        if "pipeline_us" in row:
            rows.append(row)
    except (ValueError, json.JSONDecodeError):
        continue
if not rows:
    parser.exit(1, "Nenhuma linha de métricas encontrada. Capture o monitor serial.\n")
report = {"windows": len(rows)}
for name in ("capture_read_us", "capture_copy_us", "features_us", "queue_us", "inference_us", "pipeline_us"):
    values = np.array([r[name] for r in rows])/1000
    report[name.replace("_us", "_ms")] = {"mean": float(values.mean()), "p95": float(np.percentile(values, 95)), "max": float(values.max())}
report["counters_max"] = {name: max(r[name] for r in rows) for name in ("skipped", "queue_drops", "read_errors")}
report["min_free_heap_bytes"] = min(r["heap"] for r in rows)
print(json.dumps(report, indent=2))
