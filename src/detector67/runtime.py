from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np

from .dsp import STRIDE, RATE, extract, read_audio, windows
from .model import load_model, probability


@dataclass
class EventGate:
    threshold: float
    confirmations: int = 2
    cooldown: float = 1.5
    consecutive: int = 0
    last_event: float = -1e9
    last_time: float = -1e9
    armed: bool = True
    negatives: int = 0

    def update(self, score, timestamp):
        # Frames descartados não podem contar como confirmações consecutivas.
        if timestamp - self.last_time > STRIDE / RATE * 1.5:
            self.consecutive = 0
        self.last_time = timestamp
        if score >= self.threshold:
            self.consecutive += 1
            self.negatives = 0
        else:
            self.consecutive = 0
            self.negatives += 1
            if self.negatives >= 2:
                self.armed = True
        if self.armed and self.consecutive >= self.confirmations and timestamp - self.last_event >= self.cooldown:
            self.last_event, self.armed = timestamp, False
            return True
        return False


def replay(path, model_dir, output=None, annotations=None):
    audio = read_audio(Path(path))
    params, meta = load_model(model_dir)
    gate = EventGate(meta["threshold"], meta["confirmations"], meta["cooldown_seconds"])
    rows, events = [], []
    for timestamp, clip in windows(audio):
        t0 = time.perf_counter()
        features, rms, centroid = extract(clip)
        t1 = time.perf_counter()
        score = float(probability(features[None], params)[0])
        t2 = time.perf_counter()
        event = gate.update(score, timestamp)
        if event:
            events.append(timestamp)
        rows.append(dict(time_s=timestamp, score=score, event=event, rms=rms, centroid_hz=centroid,
                         features_ms=(t1-t0)*1000, inference_ms=(t2-t1)*1000))
    report = {"audio": str(path), "duration_s": len(audio)/RATE, "events_s": events,
              "windows": rows, "timing_scope": "Computador: não representa latência do ESP32"}
    if annotations:
        truth = json.loads(Path(annotations).read_text(encoding="utf-8"))
        # Cada item contém start/end em segundos. [] significa gravação toda negativa.
        if not isinstance(truth, list) or any(not 0 <= t["start"] < t["end"] <= len(audio)/RATE for t in truth):
            raise ValueError("Anotações devem ser lista de intervalos start/end dentro do áudio.")
        used, latencies, fp = set(), [], 0
        for event in events:
            match = next((i for i, t in enumerate(truth)
                          if i not in used and t["end"] <= event <= t["end"] + 2.5), None)
            if match is None:
                fp += 1
            else:
                used.add(match)
                latencies.append(event - truth[match]["end"])
        report["event_metrics"] = {"tp": len(used), "fp": fp, "fn": len(truth)-len(used),
                                   "false_alerts_per_hour": fp/(len(audio)/RATE/3600),
                                   "latency_after_target_end_s": latencies,
                                   "matching_tolerance_s": 2.5}
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "windows"}, indent=2, ensure_ascii=False))
    return report
