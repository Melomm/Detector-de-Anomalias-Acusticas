"""Monta cenário contínuo com WAVs REAIS reservados para teste e intervalos do alvo."""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from detector67.data import load_manifest
from detector67.dsp import RATE, read_audio
from detector67.runtime import replay

parser = argparse.ArgumentParser()
parser.add_argument("--manifest", default="data/manifest.csv")
parser.add_argument("--model", default="artifacts")
parser.add_argument("--output", default="artifacts/simulation")
parser.add_argument("--seed", type=int, default=67)
args = parser.parse_args()
rows = [r for r in load_manifest(args.manifest) if r["split"] == "test"]
rng = np.random.default_rng(args.seed)
rng.shuffle(rows)
clips, annotations, cursor = [], [], 0
for row in rows:
    gap = np.zeros(int(rng.uniform(2.5, 4)*RATE), dtype=np.float32)
    clips.append(gap)
    cursor += len(gap)
    audio = read_audio(Path(row["path"]))
    if row["label"] == "alvo":
        active = np.flatnonzero(abs(audio) > max(0.003, float(abs(audio).max())*0.08))
        start, end = (int(active[0]), int(active[-1])+1) if len(active) else (0, len(audio))
        annotations.append({"start": (cursor+start)/RATE, "end": (cursor+end)/RATE})
    clips.append(audio)
    cursor += len(audio)
clips.append(np.zeros(3*RATE, dtype=np.float32))
out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
sf.write(out / "scenario.wav", np.concatenate(clips), RATE, subtype="PCM_16")
(out / "annotations.json").write_text(json.dumps(annotations, indent=2), encoding="utf-8")
replay(out / "scenario.wav", args.model, out / "report.json", out / "annotations.json")
print("Intervalos estimados por energia: revise manualmente antes de reportar latência de fala.")
