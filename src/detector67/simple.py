import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .data import load_manifest, validate_rows
from .dsp import RATE, SAMPLES, extract, read_audio, training_clip, windows
from .model import load_model, probability
from .train import metrics


def prepare(root, manifest, seed=67):
    root, manifest = Path(root).resolve(), Path(manifest).resolve()
    previous = {}
    if manifest.exists():
        with manifest.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row["session"].startswith("arquivo_"):
                    previous[row["session"]] = row["split"]
    rng = np.random.default_rng(seed)
    rows, seen = [], {}
    for folder, label in (("positivo", "alvo"), ("negativo", "negativo")):
        directory = root / folder
        directory.mkdir(parents=True, exist_ok=True)
        files = sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in (".wav", ".flac"))
        if len(files) < 5:
            raise ValueError(f"Coloque ao menos 5 WAVs/FLACs em {directory}. Sugestão inicial: 20 por pasta.")
        group = []
        for path in files:
            audio = np.asarray(training_clip(path), dtype="<f4")
            session = "arquivo_" + hashlib.sha256(audio.tobytes()).hexdigest()
            if session in seen:
                raise ValueError(f"Áudio duplicado: {path} e {seen[session]}. Use gravações independentes.")
            seen[session] = path
            group.append(dict(path=path.as_posix(), label=label, session=session, split=previous.get(session)))
        rng.shuffle(group)
        count = max(1, len(group) // 5)
        targets = {"train": len(group) - 2 * count, "val": count, "test": count}
        assigned = {s: sum(r["split"] == s for r in group) for s in targets}
        for row in group:
            if row["split"] is None:
                split = max(targets, key=lambda s: targets[s] - assigned[s])
                row["split"] = split
                assigned[split] += 1
        rows.extend(group)
    validate_rows(rows)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "session", "split"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def evaluate(manifest, model_dir):
    params, meta = load_model(model_dir)
    rows = [row for row in load_manifest(manifest) if row["split"] == "test"]
    features = np.stack([extract(training_clip(Path(row["path"])))[0] for row in rows])
    scores = probability(features, params)
    report = metrics([int(row["label"] == "alvo") for row in rows], scores, meta["threshold"],
                     [row["label"] for row in rows])
    report["files"] = [{"path": row["path"], "expected": row["label"] == "alvo",
                        "detected": bool(score >= meta["threshold"]), "score": float(score)}
                       for row, score in zip(rows, scores)]
    print(f"Teste: {len(rows)} gravações reservadas.")
    print(f"Precisão: {report['precision']:.0%} | Recall: {report['recall']:.0%} | F1: {report['f1']:.0%}")
    print(f"Falsos positivos: {report['confusion_matrix'][0][1]} | Alvos perdidos: {report['confusion_matrix'][1][0]}")
    Path(model_dir, "teste.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def predict_file(path, model_dir):
    params, meta = load_model(model_dir)
    audio = read_audio(Path(path))
    if len(audio) <= SAMPLES:
        clips = [(len(audio) / RATE, training_clip(Path(path)))]
    else:
        clips = windows(audio)
    results = []
    for end, clip in clips:
        score = float(probability(extract(clip)[0][None], params)[0])
        detected = score >= meta["threshold"]
        results.append({"end_s": end, "score": score, "detected": detected})
        print(f"{Path(path).name} | {end:.2f}s | {'6 7 identificado' if detected else 'sem 6 7'} | score={score:.3f}")
    return results
