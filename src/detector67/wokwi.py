"""Empacota WAVs e o modelo existente; nunca treina com dados de demonstração."""
import hashlib
import json
from pathlib import Path

import numpy as np

from .data import ALL_LABELS, load_manifest
from .dsp import FEATURES, RATE, read_audio
from .model import export_header, load_model

MAX_AUDIO_SECONDS = 20


def dataset_entries(manifest, per_class=1):
    if per_class < 1:
        raise ValueError("per-class deve ser pelo menos 1")
    rows = load_manifest(manifest)
    chosen = []
    for label in ALL_LABELS:
        candidates = [r for r in rows if r["split"] == "test" and r["label"] == label]
        for row in candidates[:per_class]:
            chosen.append({"path": row["path"], "label": label, "session": row["session"], "split": "test"})
    return chosen


def export_audio_bank(entries, path):
    if not entries:
        raise ValueError("Selecione pelo menos um áudio para a simulação")
    clips, offsets, lengths, names, playlist = [], [], [], [], []
    total = 0
    for i, entry in enumerate(entries):
        audio = read_audio(Path(entry["path"]))
        if total + len(audio) > MAX_AUDIO_SECONDS * RATE:
            raise ValueError("A playlist excede 20 s de áudio em flash. Use menos clipes ou um WAV menor.")
        # Padding até múltiplo do bloco; não cortar a última palavra.
        pcm = np.rint(np.clip(audio * 32768, -32768, 32767)).astype("<i2")
        name = f"wav_{i:02d}_{entry['label']}"
        offsets.append(total)
        lengths.append(len(pcm))
        names.append(name)
        clips.append(pcm)
        total += len(pcm)
        playlist.append({**entry, "id": i, "name": name, "duration_s": len(pcm)/RATE,
                         "pcm_sha256": hashlib.sha256(pcm.tobytes()).hexdigest()})
    values = np.concatenate(clips)
    lines = [", ".join(str(int(v)) for v in values[i:i+20]) for i in range(0, len(values), 20)]
    text = "// Gerado de WAVs reais. PCM16 mono 16 kHz armazenado em flash.\n#pragma once\n#include <cstdint>\n"
    text += f"static const int SIM_CLIP_COUNT = {len(clips)};\n"
    for label, data in (("OFFSETS", offsets), ("LENGTHS", lengths)):
        text += f"static const uint32_t SIM_CLIP_{label}[] = {{" + ", ".join(map(str, data)) + "};\n"
    text += "static const char* const SIM_CLIP_NAMES[] = {" + ", ".join(json.dumps(n) for n in names) + "};\n"
    text += f"static const int16_t SIM_PCM[{total}] = {{\n" + ",\n".join(lines) + "\n};\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return playlist


def prepare_model_assets(model_dir, generated):
    params, metadata = load_model(model_dir)
    expected = {"sample_rate": RATE, "features": FEATURES, "window_seconds": 2,
                "stride_seconds": .25, "confirmations": 2, "cooldown_seconds": 1.5}
    if any(metadata.get(k) != v for k, v in expected.items()):
        raise ValueError("Metadados incompatíveis com o firmware; use o treinamento deste projeto")
    shapes = {"mean": (FEATURES,), "scale": (FEATURES,), "w1": (FEATURES, 32),
              "b1": (32,), "w2": (32, 1), "b2": (1,)}
    if any(k not in params or params[k].shape != shape or not np.isfinite(params[k]).all() for k, shape in shapes.items()):
        raise ValueError("Pesos inválidos ou arquitetura incompatível")
    if np.any(params["scale"] <= 0) or not np.isfinite(metadata["threshold"]):
        raise ValueError("Normalização ou limiar inválido")
    generated = Path(generated)
    export_header(params, metadata["threshold"], generated / "model_data.h")
    return {"speech_model": True,
            "model_sha256": hashlib.sha256(Path(model_dir, "model.npz").read_bytes()).hexdigest(),
            "threshold": metadata["threshold"]}


def prepare_real_assets(entries, model_dir, generated):
    report = prepare_model_assets(model_dir, generated)
    playlist = export_audio_bank(entries, Path(generated) / "audio_data.h")
    return {**report, "mode": "audio", "playlist": playlist, "pre_silence_s": 2, "post_silence_s": 3}
