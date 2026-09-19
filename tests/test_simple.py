from collections import Counter

import numpy as np
import pytest
import soundfile as sf

from detector67.dsp import RATE, SAMPLES
from detector67.simple import evaluate, predict_file, prepare
from detector67.train import train
from detector67.wokwi import dataset_entries, prepare_real_assets


@pytest.fixture
def audio_folders(tmp_path):
    root = tmp_path / "audios"
    rng = np.random.default_rng(67)
    for folder in ("positivo", "negativo", "teste"):
        directory = root / folder
        directory.mkdir(parents=True)
        for index in range(20 if folder != "teste" else 1):
            audio = rng.normal(0, .01, SAMPLES).astype(np.float32)
            sf.write(directory / f"{index}.wav", audio, RATE)
    return root


def test_simple_split_is_balanced_and_stable(audio_folders, tmp_path):
    manifest = tmp_path / "manifest.csv"
    rows = prepare(audio_folders, manifest)
    assert len(rows) == 40
    for label in ("alvo", "negativo"):
        assert Counter(r["split"] for r in rows if r["label"] == label) == {
            "train": 12, "val": 4, "test": 4,
        }
    original = {r["session"]: r["split"] for r in rows}
    (audio_folders / "positivo/0.wav").rename(audio_folders / "positivo/renomeado.WAV")
    sf.write(audio_folders / "negativo/novo.flac", np.zeros(SAMPLES), RATE)
    updated = prepare(audio_folders, manifest)
    assert len(updated) == 41
    assert all(r["split"] == original[r["session"]] for r in updated if r["session"] in original)
    assert all("/teste/" not in r["path"] for r in updated)


def test_simple_rejects_copied_audio(audio_folders, tmp_path):
    (audio_folders / "negativo/copia.wav").write_bytes((audio_folders / "positivo/0.wav").read_bytes())
    with pytest.raises(ValueError, match="duplicado"):
        prepare(audio_folders, tmp_path / "manifest.csv")


def test_simple_train_test_and_wokwi(audio_folders, tmp_path):
    manifest, output = tmp_path / "manifest.csv", tmp_path / "model"
    prepare(audio_folders, manifest)
    train(manifest, output, tmp_path / "model_data.h", epochs=3, copies=0)
    report = evaluate(manifest, output)
    assert report["n"] == 8
    assert len(report["files"]) == 8
    assert (output / "teste.json").exists()
    predictions = predict_file(audio_folders / "teste/0.wav", output)
    assert len(predictions) == 1
    assert 0 <= predictions[0]["score"] <= 1
    entries = dataset_entries(manifest, per_class=4)
    assert Counter(e["label"] for e in entries) == {"alvo": 4, "negativo": 4}
    sim = prepare_real_assets(entries, output, tmp_path / "generated")
    assert sim["speech_model"] is True
    assert len(sim["playlist"]) == 8
