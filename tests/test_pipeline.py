import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from detector67.data import LABELS, load_manifest, make_manifest
from detector67.dsp import FEATURES, RATE, SAMPLES, augment, extract, windows
from detector67.model import export, load_model, probability
from detector67.runtime import EventGate, replay
from detector67.train import select_threshold, train


def tone_pair(reverse=False):
    audio = np.zeros(SAMPLES, dtype=np.float32)
    t = np.arange(6000)/RATE
    frequencies = (1200, 400) if reverse else (400, 1200)
    for start, freq in zip((5000, 16000), frequencies):
        audio[start:start+len(t)] = np.sin(2*np.pi*freq*t)*np.hanning(len(t))*0.2
    return audio


def test_features_preserve_order_and_silence_is_finite():
    a, rms, centroid = extract(tone_pair())
    b, _, _ = extract(tone_pair(True))
    assert a.shape == (FEATURES,)
    assert np.linalg.norm(a-b) > 5
    assert rms > 0 and centroid > 0
    assert np.isfinite(extract(np.zeros(SAMPLES))[0]).all()
    with pytest.raises(ValueError):
        extract(np.zeros(100))


def test_augmentation_does_not_cut_words():
    audio = tone_pair()
    for seed in range(10):
        result = augment(audio, np.random.default_rng(seed))
        assert result.shape == audio.shape
        assert np.count_nonzero(abs(result) > 0.02) > 5000


def test_streaming_windows_include_final_partial_window():
    audio = np.ones(SAMPLES+5000, dtype=np.float32)
    result = list(windows(audio))
    assert [time for time, _ in result] == [2, 2.25, 2.5]
    assert result[-1][1][-1] == 0
    assert len(list(windows(np.ones(100)))) == 1


def test_gate_confirmation_rearming_and_gaps():
    gate = EventGate(0.8)
    assert not gate.update(0.9, 2)
    assert gate.update(0.9, 2.25)
    assert not gate.update(0.99, 4)  # Continuar positivo não repete evento.
    gate.update(0.1, 4.25)
    gate.update(0.1, 4.5)
    assert not gate.update(0.99, 4.75)
    assert not gate.update(0.99, 5.5)  # Lacuna quebra consecutividade.
    assert gate.update(0.99, 5.75)


def test_threshold_uses_negative_examples():
    y = np.array([0, 0, 1, 1])
    score = np.array([0.1, 0.81, 0.8, 0.9], dtype=np.float32)
    threshold = select_threshold(y, score, 0)
    assert threshold > 0.81
    assert np.sum(score[y == 1] >= threshold) == 1


@pytest.fixture
def dataset(tmp_path):
    rng = np.random.default_rng(123)
    for session in ("dia1", "dia2", "dia3"):
        for index, label in enumerate(LABELS):
            folder = tmp_path / "raw" / label / session
            folder.mkdir(parents=True)
            audio = tone_pair(reverse=label != "alvo") * (1-index*0.05)
            audio += rng.normal(0, 0.0005, SAMPLES).astype(np.float32)
            sf.write(folder / "a.wav", audio, RATE, subtype="PCM_16")
    manifest = tmp_path / "manifest.csv"
    make_manifest(tmp_path / "raw", manifest, ["dia2"], ["dia3"])
    return manifest


def test_session_split_and_duplicate_protection(dataset):
    rows = load_manifest(dataset)
    assert len(rows) == 24
    assert {r["session"] for r in rows if r["split"] == "train"} == {"dia1"}
    Path(rows[-1]["path"]).write_bytes(Path(rows[0]["path"]).read_bytes())
    with pytest.raises(ValueError, match="duplicado"):
        load_manifest(dataset)


def test_training_export_and_replay_end_to_end(dataset, tmp_path):
    output = tmp_path / "model"
    report = train(dataset, output, tmp_path / "model_data.h", epochs=3, copies=0)
    assert report["test"]["n"] == 8
    assert report["onnx_max_absolute_error"] < 2e-5
    assert (output / "model.onnx").exists()
    assert "MODEL_READY 1" in (tmp_path / "model_data.h").read_text()
    params, meta = load_model(output)
    assert meta["target"] == ["seis sete", "six seven"]
    from detector67.wokwi import dataset_entries, prepare_real_assets
    playlist = dataset_entries(dataset)
    sim = prepare_real_assets(playlist, output, tmp_path / "wokwi")
    assert sim["speech_model"] is True  # Ramo real; pesos de teste isolados neste fixture.
    assert len(sim["playlist"]) == 8
    assert all(item["split"] == "test" for item in sim["playlist"])
    assert (tmp_path / "wokwi/model_data.h").read_bytes() == (tmp_path / "model_data.h").read_bytes()
    wav = tmp_path / "stream.wav"
    sf.write(wav, np.tile(tone_pair(), 3), RATE)
    annotations = tmp_path / "annotations.json"
    annotations.write_text("[]")
    result = replay(wav, output, tmp_path / "replay.json", annotations)
    assert result["duration_s"] == 6
    assert len(result["windows"]) == 17
    assert result["event_metrics"]["fn"] == 0
    assert np.isfinite(probability(extract(tone_pair())[0][None], params)).all()
