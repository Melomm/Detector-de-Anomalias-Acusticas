import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

from detector67.dsp import RATE
from detector67.wokwi import export_audio_bank, dataset_entries


def test_audio_export_preserves_pcm_and_budget(tmp_path):
    audio = np.array([-32768, -123, 0, 123, 32767], dtype=np.int16)
    wav = tmp_path / "clip.wav"
    sf.write(wav, audio, RATE, subtype="PCM_16")
    header = tmp_path / "audio_data.h"
    report = export_audio_bank([{"path": str(wav), "label": "alvo"}], header)
    text = header.read_text()
    assert "-32768, -123, 0, 123, 32767" in text
    assert "SIM_CLIP_LENGTHS[] = {5}" in text
    assert report[0]["duration_s"] == 5/RATE
    with pytest.raises(ValueError, match="pelo menos"):
        export_audio_bank([], header)
    sf.write(tmp_path / "long.wav", np.zeros(21*RATE), RATE)
    with pytest.raises(ValueError, match="20 s"):
        export_audio_bank([{"path": str(tmp_path / "long.wav"), "label": "fala"}], header)


def test_dataset_selection_only_uses_test(monkeypatch):
    rows = [{"path": f"{split}.wav", "label": "alvo", "session": split, "split": split}
            for split in ("train", "val", "test")]
    monkeypatch.setattr("detector67.wokwi.load_manifest", lambda _: rows)
    chosen = dataset_entries("unused.csv")
    assert len(chosen) == 1
    assert chosen[0]["path"] == "test.wav"


def test_demo_pipeline_and_player_controls(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        candidate = Path("C:/msys64/ucrt64/bin/g++.exe")
        compiler = str(candidate) if candidate.exists() else None
    if not compiler:
        pytest.skip("g++/clang++ necessário para teste C++ do simulador")
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / ("simulation.exe" if os.name == "nt" else "simulation")
    command = [compiler, "-std=c++17", "-O2", "-D", "SIMULATOR_DEMO=1",
               "-I", str(root / "firmware/include"), str(root / "tests/simulation_runner.cpp"), "-o", str(output)]
    if os.name == "nt":
        command.append("-static")
    subprocess.run(command, check=True, capture_output=True)
    result = subprocess.run([str(output)], check=True, capture_output=True, text=True)
    assert "clip=0 events=1" in result.stdout
    assert "clip=3 events=0" in result.stdout


def test_wokwi_diagram_buttons_match_firmware():
    root = Path(__file__).resolve().parents[1]
    diagram = json.loads((root / "diagram.json").read_text())
    assert len({p['id'] for p in diagram['parts']}) == len(diagram['parts'])
    links = {(c[0], c[1]) for c in diagram['connections']}
    for pin, button in ((18, "next"), (19, "restart"), (23, "pause")):
        assert (f"esp:D{pin}", f"{button}:1.l") in links
        assert (f"{button}:2.l", "esp:GND.1") in links
