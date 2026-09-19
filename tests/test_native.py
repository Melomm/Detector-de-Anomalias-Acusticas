"""Compila o DSP e a inferência C++ reais para comparar com Python/ONNX."""
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from detector67.dsp import FEATURES, SAMPLES, extract
from detector67.model import export, export_header, probability


def test_cpp_dsp_and_inference_match_python(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        compiler = next((str(p) for p in (Path("C:/msys64/ucrt64/bin/g++.exe"),
                                          Path("C:/msys64/mingw64/bin/g++.exe")) if p.exists()), None)
    if not compiler:
        pytest.skip("Instale g++/clang++ para testar paridade do firmware nativo")
    root = Path(__file__).resolve().parents[1]
    include = root / "firmware" / "include"
    rng = np.random.default_rng(42)
    params = {"mean": np.full(FEATURES, -12, dtype=np.float32),
              "scale": np.full(FEATURES, 5, dtype=np.float32),
              "w1": rng.normal(0, .03, (FEATURES, 32)).astype(np.float32),
              "b1": np.zeros(32, dtype=np.float32),
              "w2": rng.normal(0, .03, (32, 1)).astype(np.float32), "b2": np.zeros(1, dtype=np.float32)}
    export_header(params, .8, tmp_path / "model_data.h")
    export(params, .8, tmp_path)
    # Usa os cabeçalhos reais, mas pesos sintéticos isolados do modelo do usuário.
    shutil.copy(include / "inference.h", tmp_path / "inference.h")
    shutil.copy(root / "tests" / "native_runner.cpp", tmp_path / "runner.cpp")
    executable = tmp_path / ("runner.exe" if os.name == "nt" else "runner")
    args = [compiler, "-std=c++17", "-O2", "-I", str(include), str(tmp_path / "runner.cpp"), "-o", str(executable)]
    if os.name == "nt":
        args.append("-static")
    subprocess.run(args, check=True, capture_output=True)
    import onnxruntime as ort
    model = ort.InferenceSession(str(tmp_path / "model.onnx"), providers=["CPUExecutionProvider"])
    for audio in (np.zeros(SAMPLES), rng.normal(0, .03, SAMPLES),
                  np.sin(np.arange(SAMPLES)*0.12)*0.2):
        pcm = (audio*32768).astype("<i2")
        pcm.tofile(tmp_path / "audio.raw")
        subprocess.run([str(executable), str(tmp_path / "audio.raw"), str(tmp_path / "out.raw")], check=True)
        actual = np.fromfile(tmp_path / "out.raw", dtype="<f4")
        features, rms, centroid = extract(pcm.astype(np.float32)/32768)
        np.testing.assert_allclose(actual[:FEATURES], features, atol=0.01, rtol=0.001)
        np.testing.assert_allclose(actual[FEATURES:FEATURES+2], [rms, centroid], atol=0.02, rtol=0.001)
        score = model.run(None, {"features": features[None]})[0].item()
        assert abs(actual[-1]-score) < 0.0002
