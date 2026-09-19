"""Compila usando cache local e caminhos curtos para o toolchain ESP32 no Windows."""
import argparse
import ctypes
import os
from pathlib import Path
import subprocess
import sys

def platformio_environment(root):
    cache = root / ".platformio"
    cache.mkdir(exist_ok=True)
    cache_path = str(cache)
    if os.name == "nt":
        buffer = ctypes.create_unicode_buffer(32768)
        if ctypes.windll.kernel32.GetShortPathNameW(cache_path, buffer, len(buffer)):
            cache_path = buffer.value
    env = os.environ.copy()
    env["PLATFORMIO_CORE_DIR"] = cache_path
    env["PIP_NO_CACHE_DIR"] = "1"
    return env


def build(root, environment, upload=False, port=None):
    if upload and environment.startswith("wokwi_"):
        raise ValueError("Ambientes wokwi são para simulação. Use detector/recorder na placa física.")
    command = [sys.executable, "-m", "platformio", "run", "-d", str(root / "firmware"), "-e", environment, "-j", "2"]
    if upload:
        command += ["-t", "upload"]
    if port:
        command += ["--upload-port", port]
    subprocess.run(command, env=platformio_environment(root), check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", choices=["detector", "diagnostic", "mic_test", "recorder", "wokwi_demo", "wokwi_audio", "wokwi_mic"], default="detector")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--port")
    args = parser.parse_args()
    try:
        build(Path(__file__).resolve().parents[1], args.environment, args.upload, args.port)
    except (ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Falha no build: {exc}\n")
