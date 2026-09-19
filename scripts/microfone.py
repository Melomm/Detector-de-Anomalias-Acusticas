import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import sounddevice as sd

from detector67.microphone import stream


def main():
    parser = argparse.ArgumentParser(description="Microfone do PC para o ESP32 no Wokwi")
    parser.add_argument("--preparar", action="store_true", help="Compilar o Wokwi com o modelo existente")
    parser.add_argument("--listar", action="store_true", help="Listar microfones")
    parser.add_argument("--dispositivo", type=int, help="Índice do microfone na lista")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        if args.listar:
            print(sd.query_devices())
            return 0
        model = root / "artifacts/model.npz"
        if not model.exists():
            raise ValueError("Execute treinar.cmd primeiro.")
        if args.preparar:
            return subprocess.run([sys.executable, "-X", "utf8", str(root / "scripts/prepare_wokwi.py"), "--mic"], cwd=root).returncode
        report_path = root / "artifacts/wokwi/simulation.json"
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
        if report.get("mode") != "mic" or report.get("model_sha256") != hashlib.sha256(model.read_bytes()).hexdigest():
            raise ValueError("Execute .\\microfone.cmd --preparar e inicie Wokwi: Start Simulator.")
        stream(args.dispositivo)
        return 0
    except KeyboardInterrupt:
        print("\nMicrofone desligado.")
        return 0
    except (OSError, ValueError, RuntimeError, sd.PortAudioError) as exc:
        print(f"Não foi possível usar o microfone: {exc}\nAbra Wokwi: Start Simulator e mantenha a aba visível.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
