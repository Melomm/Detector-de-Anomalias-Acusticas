import argparse
from pathlib import Path
import subprocess
import sys

from build_firmware import build
from detector67.wokwi import prepare_model_assets


def main():
    parser = argparse.ArgumentParser(description="Incorporar um modelo treinado e compilar o firmware, sem upload.")
    parser.add_argument("--modelo", type=Path, default=Path("artifacts"),
                        help="Pasta com model.npz e metadata.json; padrão: artifacts")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    model = (root / args.modelo).resolve()
    try:
        missing = [name for name in ("model.npz", "metadata.json") if not (model / name).is_file()]
        if missing:
            raise ValueError(f"Modelo incompleto em {model}: faltam {', '.join(missing)}.")
        report = prepare_model_assets(model, root / "firmware/include")
        print(f"Modelo: {model}\nLimiar: {report['threshold']:.9f}", flush=True)
        print("Pesos incorporados em firmware/include/model_data.h. Compilando...", flush=True)
        build(root, "detector", upload=False)
        binary = root / "firmware/.pio/build/detector/firmware.bin"
        if not binary.is_file():
            raise RuntimeError("A compilação terminou sem gerar firmware.bin.")
        print(f"\nFirmware compilado: {binary}")
        print("Nenhum arquivo foi enviado ao ESP32.")
        print("Para enviar depois (substitua COM3 pela porta da placa):")
        print(r".\.venv\Scripts\python.exe scripts\build_firmware.py --environment detector --upload --port COM3")
        return 0
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Não foi possível compilar o modelo: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
