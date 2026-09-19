import os
from pathlib import Path
import subprocess
import sys

from detector67.simple import prepare
from detector67.train import train


def main():
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    try:
        rows = prepare(root / "audios", root / "data/manifest.csv")
        print(f"{len(rows)} gravações: " + ", ".join(
            f"{sum(r['split'] == split for r in rows)} {name}" for split, name in
            (("train", "treino"), ("val", "validação"), ("test", "teste"))), flush=True)
        train(root / "data/manifest.csv", root / "artifacts", root / "firmware/include/model_data.h")
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Não foi possível treinar: {exc}", file=sys.stderr)
        return 1
    print("Modelo salvo. Preparando o Wokwi...", flush=True)
    result = subprocess.run([sys.executable, str(root / "scripts/prepare_wokwi.py"),
                             "--mic"], cwd=root)
    if result.returncode:
        print("O modelo foi treinado, mas o build do Wokwi falhou. Os pesos continuam em artifacts/.", file=sys.stderr)
        return result.returncode
    print("Pronto. Para testar: testar.cmd. No VS Code: Wokwi: Start Simulator.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
