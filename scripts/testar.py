import argparse
from pathlib import Path
import sys

from detector67.simple import evaluate, predict_file


def main():
    parser = argparse.ArgumentParser(description="Testar o modelo; opcionalmente informe um WAV.")
    parser.add_argument("audio", nargs="?")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        if not (root / "artifacts/model.npz").exists():
            raise ValueError("Execute treinar.cmd primeiro.")
        if args.audio:
            predict_file(Path(args.audio), root / "artifacts")
        else:
            evaluate(root / "data/manifest.csv", root / "artifacts")
            extra = sorted(p for p in (root / "audios/teste").rglob("*") if p.suffix.lower() in (".wav", ".flac"))
            if extra:
                print("\nÁudios novos (previsão por janela; sem rótulo para medir acerto):")
                for path in extra:
                    predict_file(path, root / "artifacts")
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Não foi possível testar: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
