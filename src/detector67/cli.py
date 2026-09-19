import argparse
import json
from pathlib import Path
import time
import uuid

import numpy as np
import soundfile as sf

from .data import LABELS, identifier, make_manifest
from .dsp import RATE, SAMPLES


def record(args):
    identifier(args.session)
    if args.count < 1:
        raise ValueError("count deve ser positivo")
    folder = Path(args.data) / args.label / args.session
    folder.mkdir(parents=True, exist_ok=True)
    port = None
    if args.port:
        import serial
        port = serial.Serial(args.port, 921600, timeout=5)
        time.sleep(2)  # Reset ao abrir porta do ESP32.
        port.reset_input_buffer()
    else:
        import sounddevice as sd
    try:
        for i in range(args.count):
            input(f"[{i+1}/{args.count}] Classe {args.label}. Enter para preparar...")
            print("3", flush=True)
            time.sleep(0.5)
            print("2", flush=True)
            time.sleep(0.5)
            print("1", flush=True)
            time.sleep(0.5)
            if port:
                port.reset_input_buffer()
                port.write(b"R")
                if port.read_until(b"PCM67\n")[-6:] != b"PCM67\n":
                    raise RuntimeError("ESP32 não respondeu. Grave o ambiente 'recorder' antes.")
                print("GRAVANDO: fale agora (2 segundos)", flush=True)
                raw = port.read(SAMPLES * 2)
                if len(raw) != SAMPLES * 2:
                    raise RuntimeError("Áudio serial incompleto; verifique conexão e firmware recorder")
                audio = np.frombuffer(raw, dtype="<i2").astype(np.float32)/32768
            else:
                print("GRAVANDO: fale agora (2 segundos)", flush=True)
                audio = sd.rec(SAMPLES, samplerate=RATE, channels=1, dtype="float32", device=args.device)
                sd.wait()
                audio = audio[:, 0]
            peak = float(np.max(np.abs(audio)))
            rms = float(np.sqrt(np.mean(audio**2)))
            print(f"Pico={peak:.3f}; RMS={rms:.5f}")
            if peak >= 0.99:
                print("ATENÇÃO: possível saturação. Afaste o microfone e regrave se necessário.")
            if rms < 0.001 and args.label != "silencio":
                print("ATENÇÃO: áudio baixo. Confira microfone/canal antes de usar no treino.")
            answer = input("Enter salva; d descarta; p reproduz: ").strip().lower()
            if answer == "p":
                import sounddevice as sd
                sd.play(audio, RATE)
                sd.wait()
                keep = input("Enter salva; d descarta: ").strip().lower() != "d"
            else:
                keep = answer != "d"
            if keep:
                path = folder / f"{uuid.uuid4().hex[:12]}.wav"
                sf.write(path, audio, RATE, subtype="PCM_16")
                print(f"Salvo: {path}")
    finally:
        if port:
            port.close()


def main():
    parser = argparse.ArgumentParser(description="Detector de 'seis sete' / 'six seven' para ESP32")
    sub = parser.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("record", help="Gravar WAVs de 2 s no PC ou no INMP441")
    rec.add_argument("--label", required=True, choices=LABELS)
    rec.add_argument("--session", required=True, help="Pessoa/dia/ambiente, ex.: pessoa1_dia1_sala")
    rec.add_argument("--count", type=int, default=20)
    rec.add_argument("--data", default="data/raw")
    rec.add_argument("--device", type=int)
    rec.add_argument("--port", help="COM3 etc.; requer firmware recorder")
    sub.add_parser("devices", help="Listar microfones do computador")
    prep = sub.add_parser("prepare", help="Separar sessões inteiras, sem vazamento")
    prep.add_argument("--data", default="data/raw")
    prep.add_argument("--manifest", default="data/manifest.csv")
    prep.add_argument("--val-sessions", nargs="+", required=True)
    prep.add_argument("--test-sessions", nargs="+", required=True)
    fit = sub.add_parser("train", help="Treinar, avaliar e exportar ONNX + C++")
    fit.add_argument("--manifest", default="data/manifest.csv")
    fit.add_argument("--output", default="artifacts")
    fit.add_argument("--header", default="firmware/include/model_data.h")
    fit.add_argument("--epochs", type=int, default=150)
    fit.add_argument("--seed", type=int, default=67)
    fit.add_argument("--copies", type=int, default=2)
    fit.add_argument("--max-fpr", type=float, default=0.02)
    replay = sub.add_parser("replay", help="Detectar em WAV contínuo e medir desempenho")
    replay.add_argument("audio")
    replay.add_argument("--model", default="artifacts")
    replay.add_argument("--output", default="artifacts/replay.json")
    replay.add_argument("--annotations", help="JSON [{start: ..., end: ...}]; [] = sem alvo")
    header = sub.add_parser("dsp-header", help="Gerar constantes do DSP compartilhado")
    header.add_argument("--output", default="firmware/include/dsp_tables.h")
    args = parser.parse_args()
    try:
        if args.command == "record":
            record(args)
        elif args.command == "devices":
            import sounddevice as sd
            print(sd.query_devices())
        elif args.command == "prepare":
            rows = make_manifest(args.data, args.manifest, args.val_sessions, args.test_sessions)
            print(json.dumps({s: sum(r['split'] == s for r in rows) for s in ('train', 'val', 'test')}, indent=2))
        elif args.command == "train":
            from .train import train
            train(args.manifest, args.output, args.header, args.epochs, args.seed, args.copies, args.max_fpr)
        elif args.command == "replay":
            from .runtime import replay
            replay(args.audio, args.model, args.output, args.annotations)
        elif args.command == "dsp-header":
            from .model import export_dsp
            export_dsp(args.output)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Erro: {exc}\n")


if __name__ == "__main__":
    main()
