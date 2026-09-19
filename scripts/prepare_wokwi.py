"""Um comando gera firmware, circuito e pacote para Wokwi (web ou VS Code)."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from build_firmware import build, platformio_environment
from detector67.wokwi import dataset_entries, prepare_real_assets, prepare_model_assets


def main():
    parser = argparse.ArgumentParser(description="Preparar simulação Wokwi; padrão: demonstração sem fala")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--demo", action="store_true")
    modes.add_argument("--mic", action="store_true", help="Usar o microfone do PC com modelo no ESP32")
    modes.add_argument("--dataset", action="store_true", help="Usar áudios do split de teste e modelo treinado")
    modes.add_argument("--audio", help="Usar um WAV contínuo de até 20 segundos e modelo treinado")
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--model", default="artifacts")
    parser.add_argument("--per-class", type=int, default=1)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    real = args.dataset or bool(args.audio)
    environment = "wokwi_mic" if args.mic else "wokwi_audio" if real else "wokwi_demo"
    out = root / "artifacts" / "wokwi"
    try:
        if args.mic:
            report = prepare_model_assets(args.model, root / "firmware/generated/wokwi")
            report.update(mode="mic", sample_rate=16000, serial_baud=921600, serial_port=4000)
        elif real:
            entries = dataset_entries(args.manifest, args.per_class) if args.dataset else [
                {"path": str(Path(args.audio).resolve()), "label": "externo", "split": "external"}]
            report = prepare_real_assets(entries, args.model, root / "firmware/generated/wokwi")
        else:
            report = {"mode": "demo", "speech_model": False,
                      "note": "Tons de 440/1100 Hz; regra demonstrativa, sem reconhecimento de fala.",
                      "playlist": ["par de tons", "440 isolado", "1100 isolado", "ordem inversa", "ruido", "silencio"],
                      "pre_silence_s": 2, "post_silence_s": 3}
        build(root, environment)
        binary_dir = root / "firmware/.pio/build" / environment
        core = Path(platformio_environment(root)["PLATFORMIO_CORE_DIR"])
        esptool = core / "packages/tool-esptoolpy/esptool.py"
        # Só troca o pacote público depois de build/merge bem-sucedidos.
        out.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=out, prefix="build-") as scratch:
            merged = Path(scratch) / "firmware.bin"
            subprocess.run([sys.executable, str(esptool), "--chip", "esp32", "merge_bin", "-o", str(merged),
                            "--flash_mode", "dio", "--flash_freq", "40m", "--flash_size", "4MB",
                            "0x1000", str(binary_dir / "bootloader.bin"),
                            "0x8000", str(binary_dir / "partitions.bin"),
                            "0xe000", str(core / "packages/framework-arduinoespressif32/tools/partitions/boot_app0.bin"),
                            "0x10000", str(binary_dir / "firmware.bin")], check=True)
            shutil.copyfile(merged, out / "firmware.bin")
        shutil.copyfile(binary_dir / "firmware.elf", out / "firmware.elf")
        shutil.copyfile(root / "diagram.json", out / "diagram.json")
        (out / "wokwi.toml").write_text('[wokwi]\nversion = 1\nfirmware = "firmware.bin"\nelf = "firmware.elf"\nrfc2217ServerPort = 4000\n', encoding="utf-8")
        report["environment"] = environment
        (out / "simulation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWokwi preparado: {out}\nModo: {report['mode']} | modelo de fala: {report['speech_model']}")
        print("No VS Code, execute Wokwi: Start Simulator.")
        if args.mic:
            print("Depois execute .\\microfone.cmd no terminal, mantendo a aba do simulador visivel.")
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Não foi possível preparar o Wokwi: {exc}\n")


if __name__ == "__main__":
    main()
