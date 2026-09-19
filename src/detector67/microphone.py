import json
import math
import queue
import struct
import time
import zlib

import numpy as np
from scipy.signal import resample_poly

RATE = 16000
BLOCK = 1600
MAGIC = b"MIC67PCM"


def encode_frame(sequence, pcm):
    pcm = bytes(pcm)
    if len(pcm) != BLOCK * 2:
        raise ValueError("O bloco deve conter 1600 amostras PCM16.")
    body = struct.pack("<I", sequence & 0xffffffff) + pcm
    return MAGIC + body + struct.pack("<I", zlib.crc32(body))


def to_pcm(audio, rate):
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if rate != RATE:
        divisor = math.gcd(rate, RATE)
        audio = resample_poly(audio, RATE // divisor, rate // divisor)
    if len(audio) != BLOCK:
        raise ValueError("Tamanho inesperado do bloco do microfone.")
    return np.rint(np.clip(audio * 32768, -32768, 32767)).astype("<i2").tobytes()


class LiveBuffer:
    def __init__(self):
        self.blocks = queue.Queue(maxsize=10)
        self.sequence = 1
        self.dropped = 0

    def callback(self, data, frames, timing, status):
        if status:
            self.sequence += 1  # Mark capture gaps; firmware resets its audio window.
            self.dropped += 1
        item = (self.sequence, data.copy())
        self.sequence += 1
        try:
            self.blocks.put_nowait(item)
        except queue.Full:
            try:
                self.blocks.get_nowait()
            except queue.Empty:
                pass
            self.dropped += 1
            self.blocks.put_nowait(item)


class Link:
    def __init__(self, serial, on_line=print):
        self.serial = serial
        self.pending = bytearray()
        self.on_line = on_line

    def send(self, sequence, pcm, timeout=10):
        frame = encode_frame(sequence, pcm)
        if self.serial.write(frame) != len(frame):
            raise OSError("Envio de áudio incompleto.")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.pending.extend(self.serial.read(max(1, min(self.serial.in_waiting, 4096))))
            while b"\n" in self.pending:
                raw, _, rest = self.pending.partition(b"\n")
                self.pending = bytearray(rest)
                line = raw.decode("utf-8", errors="replace").strip()
                if line == f"MIC67_ACK {sequence & 0xffffffff}":
                    return
                if line.startswith("MIC67_ERROR"):
                    raise OSError("Falha na transmissão do áudio. Reinicie o simulador e o comando.")
                if line:
                    self.on_line(line)
            if len(self.pending) > 16384:
                raise OSError("Resposta serial inválida; confira o firmware do simulador.")
        raise TimeoutError("Wokwi não confirmou o áudio. Mantenha a aba visível e reinicie a simulação no modo mic.")


def stream(device=None):
    import serial
    import sounddevice as sd

    rate = RATE
    try:
        sd.check_input_settings(device=device, channels=1, dtype="float32", samplerate=rate)
    except sd.PortAudioError:
        rate = int(sd.query_devices(device, "input")["default_samplerate"])
        if rate % 10:
            raise ValueError("Microfone incompatível: selecione outro com --dispositivo.")
        sd.check_input_settings(device=device, channels=1, dtype="float32", samplerate=rate)

    last_print = 0.0

    def display(line):
        nonlocal last_print
        if not line.startswith("{"):
            print(line)
            return
        try:
            result = json.loads(line)
        except json.JSONDecodeError:
            return
        now = time.monotonic()
        if result.get("event") or now - last_print >= 1:
            label = "6 7 DETECTADO" if result.get("event") else "Ouvindo"
            print(f"{label} | score={result['score']:.3f} | nível={result['rms']:.4f}", flush=True)
            last_print = now

    buffer = LiveBuffer()
    with serial.serial_for_url("rfc2217://localhost:4000", baudrate=921600, timeout=.1) as connection:
        link = Link(connection, display)
        # Confirm the live firmware responds before opening the microphone.
        link.send(0, bytes(BLOCK * 2))
        name = sd.query_devices(device, "input")["name"]
        print(f"Microfone: {name}. Fale normalmente; Ctrl+C encerra. O áudio não é salvo.", flush=True)
        warned = 0
        with sd.InputStream(device=device, samplerate=rate, channels=1, dtype="float32",
                            blocksize=rate // 10, callback=buffer.callback):
            while True:
                try:
                    sequence, data = buffer.blocks.get(timeout=3)
                except queue.Empty:
                    raise OSError("O microfone não entregou áudio. Confira as permissões do Windows.") from None
                link.send(sequence, to_pcm(data, rate))
                if buffer.dropped != warned:
                    print("Áudio interrompido ou simulação atrasada; descartando blocos antigos. Mantenha o Wokwi visível.")
                    warned = buffer.dropped
