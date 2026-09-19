import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from detector67.microphone import BLOCK, Link, LiveBuffer, encode_frame, to_pcm


def test_pcm_resampling_and_clipping():
    raw = to_pcm(np.linspace(-2, 2, BLOCK), 16000)
    pcm = np.frombuffer(raw, dtype="<i2")
    assert pcm[0] == -32768 and pcm[-1] == 32767
    assert len(to_pcm(np.zeros(4800), 48000)) == BLOCK * 2
    with pytest.raises(ValueError):
        encode_frame(1, b"short")


def test_microphone_queue_bounded_and_marks_capture_gaps():
    buffer = LiveBuffer()
    audio = np.zeros((BLOCK, 1), dtype=np.float32)
    for _ in range(15):
        buffer.callback(audio, BLOCK, None, False)
    assert buffer.blocks.qsize() == 10
    assert buffer.dropped == 5
    assert buffer.blocks.get()[0] == 6
    buffer.callback(audio, BLOCK, None, True)
    items = []
    while not buffer.blocks.empty():
        items.append(buffer.blocks.get())
    assert items[-1][0] == 17


class SerialStub:
    def __init__(self, response):
        self.response = bytearray(response)
        self.sent = b""

    @property
    def in_waiting(self):
        return min(3, len(self.response))

    def read(self, n):
        result = bytes(self.response[:n])
        del self.response[:n]
        return result

    def write(self, data):
        self.sent += data
        return len(data)


def test_serial_fragmented_ack_and_telemetry():
    serial = SerialStub(b'boot\n{"event":true}\nMIC67_ACK 8\nMIC67_ACK 9\n')
    lines = []
    link = Link(serial, lines.append)
    link.send(8, bytes(BLOCK * 2))
    link.send(9, bytes(BLOCK * 2))
    assert lines == ["boot", '{"event":true}']
    assert serial.sent == encode_frame(8, bytes(BLOCK * 2)) + encode_frame(9, bytes(BLOCK * 2))
    with pytest.raises(OSError, match="transmissão"):
        Link(SerialStub(b"MIC67_ERROR crc\n")).send(1, bytes(BLOCK * 2))
    with pytest.raises(TimeoutError):
        Link(SerialStub(b"")).send(1, bytes(BLOCK * 2), timeout=.01)


def test_python_frames_decoded_by_firmware_with_crc_recovery(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        candidate = Path("C:/msys64/ucrt64/bin/g++.exe")
        compiler = str(candidate) if candidate.exists() else None
    if not compiler:
        pytest.skip("C++ compiler required")
    root = Path(__file__).resolve().parents[1]
    executable = tmp_path / ("mic.exe" if os.name == "nt" else "mic")
    command = [compiler, "-std=c++17", "-I", str(root / "firmware/include"),
               str(root / "tests/mic_runner.cpp"), "-o", str(executable)]
    if os.name == "nt":
        command.append("-static")
    subprocess.run(command, check=True, capture_output=True)
    pcm = np.linspace(-32768, 32767, BLOCK).astype("<i2").tobytes()
    broken = bytearray(encode_frame(6, pcm))
    broken[-1] ^= 1
    path = tmp_path / "frames.bin"
    path.write_bytes(b"noiseMIC" + encode_frame(5, pcm) + broken + encode_frame(7, pcm))
    result = subprocess.run([str(executable), str(path)], check=True, capture_output=True, text=True)
    assert result.stdout.splitlines() == ["5 -32768 32767", "corrupt", "7 -32768 32767"]
