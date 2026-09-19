import csv
import hashlib
import re
from pathlib import Path

LABELS = ("alvo", "seis", "sete", "invertido", "outros_numeros", "fala", "ruido", "silencio")
ALL_LABELS = LABELS + ("negativo",)


def identifier(value):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("Sessão deve conter apenas letras ASCII, números, _ e -.")
    return value


def make_manifest(root, destination, val_sessions, test_sessions):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    val_sessions, test_sessions = set(val_sessions), set(test_sessions)
    if not val_sessions or not test_sessions or val_sessions & test_sessions:
        raise ValueError("Informe sessões distintas e não vazias de validação e teste.")
    rows, found = [], set()
    for path in sorted(root.glob("*/*/*.wav")):
        label, session = path.relative_to(root).parts[:2]
        if label not in ALL_LABELS:
            raise ValueError(f"Classe desconhecida: {label}")
        found.add(session)
        split = "val" if session in val_sessions else "test" if session in test_sessions else "train"
        rows.append(dict(path=path.as_posix(), label=label, session=session, split=split))
    missing = (val_sessions | test_sessions) - found
    if missing:
        raise ValueError(f"Sessões sem WAVs: {sorted(missing)}")
    validate_rows(rows)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "session", "split"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def validate_rows(rows):
    sessions, hashes = {}, {}
    for row in rows:
        if row["label"] not in ALL_LABELS or row["split"] not in ("train", "val", "test"):
            raise ValueError(f"Rótulo ou split inválido: {row}")
        previous = sessions.setdefault(row["session"], row["split"])
        if previous != row["split"]:
            raise ValueError(f"Vazamento: sessão {row['session']} está em vários splits")
        from .dsp import training_clip
        import numpy as np
        # Hash das amostras decodificadas: detecta duplicatas com metadados distintos.
        digest = hashlib.sha256(np.asarray(training_clip(Path(row["path"])), dtype="<f4").tobytes()).hexdigest()
        if digest in hashes:
            raise ValueError(f"Áudio duplicado: {row['path']} e {hashes[digest]}")
        hashes[digest] = row["path"]
    for split in ("train", "val", "test"):
        present = {r["label"] for r in rows if r["split"] == split}
        if "alvo" not in present or not (present - {"alvo"}):
            raise ValueError(f"Split {split}: precisa de exemplos positivos e negativos.")


def load_manifest(path):
    path = Path(path).resolve()
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        item = Path(row["path"])
        row["path"] = str(item if item.is_absolute() else path.parent / item)
    validate_rows(rows)
    return rows
