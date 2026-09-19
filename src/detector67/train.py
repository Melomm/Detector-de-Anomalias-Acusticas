import copy
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, log_loss, precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from .data import load_manifest
from .dsp import augment, extract, training_clip
from .model import export, export_header, probability


def metrics(labels, scores, threshold, categories):
    truth = np.asarray(labels, dtype=int)
    detected = scores >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(truth, detected, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(truth, detected, labels=[0, 1]).ravel()
    categories = np.asarray(categories)
    return {"n": len(truth), "precision": float(precision), "recall": float(recall), "f1": float(f1),
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
            "false_positive_rate": float(fp / max(1, tn+fp)),
            "detection_rate_by_category": {c: float(detected[categories == c].mean()) for c in sorted(set(categories))}}


def select_threshold(y, scores, max_fpr):
    candidates = np.unique(np.concatenate(([0.05], scores, [np.nextafter(np.float32(1), np.float32(2))])))
    best = None
    for t in candidates:
        pred = scores >= t
        fpr = float(pred[y == 0].mean())
        recall = float(pred[y == 1].mean())
        if fpr <= max_fpr:
            rank = (recall, -fpr, float(t))
            if best is None or rank > best[0]:
                best = (rank, float(t))
    return best[1]


def train(manifest, output, header, epochs=150, seed=67, copies=2, max_fpr=0.02):
    if epochs < 1 or copies < 0 or not 0 <= max_fpr <= 1:
        raise ValueError("epochs >= 1, copies >= 0 e max-fpr entre 0 e 1 são obrigatórios")
    rows = load_manifest(manifest)
    rng = np.random.default_rng(seed)
    sets = {}
    for split in ("train", "val", "test"):
        x, y, categories = [], [], []
        for row in rows:
            if row["split"] != split:
                continue
            audio = training_clip(Path(row["path"]))
            clips = [audio] + ([augment(audio, rng) for _ in range(copies)] if split == "train" else [])
            for clip in clips:
                x.append(extract(clip)[0])
                y.append(int(row["label"] == "alvo"))
                categories.append(row["label"])
        sets[split] = (np.asarray(x, dtype=np.float32), np.array(y), categories)
    x, y, _ = sets["train"]
    scaler = StandardScaler().fit(x)
    scale = np.maximum(scaler.scale_, 0.1).astype(np.float32)
    mean = scaler.mean_.astype(np.float32)
    normalized = (x - mean) / scale
    val_x, val_y, _ = sets["val"]
    clf = MLPClassifier(hidden_layer_sizes=(32,), activation="relu", solver="adam",
                        batch_size=min(64, len(y)), learning_rate_init=0.001, alpha=0.01,
                        random_state=seed)
    best, best_loss, stale, history = None, float("inf"), 0, []
    indices = [np.flatnonzero(y == c) for c in (0, 1)]
    for epoch in range(epochs):
        # Balanceamento apenas do treino; validação/teste mantêm distribuição original.
        order = np.concatenate([rng.choice(ids, max(map(len, indices)), replace=True) for ids in indices])
        rng.shuffle(order)
        clf.partial_fit(normalized[order], y[order], classes=[0, 1])
        loss = float(log_loss(val_y, clf.predict_proba((val_x - mean) / scale), labels=[0, 1]))
        history.append({"epoch": epoch + 1, "validation_loss": loss})
        if loss < best_loss - 1e-5:
            best, best_loss, stale = copy.deepcopy(clf), loss, 0
        else:
            stale += 1
        if (epoch + 1) % 10 == 0:
            print(f"Época {epoch+1}: perda de validação={loss:.4f}", flush=True)
        if stale >= 20:
            break
    params = {"mean": mean, "scale": scale, "w1": best.coefs_[0].astype(np.float32),
              "b1": best.intercepts_[0].astype(np.float32), "w2": best.coefs_[1].astype(np.float32),
              "b2": best.intercepts_[1].astype(np.float32)}
    threshold = select_threshold(val_y, probability(val_x, params), max_fpr)
    export(params, threshold, output)
    # Valida a exportação com as mesmas entradas de teste, sem reajustar modelo/limiar.
    import onnxruntime as ort
    session = ort.InferenceSession(str(Path(output) / "model.onnx"), providers=["CPUExecutionProvider"])
    test_x = sets["test"][0]
    onnx_scores = session.run(None, {"features": test_x})[0].ravel()
    np.testing.assert_allclose(onnx_scores, probability(test_x, params), atol=2e-5, rtol=2e-5)
    export_header(params, threshold, header)
    report = {"seed": seed, "threshold": threshold, "requested_validation_max_fpr": max_fpr,
              "best_validation_loss": best_loss, "history": history,
              "note": "Métricas por clipe. Validar eventos e falsos alertas/hora em gravações contínuas.",
              "sessions": {s: sorted({r['session'] for r in rows if r['split'] == s}) for s in sets},
              "validation": metrics(val_y, probability(val_x, params), threshold, sets["val"][2]),
              "test": metrics(sets["test"][1], onnx_scores, threshold, sets["test"][2]),
              "onnx_max_absolute_error": float(np.max(np.abs(onnx_scores - probability(test_x, params))))}
    Path(output, "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["test"], indent=2, ensure_ascii=False))
    if threshold > 1 or report["validation"]["recall"] == 0:
        print("ATENÇÃO: nenhum alvo validado no limite de falsos positivos. Colete mais dados.")
    return report
