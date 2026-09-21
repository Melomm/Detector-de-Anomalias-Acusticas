"""MLP compacto; mesmos pesos em NumPy, ONNX e C++ no ESP32."""
import json
import hashlib
from pathlib import Path

import numpy as np
from scipy.special import expit

from .dsp import FEATURES, FRAME, HANN, MEL_FILTERS, MELS


def probability(x, params):
    z = (np.asarray(x, dtype=np.float32) - params["mean"]) / params["scale"]
    hidden = np.maximum(z @ params["w1"] + params["b1"], 0)
    return expit(hidden @ params["w2"] + params["b2"]).reshape(-1)


def load_model(folder):
    folder = Path(folder)
    with np.load(folder / "model.npz", allow_pickle=False) as data:
        params = {k: data[k] for k in data.files}
    meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    return params, meta


def export(params, threshold, output):
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "model.npz", **params)
    nodes = [helper.make_node(op, inputs, [name]) for op, inputs, name in [
        ("Sub", ["features", "mean"], "centered"),
        ("Div", ["centered", "scale"], "scaled"),
        ("MatMul", ["scaled", "w1"], "dense1"),
        ("Add", ["dense1", "b1"], "biased1"),
        ("Relu", ["biased1"], "hidden"),
        ("MatMul", ["hidden", "w2"], "dense2"),
        ("Add", ["dense2", "b2"], "logits"),
        ("Sigmoid", ["logits"], "probability"),
    ]]
    graph = helper.make_graph(nodes, "detector67",
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, [None, FEATURES])],
        [helper.make_tensor_value_info("probability", TensorProto.FLOAT, [None, 1])],
        [numpy_helper.from_array(v.astype(np.float32), k) for k, v in params.items()])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, output / "model.onnx")
    metadata = {"format_version": 1, "target": ["seis sete", "six seven"],
                "sample_rate": 16000, "window_seconds": 2, "stride_seconds": 0.25,
                "features": FEATURES, "threshold": float(threshold),
                "confirmations": 2, "cooldown_seconds": 1.5,
                "description": "Score de classificador; não é confiança calibrada."}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return model


def c_array(name, values):
    values = np.asarray(values, dtype=np.float32).ravel()
    lines = []
    for i in range(0, len(values), 8):
        lines.append("  " + ", ".join(f"{float(v):.9e}f" for v in values[i:i+8]))
    return f"static const float {name}[{len(values)}] = {{\n" + ",\n".join(lines) + "\n};\n"


def model_id(params, threshold):
    digest = hashlib.sha256()
    for name in sorted(params):
        values = np.asarray(params[name], dtype="<f4")
        digest.update(json.dumps([name, list(values.shape)]).encode("ascii"))
        digest.update(values.tobytes())
    digest.update(np.asarray([threshold], dtype="<f4").tobytes())
    return digest.hexdigest()


def export_header(params, threshold, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "// GERADO PELO TREINAMENTO. Não editar.\n#pragma once\n"
    text += f"#define MODEL_READY 1\n#define MODEL_HIDDEN {len(params['b1'])}\n"
    text += f'#define MODEL_ID "{model_id(params, threshold)}"\n'
    text += f"static const float MODEL_THRESHOLD = {threshold:.9e}f;\n"
    for name, data in params.items():
        text += c_array("MODEL_" + name.upper(), data)
    path.write_text(text, encoding="utf-8")


def export_dsp(path):
    # Filtros esparsos reduzem RAM/flash e multiplicações no ESP32.
    starts, lengths, weights = [], [], []
    for filt in MEL_FILTERS:
        indices = np.flatnonzero(filt)
        starts.append(int(indices[0]))
        lengths.append(len(indices))
        weights.extend(filt[indices])
    text = "// Gerado por detector67 dsp-header. Compartilhado com o DSP Python.\n#pragma once\n"
    text += c_array("DSP_HANN", HANN)
    text += f"static const int DSP_MEL_START[{MELS}] = {{" + ",".join(map(str, starts)) + "};\n"
    text += f"static const int DSP_MEL_LEN[{MELS}] = {{" + ",".join(map(str, lengths)) + "};\n"
    text += c_array("DSP_MEL_WEIGHTS", weights)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
