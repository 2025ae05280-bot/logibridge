import numpy as np


def make_interpreter(path):
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        try:
            from tensorflow.lite import Interpreter
        except ImportError as exc:
            raise ImportError("Install tflite-runtime in the container or tensorflow on the host") from exc
    return Interpreter(model_path=str(path))


def _quant(detail):
    params = detail.get("quantization_parameters", {})
    scales = np.asarray(params.get("scales", []), dtype=np.float32)
    zero_points = np.asarray(params.get("zero_points", []), dtype=np.int64)
    if scales.size:
        return float(scales.flat[0]), int(zero_points.flat[0])
    scale, zero = detail.get("quantization", (0.0, 0))
    return float(scale), int(zero)


def quantise(x, detail):
    scale, zero = _quant(detail)
    dtype = np.dtype(detail["dtype"])
    if not scale:
        return np.asarray(x, dtype=dtype)
    info = np.iinfo(dtype)
    return np.clip(np.rint(np.asarray(x, dtype=np.float32) / scale + zero), info.min, info.max).astype(dtype)


def dequantise(y, detail):
    scale, zero = _quant(detail)
    values = np.asarray(y)
    if not scale:
        return values.astype(np.float32)
    return (values.astype(np.float32) - zero) * scale
