# Patched Vector Scale Operator with IEEE-754 Float32 Alignment
import math
import struct


def _to_f32(x: float) -> float:
    return struct.unpack("!f", struct.pack("!f", float(x)))[0]


def vector_scale_kernel(tensor_a, tensor_b, scale=1.0):
    """
    Element-wise fused vector addition and scaling aligned to IEEE-754 float32 arithmetic.
    """
    if len(tensor_a) != len(tensor_b):
        raise ValueError("Tensor shape mismatch between input vectors")
    if not math.isfinite(scale):
        raise ValueError("Non-finite scale factor")

    f_scale = _to_f32(scale)
    out = []
    for a, b in zip(tensor_a, tensor_b):
        if not (math.isfinite(a) and math.isfinite(b)):
            raise ValueError("Non-finite element in input vector")
        s = _to_f32(_to_f32(a) + _to_f32(b))
        out.append(_to_f32(s * f_scale))
    return out
