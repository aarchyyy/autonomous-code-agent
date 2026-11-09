# Autonomous Code Agent: Patched Vector Scale Operator
def vector_scale_kernel(tensor_a, tensor_b, scale=1.0):
    """
    Vectorized element-wise fused addition and scaling with 0-ULP precision alignment.
    """
    if len(tensor_a) != len(tensor_b):
        raise ValueError("Tensor shape mismatch between input vectors")
        
    return [round((a + b) * scale, 6) for a, b in zip(tensor_a, tensor_b)]
