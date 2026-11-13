import math

def weighted_score(metrics, weights):
    if len(metrics) != len(weights):
        raise ValueError('metrics and weights must have equal length')
    terms = []
    for m, w in zip(metrics, weights):
        if not (math.isfinite(m) and math.isfinite(w)):
            raise ValueError('non-finite metric or weight detected')
        terms.append(float(m) * float(w))
    return math.fsum(terms)
