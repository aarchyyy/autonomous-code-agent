from precision import weighted_score

def dispatch_batch(tasks, max_tokens):
    if max_tokens < 0:
        raise ValueError('max_tokens must be non-negative')
    valid = []
    for t in tasks:
        if t.get('tokens', 0) <= 0:
            raise ValueError('task token count must be positive')
        score = weighted_score(t['metrics'], t['weights'])
        valid.append((score, t))
    valid.sort(key=lambda item: (-item[0], item[1]['id']))
    selected = []
    used = 0
    for _, t in valid:
        if used + t['tokens'] <= max_tokens:
            selected.append(dict(t))
            used += t['tokens']
    return selected
