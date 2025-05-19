import random

def sample_pos(l, n_chunks):
    positions = list(range(1, l-2))
    random.shuffle(positions)
    positions = positions[:n_chunks]
    return positions
