import numpy as np
from concrete import fhe

SCALE = 16.0
CLIP = 31
DIMS = 16

emb2 = np.load("emb2.npy").astype(np.float32)

def quantize(x: np.ndarray) -> np.ndarray:
    q = np.rint(x * SCALE).astype(np.int64)
    q = np.clip(q, -CLIP, CLIP).astype(np.int64)
    return q

# Police probe (encrypted)
z1 = quantize(emb2)[:DIMS]          # encrypted input

rng = np.random.default_rng(0)
inputset = [
    (
        rng.integers(-CLIP, CLIP + 1, size=(DIMS,), dtype=np.int64),  # z1
        rng.integers(-CLIP, CLIP + 1, size=(DIMS,), dtype=np.int64),  # zi
        np.int64(rng.integers(0, DIMS * (CLIP**2) + 1)),               # zi_norm2
    )
    for _ in range(200)
]

def dist2_mixed(z1, zi, zi_norm2):
    dot = np.sum(z1 * zi)           # encrypted * clear
    z1_norm2 = np.sum(z1 * z1)      # encrypted
    return z1_norm2 - 2 * dot + zi_norm2

compiler = fhe.Compiler(
    dist2_mixed,
    {"z1": "encrypted", "zi": "clear", "zi_norm2": "clear"},
)
circuit = compiler.compile(inputset)
circuit.keygen()

def encrypt_probe(circuit, z1: np.ndarray):
    """
    Encrypt only the probe because only z1 is marked 'encrypted' in the circuit.
    Returns an fhe.Value (ciphertext).
    """
    return circuit.encrypt(z1)