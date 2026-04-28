import numpy as np
from concrete import fhe

# ----------------------------
# CONFIG
# ----------------------------
THRESH_HOLD = 3200

SCALE = 16.0
CLIP = 31
DIMS = 16

# ----------------------------
# Load embeddings (float) -> quantize (int)
# ----------------------------
emb1 = np.load("emb1.npy").astype(np.float32)
emb2 = np.load("emb2.npy").astype(np.float32)

def quantize(x: np.ndarray) -> np.ndarray:
    q = np.rint(x * SCALE).astype(np.int64)
    q = np.clip(q, -CLIP, CLIP).astype(np.int64)
    return q

# Police probe (encrypted)
z1 = quantize(emb1)[:DIMS]          # encrypted input

# Municipality database vector (clear)
zi = quantize(emb2)[:DIMS]          # clear input
zi_norm2 = int(np.sum(zi * zi))     # clear scalar (precomputable per DB entry)

# ----------------------------
# Optimized distance:
# ||z1 - zi||^2 = ||z1||^2 - 2*(z1·zi) + ||zi||^2
# - z1 is encrypted
# - zi and ||zi||^2 are clear
# ----------------------------
def dist2_mixed(z1, zi, zi_norm2):
    dot = np.sum(z1 * zi)           # encrypted * clear
    z1_norm2 = np.sum(z1 * z1)      # encrypted
    return z1_norm2 - 2 * dot + zi_norm2

# ----------------------------
# Compile with correct annotations:
# encrypted probe, clear database
# ----------------------------
rng = np.random.default_rng(0)
inputset = [
    (
        rng.integers(-CLIP, CLIP + 1, size=(DIMS,), dtype=np.int64),  # z1
        rng.integers(-CLIP, CLIP + 1, size=(DIMS,), dtype=np.int64),  # zi
        np.int64(rng.integers(0, DIMS * (CLIP**2) + 1)),               # zi_norm2
    )
    for _ in range(200)
]

compiler = fhe.Compiler(
    dist2_mixed,
    {"z1": "encrypted", "zi": "clear", "zi_norm2": "clear"},
)

circuit = compiler.compile(inputset)
circuit.keygen()

# Encrypt probe only; database stays clear
distance2 = int(circuit.encrypt_run_decrypt(z1, zi, np.int64(zi_norm2)))

print("distance^2:", distance2)

if distance2 < THRESH_HOLD:
    print("same person")
else:
    print("different persons")