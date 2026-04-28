import numpy as np
from concrete import fhe

THRESH_HOLD = 3200

SCALE = 16.0
CLIP = 31          # much smaller range than 127
DIMS = 16          # start small (16). Later try 32, 64...

emb1 = np.load("emb1.npy").astype(np.float32)
emb2 = np.load("emb2.npy").astype(np.float32)

def quantize(x: np.ndarray) -> np.ndarray:
    q = np.rint(x * SCALE).astype(np.int64)
    q = np.clip(q, -CLIP, CLIP).astype(np.int64)
    return q

q1 = quantize(emb1)[:DIMS]
q2 = quantize(emb2)[:DIMS]

def dist2(x, y):
    d = x - y
    return np.sum(d * d)

rng = np.random.default_rng(0)
inputset = [
    (
        rng.integers(-CLIP, CLIP + 1, size=q1.shape, dtype=np.int64),
        rng.integers(-CLIP, CLIP + 1, size=q1.shape, dtype=np.int64),
    )
    for _ in range(200)
]

compiler = fhe.Compiler(dist2, {"x": "encrypted", "y": "encrypted"})
circuit = compiler.compile(inputset)

circuit.keygen()

# ✅ one-liner: encrypt -> run -> decrypt
result = circuit.encrypt_run_decrypt(q1, q2)

print("distance^2:", int(result))

if result < THRESH_HOLD:
    print("same person")
else:
    print("different persons")