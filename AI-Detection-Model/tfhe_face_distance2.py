import os
import numpy as np
from concrete import fhe

# ----------------------------
# CONFIG
# ----------------------------
THRESH_HOLD = 3200

SCALE = 16.0
CLIP = 31
DIMS = 16

QUERY_EMB = "embalage/001_5ef3e95c.npy"      # non-encrypted picture embedding
DATABASE_FOLDER = "embalage"  # folder with .npy embeddings

# ----------------------------
# Quantize
# ----------------------------
def quantize(x: np.ndarray) -> np.ndarray:
    q = np.rint(x * SCALE).astype(np.int64)
    q = np.clip(q, -CLIP, CLIP).astype(np.int64)
    return q

# ----------------------------
# Load clear query embedding
# ----------------------------
emb_query = np.load(QUERY_EMB).astype(np.float32)

z_query = quantize(emb_query)[:DIMS]      # clear / non-encrypted
z_query_norm2 = int(np.sum(z_query * z_query))

# ----------------------------
# Distance:
# query is clear
# database vector is encrypted
# ----------------------------
def dist2_database_encrypted(z_query, z_db, z_query_norm2):
    dot = np.sum(z_query * z_db)          # clear * encrypted
    z_db_norm2 = np.sum(z_db * z_db)      # encrypted
    return z_db_norm2 - 2 * dot + z_query_norm2

# ----------------------------
# Compile
# ----------------------------
rng = np.random.default_rng(0)

inputset = [
    (
        rng.integers(-CLIP, CLIP + 1, size=(DIMS,), dtype=np.int64),
        rng.integers(-CLIP, CLIP + 1, size=(DIMS,), dtype=np.int64),
        np.int64(rng.integers(0, DIMS * (CLIP**2) + 1)),
    )
    for _ in range(200)
]

compiler = fhe.Compiler(
    dist2_database_encrypted,
    {
        "z_query": "clear",
        "z_db": "encrypted",
        "z_query_norm2": "clear",
    },
)

circuit = compiler.compile(inputset)
circuit.keygen()

# ----------------------------
# Compare query to all embeddings
# ----------------------------
print("Matches:")

for filename in os.listdir(DATABASE_FOLDER):
    if not filename.endswith(".npy"):
        continue

    path = os.path.join(DATABASE_FOLDER, filename)

    emb_db = np.load(path).astype(np.float32)
    z_db = quantize(emb_db)[:DIMS]

    distance2 = int(
        circuit.encrypt_run_decrypt(
            z_query,
            z_db,
            np.int64(z_query_norm2)
        )
    )

    if distance2 < THRESH_HOLD:
        print(filename, "distance^2:", distance2)