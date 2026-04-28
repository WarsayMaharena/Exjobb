print("start")

import numpy as np
print("numpy ok")

from concrete import fhe
print("concrete ok")

from deepface import DeepFace
print("deepface ok")

print("done")
print("hello")
# ----------------------------
# CONFIG
# ----------------------------
MODEL_NAME = "Facenet"

# TFHE works on INTEGERS, so we quantize floats -> int8.
# SCALE controls how much precision you keep.
# Larger SCALE = more precision but larger distances.
SCALE = 32.0

# IMPORTANT: Your CKKS threshold (100) will NOT transfer to TFHE quantized distance.
# You must recalibrate this threshold empirically.
THRESHOLD = 2_000_000  # placeholder; change after you test

img1_path = r"AI-Detection-Model/dataset/img1.jpg"
img2_path = r"AI-Detection-Model/dataset/img2.jpg"

# ----------------------------
# 1) Extract embeddings (floats)
# ----------------------------
def get_embedding(path: str) -> np.ndarray:
    rep = DeepFace.represent(path, model_name=MODEL_NAME, enforce_detection=True)
    emb = np.array(rep[0]["embedding"], dtype=np.float32)
    return emb

# ----------------------------
# 2) Quantize to int8 for TFHE
# ----------------------------
def quantize_int8(x: np.ndarray, scale: float) -> np.ndarray:
    q = np.round(x * scale)
    q = np.clip(q, -128, 127)
    return q.astype(np.int8)

e1 = get_embedding(img1_path)
e2 = get_embedding(img2_path)

if e1.shape != e2.shape:
    raise ValueError(f"Embedding shapes differ: {e1.shape} vs {e2.shape}")

D = int(e1.shape[0])
q1 = quantize_int8(e1, SCALE)
q2 = quantize_int8(e2, SCALE)

print(f"Embedding dim: {D} | quant dtype: {q1.dtype} | SCALE={SCALE}")

# ----------------------------
# 3) Define squared Euclidean distance on int vectors
#    ||q1 - q2||^2 = sum_i (q1_i - q2_i)^2
# ----------------------------
def l2_sq(a, b):
    # a,b are int8 vectors (encrypted at runtime)
    diff = a.astype(np.int32) - b.astype(np.int32)
    return np.sum(diff * diff).astype(np.int32)

# ----------------------------
# 4) Compile TFHE circuit
#    inputset must cover realistic ranges/shapes or Concrete may compile wrong bounds.
# ----------------------------
compiler = fhe.Compiler(l2_sq, {"a": "encrypted", "b": "encrypted"})

# Representative inputset (random int8 vectors). Increase size if you hit bound issues.
inputset = [
    (
        np.random.randint(-128, 128, size=(D,), dtype=np.int8),
        np.random.randint(-128, 128, size=(D,), dtype=np.int8),
    )
    for _ in range(50)
]

print("Compilation...")
circuit = compiler.compile(inputset)

print("Key generation...")
circuit.keygen()  # keys are generated for this circuit :contentReference[oaicite:1]{index=1}

# ----------------------------
# 5) Encrypt -> Run -> Decrypt (single-file simplest workflow)
# ----------------------------
print("Encrypting inputs...")
enc_q1, enc_q2 = circuit.encrypt(q1, q2)  # :contentReference[oaicite:2]{index=2}

print("Running TFHE circuit on encrypted data...")
enc_dist = circuit.run(enc_q1, enc_q2)    # :contentReference[oaicite:3]{index=3}

print("Decrypting result...")
dist = circuit.decrypt(enc_dist)          # :contentReference[oaicite:4]{index=4}

print("Squared L2 distance (TFHE, quantized):", dist)

if dist < THRESHOLD:
    print("they are same person (by threshold)")
else:
    print("they are different persons (by threshold)")