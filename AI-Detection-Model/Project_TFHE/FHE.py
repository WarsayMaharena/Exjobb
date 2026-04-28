import argparse
import numpy as np
from concrete import fhe

# TFHE works on integers, so we quantize floats -> int8
SCALE = 32.0

# Placeholder threshold; must be calibrated for your quantization + dataset
THRESHOLD = 2_000_000

def quantize_int8(x: np.ndarray, scale: float) -> np.ndarray:
    q = np.round(x * scale)
    q = np.clip(q, -128, 127)
    return q.astype(np.int8)

def l2_sq(a, b):
    # a,b are int8 vectors (encrypted at runtime)
    diff = a.astype(np.int32) - b.astype(np.int32)
    return np.sum(diff * diff).astype(np.int32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb1", default="emb1.npy")
    ap.add_argument("--emb2", default="emb2.npy")
    ap.add_argument("--scale", type=float, default=SCALE)
    ap.add_argument("--threshold", type=int, default=THRESHOLD)
    ap.add_argument("--calib_n", type=int, default=50, help="size of compilation inputset")
    args = ap.parse_args()

    print("start")
    print("numpy ok")
    print("loading embeddings...")

    e1 = np.load(args.emb1).astype(np.float32)
    e2 = np.load(args.emb2).astype(np.float32)

    if e1.shape != e2.shape or e1.ndim != 1:
        raise ValueError(f"Embeddings must be same 1D shape. Got {e1.shape} vs {e2.shape}")

    D = int(e1.shape[0])
    q1 = quantize_int8(e1, args.scale)
    q2 = quantize_int8(e2, args.scale)

    print(f"Embedding dim: {D} | quant dtype: {q1.dtype} | SCALE={args.scale}")

    # Compile TFHE circuit
    compiler = fhe.Compiler(l2_sq, {"a": "encrypted", "b": "encrypted"})
    inputset = [
        (
            np.random.randint(-128, 128, size=(D,), dtype=np.int8),
            np.random.randint(-128, 128, size=(D,), dtype=np.int8),
        )
        for _ in range(args.calib_n)
    ]

    print("Compilation...")
    circuit = compiler.compile(inputset)

    print("Key generation...")
    circuit.keygen()

    print("Encrypting inputs...")
    enc_q1, enc_q2 = circuit.encrypt(q1, q2)

    print("Running TFHE circuit on encrypted data...")
    enc_dist = circuit.run(enc_q1, enc_q2)

    print("Decrypting result...")
    dist = int(circuit.decrypt(enc_dist))

    print("Squared L2 distance (TFHE, quantized):", dist)

    if dist < args.threshold:
        print("they are same person (by threshold)")
    else:
        print("they are different persons (by threshold)")

    print("done")

if __name__ == "__main__":
    main()