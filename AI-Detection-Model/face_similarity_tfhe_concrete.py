import os
import base64
import numpy as np
from deepface import DeepFace

from concrete import fhe  # concrete-python


# ----------------------------
# Helpers: file IO like your CKKS script
# ----------------------------
def write_bytes(path: str, data: bytes):
    with open(path, "wb") as f:
        f.write(base64.b64encode(data))


def read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return base64.b64decode(f.read())


# ----------------------------
# Step A (client-side): get embeddings (float)
# ----------------------------
def get_facenet_embedding(img_path: str) -> np.ndarray:
    rep = DeepFace.represent(img_path, model_name="Facenet", enforce_detection=True)
    emb = np.array(rep[0]["embedding"], dtype=np.float32)

    # Normalize to make quantization stable (unit-length embedding)
    emb = emb / (np.linalg.norm(emb) + 1e-12)
    return emb


# ----------------------------
# Step B (client-side): quantize float embedding -> int8 / int16
# ----------------------------
def quantize_embedding(emb: np.ndarray, scale: int = 127, dims: int = 64) -> np.ndarray:
    """
    TFHE-style FHE is integer-focused. We:
      1) optionally truncate dims to reduce FHE cost
      2) map [-1, 1] approx to [-scale, +scale] integers
    """
    emb = emb[:dims]
    q = np.round(emb * scale).astype(np.int16)

    # Safety clamp
    q = np.clip(q, -scale, scale).astype(np.int16)
    return q


# ----------------------------
# Step C (compile): integer squared L2 distance
# ----------------------------
def build_distance_circuit(dims: int = 64, scale: int = 127):
    """
    Returns a compiled circuit that computes:
        sum_i (a_i - b_i)^2
    over encrypted integer vectors.
    """

    @fhe.compiler({"a": "encrypted", "b": "encrypted"})
    def l2_sq(a, b):
        d = a - b
        return np.sum(d * d)

    # Concrete needs a representative inputset to pick parameters.
    # a,b will be in [-scale, scale] with dtype int16.
    rng = np.random.default_rng(0)
    inputset = [
        (
            rng.integers(-scale, scale + 1, size=(dims,), dtype=np.int16),
            rng.integers(-scale, scale + 1, size=(dims,), dtype=np.int16),
        )
        for _ in range(100)
    ]

    circuit = l2_sq.compile(inputset)
    return circuit


# ----------------------------
# MAIN: mirrors your CKKS flow (client keys + encrypt, server compute, client decrypt)
# ----------------------------
def main():
    # Fix your Windows paths: use raw strings or forward slashes
    img1_path = r"AI-Detection-Model\dataset\img1.jpg"
    img2_path = r"AI-Detection-Model\dataset\img2.jpg"

    dims = 64      # reduce to 32/64 to keep TFHE runtime reasonable
    scale = 127    # int quantization scale (fits int8-ish, stored in int16)

    print("[1] Client: extracting embeddings")
    e1 = get_facenet_embedding(img1_path)
    e2 = get_facenet_embedding(img2_path)

    q1 = quantize_embedding(e1, scale=scale, dims=dims)
    q2 = quantize_embedding(e2, scale=scale, dims=dims)

    print("[2] Compile TFHE-style circuit (Concrete)")
    circuit = build_distance_circuit(dims=dims, scale=scale)

    # --- Client side keygen ---
    print("[3] Client: key generation")
    circuit.keygen()

    # Keys are serializable (be careful storing them; they’re not “magically protected” when serialized).
    # Concrete’s docs explicitly warn about key handling. :contentReference[oaicite:3]{index=3}
    serialized_keys = circuit.keys.serialize()
    write_bytes("keys.b64", serialized_keys)

    # Server needs the compiled artifact (circuit) + evaluation material to run.
    # Concrete has a “deploy” workflow conceptually: client generates keys, server gets eval keys. :contentReference[oaicite:4]{index=4}
    # For a simple thesis demo, we keep everything local but still serialize like “client/server”.

    # --- Client encrypts inputs ---
    print("[4] Client: encrypt inputs")
    enc_args = circuit.encrypt(q1, q2)
    # enc_args is a tuple of ciphertext objects; serialize them one by one
    enc_a_bytes = enc_args[0].serialize()
    enc_b_bytes = enc_args[1].serialize()
    write_bytes("enc_a.b64", enc_a_bytes)
    write_bytes("enc_b.b64", enc_b_bytes)

    # --- “Server” loads public artifacts and runs ---
    print("[5] Server: run on ciphertexts (no decryption key needed)")
    # In a real deployment, server would receive evaluation keys and ciphertexts.
    # Here we reload keys into the same circuit object to keep code short.

    # Deserialize keys back (simulates transfer)
    keys2 = fhe.Keys.deserialize(read_bytes("keys.b64"))
    circuit.keys = keys2

    enc_a2 = fhe.Value.deserialize(read_bytes("enc_a.b64"))
    enc_b2 = fhe.Value.deserialize(read_bytes("enc_b.b64"))

    enc_result = circuit.run(enc_a2, enc_b2)
    write_bytes("enc_distance.b64", enc_result.serialize())

    # --- Client decrypts result ---
    print("[6] Client: decrypt result")
    enc_distance2 = fhe.Value.deserialize(read_bytes("enc_distance.b64"))
    distance_int = circuit.decrypt(enc_distance2)

    print(f"\nEncrypted L2^2 distance (integer domain) = {distance_int}")

    # ----------------------------
    # Thresholding
    # ----------------------------
    # IMPORTANT: this threshold is NOT the same as your CKKS threshold (100).
    # It depends on: dims, scale, model normalization.
    # You should calibrate with a few same-person and different-person pairs.
    #
    # Quick starting heuristic:
    # - For normalized embeddings quantized with scale=127 and dims=64,
    #   distances for same person typically smaller than different person,
    #   but you must measure on your dataset.
    threshold = 20000  # placeholder; calibrate!
    if distance_int < threshold:
        print("Decision: likely SAME person")
    else:
        print("Decision: likely DIFFERENT persons")


if __name__ == "__main__":
    main()