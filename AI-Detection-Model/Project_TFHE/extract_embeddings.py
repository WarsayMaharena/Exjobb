import argparse
import numpy as np
from deepface import DeepFace

MODEL_NAME = "Facenet"

def get_embedding(path: str) -> np.ndarray:
    rep = DeepFace.represent(path, model_name=MODEL_NAME, enforce_detection=True)
    return np.array(rep[0]["embedding"], dtype=np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img1", required=True)
    ap.add_argument("--img2", required=True)
    ap.add_argument("--out1", default="emb1.npy")
    ap.add_argument("--out2", default="emb2.npy")
    args = ap.parse_args()

    print("start")
    print("extracting embeddings with DeepFace...")

    e1 = get_embedding(args.img1)
    e2 = get_embedding(args.img2)

    if e1.shape != e2.shape:
        raise ValueError(f"Embedding shapes differ: {e1.shape} vs {e2.shape}")

    np.save(args.out1, e1)
    np.save(args.out2, e2)

    print(f"saved {args.out1} shape={e1.shape} dtype={e1.dtype}")
    print(f"saved {args.out2} shape={e2.shape} dtype={e2.dtype}")
    print("done")

if __name__ == "__main__":
    main()