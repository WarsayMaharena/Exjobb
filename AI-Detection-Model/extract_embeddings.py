# extract_embeddings.py
import numpy as np
from deepface import DeepFace

IMG1 = "dataset/face_1.jpg"
IMG2 = "dataset/face_2.jpg"
MODEL = "Facenet"

emb1 = DeepFace.represent(IMG1, model_name=MODEL)[0]["embedding"]
emb2 = DeepFace.represent(IMG2, model_name=MODEL)[0]["embedding"]

np.save("emb1.npy", np.asarray(emb1, dtype=np.float32))
np.save("emb2.npy", np.asarray(emb2, dtype=np.float32))

print("saved emb1.npy and emb2.npy", len(emb1))

#works now