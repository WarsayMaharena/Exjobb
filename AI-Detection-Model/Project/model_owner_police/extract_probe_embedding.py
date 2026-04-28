# extract_embeddings.py
import numpy as np
from deepface import DeepFace

IMG2 = "../../dataset/face_2.jpg"
MODEL = "Facenet"

emb2 = DeepFace.represent(IMG2, model_name=MODEL)[0]["embedding"]

np.save("emb2.npy", np.asarray(emb2, dtype=np.float32))

print("saved emb1.npy and emb2.npy", len(emb2))

#works now