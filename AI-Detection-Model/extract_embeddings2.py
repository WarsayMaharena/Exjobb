"""import os
# extract_embeddings.py
enforce_detection=False
import numpy as np
from deepface import DeepFace

MODEL = "Facenet"

FOLDER = "celebrity_faces/Will_Smith"

for filename in os.listdir(FOLDER):
    emb1 = DeepFace.represent("celebrity_faces/Will_Smith/"+filename, model_name=MODEL)[0]["embedding"]
    np.save("embalage/"+filename.removesuffix('.jpg')+".npy", np.asarray(emb1, dtype=np.float32))
    print("FOUND:", filename)"""

# Import the time library
import time

#
import os
import numpy as np
from deepface import DeepFace

 #Calculate the start time
start = time.time()

MODEL = "Facenet"
FOLDER = "celebrity_faces/Sandra_Bullock"

os.makedirs("embalage", exist_ok=True)

for filename in os.listdir(FOLDER):
    if not filename.lower().endswith(".jpg"):
        continue

    path = FOLDER + "/" + filename  # <-- fix here

    emb1 = DeepFace.represent(
        path,
        model_name=MODEL,
        enforce_detection=False
    )[0]["embedding"]
    try:
        name = filename.removesuffix(".jpg")
        np.save("embalage/" + name + ".npy", np.asarray(emb1, dtype=np.float32))

        print("saved:", filename)
    
    except Exception as e:
        print("skipped:", filename)

# Calculate the end time and time taken
end = time.time()
length = end - start

# Show the results : this can be altered however you like
print("It took", length, "seconds!")