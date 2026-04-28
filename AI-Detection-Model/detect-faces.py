import cv2
import os

# Input / output paths
img_path = "AI-Detection-Model/dataset/angelina-jolie-brad-pitt-2-525e409acfc9405598a161aa83ef9e69.jpg"
output_dir = "cropped_faces"

os.makedirs(output_dir, exist_ok=True)

# Load image
img = cv2.imread(img_path)
if img is None:
    raise RuntimeError("Image not found")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Load Haar cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    raise RuntimeError("Failed to load Haar cascade")

# Detect faces
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=4
)

# Iterate over detected faces
for i, (x, y, w, h) in enumerate(faces):
    # Crop face from ORIGINAL image (no rectangle drawn)
    face = img[y:y+h, x:x+w]

    # Save cropped face with unique name
    face_path = os.path.join(output_dir, f"face_{i}.jpg")
    cv2.imwrite(face_path, face)

# Iterate over detected faces
for i, (x, y, w, h) in enumerate(faces):

    # Draw rectangle ONLY on the full image
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

# Save and show result
cv2.imwrite("detected.jpg", img)
cv2.imshow("Detected Faces", img)
cv2.waitKey(0)
cv2.destroyAllWindows()