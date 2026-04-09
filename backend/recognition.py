import cv2
from mtcnn import MTCNN
from deepface import DeepFace
from pymongo import MongoClient
from scipy.spatial.distance import cosine
import numpy as np
import time
from collections import Counter

# ----------------- MongoDB Setup -----------------
MONGODB_URI = "mongodb+srv://Kamlesh-21:Guru2004@attendencesystem.nlapsic.mongodb.net/Attendencesystem?retryWrites=true&w=majority&appName=Attendencesystem"
client = MongoClient(MONGODB_URI)
db = client['facerecognition_db']
collection = db['users']

# ----------------- Config -----------------
COSINE_THRESHOLD = 0.40       # FIX 1: was 0.7 — too lenient, caused false matches
FRAME_BUFFER_SIZE = 5         # FIX 4: majority-vote across N frames
MIN_FACE_CONFIDENCE = 0.92    # FIX: ignore low-confidence MTCNN detections
MIN_EMBEDDINGS_REQUIRED = 3   # FIX 2: minimum embeddings per student for registration

# ----------------- Face Detector -----------------
detector = MTCNN()

# ----------------- FIX 3: CLAHE Preprocessing -----------------
def preprocess_face(face_img):
    """
    Apply CLAHE (adaptive histogram equalization) on the L channel.
    Normalizes brightness/contrast — critical for lighting variation.
    """
    face_resized = cv2.resize(face_img, (160, 160))
    lab = cv2.cvtColor(face_resized, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

# ----------------- Detect Faces -----------------
def detect_faces(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb_image)
    face_data = []
    for face in faces:
        # FIX: filter low-confidence detections
        if face.get('confidence', 0) < MIN_FACE_CONFIDENCE:
            continue
        x, y, w, h = face['box']
        x, y = max(0, x), max(0, y)
        # FIX: ignore tiny faces (likely noise)
        if w < 60 or h < 60:
            continue
        face_img = rgb_image[y:y+h, x:x+w]
        face_data.append({'box': (x, y, w, h), 'face': face_img})
    return face_data

# ----------------- Extract Embedding -----------------
def extract_embedding(face_img):
    try:
        # FIX 3: preprocess before embedding
        face_img = preprocess_face(face_img)
        embedding = DeepFace.represent(face_img, model_name='Facenet512', detector_backend='skip')
        return np.array(embedding[0]['embedding'], dtype=np.float32)
    except Exception as e:
        print("Error extracting embedding:", e)
        return None

# ----------------- FIX 2: Multi-Angle Registration -----------------
def auto_register_user(user_id, name, num_embeddings=8, wait_time=3):
    """
    Captures multiple embeddings across different angles/expressions.
    num_embeddings: number of distinct face snapshots to store (default 8).
    wait_time: seconds to stabilize before first capture.
    """
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print(f"Registering {name}. Look at the camera. Capturing {num_embeddings} embeddings...")

    embeddings = []
    stabilize_start = time.time()
    last_capture = 0
    CAPTURE_INTERVAL = 0.8  # seconds between captures

    while len(embeddings) < num_embeddings:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detect_faces(frame)
        now = time.time()

        if len(faces) == 1:
            x, y, w, h = faces[0]['box']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            if now - stabilize_start > wait_time and now - last_capture > CAPTURE_INTERVAL:
                embedding = extract_embedding(faces[0]['face'])
                if embedding is not None:
                    embeddings.append(embedding.tolist())
                    last_capture = now
                    print(f"  Captured embedding {len(embeddings)}/{num_embeddings} — move your head slightly")
                    cv2.putText(frame, f"Captured {len(embeddings)}/{num_embeddings}", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                remaining = max(0, wait_time - (now - stabilize_start))
                cv2.putText(frame, f"Hold still... {remaining:.1f}s", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        else:
            cv2.putText(frame, f"{len(faces)} faces detected. Show only one face.", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            stabilize_start = time.time()  # reset timer

        cv2.imshow("Registration", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(embeddings) < MIN_EMBEDDINGS_REQUIRED:
        print(f"Registration failed. Only captured {len(embeddings)} embeddings (need {MIN_EMBEDDINGS_REQUIRED}).")
        return

    user_data = {
        'user_id': user_id,
        'name': name,
        'embeddings': embeddings  # FIX 2: store list, not single embedding
    }
    collection.insert_one(user_data)
    print(f"User {name} registered with {len(embeddings)} embeddings.")

# ----------------- FIX 2+4: Recognition with voting -----------------
def live_recognition():
    users = list(collection.find())
    if not users:
        print("No users registered.")
        return

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("Live recognition started. Press 'q' to quit.")

    # FIX 4: per-face recognition buffer for majority voting
    recognition_buffer = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detect_faces(frame)

        for face_data in faces:
            x, y, w, h = face_data['box']
            face_img = face_data['face']
            embedding = extract_embedding(face_img)
            if embedding is None:
                continue

            # FIX 2: compare against ALL stored embeddings per user, take best
            best_match = None
            min_distance = float('inf')

            for user in users:
                stored_embeddings = user.get('embeddings', [])
                # backward compat: single 'embedding' key
                if not stored_embeddings and 'embedding' in user:
                    stored_embeddings = [user['embedding']]

                for stored_emb in stored_embeddings:
                    distance = cosine(embedding, stored_emb)
                    if distance < min_distance:
                        min_distance = distance
                        best_match = user

            # FIX 4: buffer this frame's result
            if min_distance < COSINE_THRESHOLD:
                frame_label = best_match['name']
            else:
                frame_label = "Unknown"

            recognition_buffer.append(frame_label)
            if len(recognition_buffer) > FRAME_BUFFER_SIZE:
                recognition_buffer.pop(0)

            # majority vote
            voted_name = Counter(recognition_buffer).most_common(1)[0][0]

            color = (0, 255, 0) if voted_name != "Unknown" else (0, 0, 255)
            label = f"{voted_name} ({min_distance:.2f})"
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow("Live Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ----------------- Main -----------------
def main():
    while True:
        print("\nFace Recognition System")
        print("1. Register User")
        print("2. Start Live Recognition")
        print("3. Exit")
        choice = input("Enter choice: ")

        if choice == '1':
            user_id = input("Enter user ID: ")
            name = input("Enter user name: ")
            auto_register_user(user_id, name)
        elif choice == '2':
            live_recognition()
        elif choice == '3':
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()
