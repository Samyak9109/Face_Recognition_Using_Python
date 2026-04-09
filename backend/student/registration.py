from flask import Blueprint, request, jsonify, current_app
import time
import base64
import numpy as np
from PIL import Image
import io
import cv2
from deepface import DeepFace
from mtcnn import MTCNN
import logging

student_registration_bp = Blueprint("student_registration", __name__)
detector = MTCNN()
logger = logging.getLogger(__name__)

# FIX: tighter confidence filter during registration
MIN_FACE_CONFIDENCE = 0.92
MIN_FACE_SIZE = 60  # pixels

def read_image_from_bytes(b):
    img = Image.open(io.BytesIO(b)).convert('RGB')
    return np.array(img)

# FIX 3: CLAHE preprocessing — same as recognition.py
def preprocess_face(face_img):
    """Normalize brightness/contrast before embedding extraction."""
    face_resized = cv2.resize(face_img, (160, 160))
    lab = cv2.cvtColor(face_resized, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def detect_faces_rgb(rgb_image):
    detections = detector.detect_faces(rgb_image)
    faces = []
    for d in detections:
        # FIX: strict confidence + size filter
        if d['confidence'] < MIN_FACE_CONFIDENCE:
            continue
        x, y, w, h = d['box']
        x, y = max(0, x), max(0, y)
        if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
            continue
        face_rgb = rgb_image[y:y+h, x:x+w]
        faces.append({'box': (x, y, w, h), 'face': face_rgb, 'confidence': d['confidence']})
    return faces

def extract_embedding(face_rgb):
    try:
        # FIX 3: apply CLAHE before extracting embedding
        face_rgb = preprocess_face(face_rgb)
        rep = DeepFace.represent(face_rgb, model_name='Facenet512', detector_backend='skip')
        return np.array(rep[0]['embedding'], dtype=np.float32)
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        return None

# FIX 2: check embedding diversity (reject duplicate captures)
def embeddings_are_diverse(embeddings, min_distance=0.05):
    """Ensure captured embeddings aren't all from same exact pose."""
    if len(embeddings) < 2:
        return True
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            from scipy.spatial.distance import cosine
            dist = cosine(embeddings[i], embeddings[j])
            if dist < min_distance:
                return False
    return True

@student_registration_bp.route('/api/register-student', methods=['POST'])
def register_student():
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "Invalid JSON data"}), 400

    db = current_app.config.get("DB")
    students_col = db.students

    required_fields = ['studentName', 'studentId', 'department', 'year',
                       'division', 'semester', 'email', 'phoneNumber', 'images']
    for field in required_fields:
        if not data.get(field):
            return jsonify({"success": False, "error": f"{field} is required"}), 400

    if students_col.find_one({'studentId': data['studentId']}):
        return jsonify({"success": False, "error": "Student ID already exists"}), 400
    if students_col.find_one({'email': data['email']}):
        return jsonify({"success": False, "error": "Email already registered"}), 400

    images = data.get('images')
    # FIX 2: require exactly 5 images (multi-angle)
    if not isinstance(images, list) or len(images) != 5:
        return jsonify({"success": False, "error": "Exactly 5 images required (different angles)"}), 400

    embeddings = []
    for idx, img_b64 in enumerate(images):
        try:
            if img_b64.startswith("data:"):
                img_b64 = img_b64.split(",", 1)[1]
            rgb = read_image_from_bytes(base64.b64decode(img_b64))
        except Exception:
            return jsonify({"success": False, "error": f"Invalid image at index {idx}"}), 400

        faces = detect_faces_rgb(rgb)
        if len(faces) != 1:
            return jsonify({
                "success": False,
                "error": f"Need exactly 1 face per image (image {idx+1} has {len(faces)})"
            }), 400

        emb = extract_embedding(faces[0]['face'])
        if emb is None:
            return jsonify({"success": False, "error": f"Failed to extract features for image {idx+1}"}), 500

        embeddings.append(emb.tolist())

    # FIX 2: reject if all images look too similar (student didn't vary pose)
    if not embeddings_are_diverse(embeddings):
        return jsonify({
            "success": False,
            "error": "Images look too similar. Capture from different angles/expressions."
        }), 400

    student_data = {
        "studentId": data['studentId'],
        "studentName": data['studentName'],
        "department": data['department'],
        "year": data['year'],
        "division": data['division'],
        "semester": data['semester'],
        "email": data['email'],
        "phoneNumber": data['phoneNumber'],
        "status": "active",
        "embeddings": embeddings,    # FIX 2: always stored as list
        "face_registered": True,
        "created_at": time.time(),
        "updated_at": time.time()
    }

    result = students_col.insert_one(student_data)
    return jsonify({"success": True, "studentId": data['studentId'], "record_id": str(result.inserted_id)})

@student_registration_bp.route('/api/students/count', methods=['GET'])
def get_student_count():
    db = current_app.config.get("DB")
    return jsonify({"success": True, "count": db.students.count_documents({})})

@student_registration_bp.route('/api/students/departments', methods=['GET'])
def get_departments():
    db = current_app.config.get("DB")
    departments = db.students.distinct("department")
    return jsonify({"success": True, "departments": departments, "count": len(departments)})
