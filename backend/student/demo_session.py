# student/demo_session.py - ACCURACY-FIXED VERSION
from flask import Blueprint, request, jsonify, current_app
import time
import base64
import numpy as np
from PIL import Image
import io
import cv2
from deepface import DeepFace
from scipy.spatial.distance import cosine
import logging
import threading
from collections import Counter

logger = logging.getLogger(__name__)
demo_session_bp = Blueprint("demo_session", __name__)

# FIX 1: strict threshold — was 0.6 in original, causes false matches
COSINE_THRESHOLD = 0.40

# FIX 4: frame-buffer size for majority vote
FRAME_BUFFER_SIZE = 5

# Per-session recognition buffers (keyed by session_id)
session_buffers = {}
session_buffers_lock = threading.Lock()

def read_image_from_bytes(b, target_size=(640, 480)):
    img = Image.open(io.BytesIO(b)).convert("RGB")
    if img.width > target_size[0] or img.height > target_size[1]:
        img.thumbnail(target_size, Image.Resampling.LANCZOS)
    return np.array(img)

# FIX 3: CLAHE preprocessing before embedding
def preprocess_face(face_img):
    """Normalize lighting with CLAHE — reduces lighting-caused mismatches."""
    face_resized = cv2.resize(face_img, (160, 160))
    lab = cv2.cvtColor(face_resized, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def detect_faces_rgb(rgb_image, detector):
    if rgb_image.shape[0] < 50 or rgb_image.shape[1] < 50:
        return []
    detections = detector.detect_faces(rgb_image)
    faces = []
    for d in detections:
        # FIX: strict confidence + size filter — same as registration
        if d["confidence"] < 0.92:
            continue
        x, y, w, h = d["box"]
        x, y = max(0, x), max(0, y)
        if w < 60 or h < 60:
            continue
        face_rgb = rgb_image[y:y+h, x:x+w]
        faces.append({"box": (x, y, w, h), "face": face_rgb, "confidence": d["confidence"]})
    return faces

def extract_embedding(face_rgb):
    try:
        # FIX 3: preprocess before embedding
        face_rgb = preprocess_face(face_rgb)
        rep = DeepFace.represent(face_rgb, model_name="Facenet512",
                                 detector_backend="skip", enforce_detection=False)
        return np.array(rep[0]["embedding"], dtype=np.float32)
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        return None

# Embedding cache with 5-minute TTL
class EmbeddingCache:
    def __init__(self):
        self.cache = None
        self.last_update = 0
        self.cache_duration = 300
        self.lock = threading.Lock()

    def get(self, students_col):
        with self.lock:
            now = time.time()
            if self.cache is None or now - self.last_update > self.cache_duration:
                students = list(students_col.find(
                    {"embeddings": {"$exists": True, "$ne": None}},
                    {"studentId": 1, "studentName": 1, "embeddings": 1}
                ))
                self.cache = []
                for s in students:
                    embeddings = s.get('embeddings', [])
                    if embeddings:
                        # FIX 2: store ALL embeddings, not just average
                        self.cache.append({
                            'studentId': s.get('studentId'),
                            'studentName': s.get('studentName'),
                            'embeddings': [np.array(e, dtype=np.float32) for e in embeddings]
                        })
                self.last_update = now
                logger.info(f"Cache refreshed: {len(self.cache)} students")
        return self.cache

    def invalidate(self):
        with self.lock:
            self.cache = None

embedding_cache = EmbeddingCache()

def find_best_match(query_embedding, students_col):
    """
    FIX 2: compare against EVERY stored embedding per student.
    FIX 1: use threshold 0.40 instead of 0.6.
    Returns (best_match_dict | None, min_distance).
    """
    cached = embedding_cache.get(students_col)
    if not cached:
        return None, float('inf')

    best_match = None
    min_distance = float('inf')

    for student in cached:
        for stored_emb in student['embeddings']:
            distance = cosine(query_embedding, stored_emb)
            if distance < min_distance:
                min_distance = distance
                best_match = student

    if min_distance < COSINE_THRESHOLD:
        return best_match, min_distance
    return None, min_distance

@demo_session_bp.route("/api/demo/recognize", methods=["POST"])
def demo_recognize():
    start_time = time.time()

    model_manager = current_app.config.get("MODEL_MANAGER")
    if not model_manager or not model_manager.is_ready():
        return jsonify({"success": False, "error": "Models not initialized"}), 503

    detector = model_manager.get_detector()
    data = request.get_json()
    db = current_app.config.get("DB")
    students_col = db.students

    # Optional: session_id enables frame-buffer voting across requests
    session_id = data.get("session_id", "default")

    image_b64 = data.get("image", "")
    if image_b64.startswith("data:"):
        image_b64 = image_b64.split(",", 1)[1]

    try:
        rgb = read_image_from_bytes(base64.b64decode(image_b64))
    except Exception as e:
        return jsonify({"success": False, "error": "Invalid base64 image"}), 400

    t_detect = time.time()
    faces = detect_faces_rgb(rgb, detector)
    detection_time = time.time() - t_detect

    if not faces:
        return jsonify({
            "success": True,
            "faces": [],
            "processing_time": round(time.time() - start_time, 3)
        })

    results = []

    # FIX 4: get/init buffer for this session
    with session_buffers_lock:
        if session_id not in session_buffers:
            session_buffers[session_id] = []
        buf = session_buffers[session_id]

    for f in faces:
        t_emb = time.time()
        emb = extract_embedding(f["face"])
        emb_time = time.time() - t_emb

        if emb is None:
            results.append({"match": None, "distance": None, "box": f["box"], "error": "Embedding failed"})
            continue

        t_search = time.time()
        best_match, min_distance = find_best_match(emb, students_col)
        search_time = time.time() - t_search

        # FIX 4: push to frame buffer, vote
        raw_label = best_match['studentName'] if best_match else "Unknown"
        with session_buffers_lock:
            buf.append(raw_label)
            if len(buf) > FRAME_BUFFER_SIZE:
                buf.pop(0)
            voted_name = Counter(buf).most_common(1)[0][0]

        if voted_name != "Unknown" and best_match:
            results.append({
                "match": {
                    "user_id": best_match["studentId"],
                    "name": voted_name
                },
                "distance": round(float(min_distance), 4),
                "confidence": round((1 - min_distance) * 100, 1),
                "box": f["box"],
                "voted": True,
                "timing": {"embedding": round(emb_time, 3), "search": round(search_time, 3)}
            })
        else:
            results.append({
                "match": None,
                "distance": round(float(min_distance), 4),
                "box": f["box"],
                "voted": True,
                "timing": {"embedding": round(emb_time, 3), "search": round(search_time, 3)}
            })

    return jsonify({
        "success": True,
        "faces": results,
        "processing_time": round(time.time() - start_time, 3),
        "detection_time": round(detection_time, 3)
    })

@demo_session_bp.route('/api/demo/session', methods=['POST'])
def create_demo_session():
    db = current_app.config.get("DB")
    session_data = {
        "session_id": f"demo_{int(time.time())}",
        "started_at": time.time(),
        "status": "active",
        "recognitions": []
    }
    result = db.demo_sessions.insert_one(session_data)
    session_data['_id'] = str(result.inserted_id)
    return jsonify({"success": True, "session": session_data})

@demo_session_bp.route('/api/demo/session/<session_id>/log', methods=['POST'])
def log_recognition(session_id):
    db = current_app.config.get("DB")
    data = request.get_json()
    db.demo_sessions.update_one(
        {"session_id": session_id},
        {"$push": {"recognitions": {
            "timestamp": time.time(),
            "result": data.get('result'),
            "confidence": data.get('confidence'),
            "processing_time": data.get('processing_time')
        }}}
    )
    return jsonify({"success": True})

@demo_session_bp.route('/api/demo/session/<session_id>/end', methods=['POST'])
def end_session(session_id):
    """FIX 4: clear frame buffer when session ends."""
    with session_buffers_lock:
        session_buffers.pop(session_id, None)
    return jsonify({"success": True, "message": f"Session {session_id} ended"})

@demo_session_bp.route('/api/demo/models/status', methods=['GET'])
def model_status():
    model_manager = current_app.config.get("MODEL_MANAGER")
    if not model_manager:
        return jsonify({"success": False, "error": "Model manager not available"}), 500
    return jsonify({
        "success": True,
        "models_ready": model_manager.is_ready(),
        "threshold": COSINE_THRESHOLD,
        "frame_buffer_size": FRAME_BUFFER_SIZE
    })

@demo_session_bp.route('/api/demo/cache/invalidate', methods=['POST'])
def invalidate_cache():
    """Force refresh embedding cache (call after new student registration)."""
    embedding_cache.invalidate()
    return jsonify({"success": True, "message": "Cache invalidated"})
