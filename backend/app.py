# app.py
import os
import time
import logging
import threading
from flask import Flask
from flask_cors import CORS
from pymongo import MongoClient
from dotenv import load_dotenv
from flask_bcrypt import Bcrypt
import numpy as np

from auth.routes import auth_bp

try:
    from student.registration import student_registration_bp
except ImportError:
    student_registration_bp = None

try:
    from student.updatedetails import student_update_bp
except ImportError:
    student_update_bp = None

try:
    from student.demo_session import demo_session_bp
except ImportError:
    demo_session_bp = None

try:
    from student.view_attendance import attendance_bp
except ImportError:
    attendance_bp = None

try:
    from teacher.attendance_records import attendance_session_bp
except ImportError:
    attendance_session_bp = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DATABASE_NAME", "facerecognition")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "students")

# FIX 1: default threshold lowered from 0.6 → 0.40
THRESHOLD = float(os.getenv("THRESHOLD", "0.40"))

client = MongoClient(MONGODB_URI)
db = client[DB_NAME]
students_collection = db[COLLECTION_NAME]
attendance_db = client["facerecognition_db"]
attendance_collection = attendance_db["attendance_records"]


class ModelManager:
    """Singleton — loads models once, shared across all requests."""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize_models()
        return cls._instance

    def _initialize_models(self):
        logger.info("Starting model initialization...")
        start_time = time.time()
        self.models_ready = False
        self.detector = None
        self.deepface_ready = False

        try:
            from mtcnn import MTCNN
            self.detector = MTCNN()
            logger.info("MTCNN loaded")

            from deepface import DeepFace
            dummy = np.zeros((160, 160, 3), dtype=np.uint8)
            DeepFace.represent(dummy, model_name='Facenet512',
                               detector_backend='skip', enforce_detection=False)
            self.deepface_ready = True
            logger.info("Facenet512 warmed up")

            self.models_ready = True
            logger.info(f"Models ready in {time.time() - start_time:.2f}s")

        except Exception as e:
            logger.error(f"Model init failed: {e}")
            raise e

    def get_detector(self):
        if not self.models_ready:
            raise RuntimeError("Models not ready")
        return self.detector

    def is_ready(self):
        return self.models_ready and self.deepface_ready

    def health_check(self):
        try:
            from deepface import DeepFace
            test_face = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
            DeepFace.represent(test_face, model_name='Facenet512',
                               detector_backend='skip', enforce_detection=False)
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False


logger.info("Initializing Model Manager...")
model_manager = ModelManager()

app = Flask(__name__)
CORS(app)

app.config["DB"] = db
app.config["COLLECTION_NAME"] = COLLECTION_NAME
app.config["THRESHOLD"] = THRESHOLD           # FIX 1: 0.40
app.config["ATTENDANCE_COLLECTION"] = attendance_collection
app.config["MODEL_MANAGER"] = model_manager
app.config["MTCNN_DETECTOR"] = model_manager.get_detector()

bcrypt = Bcrypt(app)


@app.route('/health', methods=['GET'])
def health_check():
    ready = model_manager.is_ready()
    healthy = model_manager.health_check()
    return {
        "status": "healthy" if ready and healthy else "unhealthy",
        "models_ready": ready,
        "threshold": THRESHOLD,
        "timestamp": time.time()
    }


app.register_blueprint(auth_bp)

if student_registration_bp:
    app.register_blueprint(student_registration_bp)
if student_update_bp:
    app.register_blueprint(student_update_bp)
if demo_session_bp:
    app.register_blueprint(demo_session_bp)
if attendance_bp:
    app.register_blueprint(attendance_bp)
if attendance_session_bp:
    app.register_blueprint(attendance_session_bp)

if __name__ == "__main__":
    if model_manager.is_ready():
        logger.info(f"Server starting — threshold={THRESHOLD}")
        app.run(host="0.0.0.0", port=5000, debug=False)
    else:
        logger.error("Cannot start — models not ready")
        exit(1)
