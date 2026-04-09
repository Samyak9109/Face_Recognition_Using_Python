# Face Recognition Attendance System

Automated attendance using **FaceNet512 + MTCNN** via a Next.js frontend and Flask backend.

---

## Requirements

- Python **3.12** (not 3.13 or 3.14)
- Node.js **18+**
- MongoDB Atlas account (or local MongoDB)

---

## First Time Setup

### 1 — Install system dependencies

**Fedora / RHEL:**
```bash
sudo dnf install gcc gcc-c++ python3-devel -y
```

**Ubuntu / Debian:**
```bash
sudo apt install gcc g++ python3-dev -y
```

### 2 — Install pip for Python 3.12

```bash
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3.12 get-pip.py
```

### 3 — Install Python dependencies

```bash
python3.12 -m pip install --only-binary=:all: numpy scipy
python3.12 -m pip install -r requirements.txt
```

> If TensorFlow fails, open `requirements.txt`, comment out the two `tensorflow` lines and uncomment the `torch` lines, then re-run.

### 4 — Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

### 5 — Configure environment

Create a `.env` file inside `backend/`:

```
MONGODB_URI=your_mongodb_connection_string
DATABASE_NAME=facerecognition
COLLECTION_NAME=students
THRESHOLD=0.40
```

---

## Running the Project

**Linux:**
```bash
chmod +x start.sh   # only needed once
./start.sh
```

**Windows:**
Double-click `START.bat`

**Terminal (any OS):**
```bash
python3.12 start.py
```

Opens:
- Frontend → `http://localhost:3000`
- Backend → `http://localhost:5000`

Press `Ctrl+C` to stop both.

---

## How to Use

### Register a Student
1. Open `http://localhost:3000`
2. Sign up → go to Student Registration
3. Fill details and capture **5 photos** from different angles — front, left, right, chin down, different expression
4. Submit — face embeddings stored in MongoDB

### Take Attendance (Teacher)
1. Sign in as Teacher → Start Session
2. Camera scans faces in real time
3. Attendance auto-marked for recognized students
4. View records under Attendance Records

### View Attendance (Student)
1. Sign in as Student → View Attendance

---

## Project Structure

```
Face_Recognition_Using_Python-main/
├── backend/
│   ├── app.py                  # Entry point, model loader
│   ├── recognition.py          # Standalone webcam script
│   ├── auth/routes.py          # Signup / signin
│   ├── student/
│   │   ├── registration.py     # Face registration API
│   │   ├── demo_session.py     # Live recognition API
│   │   ├── updatedetails.py
│   │   └── view_attendance.py
│   └── teacher/
│       └── attendance_records.py
├── frontend/
│   └── app/
│       ├── student/
│       └── teacher/
├── requirements.txt
├── start.py
├── START.bat
└── start.sh
```

---

## Troubleshooting

**`python3` uses wrong version** — always use `python3.12` explicitly.

**TensorFlow install fails** — switch to PyTorch backend via `requirements.txt`.

**Models load slowly on first start** — DeepFace downloads FaceNet512 weights (~90MB) once only.

**Camera not detected** — check `ls /dev/video*`

**Frontend npm warnings** — run `npm audit fix` inside `frontend/`.
