#  Potato Plant Disease Detection

An AI-powered web application that detects potato leaf diseases using a Convolutional Neural Network (CNN). Upload a photo of a potato leaf and get an instant diagnosis — **Early Blight**, **Late Blight**, or **Healthy** — along with confidence scores and treatment recommendations.

---

##  Features

- **Real-time CNN inference** via a FastAPI backend
- **EfficientNet-based model** trained on potato leaf images
- **Green pixel leaf detection** — rejects non-leaf images intelligently
- **React + Tailwind UI** with glass-morphism design, dark mode, and animations
- **Health check endpoint** for monitoring
- **Production-ready architecture** with modular FastAPI service layers

---

##  Architecture Overview

```
plant-disease-detection-using-cnn/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── endpoints.py        # FastAPI route: POST /predict
│   │   ├── core/
│   │   │   ├── config.py           # Pydantic-settings based config
│   │   │   └── logging.py          # Centralized logging setup
│   │   ├── schemas/
│   │   │   └── prediction.py       # Pydantic request/response models
│   │   ├── services/
│   │   │   ├── image_service.py    # Leaf detection + EfficientNet preprocessing
│   │   │   └── model_service.py    # Singleton Keras model loader & predictor
│   │   └── main.py                 # FastAPI app, CORS, lifespan model load
│   ├── model/
│   │   └── potato_classifier_final.h5  # Trained CNN model
│   ├── scripts/
│   │   ├── capture_camera.py           # Live webcam prediction tool
│   │   ├── prepare_potato_dataset.py   # Dataset prep script
│   │   └── train_potato_classifier.py  # Model training script
│   ├── tests/
│   │   └── test_api.py             # API endpoint tests
│   ├── .env                        # Local environment variables
│   ├── .env.example                # Template for environment setup
│   ├── requirements.txt            # Python 3.11 pinned dependencies
│   └── run.py                      # Uvicorn entrypoint
└── Frontend/
    ├── src/
    │   ├── components/
    │   │   ├── DiagnosisResult.tsx  # Displays prediction result + treatment
    │   │   ├── ImageUpload.tsx      # Drag-and-drop image uploader
    │   │   └── ui/                  # shadcn/ui component library
    │   ├── pages/
    │   │   ├── Index.tsx            # Main page
    │   │   └── NotFound.tsx         # 404 page
    │   ├── services/
    │   │   └── api.ts               # Centralized API fetch service
    │   └── hooks/
    ├── .env                         # Frontend env (VITE_API_BASE_URL)
    ├── .env.example                 # Template
    └── package.json
```

---

##  Environment Setup

> **Required**: Python 3.11 (TensorFlow is incompatible with Python 3.13 on Windows)
> Node.js 18+

### Backend Setup

```bash
# 1. Create Python 3.11 virtual environment
py -3.11 -m venv backend/venv

# 2. Activate virtual environment
backend\venv\Scripts\activate          # Windows
# source backend/venv/bin/activate    # Linux/macOS

# 3. Install dependencies
pip install -r backend/requirements.txt

# 4. Copy environment file
copy backend\.env.example backend\.env
```

### Frontend Setup

```bash
# 1. Install Node dependencies
cd Frontend
npm install

# 2. Copy environment file
copy .env.example .env
```

---

##  Running the Application

### Start the Backend (FastAPI + Uvicorn)

```bash
# From project root
backend\venv\Scripts\activate
python backend/run.py
```

The API will be available at: **http://localhost:8000**
Interactive API docs: **http://localhost:8000/docs**

### Start the Frontend (React + Vite)

```bash
cd Frontend
npm run dev
```

The frontend will be available at: **http://localhost:5173**

---

##  API Documentation

### `GET /health`
Returns the health status of the API and whether the model is loaded.

**Response:**
```json
{
  "status": "healthy",
  "project": "Potato Plant Disease Detection API",
  "version": "1.0.0",
  "model_loaded": true
}
```

### `POST /predict`
Accepts an image file and returns a disease prediction.

**Request:** `multipart/form-data` with field `file` (image)

**Response:**
```json
{
  "label": "early_blight",
  "confidence": 0.9312,
  "message": null
}
```

**Labels:**
| Label | Meaning |
|-------|---------|
| `healthy` | Plant is healthy |
| `early_blight` | Early blight detected |
| `late_blight` | Late blight detected |
| `no_leaf` | No leaf detected in image |

---

##  Running Tests

```bash
# From project root (with venv activated)
backend\venv\Scripts\python -m pytest backend/tests/ -v
```

---

##  Troubleshooting

### `ImportError: DLL load failed while importing _pywrap_tensorflow_lite_metrics_wrapper`
**Cause**: TensorFlow is not compatible with Python 3.13 on Windows.
**Fix**: Use the Python 3.11 virtual environment (`backend/venv`). Never run the backend with the system Python 3.13.

### `ModuleNotFoundError: No module named 'app'`
**Fix**: Run `python backend/run.py` from the **project root**, not from inside `backend/`.

### CORS errors in browser
**Fix**: Ensure the backend is running on port 8000 and the frontend `.env` contains `VITE_API_BASE_URL=http://127.0.0.1:8000`.

### Model not loading
**Fix**: Confirm the model file exists at `backend/model/potato_classifier_final.h5`.

---

##  Dependencies

### Backend (`requirements.txt`)
| Package | Version | Purpose |
|---------|---------|---------|
| fastapi | 0.111.0 | Web framework |
| uvicorn | 0.30.0 | ASGI server |
| tensorflow | 2.16.2 | ML inference engine |
| tf-keras | 2.16.0 | Keras model loading |
| numpy | 1.26.4 | Numerical computing |
| opencv-python-headless | 4.10.0.84 | Image processing |
| pillow | 10.4.0 | Image I/O |
| pydantic-settings | 2.3.4 | Configuration management |
| python-multipart | 0.0.9 | File upload support |
| python-dotenv | 1.0.1 | .env file loading |

---

##  Deployment Notes

- Set `CORS_ORIGINS` in `.env` to your production frontend URL.
- Set `reload=False` in `run.py` for production.
- Consider serving the frontend as static files via a CDN or Nginx.
- For Docker deployment, create a `Dockerfile` using `python:3.11-slim` as the base image.

---

## Recommended Python Version

**Python 3.11.x** — officially supported by TensorFlow 2.16+ on Windows.

```
Python 3.11.9
TensorFlow 2.16.2
Keras 3.3.3
NumPy 1.26.4
```
