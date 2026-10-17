from fastapi import FastAPI, File, UploadFile, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf
import keras
import numpy as np
from PIL import Image
import base64
import json
import io
import os
import time
import tempfile
import importlib
from datetime import datetime, timezone
import pydicom

# ── 1. App FastAPI ────────────────────────────────────────────────────────────
app = FastAPI(title="Tumor Detection API — Multi-Modal")

MONGODB_URI         = os.getenv("MONGODB_URI", "mongodb://localhost:27017/tumor_detection")
MONGODB_COLLECTION  = os.getenv("MONGODB_COLLECTION", "diagnostics")
MONGODB_FALLBACK_DB = os.getenv("MONGODB_DB", "tumor_detection")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 2. Configuration des répertoires ──────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ── 3. Focal Loss ─────────────────────────────────────────────────────────────
@keras.saving.register_keras_serializable()
def loss_fn(y_true, y_pred):
    gamma = 2.0
    alpha = 0.25
    y_pred = tf.clip_by_value(y_pred, 1e-8, 1.0)
    cross_entropy = -y_true * tf.math.log(y_pred)
    weight = alpha * y_true * tf.pow(1 - y_pred, gamma)
    return tf.reduce_sum(weight * cross_entropy, axis=1)

# ── 4. Chargement des modèles ─────────────────────────────────────────────────
print("Chargement des modèles...")

brain_classifier = keras.models.load_model(
    os.path.join(MODEL_DIR, "classifier_final.keras"),
    custom_objects={"loss_fn": loss_fn},
    compile=False
)
# Pas de recompilation inutile — modèles déjà optimisés pour inférence

brain_autoencoder = keras.models.load_model(os.path.join(MODEL_DIR, "autoencoder_final.keras"), compile=False)

with open(os.path.join(MODEL_DIR, "class_names.json")) as f:
    brain_classes = json.load(f)

breast_classifier = keras.models.load_model(os.path.join(MODEL_DIR, "breast_classifier_final.keras"), compile=False)
# Pas de recompilation inutile — modèles déjà optimisés pour inférence

with open(os.path.join(MODEL_DIR, "breast_class_names.json")) as f:
    breast_classes = json.load(f)

print("✅ Modèles chargés !")
print("Brain classes  :", brain_classes)
print("Breast classes :", breast_classes)

# ── 5. Warm-up (après chargement des modèles) ─────────────────────────────────
def warmup_models():
    print("🔥 Warm-up des modèles...")
    start = time.time()
    
    # Brain classifier warm-up
    t0 = time.time()
    brain_classifier.predict(np.zeros((1, 380, 380, 3), dtype=np.float32), verbose=0)
    t_brain = (time.time() - t0) * 1000
    print(f"  ✓ Brain classifier: {t_brain:.1f}ms")
    
    # Breast classifier warm-up
    t0 = time.time()
    breast_classifier.predict(np.zeros((1, 50, 50, 3), dtype=np.float32), verbose=0)
    t_breast = (time.time() - t0) * 1000
    print(f"  ✓ Breast classifier: {t_breast:.1f}ms")
    
    # Brain autoencoder warm-up
    t0 = time.time()
    brain_autoencoder.predict(np.zeros((1, 224, 224, 3), dtype=np.float32), verbose=0)
    t_auto = (time.time() - t0) * 1000
    print(f"  ✓ Brain autoencoder: {t_auto:.1f}ms")
    
    total = (time.time() - start) * 1000
    print(f"✅ Warm-up terminé ! Total: {total:.1f}ms")

warmup_models()

# ── 6. Helpers prétraitement ──────────────────────────────────────────────────
def preprocess_brain(image_bytes: bytes) -> np.ndarray:
    from tensorflow.keras.applications.efficientnet import preprocess_input
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((380, 380))
    return preprocess_input(np.array(img, dtype=np.float32))

def preprocess_breast(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((50, 50))
    return np.array(img, dtype=np.float32) / 255.0

_brain_gradcam_model = None
_brain_gradcam_layer = None

def get_brain_gradcam_model():
    global _brain_gradcam_model, _brain_gradcam_layer
    if _brain_gradcam_model is not None:
        return _brain_gradcam_model, _brain_gradcam_layer

    last_conv_layer = None
    for layer in reversed(brain_classifier.layers):
        if "conv" in layer.name.lower():
            last_conv_layer = layer.name
            break
    if last_conv_layer is None:
        raise ValueError("Aucune couche de convolution trouvee dans le modele")

    conv_output = brain_classifier.get_layer(last_conv_layer).output
    model_output = brain_classifier.output[0] if isinstance(brain_classifier.output, (list, tuple)) else brain_classifier.output
    _brain_gradcam_model = tf.keras.models.Model(
        inputs=brain_classifier.input,
        outputs=[conv_output, model_output],
    )
    _brain_gradcam_layer = last_conv_layer
    return _brain_gradcam_model, _brain_gradcam_layer

def jet_colormap(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0, 1)
    red = np.clip(1.5 - np.abs(4 * values - 3), 0, 1)
    green = np.clip(1.5 - np.abs(4 * values - 2), 0, 1)
    blue = np.clip(1.5 - np.abs(4 * values - 1), 0, 1)
    return np.uint8(np.stack([red, green, blue], axis=-1) * 255)

def generate_brain_gradcam_overlay(image_bytes: bytes):
    grad_model, layer_name = get_brain_gradcam_model()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_380 = img.resize((380, 380))
    img_input = np.expand_dims(preprocess_brain(image_bytes), axis=0)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_input)
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]
        pred_class = tf.argmax(predictions[0])
        loss = predictions[:, pred_class]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_out * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    heatmap_img = Image.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255), mode="L")
    heatmap_resized = np.array(heatmap_img.resize((380, 380), Image.Resampling.BILINEAR)) / 255.0
    heatmap_colored = jet_colormap(heatmap_resized)
    base_img = np.array(img_380)
    overlay = np.uint8((base_img * 0.62) + (heatmap_colored * 0.38))
    overlay_img = Image.fromarray(overlay)

    buf = io.BytesIO()
    overlay_img.save(buf, format="JPEG", quality=88, optimize=True)
    return {
        "image_base64": base64.b64encode(buf.getvalue()).decode("ascii"),
        "predicted_index": int(pred_class.numpy()),
        "predicted_class": brain_classes[int(pred_class.numpy())],
        "layer": layer_name,
    }

# ── 7. DICOM helpers ──────────────────────────────────────────────────────────
def read_dicom_as_pil(dicom_bytes: bytes):
    with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
        tmp.write(dicom_bytes)
        tmp_path = tmp.name
    try:
        ds = pydicom.dcmread(tmp_path)
        px = ds.pixel_array.astype(float)
        px -= px.min()
        if px.max() > 0:
            px /= px.max()
        px = (px * 255).astype(np.uint8)
        img = Image.fromarray(px, mode="L").convert("RGB") if len(px.shape) == 2 \
              else Image.fromarray(px).convert("RGB")
        return img, ds
    finally:
        os.unlink(tmp_path)

def extract_dicom_metadata(ds) -> dict:
    fields = {
        "PatientID": "patient_id", "PatientAge": "patient_age",
        "PatientSex": "patient_sex", "Modality": "modality_dicom",
        "StudyDescription": "study_description", "SeriesDescription": "series_description",
        "Manufacturer": "manufacturer", "InstitutionName": "institution",
        "StudyDate": "study_date", "BodyPartExamined": "body_part",
    }
    return {key: str(getattr(ds, tag, "N/A") or "N/A") for tag, key in fields.items()}

# ── 7. MongoDB helpers ────────────────────────────────────────────────────────
def get_diagnostics_collection():
    try:
        mongo_module = importlib.import_module("pymongo")
    except Exception:
        return None, "pymongo is not installed"
    try:
        client = mongo_module.MongoClient(MONGODB_URI, serverSelectionTimeoutMS=2000)
        client.admin.command("ping")
        try:
            db = client.get_default_database()
        except Exception:
            db = client[MONGODB_FALLBACK_DB]
        return db[MONGODB_COLLECTION], None
    except Exception as exc:
        return None, str(exc)

def persist_diagnostic_record(payload: dict):
    collection, error = get_diagnostics_collection()
    if collection is None:
        print(f"⚠️  MongoDB unavailable: {error}")
        return
    try:
        collection.insert_one(payload)
    except Exception as exc:
        print(f"⚠️  Could not persist: {exc}")

def serialize_history_item(doc: dict) -> dict:
    ts = doc.get("timestamp")
    return {
        "id": str(doc.get("_id")),
        "session_id": doc.get("session_id", ""),
        "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts or ""),
        "modality": doc.get("modality", ""),
        "status": doc.get("status", ""),
        "prediction": doc.get("prediction", ""),
        "confidence": doc.get("confidence"),
        "reconstruction_error": doc.get("reconstruction_error"),
        "filename": doc.get("filename", ""),
        "content_type": doc.get("content_type", ""),
        "probabilities": doc.get("probabilities", {}),
        "message": doc.get("message", ""),
    }

# ── 8. Brain inference helper ─────────────────────────────────────────────────
def _brain_inference(img_bytes: bytes, img_pil: Image.Image):
    img_array   = preprocess_brain(img_bytes)
    ae_arr      = np.array(img_pil.resize((224, 224)), dtype=np.float32)
    ae_norm     = (ae_arr - ae_arr.min()) / (ae_arr.max() - ae_arr.min() + 1e-8)
    ae_in       = np.expand_dims(ae_norm, axis=0)
    recon_error = float(np.mean(np.square(
        ae_norm - brain_autoencoder.predict(ae_in, verbose=0)
    )))
    if recon_error > 0.02:
        return None, None, None, recon_error, "SUSPECT"
    probs      = brain_classifier.predict(np.expand_dims(img_array, 0), verbose=0)[0]
    pred_class = int(np.argmax(probs))
    confidence = float(probs[pred_class])
    status     = "CONFIRMED" if confidence >= 0.7 else "UNCERTAIN"
    return probs, pred_class, confidence, recon_error, status

# ── 9. Routes ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Tumor Detection API — Multi-Modal", "status": "running",
            "modalities": ["brain_mri", "breast_histo"]}

@app.get("/health")
def health():
    return {"status": "healthy",
            "models": {"brain_mri": brain_classes, "breast_histo": breast_classes}}

@app.post("/predict")
async def predict(file: UploadFile = File(...), modality: str = Form(...)):
    t0          = time.time()
    image_bytes = await file.read()
    timestamp   = datetime.now(timezone.utc)
    img_pil     = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    if modality == "brain_mri":
        probs, pred_class, confidence, recon_error, status = _brain_inference(image_bytes, img_pil)
        if status == "SUSPECT":
            response = {
                "modality": "brain_mri", "status": "SUSPECT",
                "message": "Cas atypique — expertise humaine requise",
                "reconstruction_error": round(recon_error, 6),
                "inference_ms": round((time.time() - t0) * 1000, 2),
            }
        else:
            response = {
                "modality": "brain_mri", "status": status,
                "prediction": brain_classes[pred_class],
                "confidence": round(confidence * 100, 2),
                "probabilities": {brain_classes[i]: round(float(p)*100, 2) for i, p in enumerate(probs)},
                "reconstruction_error": round(recon_error, 6),
                "inference_ms": round((time.time() - t0) * 1000, 2),
            }
        persist_diagnostic_record({
            "timestamp": timestamp, "modality": "brain_mri",
            "status": response["status"],
            "prediction": response.get("prediction", "Atypical"),
            "confidence": response.get("confidence"),
            "reconstruction_error": round(recon_error, 6),
            "probabilities": response.get("probabilities", {}),
            "filename": file.filename, "content_type": file.content_type or "",
            "message": response.get("message", ""),
        })
        return response

    elif modality == "breast_histo":
        img_array  = preprocess_breast(image_bytes)
        probs      = breast_classifier.predict(np.expand_dims(img_array, 0), verbose=0)[0]
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])
        status     = "CONFIRMED" if confidence >= 0.7 else "UNCERTAIN"
        response   = {
            "modality": "breast_histo", "status": status,
            "prediction": breast_classes[pred_class],
            "confidence": round(confidence * 100, 2),
            "probabilities": {breast_classes[i]: round(float(p)*100, 2) for i, p in enumerate(probs)},
            "inference_ms": round((time.time() - t0) * 1000, 2),
        }
        persist_diagnostic_record({
            "timestamp": timestamp, "modality": "breast_histo",
            "status": status, "prediction": breast_classes[pred_class],
            "confidence": round(confidence * 100, 2),
            "reconstruction_error": None,
            "probabilities": response["probabilities"],
            "filename": file.filename, "content_type": file.content_type or "",
            "message": "",
        })
        return response

    return {"error": "Modalité invalide — utilise 'brain_mri' ou 'breast_histo'"}

@app.post("/gradcam")
async def gradcam(file: UploadFile = File(...), modality: str = Form(default="brain_mri")):
    if modality != "brain_mri":
        return JSONResponse(
            status_code=400,
            content={"error": "Grad-CAM est disponible uniquement pour brain_mri"},
        )
    t0 = time.time()
    image_bytes = await file.read()
    try:
        payload = generate_brain_gradcam_overlay(image_bytes)
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})
    payload.update({
        "modality": "brain_mri",
        "format": "jpeg_base64",
        "inference_ms": round((time.time() - t0) * 1000, 2),
    })
    return payload

@app.post("/predict/dicom")
async def predict_dicom(file: UploadFile = File(...), modality: str = Form(default="brain_mri")):
    """Endpoint DICOM — compatible PACS"""
    t0          = time.time()
    dicom_bytes = await file.read()
    try:
        img_pil, ds = read_dicom_as_pil(dicom_bytes)
        metadata    = extract_dicom_metadata(ds)
    except Exception as exc:
        return JSONResponse(status_code=400, content={"error": f"Fichier DICOM invalide : {exc}"})

    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    if modality == "brain_mri":
        probs, pred_class, confidence, recon_error, status = _brain_inference(img_bytes, img_pil)
        if status == "SUSPECT":
            return {"modality": "brain_mri", "format": "DICOM", "status": "SUSPECT",
                    "message": "Cas atypique — expertise humaine requise",
                    "reconstruction_error": round(recon_error, 6),
                    "dicom_metadata": metadata,
                    "inference_ms": round((time.time() - t0) * 1000, 2)}
        return {"modality": "brain_mri", "format": "DICOM", "status": status,
                "prediction": brain_classes[pred_class],
                "confidence": round(confidence * 100, 2),
                "probabilities": {brain_classes[i]: round(float(p)*100, 2) for i, p in enumerate(probs)},
                "reconstruction_error": round(recon_error, 6),
                "dicom_metadata": metadata,
                "inference_ms": round((time.time() - t0) * 1000, 2)}

    elif modality == "breast_histo":
        img_array  = preprocess_breast(img_bytes)
        probs      = breast_classifier.predict(np.expand_dims(img_array, 0), verbose=0)[0]
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])
        status     = "CONFIRMED" if confidence >= 0.7 else "UNCERTAIN"
        return {"modality": "breast_histo", "format": "DICOM", "status": status,
                "prediction": breast_classes[pred_class],
                "confidence": round(confidence * 100, 2),
                "probabilities": {breast_classes[i]: round(float(p)*100, 2) for i, p in enumerate(probs)},
                "dicom_metadata": metadata,
                "inference_ms": round((time.time() - t0) * 1000, 2)}

    return {"error": "Modalité invalide"}

@app.get("/history")
def history(session_id: str = Query(default=""), limit: int = Query(default=10, ge=1, le=50)):
    collection, error = get_diagnostics_collection()
    if collection is None:
        return {"status": "unavailable", "message": f"MongoDB unavailable: {error}", "items": []}
    query = {"session_id": session_id} if session_id else {}
    items = [serialize_history_item(doc)
             for doc in collection.find(query).sort("timestamp", -1).limit(limit)]
    return {"status": "ok", "session_id": session_id, "count": len(items), "items": items}

@app.get("/benchmark")
def benchmark():
    """Mesure la latence d'inférence — KPI < 200ms sur GPU"""
    dummy = np.random.randint(0, 255, (380, 380, 3), dtype=np.uint8)
    img   = Image.fromarray(dummy)
    buf   = io.BytesIO()
    img.save(buf, format="JPEG")
    from tensorflow.keras.applications.efficientnet import preprocess_input
    arr   = preprocess_input(np.array(img.resize((380, 380)), dtype=np.float32))
    times = []
    for _ in range(5):
        t0 = time.time()
        brain_classifier.predict(np.expand_dims(arr, 0), verbose=0)
        times.append((time.time() - t0) * 1000)
    return {
        "avg_inference_ms": round(np.mean(times), 2),
        "min_ms":           round(np.min(times), 2),
        "max_ms":           round(np.max(times), 2),
        "target_ms":        200,
        "compliant":        bool(np.mean(times) < 200),
        "note":             "200ms target validated on GPU (Kaggle T4). CPU ~800ms expected."
    }
