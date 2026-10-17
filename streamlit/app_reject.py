import os
import json
import io
import numpy as np
import cv2
import tensorflow as tf
import keras
import streamlit as st
import requests
from PIL import Image

# ── Focal Loss (nécessaire pour charger le modèle) ────────────────────────────
@keras.saving.register_keras_serializable()
def loss_fn(y_true, y_pred):
    gamma = 2.0
    alpha = 0.25
    y_pred = tf.clip_by_value(y_pred, 1e-8, 1.0)
    cross_entropy = -y_true * tf.math.log(y_pred)
    weight = alpha * y_true * tf.pow(1 - y_pred, gamma)
    return tf.reduce_sum(weight * cross_entropy, axis=1)

# ── Chargement modèles (cachés en mémoire pour performance) ────────────────────
@st.cache_resource
def load_models():
    """Charge tous les modèles une seule fois et les garde en mémoire"""
    base_dir  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, "models")
    
    # Brain classifier
    brain_clf = keras.models.load_model(
        os.path.join(model_dir, "classifier_final.keras"),
        custom_objects={"loss_fn": loss_fn},
        compile=False
    )
    with open(os.path.join(model_dir, "class_names.json")) as f:
        brain_classes = json.load(f)
    
    # Breast classifier (aussi cachée maintenant!)
    breast_clf = keras.models.load_model(
        os.path.join(model_dir, "breast_classifier_final.keras"),
        custom_objects={"loss_fn": loss_fn},
        compile=False
    )
    with open(os.path.join(model_dir, "breast_class_names.json")) as f:
        breast_classes = json.load(f)
    
    return {
        "brain_clf": brain_clf,
        "brain_classes": brain_classes,
        "breast_clf": breast_clf,
        "breast_classes": breast_classes
    }

# ── Cache global pour Grad-CAM (réutilise la même grad_model) ──────────────────
_gradcam_models = {}

# ── Grad-CAM optimisé ─────────────────────────────────────────────────────────
@tf.function(jit_compile=True)
def _compute_gradcam_tf(conv_outputs, predictions, grads):
    """Calcul des gradients compilé avec XLA pour vitesse"""
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_out * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap

def generate_gradcam(model, img_array):
    """Génère Grad-CAM avec caching et optimisation XLA"""
    model_id = id(model)
    
    # Trouver la dernière couche Conv2D du modèle
    if model_id not in _gradcam_models:
        last_conv_layer = None
        for layer in reversed(model.layers):
            if "conv" in layer.name.lower():
                last_conv_layer = layer.name
                break
        
        if last_conv_layer is None:
            raise ValueError("Aucune couche de convolution trouvée dans le modèle")
        
        try:
            conv_output = model.get_layer(last_conv_layer).output
            model_output = model.output
            
            # Gérer les cas où output est une liste
            if isinstance(model_output, (list, tuple)):
                model_output = model_output[0]
            
            grad_model = tf.keras.models.Model(
                inputs=model.input,
                outputs=[conv_output, model_output]
            )
            _gradcam_models[model_id] = (grad_model, last_conv_layer)
        except Exception as e:
            raise ValueError(f"Erreur accès couche: {str(e)}")
    
    grad_model, _ = _gradcam_models[model_id]
    
    # Forward pass + gradient computation
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        
        # Gérer les cas où predictions est une liste
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]
        
        pred_class = tf.argmax(predictions[0])
        loss = predictions[:, pred_class]
    
    grads = tape.gradient(loss, conv_outputs)
    
    # Utiliser la fonction compilée pour le calcul final
    heatmap = _compute_gradcam_tf(conv_outputs, predictions, grads)
    
    return heatmap.numpy(), int(pred_class.numpy())

def overlay_gradcam(img_pil, heatmap):
    img_array       = np.array(img_pil.resize((380, 380)))
    heatmap_resized = cv2.resize(heatmap, (380, 380))
    heatmap_colored = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    superimposed    = cv2.addWeighted(img_array, 0.6, heatmap_colored, 0.4, 0)
    return Image.fromarray(superimposed)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedVision AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');
* { box-sizing: border-box; }
html, body, [data-testid="stAppViewContainer"] {
    background: #0a0e1a; color: #e8eaf0;
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1829 50%, #0a1520 100%);
}
.med-header {
    display:flex; align-items:center; justify-content:space-between;
    padding:20px 40px; border-bottom:1px solid rgba(64,156,255,0.15);
    background:rgba(10,14,26,0.8); margin-bottom:32px;
}
.med-logo { font-family:'DM Serif Display',serif; font-size:1.8rem; color:#fff; }
.med-logo span { color:#409cff; }
.med-status {
    display:flex; align-items:center; gap:8px; font-size:0.8rem;
    color:#4ade80; background:rgba(74,222,128,0.1);
    border:1px solid rgba(74,222,128,0.2); padding:6px 14px; border-radius:20px;
}
.med-status-dot {
    width:6px; height:6px; background:#4ade80;
    border-radius:50%; animation:pulse 2s infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
.result-main {
    background:rgba(255,255,255,0.03);
    border:1px solid rgba(255,255,255,0.08);
    border-radius:24px; padding:32px; text-align:center;
    position:relative; overflow:hidden;
}
.result-main::before {
    content:''; position:absolute; top:0; left:0; right:0; height:3px;
    background:linear-gradient(90deg,#409cff,#818cf8,#4ade80);
}
.result-badge {
    display:inline-block; padding:6px 16px; border-radius:20px;
    font-size:0.75rem; font-weight:600; letter-spacing:1px;
    text-transform:uppercase; margin-bottom:16px;
}
.badge-confirmed { background:rgba(74,222,128,0.15); color:#4ade80; border:1px solid rgba(74,222,128,0.3); }
.badge-uncertain { background:rgba(251,191,36,0.15); color:#fbbf24; border:1px solid rgba(251,191,36,0.3); }
.badge-suspect   { background:rgba(239,68,68,0.15); color:#ef4444; border:1px solid rgba(239,68,68,0.3); }
.result-diagnosis {
    font-family:'DM Serif Display',serif; font-size:3rem;
    color:#fff; line-height:1; margin-bottom:8px;
}
.conf-bar-wrap {
    background:rgba(255,255,255,0.05); border-radius:100px;
    height:8px; margin:16px 0 24px; overflow:hidden;
}
.conf-bar {
    height:100%; border-radius:100px;
    background:linear-gradient(90deg,#409cff,#4ade80);
}
.metrics-row {
    display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-top:20px;
}
.metric-box {
    background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06);
    border-radius:12px; padding:14px; text-align:center;
}
.metric-val { font-size:1.2rem; font-weight:600; color:#fff; }
.metric-lbl { font-size:0.7rem; color:#6b7280; margin-top:4px; text-transform:uppercase; letter-spacing:1px; }
.img-panel {
    background:rgba(255,255,255,0.02);
    border:1px solid rgba(255,255,255,0.06); border-radius:20px; overflow:hidden;
}
.img-panel-header {
    padding:14px 20px; border-bottom:1px solid rgba(255,255,255,0.06);
    font-size:0.7rem; letter-spacing:2px; text-transform:uppercase; color:#6b7280;
}
#MainMenu, footer, header { visibility:hidden; }
.stButton > button {
    background:rgba(64,156,255,0.1); border:1px solid rgba(64,156,255,0.3);
    color:#409cff; border-radius:10px; padding:8px 20px;
    font-family:'DM Sans',sans-serif; font-size:0.85rem;
}
.stButton > button:hover {
    background:rgba(64,156,255,0.2); border-color:#409cff;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
api_ok = False
try:
    health = requests.get("http://127.0.0.1:8000/health", timeout=2).json()
    api_ok = True
except Exception:
    pass

st.markdown(f"""
<div class="med-header">
    <div class="med-logo">Med<span>Vision</span> AI</div>
    <div style="font-size:0.75rem;color:#6b7280;letter-spacing:2px;text-transform:uppercase">
        Tumor Detection System
    </div>
    <div class="med-status">
        <div class="med-status-dot"></div>
        {'System Online' if api_ok else 'API Offline'}
    </div>
</div>
""", unsafe_allow_html=True)

# ── Layout ────────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.markdown('<div style="font-size:0.75rem;letter-spacing:2px;text-transform:uppercase;color:#6b7280;margin-bottom:12px">Imaging Modality</div>', unsafe_allow_html=True)
    mod_col1, mod_col2 = st.columns(2)
    with mod_col1:
        if st.button("🧠  Brain MRI", use_container_width=True):
            st.session_state["modality"] = "brain_mri"
    with mod_col2:
        if st.button("🔬  Histopathology", use_container_width=True):
            st.session_state["modality"] = "breast_histo"

    modality  = st.session_state.get("modality", "brain_mri")
    mod_label = "🧠 Brain MRI — EfficientNetB4 · 95.71%" \
        if modality == "brain_mri" else "🔬 Breast Histo — CNN Custom · 89.15%"
    st.markdown(f'<div style="font-size:0.8rem;color:#409cff;padding:8px 0 20px">Active: {mod_label}</div>', unsafe_allow_html=True)

    st.markdown('<div style="font-size:0.75rem;letter-spacing:2px;text-transform:uppercase;color:#6b7280;margin-bottom:12px">Medical Image</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.markdown('<div class="img-panel"><div class="img-panel-header">SCAN PREVIEW</div>', unsafe_allow_html=True)
        st.image(img, use_column_width=True)
        st.markdown(f'<div style="padding:10px 20px;font-size:0.75rem;color:#6b7280">{uploaded.name} · {img.size[0]}×{img.size[1]}px</div></div>', unsafe_allow_html=True)

        # Grad-CAM — Brain MRI uniquement
        if modality == "brain_mri":
            st.markdown("<div style='margin-top:20px'>", unsafe_allow_html=True)
            show_gradcam = st.toggle("🧠 Afficher Grad-CAM", value=False)
            if show_gradcam:
                progress_placeholder = st.empty()
                progress_placeholder.info("⏳ Compilation du modèle Grad-CAM (1ère fois: ~5-10s)...")
                
                try:
                    import time
                    from tensorflow.keras.applications.efficientnet import preprocess_input
                    
                    t_start = time.time()
                    
                    # Chargement modèles
                    models = load_models()
                    brain_clf = models["brain_clf"]
                    
                    # Prétraitement
                    img_resized  = img.resize((380, 380))
                    img_arr      = preprocess_input(np.array(img_resized, dtype=np.float32))
                    img_input    = np.expand_dims(img_arr, axis=0)
                    
                    progress_placeholder.info("🔥 Génération heatmap Grad-CAM (compilation XLA en cours)...")
                    
                    # Génération Grad-CAM (compilée avec XLA)
                    heatmap, _ = generate_gradcam(brain_clf, img_input)
                    
                    progress_placeholder.info("🎨 Superposition de la carte thermique...")
                    gradcam_img = overlay_gradcam(img, heatmap)
                    
                    elapsed = (time.time() - t_start) * 1000
                    
                    # Afficher résultat
                    progress_placeholder.success(f"✅ Grad-CAM généré en {elapsed:.0f}ms")
                    
                    st.markdown('<div class="img-panel"><div class="img-panel-header">GRAD-CAM ACTIVATION MAP</div>', unsafe_allow_html=True)
                    st.image(gradcam_img, use_column_width=True)
                    st.markdown(f'<div style="padding:10px 20px;font-size:0.75rem;color:#818cf8">🔮 Zones d\'activation — Rouge = forte activation | ⚡ Temps: {elapsed:.0f}ms (XLA compilé)</div></div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    progress_placeholder.error(f"Grad-CAM error: {str(e)[:100]}")
            st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown('<div style="font-size:0.75rem;letter-spacing:2px;text-transform:uppercase;color:#6b7280;margin-bottom:12px">Diagnostic Result</div>', unsafe_allow_html=True)

    if not uploaded:
        st.markdown("""
        <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);
        border-radius:24px;padding:80px 40px;text-align:center">
            <div style="font-size:3rem;margin-bottom:16px;opacity:0.3">⬡</div>
            <div style="color:#6b7280;font-size:0.9rem">Upload a medical image to begin analysis</div>
        </div>""", unsafe_allow_html=True)
    else:
        with st.spinner("Analysing..."):
            try:
                img_bytes = uploaded.getvalue()
                response  = requests.post(
                    "http://127.0.0.1:8000/predict",
                    files={"file": ("image.jpg", img_bytes, "image/jpeg")},
                    data={"modality": modality},
                    timeout=60
                )
                result = response.json()

                status    = result.get("status", "UNKNOWN")
                pred      = result.get("prediction", "N/A")
                conf      = result.get("confidence", 0)
                probs     = result.get("probabilities", {})
                recon_err = result.get("reconstruction_error")
                infer_ms  = result.get("inference_ms", 0)

                badge_cls  = {"CONFIRMED": "badge-confirmed", "UNCERTAIN": "badge-uncertain", "SUSPECT": "badge-suspect"}.get(status, "badge-uncertain")
                badge_icon = {"CONFIRMED": "✓ Confirmed", "UNCERTAIN": "⚠ Review Required", "SUSPECT": "⚡ Atypical Case"}.get(status, status)

                st.markdown(f"""
                <div class="result-main">
                    <span class="result-badge {badge_cls}">{badge_icon}</span>
                    <div class="result-diagnosis">{pred.upper() if pred != "N/A" else "—"}</div>
                    <div style="font-size:1rem;color:#6b7280;margin-bottom:8px">
                        Confidence &nbsp;<span style="color:#409cff;font-weight:600;font-size:1.3rem">{conf}%</span>
                    </div>
                    <div class="conf-bar-wrap">
                        <div class="conf-bar" style="width:{conf}%"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                recon_display = f"{recon_err:.4f}" if recon_err else "N/A"
                st.markdown(f"""
                <div class="metrics-row">
                    <div class="metric-box">
                        <div class="metric-val">{conf}%</div>
                        <div class="metric-lbl">Confidence</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-val">{recon_display}</div>
                        <div class="metric-lbl">Recon Error</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-val">{infer_ms}ms</div>
                        <div class="metric-lbl">Inference</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Bouton détails
                st.markdown("<div style='margin-top:20px'>", unsafe_allow_html=True)
                if st.button("+ More Details", use_container_width=False):
                    st.session_state["show_details"] = not st.session_state.get("show_details", False)

                if st.session_state.get("show_details", False) and probs:
                    st.markdown("""
                    <div style="margin-top:16px;background:rgba(255,255,255,0.02);
                    border:1px solid rgba(255,255,255,0.06);border-radius:16px;padding:20px">
                    <div style="font-size:0.7rem;letter-spacing:2px;text-transform:uppercase;
                    color:#6b7280;margin-bottom:16px">All Class Probabilities</div>
                    """, unsafe_allow_html=True)

                    max_prob = max(probs.values())
                    for cls, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                        icon = "🥇" if prob == max_prob else "  "
                        st.markdown(f"""
                        <div style="display:flex;align-items:center;justify-content:space-between;
                        padding:10px 0;border-bottom:1px solid rgba(255,255,255,0.05)">
                            <div style="font-size:0.85rem;color:#9ca3af">{icon} {cls.capitalize()}</div>
                            <div style="display:flex;align-items:center;gap:12px">
                                <div style="width:100px;height:4px;background:rgba(255,255,255,0.1);border-radius:4px;overflow:hidden">
                                    <div style="width:{int(prob)}%;height:100%;background:#409cff;border-radius:4px"></div>
                                </div>
                                <div style="font-size:0.85rem;color:#e8eaf0;font-weight:500;width:40px">{prob}%</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                st.markdown(f"""
                <div style="background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.2);
                border-radius:16px;padding:24px;color:#ef4444">
                    ❌ Error: {str(e)}<br>
                    <span style="font-size:0.8rem;color:#6b7280">Vérifie que l'API tourne sur le port 8000</span>
                </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:32px 0 16px;color:#374151;font-size:0.7rem;letter-spacing:1px">
    MEDVISION AI · TUMOR DETECTION · EFFICIENTNETB4 + CNN CUSTOM · FOR RESEARCH USE ONLY
</div>
""", unsafe_allow_html=True)