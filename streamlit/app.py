import base64
import glob
import hashlib
import html
import io
import os
from datetime import datetime

import numpy as np
import requests
import streamlit as st
from PIL import Image


API_URL = os.getenv("TUMOR_API_URL", "http://127.0.0.1:8000")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "logo.jpg")
CONFIDENCE_THRESHOLD = 70.0
RECON_THRESHOLD = 0.02

MODALITIES = {
    "brain_mri": {
        "label": "IRM cérébrale",
        "short": "IRM",
        "model": "EfficientNetB4",
        "accuracy": "95.71%",
        "input": "380 × 380 RGB",
        "classes": 4,
        "gradcam": True,
        "samples": ["samples/brain_mri/*.jpg"],
        "description": "Classification IRM en 4 classes, filtrage des cas atypiques par autoencodeur et explicabilité Grad-CAM.",
    },
    "breast_histo": {
        "label": "Histopathologie",
        "short": "Histo",
        "model": "CNN personnalisé",
        "accuracy": "89.15%",
        "input": "50 × 50 RGB",
        "classes": 2,
        "gradcam": False,
        "samples": ["samples/histopathology/*.png"],
        "description": "Classification de patchs de tissu mammaire : tissu sain ou carcinome.",
    },
}

# label, severity level (0 = sain), stage text, color
CLASS_INFO = {
    "notumor": ("Absence de tumeur", 0, "Sain", "#12946b"),
    "meningioma": ("Méningiome", 1, "Stade I · bénin", "#d4a017"),
    "pituitary": ("Tumeur hypophysaire", 2, "Stade II · modéré", "#e07a1f"),
    "glioma": ("Gliome", 3, "Stade III/IV · agressif", "#d64545"),
    "sain": ("Tissu sain", 0, "Sain", "#12946b"),
    "carcinome": ("Carcinome", 3, "Malin", "#d64545"),
}

SEVERITY_SCALE = {
    "brain_mri": ["notumor", "meningioma", "pituitary", "glioma"],
    "breast_histo": ["sain", "carcinome"],
}

STATUS_INFO = {
    "CONFIRMED": {
        "label": "Diagnostic confirmé",
        "cls": "ok",
        "icon": "✓",
        "note": "La confiance du modèle dépasse le seuil de confirmation de 70 %.",
    },
    "UNCERTAIN": {
        "label": "Revue clinique recommandée",
        "cls": "warn",
        "icon": "!",
        "note": "La confiance est sous le seuil de 70 %. Une relecture par un spécialiste est recommandée.",
    },
    "SUSPECT": {
        "label": "Cas atypique détecté",
        "cls": "alert",
        "icon": "?",
        "note": "L'autoencodeur signale une image hors distribution. L'expertise humaine est requise.",
    },
}


# ── Assets ───────────────────────────────────────────────────────────────────

def to_data_uri(image, fmt="PNG", quality=88):
    buf = io.BytesIO()
    if fmt == "JPEG":
        image.convert("RGB").save(buf, format="JPEG", quality=quality)
    else:
        image.save(buf, format=fmt)
    mime = "jpeg" if fmt == "JPEG" else fmt.lower()
    return f"data:image/{mime};base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


@st.cache_resource(show_spinner=False)
def load_brand_icon():
    """Crop the brain emblem out of the wide logo file (drops the margins and the wordmark)."""
    if not os.path.exists(LOGO_PATH):
        return None
    logo = Image.open(LOGO_PATH).convert("RGB")
    ink = np.asarray(logo).min(axis=2) < 200
    rows = np.where(ink.any(axis=1))[0]
    if rows.size == 0:
        return logo
    # The emblem is the first block of ink rows; the wordmark follows after a blank gap.
    end = rows[0]
    for r in rows[1:]:
        if r - end > 8:
            break
        end = r
    cols = np.where(ink[rows[0]:end + 1].any(axis=0))[0]
    pad = 18
    box = (
        max(cols.min() - pad, 0),
        max(rows[0] - pad, 0),
        min(cols.max() + pad, logo.width),
        min(end + pad, logo.height),
    )
    icon = logo.crop(box)
    icon.thumbnail((256, 256))
    return icon


def sample_images(modality):
    paths = []
    for pattern in MODALITIES[modality]["samples"]:
        paths.extend(glob.glob(os.path.join(BASE_DIR, pattern)))
    return sorted(paths)


# ── API ──────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=5)
def check_api_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        response.raise_for_status()
        return True, response.json()
    except Exception as exc:
        return False, {"error": str(exc)}


@st.cache_data(show_spinner=False)
def run_prediction(image_bytes, modality, file_name, image_hash):
    del image_hash
    response = requests.post(
        f"{API_URL}/predict",
        files={"file": (file_name or "image.jpg", image_bytes, "image/jpeg")},
        data={"modality": modality},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


@st.cache_data(show_spinner=False)
def run_gradcam(image_bytes, file_name, image_hash):
    del image_hash
    response = requests.post(
        f"{API_URL}/gradcam",
        files={"file": (file_name or "image.jpg", image_bytes, "image/jpeg")},
        data={"modality": "brain_mri"},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    image = Image.open(io.BytesIO(base64.b64decode(payload["image_base64"]))).convert("RGB")
    return image, payload


# ── Helpers ──────────────────────────────────────────────────────────────────

def render(markup):
    """Render HTML as one line so Markdown never turns indented lines into code blocks."""
    st.markdown(" ".join(line.strip() for line in markup.splitlines() if line.strip()), unsafe_allow_html=True)


def class_label(key):
    return CLASS_INFO.get(str(key).lower(), (str(key).title(), None, None, "#0085c3"))[0]


def class_color(key):
    return CLASS_INFO.get(str(key).lower(), (None, None, None, "#0085c3"))[3]


def esc(value):
    return html.escape(str(value))


def build_report(result, modality, file_name):
    status = result.get("status", "UNKNOWN")
    lines = [
        "MedVision AI — Rapport d'analyse",
        "=" * 40,
        f"Date            : {datetime.now():%d/%m/%Y %H:%M}",
        f"Fichier         : {file_name}",
        f"Modalité        : {MODALITIES[modality]['label']} ({MODALITIES[modality]['model']})",
        f"Statut          : {STATUS_INFO.get(status, {}).get('label', status)}",
    ]
    if result.get("prediction"):
        info = CLASS_INFO.get(str(result["prediction"]).lower())
        lines.append(f"Résultat        : {class_label(result['prediction'])}")
        if info:
            lines.append(f"Stade indicatif : {info[2]}")
        lines.append(f"Confiance       : {float(result.get('confidence') or 0):.2f} %")
    if isinstance(result.get("reconstruction_error"), (int, float)):
        lines.append(f"Erreur recons.  : {result['reconstruction_error']:.4f} (seuil {RECON_THRESHOLD})")
    lines.append(f"Inférence       : {result.get('inference_ms', 0)} ms")
    probabilities = result.get("probabilities") or {}
    if probabilities:
        lines += ["", "Probabilités par classe :"]
        for cls, prob in sorted(probabilities.items(), key=lambda item: item[1], reverse=True):
            lines.append(f"  - {class_label(cls):<22} {float(prob):6.2f} %")
    lines += [
        "",
        STATUS_INFO.get(status, {}).get("note", ""),
        "",
        "Outil de recherche — validation clinique requise avant tout usage opérationnel.",
    ]
    return "\n".join(lines)


def record_history(result, modality, file_name, image, image_hash):
    history = st.session_state.setdefault("history", [])
    key = f"{image_hash}:{modality}"
    if any(item["key"] == key for item in history):
        return
    thumb = image.copy()
    thumb.thumbnail((120, 120))
    history.insert(0, {
        "key": key,
        "time": datetime.now().strftime("%H:%M:%S"),
        "file": file_name,
        "modality": modality,
        "status": result.get("status", "UNKNOWN"),
        "prediction": result.get("prediction"),
        "confidence": result.get("confidence"),
        "thumb": to_data_uri(thumb, "JPEG", 80),
    })
    del history[24:]


# ── Styles ───────────────────────────────────────────────────────────────────

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Montserrat:wght@300;400;600;700;800&display=swap');

:root {
    --navy-950: #061a33;
    --navy-900: #0a2445;
    --navy-800: #0e3460;
    --navy-700: #174a82;
    --blue: #0085c3;
    --blue-bright: #1ea7e8;
    --glow: #5cc8ff;
    --ink: #0d2240;
    --muted: #5d6f86;
    --line: #dbe5ef;
    --canvas: #f2f6fa;
    --panel: #ffffff;
    --soft: #f6f9fc;
    --ok: #12946b;
    --ok-soft: #e6f6f0;
    --warn: #c98a12;
    --warn-soft: #fdf4e1;
    --alert: #d64545;
    --alert-soft: #fdecec;
    --radius: 14px;
    --shadow: 0 1px 2px rgba(10, 36, 69, .06), 0 12px 32px rgba(10, 36, 69, .08);
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background: var(--canvas) !important;
    color: var(--ink);
    font-family: Inter, system-ui, -apple-system, "Segoe UI", sans-serif;
}
[data-testid="stHeader"], #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }
[data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none !important; }
.block-container { max-width: 1340px !important; padding: 1rem 1.5rem 2rem !important; }
h1, h2, h3, h4, p { margin: 0; }
.stMarkdown p { color: inherit; }

@keyframes fadeUp { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }
@keyframes grow { from { width: 0; } }
@keyframes ringIn { from { stroke-dashoffset: 326.73; } }
@keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(18, 148, 107, .55); } 70% { box-shadow: 0 0 0 8px rgba(18, 148, 107, 0); } 100% { box-shadow: 0 0 0 0 rgba(18, 148, 107, 0); } }
@keyframes scan { 0% { top: 0; opacity: 0; } 8% { opacity: 1; } 92% { opacity: 1; } 100% { top: 100%; opacity: 0; } }
@keyframes orbit { to { transform: rotate(360deg); } }
@keyframes breathe { 0%, 100% { opacity: .55; transform: scale(1); } 50% { opacity: .9; transform: scale(1.06); } }

/* ── Navbar ── */
.nav {
    display: flex; align-items: center; justify-content: space-between; gap: 16px;
    padding: 10px 14px 10px 10px; background: var(--panel);
    border: 1px solid var(--line); border-radius: var(--radius); box-shadow: var(--shadow);
}
.brand { display: flex; align-items: center; gap: 12px; }
.brand-mark {
    width: 48px; height: 48px; border-radius: 12px; background: #fff;
    border: 1px solid var(--line); display: grid; place-items: center; overflow: hidden;
    box-shadow: 0 6px 18px rgba(0, 133, 195, .18);
}
.brand-mark img { width: 40px; height: 40px; object-fit: contain; }
.brand-name { font-family: Montserrat, Inter, sans-serif; font-size: 1.32rem; color: var(--navy-800); line-height: 1; letter-spacing: -.01em; }
.brand-name .l { font-weight: 300; }
.brand-name .b { font-weight: 800; }
.brand-tag { font-size: .74rem; color: var(--muted); margin-top: 4px; letter-spacing: .02em; }
.nav-right { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }
.chip {
    display: inline-flex; align-items: center; gap: 7px; padding: 6px 11px; border-radius: 999px;
    font-size: .76rem; font-weight: 600; color: var(--muted); background: var(--soft); border: 1px solid var(--line); white-space: nowrap;
}
.chip b { color: var(--ink); font-weight: 700; }
.dot { width: 8px; height: 8px; border-radius: 50%; background: var(--alert); }
.chip.online { color: var(--ok); background: var(--ok-soft); border-color: #bfe6d7; }
.chip.online .dot { background: var(--ok); animation: pulse 2s infinite; }
.chip.offline { color: var(--alert); background: var(--alert-soft); border-color: #f3c4c4; }

/* ── Hero ── */
.hero {
    position: relative; overflow: hidden; margin-top: 14px; border-radius: 18px;
    padding: 30px 34px; color: #fff;
    background:
        radial-gradient(circle at 82% 50%, rgba(30, 167, 232, .35), transparent 38%),
        linear-gradient(120deg, var(--navy-950) 0%, var(--navy-800) 60%, #0f4f8a 100%);
    display: grid; grid-template-columns: minmax(0, 1fr) 220px; gap: 24px; align-items: center;
    animation: fadeUp .5s ease both;
}
.hero::before {
    content: ""; position: absolute; inset: 0; pointer-events: none;
    background-image: linear-gradient(rgba(92, 200, 255, .07) 1px, transparent 1px), linear-gradient(90deg, rgba(92, 200, 255, .07) 1px, transparent 1px);
    background-size: 28px 28px;
    mask-image: linear-gradient(90deg, transparent, #000 45%);
    -webkit-mask-image: linear-gradient(90deg, transparent, #000 45%);
}
.hero > * { position: relative; }
.eyebrow { font-size: .72rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; color: var(--glow); }
.hero-title { font-family: Montserrat, Inter, sans-serif; font-size: clamp(1.7rem, 3.2vw, 2.55rem); font-weight: 700; line-height: 1.12; margin-top: 10px; color: #fff; }
.hero-title .accent { color: var(--glow); }
.hero p.lead { color: #c4d6ea; font-size: .97rem; line-height: 1.6; margin-top: 10px; max-width: 640px; }
.hero-stats { display: flex; gap: 10px; margin-top: 20px; flex-wrap: wrap; }
.hstat { padding: 10px 14px; border-radius: 12px; background: rgba(255, 255, 255, .07); border: 1px solid rgba(255, 255, 255, .12); backdrop-filter: blur(4px); min-width: 120px; }
.hstat b { display: block; font-family: Montserrat, Inter, sans-serif; font-size: 1.2rem; font-weight: 700; color: #fff; }
.hstat span { font-size: .72rem; color: #a9c1db; }
.emblem { position: relative; width: 200px; height: 200px; margin: 0 auto; display: grid; place-items: center; }
.emblem .halo { position: absolute; inset: 18px; border-radius: 50%; background: radial-gradient(circle, rgba(92, 200, 255, .45), transparent 70%); animation: breathe 4s ease-in-out infinite; }
.emblem .ring { position: absolute; inset: 0; border-radius: 50%; border: 1px dashed rgba(92, 200, 255, .45); animation: orbit 30s linear infinite; }
.emblem .ring::after { content: ""; position: absolute; top: -4px; left: 50%; width: 8px; height: 8px; border-radius: 50%; background: var(--glow); box-shadow: 0 0 12px var(--glow); }
.emblem .disc { position: relative; width: 132px; height: 132px; border-radius: 50%; background: #fff; display: grid; place-items: center; box-shadow: 0 0 0 6px rgba(255, 255, 255, .08), 0 18px 40px rgba(0, 0, 0, .35); }
.emblem .disc img { width: 96px; height: 96px; object-fit: contain; }

/* ── Panels ── */
.panel { background: var(--panel); border: 1px solid var(--line); border-radius: var(--radius); box-shadow: var(--shadow); animation: fadeUp .45s ease both; }
.section-title { display: flex; align-items: center; gap: 10px; margin: 18px 0 10px; }
.section-title .num { width: 26px; height: 26px; border-radius: 8px; display: grid; place-items: center; font-size: .78rem; font-weight: 800; color: #fff; background: linear-gradient(135deg, var(--navy-800), var(--blue)); }
.section-title h3 { font-size: 1rem; font-weight: 700; color: var(--ink); }
.section-title small { color: var(--muted); font-size: .8rem; margin-left: auto; }

.model-card { padding: 14px 16px; display: flex; gap: 14px; align-items: center; }
.model-icon { width: 46px; height: 46px; border-radius: 12px; flex: none; display: grid; place-items: center; background: linear-gradient(135deg, var(--navy-800), var(--blue)); color: #fff; font-size: .72rem; font-weight: 800; letter-spacing: .04em; box-shadow: 0 6px 16px rgba(0, 133, 195, .25); }
.model-card .t { font-weight: 700; font-size: .95rem; }
.model-card .d { color: var(--muted); font-size: .8rem; line-height: 1.45; margin-top: 2px; }
.model-specs { display: flex; gap: 6px; flex-wrap: wrap; margin-left: auto; }

/* ── Viewer ── */
.viewer { overflow: hidden; }
.viewer-head { display: flex; justify-content: space-between; align-items: center; gap: 10px; padding: 11px 14px; border-bottom: 1px solid var(--line); }
.viewer-head .t { font-weight: 700; font-size: .88rem; }
.viewer-head .m { color: var(--muted); font-size: .76rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 60%; }
.stage { background: radial-gradient(circle at 50% 40%, #0f2a4d, #050f1f 75%); padding: 16px; display: grid; gap: 12px; grid-template-columns: 1fr; }
.stage.compare { grid-template-columns: 1fr 1fr; }
.frame { position: relative; margin: 0; border-radius: 10px; overflow: hidden; aspect-ratio: 1 / 1; background: #000; }
.frame img { width: 100%; height: 100%; object-fit: contain; display: block; }
.frame .scanline { position: absolute; left: 0; right: 0; height: 3px; background: linear-gradient(90deg, transparent, var(--glow), transparent); box-shadow: 0 0 18px 4px rgba(92, 200, 255, .55); animation: scan 2.2s ease-in-out 2 both; }
.frame .c { position: absolute; width: 18px; height: 18px; border: 2px solid var(--glow); opacity: .85; }
.frame .tl { top: 8px; left: 8px; border-right: 0; border-bottom: 0; border-radius: 4px 0 0 0; }
.frame .tr { top: 8px; right: 8px; border-left: 0; border-bottom: 0; border-radius: 0 4px 0 0; }
.frame .bl { bottom: 8px; left: 8px; border-right: 0; border-top: 0; border-radius: 0 0 0 4px; }
.frame .br { bottom: 8px; right: 8px; border-left: 0; border-top: 0; border-radius: 0 0 4px 0; }
.frame figcaption { position: absolute; left: 10px; bottom: 10px; padding: 3px 8px; border-radius: 6px; font-size: .7rem; font-weight: 700; color: #fff; background: rgba(6, 26, 51, .75); letter-spacing: .04em; text-transform: uppercase; }
.viewer-foot { padding: 11px 14px; font-size: .78rem; color: var(--muted); line-height: 1.5; border-top: 1px solid var(--line); background: var(--soft); }
.heat-scale { display: flex; align-items: center; gap: 10px; margin-top: 6px; }
.heat-bar { flex: 1; height: 8px; border-radius: 999px; background: linear-gradient(90deg, #0033cc, #00b3ff, #33dd66, #ffdd00, #ff3300); }

/* ── Empty states ── */
.empty { padding: 34px 24px; text-align: center; border: 1.5px dashed #c5d4e3; border-radius: var(--radius); background: rgba(255, 255, 255, .7); color: var(--muted); }
.empty .ico { width: 56px; height: 56px; margin: 0 auto 12px; border-radius: 16px; display: grid; place-items: center; font-size: 1.5rem; background: linear-gradient(135deg, #e3f3fb, #eef4fb); color: var(--navy-800); }
.empty strong { display: block; color: var(--ink); font-size: 1rem; margin-bottom: 4px; }
.steps { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 20px; text-align: left; }
.step { padding: 12px; border-radius: 12px; background: var(--panel); border: 1px solid var(--line); font-size: .8rem; }
.step b { display: block; color: var(--blue); font-size: .72rem; letter-spacing: .08em; text-transform: uppercase; margin-bottom: 4px; }

/* ── Result ── */
.result { overflow: hidden; }
.banner { display: flex; align-items: center; gap: 12px; padding: 12px 18px; font-weight: 700; font-size: .9rem; }
.banner .ic { width: 26px; height: 26px; border-radius: 50%; display: grid; place-items: center; color: #fff; font-weight: 800; font-size: .85rem; flex: none; }
.banner .mod { margin-left: auto; font-size: .74rem; font-weight: 600; opacity: .85; }
.banner.ok { background: var(--ok-soft); color: var(--ok); } .banner.ok .ic { background: var(--ok); }
.banner.warn { background: var(--warn-soft); color: var(--warn); } .banner.warn .ic { background: var(--warn); }
.banner.alert { background: var(--alert-soft); color: var(--alert); } .banner.alert .ic { background: var(--alert); }
.verdict { display: grid; grid-template-columns: minmax(0, 1fr) 150px; gap: 18px; align-items: center; padding: 20px 22px 6px; }
.verdict .k { font-size: .72rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); }
.verdict .dx { font-family: Montserrat, Inter, sans-serif; font-size: clamp(1.8rem, 3.4vw, 2.6rem); font-weight: 800; line-height: 1.08; margin: 6px 0 10px; }
.verdict .note { color: var(--muted); font-size: .86rem; line-height: 1.55; }
.stage-tag { display: inline-flex; align-items: center; gap: 8px; padding: 5px 11px; border-radius: 999px; font-size: .78rem; font-weight: 700; border: 1px solid currentColor; margin-bottom: 10px; }
.stage-tag i { width: 8px; height: 8px; border-radius: 50%; background: currentColor; }
.gauge { position: relative; width: 150px; height: 150px; }
.gauge svg { width: 150px; height: 150px; transform: rotate(-90deg); }
.gauge .track { fill: none; stroke: #e6edf4; stroke-width: 11; }
.gauge .val { fill: none; stroke-width: 11; stroke-linecap: round; stroke-dasharray: 326.73; animation: ringIn 1.1s cubic-bezier(.2, .8, .2, 1) both; }
.gauge .thr { stroke: var(--ink); stroke-width: 2; opacity: .45; }
.gauge .lbl { position: absolute; inset: 0; display: grid; place-items: center; text-align: center; }
.gauge .lbl b { display: block; font-family: Montserrat, Inter, sans-serif; font-size: 1.5rem; font-weight: 800; color: var(--ink); line-height: 1; }
.gauge .lbl span { font-size: .7rem; color: var(--muted); font-weight: 600; letter-spacing: .06em; text-transform: uppercase; }

.block { padding: 14px 22px; border-top: 1px solid var(--line); }
.block-title { font-size: .72rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); margin-bottom: 10px; }
.sev { display: grid; gap: 6px; }
.sev-seg { position: relative; padding: 10px 8px 8px; border-radius: 10px; text-align: center; font-size: .74rem; font-weight: 600; color: #8a9aae; background: var(--soft); border: 1px solid var(--line); transition: all .3s; }
.sev-seg::before { content: ""; position: absolute; top: 0; left: 0; right: 0; height: 4px; border-radius: 10px 10px 0 0; background: var(--c); opacity: .35; }
.sev-seg.on { color: #fff; background: var(--c); border-color: var(--c); transform: translateY(-2px); box-shadow: 0 8px 20px -6px var(--c); }
.sev-seg.on::before { opacity: 0; }
.sev-seg small { display: block; font-weight: 500; font-size: .68rem; opacity: .85; margin-top: 2px; }

.prob { display: grid; grid-template-columns: 150px minmax(0, 1fr) 62px; gap: 12px; align-items: center; padding: 7px 0; }
.prob .n { font-size: .85rem; font-weight: 600; display: flex; align-items: center; gap: 8px; }
.prob .n i { width: 9px; height: 9px; border-radius: 3px; flex: none; }
.prob .tr { height: 10px; border-radius: 999px; background: #edf2f7; overflow: hidden; }
.prob .fl { height: 100%; border-radius: 999px; animation: grow 1s cubic-bezier(.2, .8, .2, 1) both; }
.prob .v { text-align: right; font-weight: 700; font-size: .85rem; font-variant-numeric: tabular-nums; color: var(--muted); }
.prob.top .n, .prob.top .v { color: var(--ink); }
.prob.top .tr { height: 14px; }

.metrics { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }
.metric { padding: 12px; border-radius: 12px; background: var(--soft); border: 1px solid var(--line); }
.metric b { display: block; font-size: 1.05rem; font-weight: 800; font-variant-numeric: tabular-nums; }
.metric span { font-size: .72rem; color: var(--muted); }
.mini { height: 5px; border-radius: 999px; background: #e3eaf2; margin-top: 8px; overflow: hidden; }
.mini i { display: block; height: 100%; border-radius: 999px; animation: grow 1s ease both; }

.disclaimer { font-size: .74rem; color: var(--muted); padding: 12px 22px; background: var(--soft); border-top: 1px solid var(--line); }

/* ── History ── */
.hist-summary { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin-bottom: 14px; }
.hist-summary .metric b { font-size: 1.4rem; font-family: Montserrat, Inter, sans-serif; }
.hist { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 12px; }
.hcard { display: flex; gap: 12px; padding: 10px; align-items: center; }
.hcard img { width: 64px; height: 64px; border-radius: 10px; object-fit: cover; background: #000; flex: none; }
.hcard .t { font-weight: 700; font-size: .9rem; }
.hcard .s { font-size: .74rem; color: var(--muted); margin-top: 2px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 170px; }
.pill { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: .68rem; font-weight: 700; margin-top: 5px; }
.pill.ok { background: var(--ok-soft); color: var(--ok); } .pill.warn { background: var(--warn-soft); color: var(--warn); } .pill.alert { background: var(--alert-soft); color: var(--alert); }

/* ── Models tab ── */
.models { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.mcard { padding: 20px; }
.mcard h4 { font-size: 1.05rem; font-weight: 700; }
.mcard .acc { font-family: Montserrat, Inter, sans-serif; font-size: 2.2rem; font-weight: 800; color: var(--navy-800); margin: 10px 0 2px; }
.mcard .row { display: flex; justify-content: space-between; padding: 9px 0; border-top: 1px solid var(--line); font-size: .84rem; }
.mcard .row span:first-child { color: var(--muted); }
.mcard .row span:last-child { font-weight: 600; text-align: right; }
.pipeline { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0; margin-top: 14px; padding: 18px; }
.pstep { position: relative; padding: 0 14px; text-align: center; font-size: .8rem; color: var(--muted); }
.pstep b { display: block; color: var(--ink); font-size: .88rem; margin: 8px 0 3px; }
.pstep .bubble { width: 40px; height: 40px; margin: 0 auto; border-radius: 50%; display: grid; place-items: center; color: #fff; font-weight: 800; background: linear-gradient(135deg, var(--navy-800), var(--blue)); }
.pstep:not(:last-child)::after { content: ""; position: absolute; top: 20px; left: calc(50% + 26px); right: calc(-50% + 26px); height: 2px; background: linear-gradient(90deg, var(--blue), var(--line)); }

.footer { display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-top: 28px; padding: 14px 4px 0; border-top: 1px solid var(--line); font-size: .74rem; color: var(--muted); }
.footer img { width: 22px; height: 22px; object-fit: contain; vertical-align: middle; margin-right: 6px; }

/* ── Streamlit widgets ── */
[data-testid="stTabs"] [role="tablist"] { gap: 4px; border-bottom: 1px solid var(--line); }
[data-testid="stTabs"] button[role="tab"] { padding: 10px 16px; border-radius: 10px 10px 0 0; }
[data-testid="stTabs"] button[role="tab"] p { font-weight: 600; font-size: .92rem; color: var(--muted); }
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p { color: var(--navy-800); font-weight: 700; }
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background: var(--blue) !important; height: 3px; }
[data-testid="stSegmentedControl"] button { border-radius: 10px !important; font-weight: 600 !important; }
[data-testid="stSegmentedControl"] button[aria-checked="true"],
[data-testid="stSegmentedControl"] button[kind="segmented_controlActive"] {
    background: var(--navy-800) !important; color: #fff !important; border-color: var(--navy-800) !important;
}
[data-testid="stFileUploader"] section { border: 1.5px dashed #a9c3dc !important; border-radius: var(--radius) !important; background: #fff !important; padding: 18px !important; transition: border-color .2s, background .2s; }
[data-testid="stFileUploader"] section:hover { border-color: var(--blue) !important; background: #f3faff !important; }
.stButton > button, .stDownloadButton > button { border-radius: 10px !important; font-weight: 600 !important; }
.stDownloadButton > button { background: var(--navy-800) !important; color: #fff !important; border: 0 !important; }
.stDownloadButton > button:hover { background: var(--blue) !important; }
[data-testid="stWidgetLabel"] p { font-weight: 600; color: var(--ink); font-size: .84rem; }

@media (max-width: 900px) {
    .hero { grid-template-columns: 1fr; padding: 22px; }
    .emblem { display: none; }
    .verdict { grid-template-columns: 1fr; }
    .gauge { margin: 0 auto; }
    .metrics, .hist-summary, .steps, .models, .pipeline { grid-template-columns: 1fr; }
    .pstep { padding: 10px 0; } .pstep::after { display: none; }
    .prob { grid-template-columns: 1fr 52px; } .prob .tr { grid-column: 1 / -1; grid-row: 2; }
    .nav { flex-direction: column; align-items: flex-start; }
    .model-card { flex-wrap: wrap; } .model-specs { margin-left: 0; }
}
</style>
"""


# ── Components ───────────────────────────────────────────────────────────────

def render_navbar(icon_uri, api_online):
    status = '<span class="chip online"><span class="dot"></span>API en ligne</span>' if api_online \
        else '<span class="chip offline"><span class="dot"></span>API hors ligne</span>'
    mark = f'<img src="{icon_uri}" alt="MedVision AI">' if icon_uri else "<b>MV</b>"
    render(f"""
<div class="nav">
    <div class="brand">
        <div class="brand-mark">{mark}</div>
        <div>
            <div class="brand-name"><span class="l">Med</span><span class="b">Vision</span><span class="l"> AI</span></div>
            <div class="brand-tag">Détection de tumeurs &amp; explicabilité</div>
        </div>
    </div>
    <div class="nav-right">
        <span class="chip">Modèles <b>2</b></span>
        <span class="chip">Endpoint <b>{esc(API_URL.replace("http://", ""))}</b></span>
        {status}
    </div>
</div>
""")


def render_hero(icon_uri):
    emblem = f'<img src="{icon_uri}" alt="">' if icon_uri else ""
    render(f"""
<section class="hero">
    <div>
        <div class="eyebrow">Analyse d'imagerie médicale par IA</div>
        <div class="hero-title" role="heading" aria-level="1">Un second regard <span class="accent">intelligent</span><br>sur chaque image.</div>
        <p class="lead">Importez une IRM cérébrale ou une coupe histopathologique : le modèle classe l'image,
        mesure sa confiance, filtre les cas atypiques et localise les zones déterminantes avec Grad-CAM.</p>
        <div class="hero-stats">
            <div class="hstat"><b>95.71%</b><span>Précision IRM</span></div>
            <div class="hstat"><b>89.15%</b><span>Précision histo</span></div>
            <div class="hstat"><b>6</b><span>Classes détectées</span></div>
            <div class="hstat"><b>Grad-CAM</b><span>Explicabilité</span></div>
        </div>
    </div>
    <div class="emblem"><div class="halo"></div><div class="ring"></div><div class="disc">{emblem}</div></div>
</section>
""")


def render_section(num, title, hint=""):
    render(f'<div class="section-title"><span class="num">{num}</span><h3>{title}</h3><small>{hint}</small></div>')


def render_model_card(modality):
    meta = MODALITIES[modality]
    render(f"""
<div class="panel model-card">
    <div class="model-icon">{meta["short"].upper()}</div>
    <div>
        <div class="t">{meta["model"]} · {meta["label"]}</div>
        <div class="d">{meta["description"]}</div>
    </div>
    <div class="model-specs">
        <span class="chip">Précision <b>{meta["accuracy"]}</b></span>
        <span class="chip">Entrée <b>{meta["input"]}</b></span>
        <span class="chip">Classes <b>{meta["classes"]}</b></span>
    </div>
</div>
""")


def frame(uri, caption, scan=False):
    scanline = '<span class="scanline"></span>' if scan else ""
    return (
        f'<figure class="frame"><img src="{uri}" alt="{esc(caption)}">{scanline}'
        '<i class="c tl"></i><i class="c tr"></i><i class="c bl"></i><i class="c br"></i>'
        f"<figcaption>{esc(caption)}</figcaption></figure>"
    )


def render_viewer(image, file_name, mode, gradcam_image=None, gradcam_payload=None):
    display = image.copy()
    display.thumbnail((760, 760))
    original = frame(to_data_uri(display, "JPEG"), "Original", scan=mode == "Original")
    titles = {"Original": "Image importée", "Grad-CAM": "Carte d'activation Grad-CAM", "Comparaison": "Original vs Grad-CAM"}

    if mode != "Original" and gradcam_image is not None:
        heat = frame(to_data_uri(gradcam_image, "JPEG"), "Grad-CAM", scan=True)
        stage = f'<div class="stage compare">{original}{heat}</div>' if mode == "Comparaison" else f'<div class="stage">{heat}</div>'
        layer = esc(gradcam_payload.get("layer", "dernière couche convolutive")) if gradcam_payload else ""
        elapsed = gradcam_payload.get("inference_ms", 0) if gradcam_payload else 0
        foot = f"""
<div class="viewer-foot">
    Les zones chaudes indiquent les régions qui ont le plus influencé la décision du modèle.
    <div class="heat-scale"><span>Faible</span><div class="heat-bar"></div><span>Forte</span></div>
    <div style="margin-top:6px;font-size:.72rem">Couche : {layer} · {elapsed} ms</div>
</div>"""
    else:
        stage = f'<div class="stage">{original}</div>'
        foot = '<div class="viewer-foot">Image exacte transmise au modèle sélectionné.</div>'

    render(f"""
<div class="panel viewer">
    <div class="viewer-head">
        <span class="t">{titles.get(mode, titles["Original"])}</span>
        <span class="m">{esc(file_name)} · {image.size[0]} × {image.size[1]} px</span>
    </div>
    {stage}
    {foot}
</div>
""")


def render_empty_input():
    render("""
<div class="empty">
    <div class="ico">⤒</div>
    <strong>En attente d'une image</strong>
    Glissez-déposez un fichier JPG ou PNG ci-dessus, ou choisissez un exemple.
</div>
""")


def render_empty_result():
    render("""
<div class="empty">
    <div class="ico">◎</div>
    <strong>Aucune analyse pour le moment</strong>
    Le diagnostic apparaîtra ici dès qu'une image sera importée.
    <div class="steps">
        <div class="step"><b>Étape 1</b>Choisissez la modalité d'imagerie.</div>
        <div class="step"><b>Étape 2</b>Importez une image ou un exemple.</div>
        <div class="step"><b>Étape 3</b>Lisez le résultat et la carte Grad-CAM.</div>
    </div>
</div>
""")


def gauge_html(confidence, color):
    circumference = 326.73
    value = max(0.0, min(confidence, 100.0))
    offset = circumference * (1 - value / 100)
    # threshold tick at 70 %
    angle = np.deg2rad(360 * CONFIDENCE_THRESHOLD / 100)
    x1, y1 = 60 + 46 * np.cos(angle), 60 + 46 * np.sin(angle)
    x2, y2 = 60 + 58 * np.cos(angle), 60 + 58 * np.sin(angle)
    return f"""
<div class="gauge">
    <svg viewBox="0 0 120 120">
        <circle class="track" cx="60" cy="60" r="52"></circle>
        <circle class="val" cx="60" cy="60" r="52" stroke="{color}" stroke-dashoffset="{offset:.2f}"></circle>
        <line class="thr" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"></line>
    </svg>
    <div class="lbl"><div><b>{value:.2f}%</b><span>Confiance</span></div></div>
</div>"""


def severity_html(modality, prediction):
    scale = SEVERITY_SCALE[modality]
    segments = []
    for key in scale:
        label, _, stage, color = CLASS_INFO[key]
        on = " on" if key == prediction else ""
        segments.append(f'<div class="sev-seg{on}" style="--c:{color}">{label}<small>{stage}</small></div>')
    return (
        '<div class="block"><div class="block-title">Échelle de sévérité indicative</div>'
        f'<div class="sev" style="grid-template-columns:repeat({len(scale)},1fr)">{"".join(segments)}</div></div>'
    )


def probabilities_html(probabilities, prediction):
    rows = []
    ordered = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    for index, (cls, prob) in enumerate(ordered):
        value = max(0.0, min(float(prob), 100.0))
        color = class_color(cls)
        top = " top" if str(cls).lower() == prediction else ""
        rows.append(
            f'<div class="prob{top}"><div class="n"><i style="background:{color}"></i>{esc(class_label(cls))}</div>'
            f'<div class="tr"><div class="fl" style="width:{value:.2f}%;background:{color};animation-delay:{index * 0.08:.2f}s"></div></div>'
            f'<div class="v">{value:.2f}%</div></div>'
        )
    return f'<div class="block"><div class="block-title">Probabilités par classe</div>{"".join(rows)}</div>'


def metrics_html(result, modality):
    confidence = float(result.get("confidence") or 0)
    recon = result.get("reconstruction_error")
    inference = result.get("inference_ms", 0)
    conf_color = "var(--ok)" if confidence >= CONFIDENCE_THRESHOLD else "var(--warn)"
    if result.get("confidence") is None:
        cells = ['<div class="metric"><b>Non évaluée</b><span>Confiance · image non classée</span></div>']
    else:
        cells = [
            f'<div class="metric"><b>{confidence:.2f}%</b><span>Confiance · seuil 70 %</span>'
            f'<div class="mini"><i style="width:{min(confidence, 100):.1f}%;background:{conf_color}"></i></div></div>'
        ]
    if isinstance(recon, (int, float)):
        ratio = min(recon / RECON_THRESHOLD, 1.0) * 100
        recon_color = "var(--alert)" if recon > RECON_THRESHOLD else "var(--ok)"
        cells.append(
            f'<div class="metric"><b>{recon:.4f}</b><span>Erreur recons. · seuil {RECON_THRESHOLD}</span>'
            f'<div class="mini"><i style="width:{ratio:.1f}%;background:{recon_color}"></i></div></div>'
        )
    else:
        cells.append(f'<div class="metric"><b>{MODALITIES[modality]["model"]}</b><span>Modèle utilisé</span></div>')
    cells.append(f'<div class="metric"><b>{inference} ms</b><span>Temps d\'inférence</span></div>')
    return f'<div class="block"><div class="block-title">Indicateurs techniques</div><div class="metrics">{"".join(cells)}</div></div>'


def render_result(result, modality):
    status = result.get("status", "UNKNOWN")
    info = STATUS_INFO.get(status, {"label": str(status).title(), "cls": "warn", "icon": "!", "note": "Réponse inattendue de l'API."})
    prediction = str(result.get("prediction") or "").lower()
    confidence = float(result.get("confidence") or 0)
    probabilities = result.get("probabilities") or {}

    if prediction:
        label, _, stage, color = CLASS_INFO.get(prediction, (prediction.title(), None, None, "#0085c3"))
        stage_tag = f'<span class="stage-tag" style="color:{color}"><i></i>{stage}</span>' if stage else ""
        gauge = gauge_html(confidence, color)
    else:
        label, color = "Analyse non concluante", "var(--alert)"
        stage_tag = '<span class="stage-tag" style="color:var(--alert)"><i></i>Hors distribution</span>'
        gauge = '<div class="gauge"><div class="lbl"><div><b style="color:var(--alert)">—</b><span>Non classé</span></div></div></div>'

    body = [
        f'<div class="banner {info["cls"]}"><span class="ic">{info["icon"]}</span>{info["label"]}'
        f'<span class="mod">{MODALITIES[modality]["label"]}</span></div>',
        f"""<div class="verdict"><div>
            <div class="k">Résultat principal</div>
            <div class="dx" style="color:{color}">{esc(label)}</div>
            {stage_tag}
            <p class="note">{info["note"]}</p>
        </div>{gauge}</div>""",
    ]
    if prediction in SEVERITY_SCALE[modality]:
        body.append(severity_html(modality, prediction))
    if probabilities:
        body.append(probabilities_html(probabilities, prediction))
    body.append(metrics_html(result, modality))
    body.append('<div class="disclaimer">⚕ Outil d\'aide à la décision — ne remplace pas l\'avis d\'un professionnel de santé.</div>')
    render(f'<div class="panel result">{"".join(body)}</div>')


def render_history():
    history = st.session_state.get("history", [])
    if not history:
        render("""
<div class="empty" style="margin-top:12px">
    <div class="ico">⟲</div>
    <strong>Historique vide</strong>
    Les analyses réalisées pendant cette session apparaîtront ici.
</div>
""")
        return
    confirmed = sum(1 for item in history if item["status"] == "CONFIRMED")
    flagged = len(history) - confirmed
    confidences = [float(item["confidence"]) for item in history if item.get("confidence") is not None]
    mean_conf = f"{np.mean(confidences):.1f}%" if confidences else "—"
    render(f"""
<div class="hist-summary" style="margin-top:12px">
    <div class="metric"><b>{len(history)}</b><span>Analyses</span></div>
    <div class="metric"><b style="color:var(--ok)">{confirmed}</b><span>Confirmées</span></div>
    <div class="metric"><b style="color:var(--warn)">{flagged}</b><span>À revoir</span></div>
    <div class="metric"><b>{mean_conf}</b><span>Confiance moyenne</span></div>
</div>
""")
    cards = []
    for item in history:
        info = STATUS_INFO.get(item["status"], {"label": item["status"], "cls": "warn"})
        title = class_label(item["prediction"]) if item.get("prediction") else "Cas atypique"
        conf = f' · {float(item["confidence"]):.1f}%' if item.get("confidence") is not None else ""
        cards.append(f"""
<div class="panel hcard">
    <img src="{item["thumb"]}" alt="">
    <div style="min-width:0">
        <div class="t">{esc(title)}{conf}</div>
        <div class="s">{MODALITIES[item["modality"]]["short"]} · {item["time"]} · {esc(item["file"])}</div>
        <span class="pill {info["cls"]}">{info["label"]}</span>
    </div>
</div>""")
    render(f'<div class="hist">{"".join(cards)}</div>')
    st.write("")
    if st.button("Effacer l'historique", icon=":material/delete:"):
        st.session_state.history = []
        st.rerun()


def render_models_tab(health):
    cards = []
    for key, meta in MODALITIES.items():
        classes = ", ".join(CLASS_INFO[c][0] for c in SEVERITY_SCALE[key])
        cards.append(f"""
<div class="panel mcard">
    <div class="eyebrow" style="color:var(--blue)">{meta["label"]}</div>
    <h4>{meta["model"]}</h4>
    <div class="acc">{meta["accuracy"]}</div>
    <div style="color:var(--muted);font-size:.78rem;margin-bottom:12px">Précision de référence sur le jeu de test</div>
    <div class="row"><span>Entrée</span><span>{meta["input"]}</span></div>
    <div class="row"><span>Classes</span><span>{classes}</span></div>
    <div class="row"><span>Seuil de confirmation</span><span>{CONFIDENCE_THRESHOLD:.0f} %</span></div>
    <div class="row"><span>Filtre d'anomalies</span><span>{"Autoencodeur · seuil " + str(RECON_THRESHOLD) if key == "brain_mri" else "—"}</span></div>
    <div class="row"><span>Grad-CAM</span><span>{"Disponible" if meta["gradcam"] else "Non disponible"}</span></div>
</div>""")
    render(f'<div class="models" style="margin-top:12px">{"".join(cards)}</div>')
    render("""
<div class="panel pipeline" style="margin-top:14px">
    <div class="pstep"><div class="bubble">1</div><b>Import</b>JPG / PNG redimensionné à l'entrée du modèle</div>
    <div class="pstep"><div class="bubble">2</div><b>Filtrage</b>L'autoencodeur écarte les images atypiques</div>
    <div class="pstep"><div class="bubble">3</div><b>Classification</b>Probabilités et niveau de confiance</div>
    <div class="pstep"><div class="bubble">4</div><b>Explicabilité</b>Grad-CAM localise les zones décisives</div>
</div>
""")
    if isinstance(health, dict) and "models" in health:
        st.caption(f"Classes exposées par l'API : {health['models']}")


def render_footer(icon_uri):
    img = f'<img src="{icon_uri}" alt="">' if icon_uri else ""
    render(f"""
<div class="footer">
    <span>{img}<b>MedVision AI</b> · Interface de recherche</span>
    <span>Validation clinique requise avant tout usage opérationnel</span>
</div>
""")


# ── Analysis tab ─────────────────────────────────────────────────────────────

def render_analysis(api_online):
    top_left, top_right = st.columns([0.9, 2.1], vertical_alignment="center")
    with top_left:
        selected = st.segmented_control(
            "Modalité d'imagerie",
            options=list(MODALITIES),
            format_func=lambda key: (":material/neurology: " if key == "brain_mri" else ":material/biotech: ") + MODALITIES[key]["label"],
            default="brain_mri",
            key="modality",
        )
    modality = selected or "brain_mri"
    with top_right:
        render_model_card(modality)

    left, right = st.columns([1, 1.08], gap="large")

    with left:
        render_section(1, "Image à analyser", "JPG · JPEG · PNG")
        uploaded = st.file_uploader(
            "Importer une image médicale",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            key=f"upload_{modality}",
        )
        samples = sample_images(modality)
        sample = None
        if samples and not uploaded:
            sample = st.selectbox(
                "Ou essayer avec une image d'exemple",
                options=[None] + samples,
                format_func=lambda p: "— Choisir un exemple —" if p is None else os.path.basename(p),
                key=f"sample_{modality}",
            )

        image = image_bytes = image_hash = file_name = None
        if uploaded:
            image_bytes, file_name = uploaded.getvalue(), uploaded.name
        elif sample:
            with open(sample, "rb") as f:
                image_bytes = f.read()
            file_name = os.path.basename(sample)

        if image_bytes:
            image_hash = hashlib.sha256(image_bytes).hexdigest()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            modes = ["Original", "Grad-CAM", "Comparaison"] if MODALITIES[modality]["gradcam"] else ["Original"]
            mode = st.segmented_control(
                "Affichage",
                options=modes,
                default="Original",
                key=f"view_{modality}",
                disabled=len(modes) == 1,
                help=None if len(modes) > 1 else "Grad-CAM est disponible uniquement pour l'IRM cérébrale.",
            ) or "Original"

            gradcam_image = gradcam_payload = None
            if mode != "Original":
                if not api_online:
                    st.error("L'API est hors ligne : Grad-CAM nécessite le backend FastAPI sur le port 8000.")
                    mode = "Original"
                else:
                    with st.spinner("Génération de la carte Grad-CAM…"):
                        try:
                            gradcam_image, gradcam_payload = run_gradcam(image_bytes, file_name, image_hash)
                        except Exception as exc:
                            st.error(f"Impossible de générer Grad-CAM : {exc}")
                            mode = "Original"
            render_viewer(image, file_name, mode, gradcam_image, gradcam_payload)
        else:
            render_empty_input()

    with right:
        render_section(2, "Résultat du diagnostic", MODALITIES[modality]["model"])
        if not image_bytes:
            render_empty_result()
            return
        if not api_online:
            st.error("L'API est hors ligne. Démarrez FastAPI sur le port 8000 puis relancez l'analyse.")
            return
        with st.spinner("Analyse de l'image en cours…"):
            try:
                result = run_prediction(image_bytes, modality, file_name, image_hash)
            except Exception as exc:
                st.error(f"Échec de la prédiction : {exc}")
                return
        if "error" in result:
            st.error(f"Erreur de l'API : {result['error']}")
            return
        render_result(result, modality)
        record_history(result, modality, file_name, image, image_hash)
        st.write("")
        st.download_button(
            "Télécharger le rapport",
            data=build_report(result, modality, file_name),
            file_name=f"rapport_medvision_{os.path.splitext(file_name)[0]}.txt",
            mime="text/plain",
            icon=":material/download:",
            width="stretch",
        )


def main():
    icon = load_brand_icon()
    st.set_page_config(
        page_title="MedVision AI · Détection de tumeurs",
        page_icon=icon if icon is not None else "🧠",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    icon_uri = to_data_uri(icon) if icon is not None else ""
    api_online, health = check_api_health()

    render_navbar(icon_uri, api_online)
    render_hero(icon_uri)
    if not api_online:
        st.warning(f"Le backend n'est pas joignable sur {API_URL}. Lancez run_backend.bat avant d'analyser une image.", icon=":material/cloud_off:")

    tab_analysis, tab_history, tab_models = st.tabs(["Analyse", "Historique de session", "Modèles"])
    with tab_analysis:
        render_analysis(api_online)
    with tab_history:
        render_history()
    with tab_models:
        render_models_tab(health)

    render_footer(icon_uri)


if __name__ == "__main__":
    main()
