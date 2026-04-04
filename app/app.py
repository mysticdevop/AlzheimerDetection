import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
from PIL import Image
import io
import time
import torch
import torch.nn as nn
from monai.networks.nets import resnet18
import matplotlib.pyplot as plt

# ─── Page config must be first ───────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroScan AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

html, body, [class*="css"] {
  font-family: 'Inter', 'Segoe UI', sans-serif;
}

.stApp {
  background: linear-gradient(135deg, #141e30, #243b55);
}

.stApp::before {
  content: "";
  position: fixed;
  top: 0; left: 0;
  width: 100%; height: 100%;
  background: radial-gradient(circle at 80% 10%, rgba(100,181,246,0.15), transparent 40%);
  pointer-events: none;
}

#MainMenu, footer, header { visibility: hidden; }

section.main > div {
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.block-container {
  padding-top: 2rem;
  padding-bottom: 3rem;
  max-width: 1100px;
}

.ns-header {
  background: linear-gradient(135deg, #2c5364, #203a43);
  border-radius: 22px;
  padding: 2.5rem;
  margin-bottom: 10px;
  box-shadow: 0 12px 30px rgba(0,0,0,0.25);
}

.ns-header h1 {
  font-family: 'DM Serif Display', serif;
  font-size: 2.2rem;
  color: #ffffff;
  margin: 0.4rem 0 0.3rem 0;
}

.ns-header p {
  color: rgba(255,255,255,0.75);
  font-size: 0.95rem;
  font-weight: 300;
  line-height: 1.6;
  margin: 0;
}

.ns-badge {
  display: inline-block;
  background: rgba(255,255,255,0.12);
  color: #fff;
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 1.2px;
  text-transform: uppercase;
  padding: 4px 12px;
  border-radius: 20px;
  margin-bottom: 0.8rem;
  border: 1px solid rgba(255,255,255,0.2);
}

.ns-card {
  background: rgba(255,255,255,0.07);
  backdrop-filter: blur(12px);
  border-radius: 16px;
  padding: 1.5rem;
  border: 1px solid rgba(255,255,255,0.12);
  box-shadow: 0 6px 24px rgba(0,0,0,0.25);
  margin-bottom: 10px;
  animation: fadeIn 0.3s ease;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(5px); }
  to   { opacity: 1; transform: translateY(0); }
}

.ns-card-title {
  font-size: 0.76rem;
  font-weight: 600;
  letter-spacing: 1.1px;
  text-transform: uppercase;
  color: #9fb7ff;
  margin-bottom: 1rem;
}

.stButton { margin-top: 12px; }

div.stButton > button {
  width: 100%;
  background: linear-gradient(135deg, #4facfe, #00c6ff);
  color: white;
  border-radius: 10px;
  font-weight: 600;
  border: none;
  padding: 0.7rem;
}

div.stButton > button:hover { transform: translateY(-1px); }

[data-testid="stFileUploader"] {
  background: #1b2a38;
  border: 2px dashed #5fa8d3;
  border-radius: 14px;
  padding: 1rem;
  margin-top: 10px;
}

.result-label {
  font-family: 'DM Serif Display', serif;
  font-size: 2rem;
  color: white;
}

.result-sub { color: #9fb7ff; font-size: 0.9rem; }

.conf-track {
  background: rgba(255,255,255,0.15);
  border-radius: 10px;
  height: 10px;
  margin: 0.6rem 0 0.3rem;
  overflow: hidden;
}

.conf-fill {
  height: 10px;
  border-radius: 10px;
}

.prob-row {
  display: flex;
  align-items: center;
  margin-bottom: 0.7rem;
  gap: 10px;
}

.prob-label {
  width: 48px;
  font-size: 0.8rem;
  font-weight: 600;
  color: #cfd8ff;
}

.prob-bar-wrap {
  flex: 1;
  background: rgba(255,255,255,0.1);
  border-radius: 6px;
  height: 9px;
  overflow: hidden;
}

.prob-bar-fill { height: 9px; border-radius: 6px; }

.prob-pct {
  width: 44px;
  text-align: right;
  font-size: 0.8rem;
  color: #9fb7ff;
}

.info-strip {
  display: flex;
  flex-wrap: wrap;
  gap: 0.7rem;
  margin-bottom: 0.5rem;
}

.info-chip {
  background: rgba(255,255,255,0.1);
  color: #9fb7ff;
  border-radius: 20px;
  padding: 5px 14px;
  font-size: 0.78rem;
  font-weight: 500;
}

.disclaimer {
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.15);
  border-radius: 10px;
  padding: 0.85rem 1.2rem;
  font-size: 0.82rem;
  color: #9fb7ff;
  line-height: 1.6;
}

img {
  display: block;
  max-width: 100%;
  border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# SECTION 1 — CONSTANTS & LABELS
# ═══════════════════════════════════════════════════════════════════

IMG_SIZE = (224, 224)

BINARY_CLASSES = ["CN (Cognitively Normal)", "AD (Alzheimer's Disease)"]
MULTI_CLASSES  = ["CN (Cognitively Normal)", "MCI (Mild Cognitive Impairment)", "AD (Alzheimer's Disease)"]
BINARY_SHORT   = ["CN", "AD"]
MULTI_SHORT    = ["CN", "MCI", "AD"]

BAR_COLORS = {
    "CN":  "#2ecc71",
    "MCI": "#f0a500",
    "AD":  "#e85d5d",
}


# ═══════════════════════════════════════════════════════════════════
# SECTION 2 — CORE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Convert PIL image to grayscale, resize to 224x224,
    normalize to [0,1], return as (1, 1, 224, 224) tensor.
    """
    img = image.convert("L")                                        # grayscale
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)       # normalize
    arr = np.expand_dims(arr, axis=0)                               # (1, 224, 224)
    arr = np.expand_dims(arr, axis=0)                               # (1, 1, 224, 224)
    return torch.tensor(arr, dtype=torch.float32)


@st.cache_resource(show_spinner=False)
def load_models():
    """
    Load binary and multi-class MONAI ResNet18 models from disk.
    Returns (binary_model, multi_model, loaded_flag).
    Falls back to demo mode if weight files are missing.
    """
    device = torch.device("cpu")
    try:
        binary_model = resnet18(spatial_dims=2, n_input_channels=1, num_classes=2)
        binary_model.load_state_dict(
            torch.load("binary_resnet_monai.pth", map_location=device)
        )
        binary_model.eval()

        multi_model = resnet18(spatial_dims=2, n_input_channels=1, num_classes=3)
        multi_model.load_state_dict(
            torch.load("best_multiclass_model.pth", map_location=device)
        )
        multi_model.eval()

        return binary_model, multi_model, True

    except Exception:
        return None, None, False


def predict_binary(model, image_tensor: torch.Tensor) -> dict:
    """Binary classification: CN vs AD."""
    with torch.no_grad():
        outputs = model(image_tensor)
        probs   = torch.softmax(outputs, dim=1).numpy()[0]

    idx = int(np.argmax(probs))
    return {
        "class":      BINARY_CLASSES[idx],
        "short":      BINARY_SHORT[idx],
        "confidence": float(probs[idx]) * 100,
        "probs":      probs,
        "labels":     BINARY_SHORT,
    }


def predict_multiclass(model, image_tensor: torch.Tensor) -> dict:
    """Multi-class classification: CN vs MCI vs AD."""
    with torch.no_grad():
        outputs = model(image_tensor)
        probs   = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    idx = int(np.argmax(probs))
    return {
        "class":      MULTI_CLASSES[idx],
        "short":      MULTI_SHORT[idx],
        "confidence": float(probs[idx]) * 100,
        "probs":      probs,
        "labels":     MULTI_SHORT,
    }


def _demo_predict(mode: str) -> dict:
    """Synthetic predictions for demo mode when model files are absent."""
    rng = np.random.default_rng(seed=int(time.time()) % 100)
    if mode == "binary":
        probs = rng.dirichlet([4, 1])
        idx   = int(np.argmax(probs))
        return {
            "class":      BINARY_CLASSES[idx],
            "short":      BINARY_SHORT[idx],
            "confidence": float(probs[idx]) * 100,
            "probs":      probs,
            "labels":     BINARY_SHORT,
        }
    else:
        probs = rng.dirichlet([5, 2, 1])
        idx   = int(np.argmax(probs))
        return {
            "class":      MULTI_CLASSES[idx],
            "short":      MULTI_SHORT[idx],
            "confidence": float(probs[idx]) * 100,
            "probs":      probs,
            "labels":     MULTI_SHORT,
        }


# ═══════════════════════════════════════════════════════════════════
# SECTION 3 — UI RENDERING HELPERS
# ═══════════════════════════════════════════════════════════════════

def render_probability_bars(result: dict):
    """Render color-coded HTML probability bars for each class."""
    html = ""
    for lbl, p in zip(result["labels"], result["probs"]):
        pct   = float(p) * 100
        color = BAR_COLORS.get(lbl, "#4facfe")
        html += f"""
        <div class="prob-row">
          <span class="prob-label">{lbl}</span>
          <div class="prob-bar-wrap">
            <div class="prob-bar-fill" style="width:{pct:.1f}%; background:{color};"></div>
          </div>
          <span class="prob-pct">{pct:.1f}%</span>
        </div>"""
    st.markdown(html, unsafe_allow_html=True)


def render_result_card(result: dict):
    """Render the main prediction result card with colored accent."""
    conf   = result["confidence"]
    short  = result["short"]
    label  = result["class"]
    accent = {"AD": "#e85d5d", "MCI": "#f0a500", "CN": "#2ecc71"}.get(short, "#4facfe")

    st.markdown(f"""
    <div class="ns-card" style="border-left: 4px solid {accent};">
      <div class="ns-card-title">🔬 Prediction Result</div>
      <div class="result-label">{short}</div>
      <div class="result-sub">{label}</div>
      <div class="conf-track">
        <div class="conf-fill" style="width:{conf:.1f}%; background:{accent};"></div>
      </div>
      <div style="font-size:0.82rem; color:#9fb7ff; margin-top:4px;">
        Confidence: <strong style="color:#ffffff;">{conf:.1f}%</strong>
      </div>
    </div>
    """, unsafe_allow_html=True)

def plot_probabilities(class_names, probs):
    fig, ax = plt.subplots()

    ax.bar(class_names, probs)
    ax.set_ylabel("Probability")
    ax.set_title("Class Prediction Confidence")

    for i, v in enumerate(probs):
        ax.text(i, v + 0.01, f"{v*100:.1f}%", ha='center')

    return fig
# ═══════════════════════════════════════════════════════════════════
# SECTION 4 — MAIN APP
# ═══════════════════════════════════════════════════════════════════

def main():

    # ── Header ────────────────────────────────────────────────────
    st.markdown("""
    <div class="ns-header">
      <div class="ns-badge">🧠 Clinical AI Research Tool</div>
      <h1>NeuroScan AI</h1>
      <p>Upload a brain MRI scan to receive an AI-powered screening assessment for cognitive decline patterns.
         The system uses deep learning to classify MRI scans across cognitive categories.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)

    # ── Load models (cached) ──────────────────────────────────────
    with st.spinner("Loading models..."):
        binary_model, multiclass_model, models_loaded = load_models()

    if not models_loaded:
        st.info(
            "⚠️ **Demo Mode:** Model files not found (`binary_resnet_monai.pth` / "
            "`best_multiclass_model.pth`). Running with synthetic predictions. "
            "Place your trained `.pth` files in the same directory to enable real inference."
        )

    # ── Info chips ───────────────────────────────────────────────
    st.markdown("""
    <div class="info-strip">
      <span class="info-chip">📐 Input: 224 × 224 px</span>
      <span class="info-chip">🔢 Binary &amp; Multi-class</span>
      <span class="info-chip">⚡ PyTorch · MONAI ResNet18</span>
      <span class="info-chip">🩶 Grayscale MRI</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)

    # ── Two-column layout ─────────────────────────────────────────
    left_col, right_col = st.columns([1, 1.55], gap="large")

    # ════════════════════
    # LEFT COLUMN
    # ════════════════════
    with left_col:

        # Upload card
        st.markdown('<div class="ns-card">', unsafe_allow_html=True)
        st.markdown('<div class="ns-card-title">📂 Upload MRI Scan</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Upload MRI",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )
        st.caption("Accepted formats: JPG, PNG")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # Image preview — only shown after upload
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.markdown('<div class="ns-card">', unsafe_allow_html=True)
            st.markdown('<div class="ns-card-title">🖼 Image Preview</div>', unsafe_allow_html=True)
            st.image(image, use_container_width=True, caption=uploaded.name)
            w, h = image.size
            st.caption(f"Original: {w} × {h} px  →  model input: 224 × 224")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # Model selection + run button
        st.markdown('<div class="ns-card">', unsafe_allow_html=True)
        st.markdown('<div class="ns-card-title">⚙️ Classification Mode</div>', unsafe_allow_html=True)
        mode = st.radio(
            "",
            ["Binary (CN vs AD)", "Multi-class (CN vs MCI vs AD)"],
        )
        analyze_btn = st.button(
            "🔍 Run Screening",
            use_container_width=True,
            disabled=(uploaded is None),
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ════════════════════
    # RIGHT COLUMN
    # ════════════════════
    with right_col:

        # Empty state
        if uploaded is None:
            st.markdown("""
            <div class="ns-card" style="min-height:320px;display:flex;flex-direction:column;
                 align-items:center;justify-content:center;text-align:center;gap:0.8rem;">
              <div style="font-size:3rem;">🧠</div>
              <div style="color:#cfd8ff;font-weight:600;font-size:1.05rem;">Ready for Analysis</div>
              <div style="color:#9fb7ff;font-size:0.88rem;max-width:280px;line-height:1.6;">
                Upload a brain MRI scan on the left to begin AI-powered cognitive screening.
              </div>
            </div>
            """, unsafe_allow_html=True)

        # Uploaded but not yet run
        elif not analyze_btn:
            st.markdown("""
            <div class="ns-card" style="min-height:200px;display:flex;flex-direction:column;
                 align-items:center;justify-content:center;text-align:center;gap:0.6rem;">
              <div style="font-size:2rem;">⬅️</div>
              <div style="color:#9fb7ff;font-size:0.92rem;">
                Image uploaded. Select a mode and click
                <strong style="color:#cfd8ff;">Run Screening</strong>.
              </div>
            </div>
            """, unsafe_allow_html=True)

        # Results
        else:
            with st.spinner("Analyzing scan..."):
                image_tensor = preprocess_image(image)
                is_binary    = "Binary" in mode

                if models_loaded:
                    if is_binary:
                        result = predict_binary(binary_model, image_tensor)
                    else:
                        result = predict_multiclass(multiclass_model, image_tensor)
                else:
                    result = _demo_predict("binary" if is_binary else "multi")

            # Prediction result card
            render_result_card(result)

            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

            # Class probability bars
            st.markdown('<div class="ns-card">', unsafe_allow_html=True)
            st.markdown('<div class="ns-card-title">📊 Class Probabilities</div>', unsafe_allow_html=True)
            render_probability_bars(result)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

            # Disclaimer
            st.markdown("""
            <div class="disclaimer">
              ⚕️ <strong>Medical Disclaimer:</strong> This tool is for screening purposes only and is
              <em>not</em> a clinical diagnosis. Results must be reviewed by a qualified healthcare professional.
              Do not make medical decisions based solely on these outputs.
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()