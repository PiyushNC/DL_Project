import json
import logging
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CONFIG
from src.disease_info import get_disease_information
from src.predict import PredictionService
from src.preprocessing import load_validated_image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

st.set_page_config(
    page_title="LeafLens",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    html, body, [class*="css"] { font-family: "Inter", "Segoe UI", sans-serif; }
    .stApp { background: #0B1712; color: #EDF5EC; }
    [data-testid="stHeader"] {
        background: rgba(11,23,18,0.92);
    }
    h1, h2, h3, h4, h5, h6, p, label, .stCaption {
        color: #EDF5EC !important;
    }
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li {
        color: #C8D7C9;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #173F2B 0%, #24543A 100%);
    }
    [data-testid="stSidebar"] * { color: #F7F8F2 !important; }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        border-radius: 12px;
        padding: 7px 10px;
        margin: 3px 0;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
        background: rgba(255,255,255,0.13);
    }
    .hero {
        background: linear-gradient(135deg, #173F2B 0%, #1D4A38 58%, #263C2E 100%);
        border: 1px solid #3D5B49;
        border-radius: 24px;
        padding: 42px;
        box-shadow: 0 12px 32px rgba(37,84,58,0.08);
    }
    .hero h1 { color: #F2F8F0; font-size: 3rem; margin-bottom: 8px; }
    .hero h3 { color: #C6DBB9 !important; }
    .hero p { color: #D0DDD2; font-size: 1.12rem; max-width: 680px; }
    .brand { font-size: 1.25rem; font-weight: 800; letter-spacing: 0.02em; }
    .card {
        background: #13251D;
        border: 1px solid #30483A;
        border-radius: 18px;
        padding: 22px;
        box-shadow: 0 8px 24px rgba(37,84,58,0.06);
        height: 100%;
    }
    .card h3 { color: #EDF5EC; margin-bottom: 8px; }
    .metric-card {
        background: #13251D;
        border: 1px solid #30483A;
        border-radius: 16px;
        padding: 18px;
        text-align: center;
    }
    .status-good {
        background: #173F2B; color: #BFE0AA; border-radius: 12px;
        padding: 10px 14px; font-weight: 700;
    }
    .status-alert {
        background: #4A3522; color: #FFD59A; border-radius: 12px;
        padding: 10px 14px; font-weight: 700;
    }
    .disclaimer {
        background: #14271F; border-left: 4px solid #78A85F;
        border-radius: 8px; padding: 12px 16px; color: #CBD9CC;
    }
    .stButton > button {
        border-radius: 12px; background: #2F6650; color: white;
        border: none; padding: 0.65rem 1.1rem; font-weight: 700;
    }
    .stButton > button:hover { background: #24543A; color: white; }
    </style>
    """,
    unsafe_allow_html=True,
)


if "page" not in st.session_state:
    st.session_state.page = "Home"
if "uploaded_version" not in st.session_state:
    st.session_state.uploaded_version = 0
if "prediction" not in st.session_state:
    st.session_state.prediction = None


@st.cache_resource
def load_prediction_service() -> PredictionService:
    return PredictionService(
        model_path=CONFIG.model_path,
        class_names_path=CONFIG.class_names_path,
        image_size=CONFIG.image_size,
    )


def model_available() -> bool:
    return CONFIG.model_path.exists() and CONFIG.class_names_path.exists()


def model_status_badge() -> None:
    if model_available():
        st.success("Trained model available")
    else:
        st.warning("Trained model not available")


def page_home() -> None:
    st.markdown(
        """
        <div class="hero">
            <div style="font-size:2.2rem;">🌿</div>
            <h1>LeafLens</h1>
            <h3 style="color:#466653;">AI-Powered Plant Disease Classification</h3>
            <p>Explore leaf health with a lightweight MobileNetV2 model trained for
            PlantVillage-style plant disease image classification.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="card"><h3>1 · Upload</h3><p>Select a clear image of a plant leaf.</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card"><h3>2 · Analyze</h3><p>MobileNetV2 extracts features and ranks disease classes.</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="card"><h3>3 · Review</h3><p>Inspect confidence, alternatives, and general educational guidance.</p></div>', unsafe_allow_html=True)

    st.write("")
    left, right = st.columns([2, 1])
    with left:
        if st.button("Start Image Analysis", use_container_width=True):
            st.session_state.page = "Disease Detection"
            st.rerun()
    with right:
        model_status_badge()


def page_detection() -> None:
    st.title("Disease Detection")
    st.caption("Upload a clear, well-lit leaf image for model-based analysis.")
    model_status_badge()

    uploader_key = f"leaf_uploader_{st.session_state.uploaded_version}"
    uploaded_file = st.file_uploader(
        "Upload a leaf image",
        type=["jpg", "jpeg", "png"],
        key=uploader_key,
    )

    if uploaded_file is None:
        st.info("Choose a JPG or PNG image to begin.")
        st.session_state.prediction = None
        return

    try:
        image = load_validated_image(uploaded_file)
    except ValueError as exc:
        st.error(str(exc))
        return

    preview, action = st.columns([1.15, 1])
    with preview:
        st.image(image, caption="Uploaded leaf", use_container_width=True)
    with action:
        st.write("**Analysis controls**")
        analyze = st.button("Analyze Image", use_container_width=True)
        if st.button("Upload Another Image", use_container_width=True):
            st.session_state.uploaded_version += 1
            st.session_state.prediction = None
            st.rerun()

    if analyze:
        if not model_available():
            st.error(
                "The trained model or class mapping is missing. "
                "Run `python -m src.train --dataset data/raw` before analysis."
            )
            st.session_state.prediction = None
            return

        try:
            with st.spinner("Analyzing leaf features…"):
                service = load_prediction_service()
                label, confidence, top_three = service.predict(image, top_k=3)
            st.session_state.prediction = {
                "label": label,
                "confidence": confidence,
                "top_three": top_three,
            }
        except Exception as exc:
            st.error(f"Prediction could not be completed: {exc}")
            st.session_state.prediction = None
            return

    prediction = st.session_state.prediction
    if prediction is None:
        return

    info = get_disease_information(prediction["label"])
    st.divider()
    result_left, result_right = st.columns([1, 1])
    with result_left:
        st.subheader("Predicted Species")
        st.markdown(f"### {info.species}")
        st.subheader("Disease / Condition")
        st.markdown(f"### {info.disease}")
        status_class = "status-good" if info.status == "Healthy" else "status-alert"
        st.markdown(
            f'<div class="{status_class}">Status: {info.status}</div>',
            unsafe_allow_html=True,
        )
    with result_right:
        st.subheader("Model Confidence")
        st.metric(
            "Top probability",
            f"{prediction['confidence'] * 100:.2f}%",
        )
        st.caption(
            "Confidence is a model score, not scientific or medical certainty."
        )

    st.subheader("Top Three Predictions")
    probabilities = pd.DataFrame(
        prediction["top_three"],
        columns=["Plant Class", "Probability"],
    )
    probabilities["Probability"] = probabilities["Probability"].astype(float)
    probabilities["Probability"] = probabilities["Probability"].map(
        lambda value: f"{value:.2%}"
    )
    st.dataframe(
        probabilities,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Disease Information")
    info_left, info_right = st.columns(2)
    with info_left:
        st.markdown('<div class="card"><h3>Possible symptoms</h3>', unsafe_allow_html=True)
        st.write(info.symptoms)
        st.markdown("</div>", unsafe_allow_html=True)
    with info_right:
        st.markdown('<div class="card"><h3>General prevention</h3>', unsafe_allow_html=True)
        st.write(info.prevention)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown(
        f'<div class="disclaimer"><strong>Educational notice:</strong> {info.note}</div>',
        unsafe_allow_html=True,
    )


def load_evaluation_metrics() -> dict | None:
    path = CONFIG.metrics_dir / "test_metrics.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        LOGGER.warning("Could not read evaluation metrics: %s", exc)
        return None


def page_performance() -> None:
    st.title("Model Performance")
    st.caption("Results loaded only from saved artifacts generated by evaluation.")

    metrics = load_evaluation_metrics()
    if metrics is None:
        st.warning(
            "No evaluation results are available. Train the model and run "
            "`python -m src.evaluate --dataset data/raw`."
        )

    if metrics:
        cards = [
            ("Test Accuracy", f"{metrics.get('test_accuracy', 0) * 100:.2f}%"),
            ("Macro Precision", f"{metrics.get('macro_precision', 0) * 100:.2f}%"),
            ("Macro Recall", f"{metrics.get('macro_recall', 0) * 100:.2f}%"),
            ("Macro F1-score", f"{metrics.get('macro_f1', 0) * 100:.2f}%"),
        ]
        columns = st.columns(4)
        for column, (title, value) in zip(columns, cards):
            with column:
                st.markdown(
                    f'<div class="metric-card"><small>{title}</small><h2>{value}</h2></div>',
                    unsafe_allow_html=True,
                )

        st.write("")
        detail_left, detail_right = st.columns(2)
        with detail_left:
            st.metric("Parameters", f"{metrics.get('model_parameters', 0):,}")
        with detail_right:
            st.metric(
                "Model File Size",
                f"{metrics.get('model_file_size_mb', 0):.3f} MB",
            )

    st.subheader("Confusion Matrix")
    confusion_path = CONFIG.plots_dir / "confusion_matrix.png"
    if confusion_path.exists():
        st.image(str(confusion_path), use_container_width=True)
    else:
        st.info("No saved confusion matrix is available.")

    st.subheader("Training and Validation Curves")
    training_path = CONFIG.plots_dir / "training_history.png"
    if training_path.exists():
        st.image(str(training_path), use_container_width=True)
    else:
        st.info("No training history graph is available.")

    if metrics:
        with st.expander("View per-class classification report"):
            report = metrics.get("classification_report", {})
            rows = []
            for class_name, values in report.items():
                if isinstance(values, dict):
                    rows.append({
                        "Class": class_name,
                        "Precision": values.get("precision"),
                        "Recall": values.get("recall"),
                        "F1-score": values.get("f1-score"),
                        "Support": values.get("support"),
                    })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def page_about() -> None:
    st.title("About LeafLens")

    left, right = st.columns(2)
    with left:
        st.markdown(
            """
            ### Problem
            Plant diseases can reduce crop quality and yield. Early visual screening
            can help users prioritize closer inspection and expert support.

            ### Solution
            LeafLens uses a pretrained MobileNetV2 model with a custom plant-disease
            classification head. It accepts a leaf image and ranks the possible
            PlantVillage classes.

            ### Reproducibility
            The project uses fixed random seeds where supported, a saved split
            manifest, fixed preprocessing, and saved model/class artifacts.
            """
        )
    with right:
        st.markdown(
            """
            ### Transfer Learning
            ImageNet-pretrained convolutional features are reused rather than
            training all parameters from scratch. Initially, the MobileNetV2 base
            is frozen while a lightweight head learns disease classes.

            ### MobileNetV2
            MobileNetV2 is suitable for a college laboratory because it combines
            depthwise separable convolutions with an efficient lightweight
            architecture and supports optional upper-layer fine-tuning.

            ### PlantVillage
            PlantVillage contains leaf images organized into plant/disease folders.
            The application discovers classes from the dataset and preserves their
            mapping for training and inference.
            """
        )

    st.subheader("Architecture")
    st.code(
        "Leaf image → Resize 224×224 → MobileNetV2 preprocessing\n"
        "→ MobileNetV2 feature base → Global Average Pooling\n"
        "→ Dropout → Softmax disease classes",
        language="text",
    )

    st.subheader("Technologies")
    st.write("Python · TensorFlow/Keras · MobileNetV2 · Streamlit · NumPy · pandas · scikit-learn")
    st.markdown(
        '<div class="disclaimer"><strong>Scope:</strong> LeafLens is an educational '
        'machine-learning demonstration, not a replacement for agricultural, botanical, '
        'or professional diagnosis.</div>',
        unsafe_allow_html=True,
    )


with st.sidebar:
    st.markdown('<div class="brand">🌿 LeafLens</div>', unsafe_allow_html=True)
    st.caption("AI-Powered Plant Disease Classification")
    st.write("")
    page = st.radio(
        "Navigation",
        ["Home", "Disease Detection", "Model Performance", "About Project"],
        index=["Home", "Disease Detection", "Model Performance", "About Project"].index(
            st.session_state.page
        ),
        label_visibility="collapsed",
    )
    st.session_state.page = page
    st.write("")
    st.caption("Model Status")
    model_status_badge()

if st.session_state.page == "Home":
    page_home()
elif st.session_state.page == "Disease Detection":
    page_detection()
elif st.session_state.page == "Model Performance":
    page_performance()
else:
    page_about()
