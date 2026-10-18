from pathlib import Path
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageFilter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "sample_fingerprint_features.csv"
FEATURES = [
    "ridge_clarity", "pore_visibility", "texture_entropy",
    "ridge_continuity", "specular_reflection", "surface_uniformity"
]
LABEL = "label"

st.set_page_config(page_title="Fingerprint Anti-Spoofing Demo", page_icon="🔎", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def train_model():
    data = load_data()
    X = data[FEATURES]
    y = data[LABEL]
    model = RandomForestClassifier(n_estimators=180, max_depth=8, random_state=42, class_weight="balanced")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return model, X_test, y_test, preds

def image_features(uploaded_file):
    """Compute simple visual proxies from an uploaded grayscale image.
    These are educational image statistics, not validated fingerprint liveness features.
    """
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image).convert("L")
    image.thumbnail((700, 700))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    if arr.size == 0:
        raise ValueError("The uploaded image is empty.")
    # Gradient magnitude as a rough proxy for ridge-edge strength.
    gy, gx = np.gradient(arr)
    grad = np.sqrt(gx * gx + gy * gy)
    # Local contrast and histogram entropy.
    contrast = float(np.std(arr))
    hist, _ = np.histogram(arr, bins=32, range=(0, 1), density=True)
    p = hist / (hist.sum() + 1e-12)
    entropy = float(-(p * np.log2(p + 1e-12)).sum() / 5.0)
    # Image-level proxies normalized to a nominal 0..1 scale.
    ridge_clarity = float(np.clip(np.mean(grad) * 4.0, 0, 1))
    pore_visibility = float(np.clip(np.percentile(grad, 90) * 5.0, 0, 1))
    texture_entropy = float(np.clip(entropy, 0, 1))
    ridge_continuity = float(np.clip(1.0 - np.mean(np.abs(np.diff(arr, axis=1))) * 3.0, 0, 1))
    # Bright, low-texture pixels can mimic specular regions; this is not a physical measurement.
    specular_reflection = float(np.clip(np.mean(arr > 0.92) * 4.0, 0, 1))
    surface_uniformity = float(np.clip(1.0 - contrast * 2.0, 0, 1))
    return {
        "ridge_clarity": ridge_clarity,
        "pore_visibility": pore_visibility,
        "texture_entropy": texture_entropy,
        "ridge_continuity": ridge_continuity,
        "specular_reflection": specular_reflection,
        "surface_uniformity": surface_uniformity,
    }, image

def make_gauge(value):
    return f"{value:.2f}"

st.title("🔎 Intelligent Fingerprint Anti-Spoofing System")
st.caption("College mini-project prototype • Genuine vs. silicone-spoof classification")
st.warning(
    "Research/demo limitation: the bundled dataset is synthetically generated. "
    "This application demonstrates a machine-learning workflow; it is not a validated biometric security system "
    "and must not be used to grant or deny real-world access."
)

with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to", ["Overview", "Analyze sample features", "Analyze an image", "Dataset & model", "Project documentation"])
    st.divider()
    st.caption("Model: Random Forest • Dataset: synthetic feature records")

model, X_test, y_test, preds = train_model()
data = load_data()
test_acc = accuracy_score(y_test, preds)

if page == "Overview":
    st.subheader("Project overview")
    st.write(
        "The prototype illustrates how a classifier could use fingerprint-image texture indicators to distinguish "
        "a genuine finger from a presentation attack made with a silicone replica. In a real system, features and "
        "training data must come from a carefully collected, labeled fingerprint presentation-attack dataset."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Sample records", f"{len(data):,}")
    c2.metric("Feature indicators", len(FEATURES))
    c3.metric("Held-out demo accuracy", f"{test_acc:.1%}")
    st.caption("Accuracy is measured on a held-out portion of synthetic data only; it does not estimate real-world performance.")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Workflow")
        st.markdown("1. Collect or upload a fingerprint sample\n2. Extract/enter visual feature indicators\n3. Run the Random Forest classifier\n4. Display the predicted class and feature values")
    with right:
        st.markdown("#### Classes")
        st.markdown("- **Genuine:** sample labeled as a real finger in the training data.\n- **Spoof:** sample labeled as a silicone replica in the training data.")
    st.info("Use the feature-input page for a reproducible demonstration. The image page extracts rough image statistics and passes them to the demo model; this is illustrative only.")

elif page == "Analyze sample features":
    st.subheader("Analyze fingerprint feature indicators")
    st.write("Adjust the example values to see how the model responds. Values range from 0.00 to 1.00.")
    defaults = {
        "ridge_clarity": 0.72, "pore_visibility": 0.68, "texture_entropy": 0.63,
        "ridge_continuity": 0.70, "specular_reflection": 0.18, "surface_uniformity": 0.42
    }
    spoof_defaults = {
        "ridge_clarity": 0.48, "pore_visibility": 0.31, "texture_entropy": 0.42,
        "ridge_continuity": 0.80, "specular_reflection": 0.60, "surface_uniformity": 0.76
    }
    preset = st.selectbox("Choose a starting example", ["Genuine-like example", "Silicone-spoof-like example", "Custom"])
    values = defaults if preset == "Genuine-like example" else spoof_defaults if preset == "Silicone-spoof-like example" else defaults
    entered = {}
    col1, col2 = st.columns(2)
    for i, feature in enumerate(FEATURES):
        target = col1 if i < 3 else col2
        with target:
            entered[feature] = st.slider(feature.replace("_", " ").title(), 0.0, 1.0, float(values[feature]), 0.01)
    if st.button("Classify sample", type="primary", use_container_width=True):
        row = pd.DataFrame([entered], columns=FEATURES)
        pred = model.predict(row)[0]
        proba = model.predict_proba(row)[0]
        classes = list(model.classes_)
        score = float(proba[classes.index(pred)])
        if pred == "Genuine":
            st.success(f"Model prediction: {pred}")
        else:
            st.error(f"Model prediction: {pred} (possible spoof)")
        st.metric("Model confidence on this synthetic-feature model", f"{score:.1%}")
        st.dataframe(pd.DataFrame({"Feature": list(entered.keys()), "Value": list(entered.values())}), use_container_width=True, hide_index=True)
        st.caption("The confidence is a model score, not a calibrated probability of biometric authenticity.")

elif page == "Analyze an image":
    st.subheader("Analyze an uploaded fingerprint image")
    st.write("Upload a grayscale or color fingerprint image. The app computes simple image statistics as feature proxies.")
    st.caption("For privacy, use only images you have permission to use. Avoid uploading identifiable biometric data.")
    uploaded = st.file_uploader("Choose a PNG, JPG, or JPEG image", type=["png", "jpg", "jpeg"])
    if uploaded:
        try:
            features, image = image_features(uploaded)
            preview_col, result_col = st.columns([1, 1])
            with preview_col:
                st.image(image, caption="Uploaded image (grayscale preview)", use_container_width=True)
            with result_col:
                st.markdown("#### Extracted image statistics")
                feature_df = pd.DataFrame({"Feature proxy": list(features.keys()), "Value": list(features.values())})
                st.dataframe(feature_df, hide_index=True, use_container_width=True)
                if st.button("Run demo classifier", type="primary"):
                    row = pd.DataFrame([features], columns=FEATURES)
                    pred = model.predict(row)[0]
                    score = float(np.max(model.predict_proba(row)[0]))
                    if pred == "Genuine":
                        st.success(f"Demo model prediction: {pred}")
                    else:
                        st.error(f"Demo model prediction: {pred}")
                    st.metric("Model score", f"{score:.1%}")
                    st.warning("This result is not a reliable determination of whether the image is genuine or a silicone replica. The model was trained on synthetic tabular features, not real fingerprint images.")
        except Exception as exc:
            st.error(f"Could not process this image: {exc}")

elif page == "Dataset & model":
    st.subheader("Dataset preview")
    st.write("The included CSV is synthetic and intended for demonstrating data preparation, model training, and evaluation.")
    st.dataframe(data.head(20), use_container_width=True, hide_index=True)
    st.download_button("Download sample dataset CSV", data=data.to_csv(index=False).encode("utf-8"), file_name="sample_fingerprint_features.csv", mime="text/csv")
    st.markdown("#### Class distribution")
    counts = data[LABEL].value_counts().rename_axis("Class").reset_index(name="Records")
    st.bar_chart(counts.set_index("Class"))
    st.markdown("#### Feature importance")
    importance = pd.DataFrame({"Feature": FEATURES, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)
    st.bar_chart(importance.set_index("Feature"))
    st.markdown("#### Held-out test metrics (synthetic data)")
    st.metric("Accuracy", f"{test_acc:.1%}")
    cm = confusion_matrix(y_test, preds, labels=["Genuine", "Spoof"])
    cm_df = pd.DataFrame(cm, index=["Actual Genuine", "Actual Spoof"], columns=["Predicted Genuine", "Predicted Spoof"])
    st.dataframe(cm_df, use_container_width=True)
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)
    st.caption("Do not cite these synthetic-data metrics as evidence of real biometric performance.")

elif page == "Project documentation":
    st.subheader("Project documentation")
    st.markdown("#### Abstract")
    st.write("Fingerprint authentication is widely used, but a system may be attacked using an artificial replica. This mini project demonstrates an intelligent anti-spoofing workflow using a Random Forest classifier and six illustrative texture-related feature indicators. A Streamlit interface lets users inspect the sample dataset, enter feature values, upload an image for basic image-statistic extraction, and view a predicted class. The bundled records are synthetic, so the application is an educational prototype rather than a validated fingerprint liveness detector.")
    st.markdown("#### Problem statement")
    st.write("A conventional fingerprint matcher may accept a replica if it only compares ridge patterns. The project demonstrates the idea of using additional texture indicators to flag a possible presentation attack.")
    st.markdown("#### Objectives")
    st.markdown("- Demonstrate a machine-learning classification pipeline.\n- Provide a simple interactive Streamlit interface.\n- Show feature inputs, predictions, dataset exploration, and basic evaluation.\n- Document the limitations of synthetic training data.")
    st.markdown("#### System architecture")
    st.code("Sample CSV / Uploaded Image\\n          ↓\\nData loading / image-statistic extraction\\n          ↓\\nFeature vector (6 indicators)\\n          ↓\\nRandom Forest classifier\\n          ↓\\nPrediction + score + evaluation display", language="text")
    st.markdown("#### Modules")
    st.markdown("1. **Dataset module:** loads the sample CSV.\n2. **Model module:** trains a Random Forest and evaluates held-out synthetic records.\n3. **Feature input module:** accepts manual feature values.\n4. **Image analysis module:** calculates basic grayscale statistics from an uploaded image.\n5. **Dashboard module:** displays predictions, charts, and metrics.")
    st.markdown("#### Technologies")
    st.markdown("- Python\n- Streamlit\n- pandas and NumPy\n- scikit-learn\n- Pillow\n- Matplotlib")
    st.markdown("#### Limitations and future scope")
    st.markdown("- Synthetic records do not represent real fingerprints.\n- Image statistics are not validated liveness features.\n- Future work should use a lawful, labeled public fingerprint presentation-attack dataset, image-level train/test separation, stronger ridge/pore analysis, and evaluation with APCER/BPCER or other domain-appropriate metrics.")
    st.markdown("#### Ethical and privacy note")
    st.write("Fingerprint images are sensitive biometric data. Use consented samples, minimize storage, and do not use this prototype for identity verification or access-control decisions.")

st.divider()
st.caption("Educational prototype • Not for production biometric security")
