from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model" / "fire_smoke.keras"
TRAIN_DIR = APP_DIR / "train"
IMG_SIZE = (224, 224)


def get_class_names() -> list[str]:
    if TRAIN_DIR.exists():
        classes = sorted([p.name for p in TRAIN_DIR.iterdir() if p.is_dir()])
        if classes:
            return classes
    return ["fire", "smoke"]


@st.cache_resource
def load_model(model_path: Path) -> tf.keras.Model:
    return tf.keras.models.load_model(model_path)


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    array = np.array(image, dtype=np.float32)
    return np.expand_dims(array, axis=0)


def predict_image(model: tf.keras.Model, image: Image.Image, class_names: list[str]) -> tuple[str, float]:
    x = preprocess_image(image)
    pred = model.predict(x, verbose=0)

    if pred.shape[-1] == 1:
        smoke_prob = float(pred[0][0])
        fire_prob = 1.0 - smoke_prob

        labels = class_names if len(class_names) >= 2 else ["fire", "smoke"]
        probs = [fire_prob, smoke_prob]
        idx = int(np.argmax(probs))
        return labels[idx], float(probs[idx])

    probs = tf.nn.softmax(pred[0]).numpy()
    idx = int(np.argmax(probs))
    label = class_names[idx] if idx < len(class_names) else str(idx)
    return label, float(probs[idx])


def main() -> None:
    st.set_page_config(page_title="Fire/Smoke Classifier", page_icon="🔥", layout="centered")
    st.title("Fire vs Smoke Classifier")
    st.write("Upload an image and run inference with your saved model.")

    if not MODEL_PATH.exists():
        st.error(f"Model file not found at: {MODEL_PATH}")
        st.stop()

    class_names = get_class_names()
    model = load_model(MODEL_PATH)

    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("Upload a JPG/PNG image to get a prediction.")
        return

    image = Image.open(uploaded)
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Predict", type="primary"):
        label, confidence = predict_image(model, image, class_names)
        st.success(f"Prediction: {label.capitalize()}")
        st.metric("Confidence", f"{confidence * 100:.2f}%")


if __name__ == "__main__":
    main()