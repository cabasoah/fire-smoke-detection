from pathlib import Path
import tempfile

import numpy as np
import cv2
import streamlit as st
import tensorflow as tf
from PIL import Image


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model" / "fire_smoke4.keras"
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


def predict_from_array(
    model: tf.keras.Model,
    x: np.ndarray,
    class_names: list[str],
) -> tuple[str, float, dict[str, float]]:
    pred = model.predict(x, verbose=0)

    if pred.shape[-1] == 1:
        smoke_prob = float(pred[0][0])
        fire_prob = 1.0 - smoke_prob

        labels = class_names if len(class_names) >= 2 else ["fire", "smoke"]
        probs = [fire_prob, smoke_prob]
        idx = int(np.argmax(probs))
        return labels[idx], float(probs[idx]), {"fire": fire_prob, "smoke": smoke_prob}

    probs = tf.nn.softmax(pred[0]).numpy()
    idx = int(np.argmax(probs))
    label = class_names[idx] if idx < len(class_names) else str(idx)
    prob_map = {class_names[i] if i < len(class_names) else str(i): float(probs[i]) for i in range(len(probs))}
    return label, float(probs[idx]), prob_map


def predict_image(model: tf.keras.Model, image: Image.Image, class_names: list[str]) -> tuple[str, float, dict[str, float]]:
    x = preprocess_image(image)
    return predict_from_array(model, x, class_names)


def process_video_file(
    model: tf.keras.Model,
    video_bytes: bytes,
    class_names: list[str],
    sample_every: int = 10,
    max_frames: int = 60,
) -> dict[str, object]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(video_bytes)
        temp_path = tmp_file.name

    capture = cv2.VideoCapture(temp_path)
    sampled_frames: list[dict[str, object]] = []
    fire_scores: list[float] = []
    smoke_scores: list[float] = []

    frame_index = 0
    try:
        while capture.isOpened() and len(sampled_frames) < max_frames:
            success, frame = capture.read()
            if not success:
                break

            if frame_index % sample_every == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                label, confidence, probs = predict_image(model, image, class_names)

                fire_score = float(probs.get("fire", 0.0))
                smoke_score = float(probs.get("smoke", 0.0))
                fire_scores.append(fire_score)
                smoke_scores.append(smoke_score)
                sampled_frames.append(
                    {
                        "frame": frame_index,
                        "label": label,
                        "confidence": confidence,
                        "fire": fire_score,
                        "smoke": smoke_score,
                    }
                )

            frame_index += 1
    finally:
        capture.release()
        Path(temp_path).unlink(missing_ok=True)

    if fire_scores:
        avg_fire = float(np.mean(fire_scores))
        avg_smoke = float(np.mean(smoke_scores))
        aggregate_label = "fire" if avg_fire >= avg_smoke else "smoke"
        aggregate_confidence = max(avg_fire, avg_smoke)
    else:
        avg_fire = 0.0
        avg_smoke = 0.0
        aggregate_label = "unknown"
        aggregate_confidence = 0.0

    return {
        "frames": sampled_frames,
        "aggregate_label": aggregate_label,
        "aggregate_confidence": aggregate_confidence,
        "avg_fire": avg_fire,
        "avg_smoke": avg_smoke,
        "sampled_count": len(sampled_frames),
    }


def main() -> None:
    st.set_page_config(page_title="Fire/Smoke Classifier", page_icon="🔥", layout="centered")
    st.title("Fire vs Smoke Classifier")
    st.write("Upload an image or video and run inference with your saved model.")

    if not MODEL_PATH.exists():
        st.error(f"Model file not found at: {MODEL_PATH}")
        st.stop()

    class_names = get_class_names()
    model = load_model(MODEL_PATH)

    input_mode = st.radio("Input type", ["Image", "Video"], horizontal=True)

    if input_mode == "Image":
        uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        if uploaded is None:
            st.info("Upload a JPG/PNG image to get a prediction.")
            return

        image = Image.open(uploaded)
        st.image(image, caption="Uploaded image", use_container_width=True)

        if st.button("Predict image", type="primary"):
            label, confidence, probs = predict_image(model, image, class_names)
            st.success(f"Prediction: {label.capitalize()}")
            st.metric("Confidence", f"{confidence * 100:.2f}%")
            st.write({name: f"{score * 100:.2f}%" for name, score in probs.items()})
    else:
        uploaded_video = st.file_uploader("Choose a video", type=["mp4", "mov", "avi", "mkv"])
        if uploaded_video is None:
            st.info("Upload an MP4, MOV, AVI, or MKV file to analyze a video feed.")
            return

        st.video(uploaded_video)
        sample_every = st.slider("Analyze every Nth frame", min_value=1, max_value=30, value=10)
        max_frames = st.slider("Maximum sampled frames", min_value=10, max_value=200, value=60, step=10)

        if st.button("Analyze video", type="primary"):
            result = process_video_file(
                model=model,
                video_bytes=uploaded_video.getvalue(),
                class_names=class_names,
                sample_every=sample_every,
                max_frames=max_frames,
            )

            st.success(f"Video-level prediction: {str(result['aggregate_label']).capitalize()}")
            st.metric("Average confidence", f"{float(result['aggregate_confidence']) * 100:.2f}%")
            st.write(
                {
                    "Average fire score": f"{float(result['avg_fire']) * 100:.2f}%",
                    "Average smoke score": f"{float(result['avg_smoke']) * 100:.2f}%",
                    "Sampled frames": int(result['sampled_count']),
                }
            )

            frames = result["frames"]
            if frames:
                st.subheader("Sampled frame predictions")
                for frame_result in frames[:8]:
                    st.write(
                        f"Frame {frame_result['frame']}: {str(frame_result['label']).capitalize()} "
                        f"({float(frame_result['confidence']) * 100:.2f}%)"
                    )


if __name__ == "__main__":
    main()