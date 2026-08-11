from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image
import numpy as np
from pathlib import Path
import uuid
import os

# ============================================================
# Smart Waste Classification - Flask + Keras
# ============================================================

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "smart_waste_model.keras"
UPLOAD_FOLDER = BASE_DIR / "uploads"

IMG_SIZE = 224
NUM_CLASSES = 3

CLASS_NAMES = ["glass", "metal", "paper"]

RECYCLABLE = {
    "glass": "Recyclable",
    "metal": "Recyclable",
    "paper": "Recyclable",
}

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

UPLOAD_FOLDER.mkdir(exist_ok=True)

# ------------------------------------------------------------
# Model architecture
# IMPORTANT:
# This matches the architecture shown in your Colab summary:
# EfficientNetB0 -> GAP -> BatchNormalization -> Dropout -> Dense(3)
# ------------------------------------------------------------

def build_model():
    data_augmentation = models.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.10),
            layers.RandomContrast(0.10),
        ],
        name="data_augmentation",
    )

    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights=None,
        name="efficientnetb0",
    )

    base_model.trainable = False

    model = models.Sequential(
        [
            layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_layer"),
            data_augmentation,
            base_model,
            layers.GlobalAveragePooling2D(name="global_average_pooling2d"),
            layers.BatchNormalization(name="batch_normalization"),
            layers.Dropout(0.3, name="dropout"),
            layers.Dense(NUM_CLASSES, activation="softmax", name="dense"),
        ],
        name="smart_waste_classifier",
    )

    return model


print("=" * 60)
print("Loading Smart Waste Classification model...")
print(f"TensorFlow: {tf.__version__}")
print(f"Keras: {tf.keras.__version__}")
print(f"Weights: {MODEL_PATH}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "\nModel weights not found.\n"
        f"Copy your Colab file:\n"
        "  /content/smart_waste_weights.weights.h5\n"
        "to:\n"
        f"  {MODEL_PATH}\n"
    )

model = build_model()

try:
    model.load_weights(str(MODEL_PATH))
except Exception as exc:
    raise RuntimeError(
        "\nCould not load the weights file.\n"
        "The architecture in app.py must match the architecture used during training.\n"
        f"Original error: {exc}"
    ) from exc

print("Model weights loaded successfully!")
print("=" * 60)


def allowed_file(filename):
    return (
        bool(filename)
        and "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def predict_waste(img_path):
    image = Image.open(img_path).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))

    # EfficientNetB0 from tf.keras includes its own preprocessing.
    # Therefore DO NOT divide the image by 255 here.
    image_array = np.asarray(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)[0]

    predicted_index = int(np.argmax(predictions))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(predictions[predicted_index] * 100.0)

    return {
        "class": predicted_class,
        "confidence": round(confidence, 2),
        "recyclability": RECYCLABLE[predicted_class],
        "scores": {
            CLASS_NAMES[i]: round(float(predictions[i] * 100.0), 2)
            for i in range(NUM_CLASSES)
        },
    }


@app.errorhandler(413)
def too_large(_error):
    return render_template(
        "index.html",
        result=None,
        error="Image is too large. Maximum file size is 10 MB.",
    ), 413


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        if "image" not in request.files:
            error = "Please select an image."
            return render_template("index.html", result=None, error=error)

        file = request.files["image"]

        if not file or not file.filename:
            error = "Please select an image."
            return render_template("index.html", result=None, error=error)

        if not allowed_file(file.filename):
            error = "Invalid file type. Use JPG, JPEG, PNG, or WEBP."
            return render_template("index.html", result=None, error=error)

        original_name = secure_filename(file.filename)
        extension = Path(original_name).suffix.lower()
        filename = f"{uuid.uuid4().hex}{extension}"
        file_path = UPLOAD_FOLDER / filename

        try:
            file.save(str(file_path))

            # Validate that the uploaded file is actually an image.
            with Image.open(file_path) as test_image:
                test_image.verify()

            result = predict_waste(file_path)
            result["filename"] = filename
            result["original_name"] = original_name

        except Exception as exc:
            error = f"Prediction error: {exc}"

            if file_path.exists():
                file_path.unlink(missing_ok=True)

    return render_template("index.html", result=result, error=error)


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(str(UPLOAD_FOLDER), filename)


@app.route("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "tensorflow": tf.__version__,
        "classes": CLASS_NAMES,
    }


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
