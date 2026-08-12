from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
import tensorflow as tf
from PIL import Image
import numpy as np
from pathlib import Path
import uuid


# ============================================================
# SMART WASTE CLASSIFICATION
# Flask + TensorFlow/Keras
# ============================================================

app = Flask(__name__)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "smart_waste_model.keras"
UPLOAD_FOLDER = BASE_DIR / "uploads"


# ============================================================
# MODEL SETTINGS
# ============================================================

IMG_SIZE = 224

CLASS_NAMES = [
    "glass",
    "metal",
    "paper"
]

RECYCLABLE = {
    "glass": "Recyclable",
    "metal": "Recyclable",
    "paper": "Recyclable"
}


# ============================================================
# FILE SETTINGS
# ============================================================

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "webp"
}

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE


# ============================================================
# LOAD KERAS MODEL
# ============================================================

print("=" * 60)
print("SMART WASTE CLASSIFICATION")
print("=" * 60)

print(f"TensorFlow : {tf.__version__}")
print(f"Keras      : {tf.keras.__version__}")
print(f"Model path : {MODEL_PATH}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"\nModel not found!\n"
        f"Expected location:\n{MODEL_PATH}\n\n"
        f"Make sure your project contains:\n"
        f"models/smart_waste_model.keras"
    )


try:
    # .keras is a COMPLETE Keras model.
    # Therefore use load_model().
    model = tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False
    )

    print("Model loaded successfully!")

except Exception as error:
    print("\nERROR: Could not load model.")
    print(error)
    raise


print("=" * 60)


# ============================================================
# ALLOWED FILE CHECK
# ============================================================

def allowed_file(filename):

    return (
        filename
        and "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ============================================================
# IMAGE PREDICTION
# ============================================================

def predict_waste(image_path):

    # Open image
    image = Image.open(image_path).convert("RGB")

    # Resize
    image = image.resize(
        (IMG_SIZE, IMG_SIZE)
    )

    # Convert image to NumPy
    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # --------------------------------------------------------
    # IMPORTANT
    # --------------------------------------------------------
    # Do NOT divide by 255 here if your trained EfficientNet
    # model already contains the preprocessing internally.
    # --------------------------------------------------------

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    # Predicted class index
    predicted_index = int(
        np.argmax(predictions)
    )

    # Predicted class
    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    # Confidence
    confidence = float(
        predictions[predicted_index] * 100
    )

    # Scores for all classes
    scores = {}

    for i, class_name in enumerate(CLASS_NAMES):

        scores[class_name] = round(
            float(predictions[i] * 100),
            2
        )

    return {
        "class": predicted_class,
        "confidence": round(confidence, 2),
        "recyclability": RECYCLABLE[predicted_class],
        "scores": scores
    }


# ============================================================
# FILE SIZE ERROR
# ============================================================

@app.errorhandler(413)
def file_too_large(error):

    return render_template(
        "index.html",
        result=None,
        error="Image is too large. Maximum size is 10 MB."
    ), 413


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def index():

    result = None
    error = None

    if request.method == "POST":

        # ----------------------------------------------------
        # Check image field
        # ----------------------------------------------------

        if "image" not in request.files:

            error = "Please select an image."

            return render_template(
                "index.html",
                result=None,
                error=error
            )

        file = request.files["image"]

        # ----------------------------------------------------
        # Check filename
        # ----------------------------------------------------

        if not file or not file.filename:

            error = "Please select an image."

            return render_template(
                "index.html",
                result=None,
                error=error
            )

        # ----------------------------------------------------
        # Check extension
        # ----------------------------------------------------

        if not allowed_file(file.filename):

            error = (
                "Invalid file type. "
                "Please use JPG, JPEG, PNG, or WEBP."
            )

            return render_template(
                "index.html",
                result=None,
                error=error
            )

        # ----------------------------------------------------
        # Secure original filename
        # ----------------------------------------------------

        original_name = secure_filename(
            file.filename
        )

        # ----------------------------------------------------
        # Generate unique filename
        # ----------------------------------------------------

        extension = Path(
            original_name
        ).suffix.lower()

        filename = (
            f"{uuid.uuid4().hex}"
            f"{extension}"
        )

        file_path = (
            UPLOAD_FOLDER / filename
        )

        try:

            # ------------------------------------------------
            # Save image
            # ------------------------------------------------

            file.save(
                str(file_path)
            )

            # ------------------------------------------------
            # Validate actual image
            # ------------------------------------------------

            with Image.open(file_path) as image:

                image.verify()

            # ------------------------------------------------
            # Predict
            # ------------------------------------------------

            result = predict_waste(
                file_path
            )

            # Add filename information
            result["filename"] = filename

            result["original_name"] = (
                original_name
            )

        except Exception as error_message:

            error = (
                f"Prediction error: "
                f"{error_message}"
            )

            # Delete invalid file
            if file_path.exists():

                file_path.unlink(
                    missing_ok=True
                )

    return render_template(
        "index.html",
        result=result,
        error=error
    )


# ============================================================
# SERVE UPLOADED IMAGE
# ============================================================

@app.route("/uploads/<filename>")
def uploaded_file(filename):

    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return {
        "status": "ok",
        "model_loaded": model is not None,
        "tensorflow": tf.__version__,
        "keras": tf.keras.__version__,
        "classes": CLASS_NAMES
    }


# ============================================================
# RUN FLASK
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )