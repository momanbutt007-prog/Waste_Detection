# ♻ Smart Waste AI

A clean Flask web application for **Smart Waste Classification** using a trained **Keras / TensorFlow EfficientNetB0** model.

The application classifies an uploaded image into:

- 🟢 Glass
- 🟢 Metal
- 🟢 Paper

It also shows the model confidence and recyclability.

## Project structure

```text
Waste_Detection/
│
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
│
├── models/
│   └── smart_waste_model.keras
│
├── static/
│   └── style.css
│
├── templates/
│   └── index.html
│
└── uploads/
```

## Model

The project is designed around the architecture you showed from Colab:

```text
Input (224, 224, 3)
        ↓
Data Augmentation
        ↓
EfficientNetB0
        ↓
GlobalAveragePooling2D
        ↓
BatchNormalization
        ↓
Dropout
        ↓
Dense(3, softmax)
```

The app loads the **weights file**, not the incompatible full `.keras` file. This avoids the `quantization_config` deserialization problem you were getting locally.

### Required model file

Copy your Colab weights:

```text
/content/smart_waste_model.keras
```

into:

Do not rename it unless you also change `MODEL_PATH` in `app.py`.

## Python version

Recommended:

```text
Python 3.10.x
```

Your current environment is Python 3.10.11, so it is suitable.

## Installation

Create and activate a virtual environment:

### Windows PowerShell

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install the exact project dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Verify:

```powershell
python -c "import tensorflow as tf; import keras; print('TensorFlow:', tf.__version__); print('Keras:', keras.__version__)"
```

Expected:

```text
TensorFlow: 2.20.0
Keras: 3.12.4
```

## Run

```powershell
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

Health check:

```text
http://127.0.0.1:5000/health
```

## Important preprocessing note

The app uses `tf.keras.applications.EfficientNetB0`.

Keras EfficientNet includes its preprocessing internally, so the app intentionally **does not divide the image by 255** before prediction.

If your original training notebook used a different preprocessing pipeline, keep that preprocessing exactly the same in the Flask app.

## GitHub

Do not commit the large model file if you do not want it in Git.

The included `.gitignore` ignores `models/*`.

If you want the model available after cloning, use Git LFS or a model-storage service and document the download location in this README.

## Tech stack

- Python 3.10
- Flask 3.1.2
- TensorFlow 2.20.0
- Keras 3.12.4
- EfficientNetB0
- NumPy 2.2.6
- Pillow 11.3.0
- HTML5
- CSS3
- JavaScript

## Troubleshooting

### `FileNotFoundError`

Make sure this exists:

```text
models\smart_waste_model.keras
```

### `Layer expected variables but received different variables`

This means the architecture in `app.py` does not match the architecture used when the weights were created. Do not switch the model to MobileNetV2 unless the weights were actually trained with MobileNetV2.

### `quantization_config` error

Use the weights-based loading approach in this project instead of:

```python
tf.keras.models.load_model("smart_waste_model.keras")
```

The full model file that produced that error should not be used for this deployment setup.

---

Made with Flask + TensorFlow + Keras.
