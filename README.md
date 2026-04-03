# Fire vs Smoke Classifier

This project classifies images as **fire** or **smoke** using a TensorFlow/Keras model and provides a simple Streamlit web app for prediction.

## What is in this project

- A trained model file: `model/fire_smoke.keras`
- A Streamlit app for inference: `app.py`
- Data helper scripts:
  - `augmentation.py` for data augmentation
  - `test_split.py` for creating a test split
- A training notebook: `training.ipynb`

## Project structure

```text
.
├── app.py
├── augmentation.py
├── test_split.py
├── training.ipynb
├── requirements.txt
├── model/
│   └── fire_smoke.keras
├── train/
│   ├── fire/
│   └── smoke/
└── test/
    ├── fire/
    └── smoke/
```

## Prerequisites

- Python 3.10+ (recommended)
- A virtual environment (you already use `.ven`)

## Setup

1. Open a terminal in the project folder.
2. Activate your virtual environment.
3. Install dependencies.

### Windows (Git Bash)

```bash
source .venv/Scripts/activate
pip install -r requirements.txt
```

### Windows (PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the Streamlit app

```bash
streamlit run app.py
```

Then open the local URL shown in terminal (usually `http://localhost:8501`).

If `streamlit` is not recognized:

```bash
python -m streamlit run app.py
```

## How to use the app

1. Upload an image (`.jpg`, `.jpeg`, or `.png`).
2. Click **Predict**.
3. View:
   - predicted class (`Fire` or `Smoke`)
   - confidence score

## Notes about model input/output

- The app resizes input images to `224 x 224`.
- The saved model is loaded from `model/fire_smoke.keras`.
- Labels are inferred from folders in `train/` (sorted alphabetically).
  - With your current folders, this maps to `fire` and `smoke`.

## Utility scripts

### `augmentation.py`

Applies Albumentations-based transforms and writes generated images for inspection.

Run:

```bash
python augmentation.py
```

### `test_split.py`

Moves a sample of images from `train/` into `test/`.

Run:

```bash
python test_split.py
```

## Training

Open and run `training.ipynb` to train/evaluate and save the model.

Model save path in notebook:

```python
model.save('./model/fire_smoke.keras')
```

## Troubleshooting

- **Model file not found**
  - Ensure `model/fire_smoke.keras` exists.
- **Import errors**
  - Re-run `pip install -r requirements.txt` inside the active virtual environment.
- **Wrong Python environment**
  - Verify your terminal shows the `.venv-tf` environment is activated.
