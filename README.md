# ManyKGs

**Height → Weight Regression** — a Streamlit dashboard around an exported scikit-learn model.

## Features

- Height-based weight prediction
- StandardScaler preprocessing
- Pickled ML model (type detected dynamically from the pickle)
- Interactive Plotly visualizations (gauge + regression curve)
- Model information dashboard with sticky note and diagnostics
- Responsive Streamlit UI

## Install

    pip install -r requirements.txt

## Run

    streamlit run app.py

## Model files

    models/best_model.pkl
    models/scaler.pkl

Prediction flow: `height_cm → scaler.transform → model.predict → weight_kg`.
The app only loads these files; it never retrains or simulates data. The Model Lab lists the
models compared in the notebook (names only; no metrics are stored in the pickles).

Dataset: [SOCR Height/Weight (Kaggle)](https://www.kaggle.com/datasets/burnoutminer/heights-and-weights-dataset),
converted with `cm = in × 2.54` and `kg = lb × 0.45359237`.

> The pickles were saved with scikit-learn 1.6.1. For exact fidelity, `pip install scikit-learn==1.6.1`
> (newer versions load it but may warn).

## Disclaimer

This is an academic regression demonstration, **not a medical tool**. Outputs are model estimates only.
