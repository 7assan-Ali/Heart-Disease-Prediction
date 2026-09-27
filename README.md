# Heart Disease Prediction

Binary classification workflow for predicting the presence of heart disease from clinical measurements.

## Models
Logistic Regression · SVM · Random Forest

## Metrics
Accuracy · Precision · Recall · F1 · ROC-AUC

## Dataset
https://raw.githubusercontent.com/plotly/datasets/master/heart.csv

No fixed performance numbers are claimed; run the training script to reproduce the evaluation.

## Run
```bash
python -m venv .venv
.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
python src/train.py
```

## Methodology
The preprocessing pipeline is fitted only on the training split and reused on held-out data.

## Author
Hassan Ali — Computer Science student focused on Machine Learning and AI Engineering.
