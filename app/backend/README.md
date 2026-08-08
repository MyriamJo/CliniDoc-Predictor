# CliniDoc Predictor — Backend

Django + Django REST Framework API that serves the Voting Classifier (SVM + KNN, TF-IDF features) trained in `../../notebooks/01_traditional_ml_classification.ipynb`. This is the model the thesis identifies as the best performer overall (accuracy 0.89, F1 0.888).

## Setup

```
pip install -r requirements.txt
python -m spacy download en_core_web_lg
python manage.py migrate
python manage.py runserver
```

Runs on `http://localhost:8000`. CORS is currently open (`CORS_ORIGIN_ALLOW_ALL = True` in `myriam/settings.py`) for local development with the React frontend on port 3000 — tighten this before any real deployment.
