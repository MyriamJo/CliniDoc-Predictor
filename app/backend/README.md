# CliniDoc Predictor — Backend

Django + Django REST Framework API that serves the Voting Classifier (SVM + KNN, TF-IDF features) trained in `../../notebooks/01_traditional_ml_classification.ipynb`. This is the model the thesis identifies as the best performer overall (accuracy 0.89, F1 0.888).

## Structure

```
app/backend/
├── myriam/                  Django project config (settings, root urls)
├── myapp/                   The classifier API
│   ├── views.py             POST /api/predict/ -> runs the classifier on an uploaded document
│   ├── clinical_document_classifier.py   Preprocessing + inference wrapper (mirrors notebooks/src/preprocessing.py)
│   ├── ml_artifacts/        Exported model + vectorizer (.joblib)
│   ├── models.py            Document model (uploaded file record)
│   └── urls.py
├── media/                   Uploaded documents (runtime data, see .gitignore)
├── manage.py
└── requirements.txt
```

## Setup

```
pip install -r requirements.txt
python -m spacy download en_core_web_lg
python manage.py migrate
python manage.py runserver
```

Runs on `http://localhost:8000`. CORS is currently open (`CORS_ORIGIN_ALLOW_ALL = True` in `myriam/settings.py`) for local development with the React frontend on port 3000 — tighten this before any real deployment.

## API

`POST /api/predict/` — multipart form, field name `document` (a `.txt` file). Returns `{"prediction": "<label>"}`.

## Notes on this cleanup

- `myapp/clinical_document_classifier.py` was renamed from `Voting_Classifier.py` and its class from `Voting_Classifier` to `ClinicalDocumentClassifier`.
- `myapp/VC_model.joblib` / `vectorizer.joblib` moved into `myapp/ml_artifacts/` and renamed to `voting_classifier_model.joblib` / `tfidf_vectorizer.joblib`.
- `views.py` previously hardcoded an absolute Windows path to the model files and reloaded spaCy + both joblib files on *every* request. Fixed to resolve paths relative to the app directory and to load the classifier once per process.
- `myapp/classifier.py` was dead code (loaded a model from a placeholder path, never imported anywhere) — moved to `../../archive/dead_code/`.
- The Django app label (`myapp`) and project package name (`myriam`) were **left unchanged**. Renaming them would change Django's expected DB table names (`myapp_document` → e.g. `classifier_api_document`) and break `db.sqlite3`'s existing migration history and uploaded-document records, which isn't safe to do without being able to run `manage.py migrate` and verify it. If you want to rename these too, do it as its own change with a fresh migration.
- There was a second, parallel Django backend (`controller/controller`, app label `api`) that referenced a `ml_model.py` module that doesn't exist and was never wired to the frontend. It's been moved to `../../archive/controller_django_scaffold_unused/` rather than deleted.
