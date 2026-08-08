# CliniDoc Predictor — Clinical Document Classification

Project to automatically classifying narrative clinical documents into 5 categories using machine learning, as a proof of concept for automating document entry and retrieval in hospital record systems.

## Results summary

| Category | Best model | Accuracy | F1 |
|---|---|---|---|
| Traditional ML | **Voting Classifier** (SVM + KNN, TF-IDF) | **0.89** | **0.888** |
| Sequence models | GRU | 0.88 | 0.878 |
| Transformers | ClinicalBERT | 0.85 | 0.84 |

The Voting Classifier is the overall winner and is what's deployed in `app/`

## Dataset

Derived from [Clinical Documents on Syndromes Disease](https://www.kaggle.com/datasets/muhammadimran112233/clinical-documents-on-syndromes-disease/data)
(Kaggle), extended with data augmentation to ~100 documents per category (493 total). Layout: one
folder per class under `data/clinical_documents/`, each containing plain-text documents

## Running the app (CliniDoc Predictor)

See [`app/backend/README.md`](app/backend/README.md) and [`app/frontend/README.md`](app/frontend/README.md).
Short version: `pip install -r app/backend/requirements.txt && python app/backend/manage.py runserver`,
then `npm install && npm start` in `app/frontend/`. A demo recording of the app end-to-end (upload →
content preview → predicted label) is here: https://www.youtube.com/watch?v=OLQi3J8RsVM