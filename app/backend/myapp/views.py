import os

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .clinical_document_classifier import ClinicalDocumentClassifier

# Model artifacts are trained and exported in
# notebooks/01_traditional_ml_classification.ipynb and shipped alongside the
# app so the API has no external dependency at request time.
MODEL_PATH = os.path.join(os.path.dirname(__file__), "ml_artifacts", "voting_classifier_model.joblib")
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), "ml_artifacts", "tfidf_vectorizer.joblib")

# Loaded once at process start (spaCy + joblib load are too slow to redo per request).
_classifier = None


def get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = ClinicalDocumentClassifier(MODEL_PATH, VECTORIZER_PATH)
    return _classifier


@csrf_exempt
def upload_view(request):
    """Accept an uploaded clinical document (multipart field "document") and return its predicted class."""
    if request.method == 'POST' and request.FILES.get('document'):
        document = request.FILES['document'].read().decode('utf-8')

        prediction = get_classifier().predict(document)

        return JsonResponse({'prediction': prediction})
    else:
        return JsonResponse({'error': 'No document uploaded. POST a file under the "document" field.'}, status=400)



