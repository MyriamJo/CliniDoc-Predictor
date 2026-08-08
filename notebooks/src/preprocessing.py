from __future__ import annotations

import re
from pathlib import Path

import spacy
from sklearn.datasets import load_files

CLASS_NAMES = [
    "Discharge Summary",
    "Gastroenterology",
    "General Medicine",
    "Neurology",
    "Radiology",
]

# Common clinical abbreviations expanded before any other cleaning, so that downstream
# stopword removal and lemmatization operate on the full clinical term (e.g. "MRI" ->
# "magnetic resonance imaging") instead of an opaque acronym.
MEDICAL_ABBREVIATIONS = {
    'MRI': 'magnetic resonance imaging', 'CT': 'computed tomography', 'CBC': 'complete blood count',
    'HB': 'hemoglobin', 'WBC': 'white blood cell', 'EKG': 'electrocardiogram', 'BP': 'blood pressure',
    'CXR': 'chest x-ray', 'US': 'ultrasound', 'GI': 'gastrointestinal', 'IV': 'intravenous',
    'IVP': 'intravenous pyelogram', 'IM': 'intramuscular', 'PT': 'physical therapy',
    'OT': 'occupational therapy', 'CTA': 'computed tomography angiography', 'SV': 'Stroke Volume',
    'MRA': 'magnetic resonance angiography', 'CC': 'chief complaint', 'HX': 'history',
    'OD': 'oculus dexter', 'PMH': 'past medical history', 'FHX': 'family history',
    'CAD': 'coronary artery disease', 'SHX': 'social history', 'HR': 'heart rate',
    'RR': 'respiratory rate', 'WT': 'weight', 'MS': 'multiple sclerosis', 'HPI': 'history of present illness',
    'CN': 'cranial nerves', 'EMG': 'electromyography', 'NCV': 'nerve conduction velocity',
    'XRT': 'Radiotherapy', 'MDCT': 'multi-detector computed tomography',
    'LM': 'left main coronary artery', 'RCA': 'Right Coronary Artery', 'RIS': 'radiology information system',
    'LAD': 'left anterior descending coronary artery', 'CX': 'circumflex coronary artery',
    'EF': 'ejection fraction', 'HCT': 'hematocrit', 'PP': 'pinprick',
    'COORD': 'coordination examination', 'GCS': 'glasgow coma scale', 'EXAM': 'examination',
    'GEN': 'general', 'RMCA': 'right middle cerebral artery', 'IR': 'interventional radiology',
    'LUE': 'left upper extremity', 'PEG': 'percutaneous endoscopic gastrostomy',
    'LLE': 'left lower extremity', 'DPH': 'Diphenhydramine', 'LOC': 'loss of consciousness',
    'PMH/FHX/SHX': 'past medical history family history social history', 'ETOH': 'ethanol',
    'EOM': 'extraocular movements', 'ESR': 'erythrocyte sedimentation rate',
    'PTT': 'partial thromboplastin time', 'GS': 'glucose', 'CSF': 'cerebrospinal fluid',
    'Cx': 'culture', 'ANA': 'antinuclear antibody', 'EEG': 'electroencephalogram',
    'LBBB': 'left bundle branch block', 'Screen': 'screening', 'ASA': 'aspirin', 'ICU': 'intensive care unit',
    'HCTZ': 'hydrochlorothiazide', 'BPD': 'biparietal diameter', 'HC': 'head circumference',
    'AC': 'abdominal circumference', 'FL': 'femur length', 'AFI': 'amniotic fluid index',
    'BPM': 'beats per minute', 'AP': 'anteroposterior', 'OB': 'obstetric',
    'CVA': 'cerebrovascular accident', 'MCA': 'middle cerebral artery',
    'LVEF': 'left ventricular ejection fraction', 'LA': 'left atrium', 'RA': 'right atrium',
    'RV': 'right ventricle', 'LV': 'left ventricle', 'PA': 'pulmonary artery',
    'RVSP': 'right ventricular systolic pressure', 'ERCP': 'endoscopic retrograde cholangiopancreatography',
    'MRCP': 'magnetic resonance cholangiopancreatography', 'MRSA': 'methicillin-resistant Staphylococcus aureus',
    'MR': 'magnetic resonance', 'AI': 'aortic insufficiency', 'TR': 'tricuspid regurgitation',
    'ASM': 'anti-seizure medication', 'SPECT': 'single-photon emission computed tomography',
    'ASD': 'atrial septal defect', 'ER': 'emergency room', 'LVH': 'left ventricular hypertrophy',
    'MEDS': 'medication', 'ROS': 'review of system', 'MOTOR': 'motor examination',
    'SENSORY': 'sensory examination', 'STATION': 'station examination', 'GAIT': 'gait examination',
    'REFLEXES': 'reflex examination', 'AD': "alzheimer's disease", 'ALS': 'amyotrophic lateral sclerosis',
    'CNS': 'central nervous system', 'PD': "parkinson's disease", 'PNS': 'peripheral nervous system',
    'TIA': 'transient ischemic attack', 'HIV': 'human immunodeficiency virus',
    'ASPECTS': 'alberta stroke program early ct score', 'CBF': 'cerebral blood flow',
    'ED': 'emergency department', 'DWI': 'diffusion-weighted imaging', 'ICA': 'internal carotid artery',
    'LP': 'Lumbar Puncture', 'IVH': 'intraventricular hemorrhage', 'PCA': 'posterior cerebral artery',
    'NCS': 'nerve conduction studies', 'SAH': 'subarachnoid hemorrhage', 'TBI': 'traumatic brain injury',
    'VA': 'visual acuity', 'DVT': 'deep vein thrombosis', 'Dx': 'diagnosis',
    'C6': 'cervical vertebra 6', 'MCI': 'mild cognitive impairment', 'C5': 'cervical vertebra 5',
    'C7': 'cervical vertebra 7', 'G': 'gravida', 'P': 'para', 'C': 'celsius',
    'yr': 'year', 'mo': 'month', 'mg': 'milligram', 'kg': 'kilogram', 'cm': 'centimeter',
    'EMS': 'emergency medical service', 'min': 'minute', 'yr/o': 'year-old', 'ENT': 'Ear Nose Throat',
    'Botox': 'botulinum toxin', 'ASAP': 'as soon as possible', 'BMI': 'body mass index',
    'PE': 'physical examination', 'ADL': 'activities of daily living', 'SC': 'subcutaneous',
    'NSTEMI': 'non-st-elevation myocardial infarction', 'STEMI': 'st-elevation myocardial infarction',
    'COPD': 'chronic obstructive pulmonary disease', 'DM': 'diabetes mellitus', 'HTN': 'hypertension',
    'CHF': 'congestive heart failure', 'PCR': 'polymerase chain reaction', 'CKD': 'chronic kidney disease',
    'GERD': 'gastroesophageal reflux disease', 'RNA': 'ribonucleic acid', 'IBD': 'inflammatory bowel disease',
    'IBS': 'irritable bowel syndrome', 'UC': 'ulcerative colitis', 'CD': "crohn's disease",
    'EGD': 'Esophagogastroduodenoscopy', 'HIDA': 'hepatobiliary iminodiacetic acid',
    'SBS': 'short bowel syndrome', 'EUS': 'endoscopic ultrasound', 'GIB': 'gastrointestinal bleeding',
    'H.pylori': 'helicobacter pylori', 'LFT': 'liver function test', 'PET': 'positron emission tomography',
    'PPI': 'proton pump inhibitor', 'SBFT': 'small bowel follow through',
    'TIPS': 'transjugular intrahepatic portosystemic shunt', 'BE': 'barium enema',
    'FMT': 'fecal microbiota transplantation', 'EMR': 'endoscopic mucosal resection',
    'HPV': 'human papillomavirus', 'SIBO': 'small intestinal bacterial overgrowth', 'dc': 'discharge',
    'po': 'by mouth', 'npo': 'nothing by mouth', 'DNA': 'deoxyribonucleic acid',
    'AIDS': 'acquired immunodeficiency syndrome', 'prn': 'as needed', 'q': 'every', 'qd': 'every day',
    'bid': 'twice a day', 'tid': 'three times a day', 'qid': 'four times a day', 'qhs': 'every night',
    'q4h': 'every 4 hours', 'q6h': 'every 6 hours', 'qod': 'every other day',
    'bka': 'below knee amputation', 'aka': 'above knee amputation', 'MI': 'myocardial infarction',
    'UTI': 'urinary tract infection',
}

# General-English filler words that don't carry class-discriminative signal for this task,
# on top of spaCy's built-in stopword list.
GENERAL_STOP_WORDS = [
    'one', 'two', 'three', 'four', 'addition', 'furthermore', 'at', 'in', 'on', 'by', 'with', 'and', 'or', 'but', 'so',
    'yet', 'it', 'he', 'she', 'they', 'his', 'her', 'their', 'him', 'our', 'very', 'much', 'more', 'most', 'quite', 'always',
    'never', 'to', 'when', 'while', 'the', 'of', 'from', 'do', 'does', 'make', 'take', 'get', 'have', 'say', 'call', 'go', 'come',
    'see', 'seem', 'could', 'would', 'should', 'must', 'be', 'been', 'is', 'was', 'were', 'might', 'shall', 'has', 'had', 'this',
    'these', 'that', 'those', 'all', 'difference', 'may', 'for', 'not', 'without', 'as', 'past', 'present',
    'sometimes', 'if', 'thus', 'however', 'during', 'into', 'via', 'we', 'because', 'also', 'less', 'than', 'many', 'slowly',
    'increasing', 'you', 'i', 'decreasing', 'become', 'frequently', 'good', 'bad', 'nice', 'beautiful',
    'ugly', 'big', 'small', 'large', 'tiny', 'huge', 'old', 'new', 'young', 'high', 'low', 'deep', 'shallow', 'wide', 'narrow',
    'long', 'short', 'tall', 'really', 'often', 'already', 'still', 'too', 'here', 'there', 'now', 'about', 'above', 'below',
    'between', 'among', 'through', 'under', 'over', 'before', 'after', 'behind', 'beside', 'within', 'nor', 'who', 'what',
    'which', 'whose', 'whom', 'mine', 'yours', 'hers', 'ours', 'theirs', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
    'additionally', 'moreover', 'besides', 'forward', 'backward', 'upward', 'downward', 'left', 'right', 'clockwise',
    'counterclockwise', 'near', 'far', 'adjacent', 'distant', 'close', 'apart', 'few', 'several', 'little',
    'fewest', 'least', 'numerous', 'later', 'earlier', 'subsequent', 'preceding', 'initial', 'final', 'alike', 'similar',
    'different', 'distinct', 'equivalent', 'comparable', 'cause', 'effect', 'result', 'association', 'correlation',
    'related', 'somewhat', 'neighboring', 'thereafter', 'further', 'ultimate', 'time', 'a', 'again',
    'against', 'am', 'an', 'any', 'are', "aren't", 'because', 'been', 'before', 'being', 'both', 'but',
    'can', "can't", 'cannot', "couldn't", 'did', "didn't", "doesn't", 'doing', "don't", 'down', 'each',
    'further', "hadn't", "hasn't", "haven't", 'having', "he'd", "he'll", "he's", "here's", 'herself', 'himself',
    "how's", "i'd", "i'll", "i'm", "i've", "isn't", "it's", 'its', 'itself', "let's", 'me', "mustn't", 'my', 'myself',
    'no', 'day', 'week', 'month', 'year', 'today', 'tomorrow', 'yesterday', 'hour', 'minute', 'second', 'some', 'every', 'other',
    'likelihood', 'identical', 'like', 'same', 'equal', 'matching', 'parallel', 'corresponding',
    'multiple', 'hope', 'wish', 'desire', 'aspiration', 'optimism', 'expectation', 'anticipation',
    'confidence', 'trust', 'belief', 'faith', 'useful', 'beneficial', 'benefit', 'help', 'helpful', 'value', 'valuable',
    'effective', 'advantage', 'advantageous', 'practical', 'functional', 'handy', 'convenient', 'productive', 'immediate',
]

# Domain-specific stopwords: words that appear in nearly every clinical document regardless of
# category (e.g. "patient", "hospital") and therefore add noise rather than signal.
MEDICAL_STOP_WORDS = [
    'patient', 'doctor', 'hospital', 'clinic', 'health', 'accuracy',
    'healthy', 'physician', 'healthcare', 'nurse', 'specialist',
    'sick', 'diseased', 'stable', 'unstable', 'response', 'revision', 'severely',
    'case', 'cases', 'participant', 'unit', 'department', 'findings', 'assessment', 'evaluation', 'analysis',
    'conclusions', 'human', 'common', 'indication', 'sign', 'marker', 'manifestation',
    'feature', 'criteria', 'presentation', 'index', 'pointer', 'cue', 'type', 'kind', 'version', 'document', 'documents',
    'record', 'sheet', 'format', 'variety', 'emergency', 'critical', 'urgency', 'urgent', 'crisis', 'medical', 'services',
    'service', 'men', 'women', 'man', 'woman', 'kid', 'abc', 'ab', 'abcd', 'abcg', 'abcmedical', 'abd', 'aca', 'af', 'ad',
    'yes', 'no', 'yield', 'yo', 'yr', 'yrs', 'yy', 'yyyy', 'zzz', 'xyz', 'xx', 'xxx',
    'et', 'etc', 'ii', 'iih', 'iii', 'xii', 'xl', 'xq', 'xr',
]


def load_spacy_model(name: str = "en_core_web_lg"):
    """Load spaCy and register the project's extra stopwords on it (once)."""
    nlp = spacy.load(name)
    for word in GENERAL_STOP_WORDS + MEDICAL_STOP_WORDS:
        nlp.Defaults.stop_words.add(word)
        nlp.vocab[word].is_stop = True
    return nlp


# Pipeline steps

def expand_abbreviations(document: str, nlp) -> str:
    """Replace clinical abbreviations (token-for-token) with their expanded form."""
    doc = nlp(document)
    expanded = [MEDICAL_ABBREVIATIONS.get(token.text, token.text) for token in doc]
    return " ".join(expanded)


def deidentify(document: str) -> str:
    """Strip patient/doctor names, hospital names, and titles (Mr./Mrs./Dr./...)."""
    text = document
    text = re.sub(r'\b(?:Mr\.|Mrs\.|Ms\.)\s[A-Z][a-z]*\b', ' ', text)
    text = re.sub(r'\bDr\.\s[A-Z][a-z]*\b', ' ', text)
    text = re.sub(r'\b[A-Z][a-zA-Z]*\s+hospital\b', ' ', text)
    text = re.sub(r'\b(?:Mr|Ms|Mrs|Dr)\.?\b', ' ', text)
    return text


def clean_and_lemmatize(document: str, nlp) -> str:
    """Strip dates/digits/punctuation and lemmatize what's left."""
    text = document
    text = re.sub(r'\b(?:\d{1,2}|MM|mm)\s*[/-]\s*(?:\d{1,2}|DD|dd)\s*[/-]\s*(?:\d{4}|YYYY|yyyy)\b', ' ', text)  # dates
    text = re.sub(r'\d', ' ', text)  # remaining digits
    text = re.sub(r'[\n\r]', ' ', text)
    text = re.sub(r'[^\w/\'-]', ' ', text)  # punctuation, keep word chars / ' -
    text = re.sub(r'\b(?![a-zA-Z]-)[a-zA-Z]\b', ' ', text)  # single characters
    text = re.sub(r'^b\s+', ' ', text)  # stray leading b (bytes-decoding artifact)

    doc = nlp(text)
    return ' '.join(token.lemma_ for token in doc)


def remove_stopwords(document: str, nlp) -> str:
    """Drop stopwords/punctuation tokens and collapse whitespace."""
    doc = nlp(document)
    kept = [w.text.lower() for w in doc if w.text.lower() not in nlp.Defaults.stop_words and not w.is_punct]
    text = ' '.join(kept)
    text = re.sub(r'\b(?![a-zA-Z]-)[a-zA-Z]\b', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess_document(document: str, nlp) -> str:
    """Run the full cleaning pipeline: abbreviations -> de-identify -> clean/lemmatize -> stopwords."""
    document = expand_abbreviations(document, nlp)
    document = deidentify(document)
    document = clean_and_lemmatize(document, nlp)
    document = remove_stopwords(document, nlp)
    return document


# Data loading
def load_clinical_documents(data_dir: str | Path):
    """Load the (augmented) clinical documents dataset from ``data_dir``.

    Expects the standard sklearn ``load_files`` layout: one sub-folder per class
    (matching ``CLASS_NAMES``, e.g. ``Discharge Summary/``, ``Radiology/``, ...),
    each containing plain-text documents.

    Returns a DataFrame with columns ``label`` (class name, string) and ``document``
    (decoded text).
    """
    import pandas as pd

    bunch = load_files(str(data_dir), encoding="utf-8", decode_error="replace")
    df = pd.DataFrame({"label": bunch.target, "document": bunch.data})
    # load_files assigns integer targets alphabetically by folder name, which matches
    # CLASS_NAMES sorted order (Discharge Summary, Gastroenterology, General Medicine,
    # Neurology, Radiology) -- swap in the human-readable label.
    target_names = sorted(CLASS_NAMES)
    df["label"] = df["label"].map(lambda i: target_names[i])
    return df
