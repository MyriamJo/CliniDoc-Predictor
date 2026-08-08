"""Inference wrapper around the deployed Voting Classifier (SVM + KNN, TF-IDF features).

This mirrors the preprocessing pipeline developed and validated in
``notebooks/01_traditional_ml_classification.ipynb`` (see also the shared
``notebooks/src/preprocessing.py`` module). Any change to the cleaning steps
there should be mirrored here, since this is the copy that actually runs
in production for CliniDoc Predictor.
"""

import joblib
import re
import spacy

nlp = spacy.load('en_core_web_lg')
from nltk.stem import WordNetLemmatizer


class ClinicalDocumentClassifier:
    def __init__(self, model_path, vectorizer_path):
         self.model = joblib.load(model_path)
         self.vectorizer = joblib.load(vectorizer_path)


    def preprocess_input(self, document):
        document = self.expand_abbreviations(document)
        document = self.deidentify(document)
        document = self.remove(document)
        document = self.remove_stopwords(document)
        return document
    
    def expand_abbreviations(self, document):
        medical_abbreviations = {
        'MRI': 'magnetic resonance imaging', 'CT': 'computed tomography', 'CBC': 'complete blood count',
        'HB': 'hemoglobin', 'WBC': 'white blood cell', 'EKG': 'electrocardiogram', 'BP': 'blood pressure',
        'CXR': 'chest x-ray', 'US': 'ultrasound', 'GI': 'gastrointestinal', 'IV': 'intravenous',
        'IVP': 'intravenous pyelogram', 'IM': 'intramuscular', 'PT': 'physical therapy',
        'OT': 'occupational therapy', 'CTA': 'computed tomography angiography','SV':'Stroke Volume',
        'MRA': 'magnetic resonance angiography', 'CC': 'chief complaint', 'HX': 'history',
        'OD': 'oculus dexter', 'PMH': 'past medical history', 'FHX': 'family history',
        'CAD': 'coronary artery disease', 'SHX': 'social history', 'HR': 'heart rate',
        'RR': 'respiratory rate', 'WT': 'weight', 'MS': 'multiple sclerosis',
        'CN': 'cranial nerves', 'EMG': 'electromyography', 'NCV': 'nerve conduction velocity',
        'XRT': 'Radiotherapy', 'MDCT': 'multi-detector computed tomography',
        'LM': 'left main coronary artery', 'RCA': 'Right Coronary Artery','RIS':'radiology information system',
        'LAD': 'left anterior descending coronary artery', 'CX': 'circumflex coronary artery',
        'EF': 'ejection fraction', 'HCT': 'hematocrit', 'PP': 'pinprick',
        'COORD': 'coordination examination', 'GCS': 'glasgow coma scale', 'EXAM': 'examination',
        'GEN': 'general', 'RMCA': 'right middle cerebral artery','IR':'interventional radiology',
        'LUE': 'left upper extremity', 'PEG': 'percutaneous endoscopic gastrostomy',
        'LLE': 'left lower extremity', 'DPH': 'Diphenhydramine', 'LOC': 'loss of consciousness',
        'PMH/FHX/SHX': 'past medical history family history social history', 'ETOH': 'ethanol',
        'EOM': 'extraocular movements', 'ESR': 'erythrocyte sedimentation rate',
        'PTT': 'partial thromboplastin time', 'GS': 'glucose', 'CSF': 'cerebrospinal fluid',
        'Cx': 'culture', 'ANA': 'antinuclear antibody', 'EEG': 'electroencephalogram',
        'LBBB': 'left bundle branch block', 'Screen': 'screening', 'PT': 'prothrombin time',
        'PTT': 'artial thromboplastin time', 'ASA': 'aspirin','ICU':'intensive care unit',
        'HCTZ': 'hydrochlorothiazide', 'BPD': 'biparietal diameter', 'HC': 'head circumference',
        'AC': 'abdominal circumference', 'FL': 'femur length', 'AFI': 'amniotic fluid index',
        'BPM': 'beats per minute', 'AP': 'anteroposterior', 'OB': 'obstetric',
        'CVA': 'cerebrovascular accident', 'MCA': 'middle cerebral artery','GI':'Gastrointestinal',
        'LVEF': 'left ventricular ejection fraction', 'LA': 'left atrium', 'RA': 'right atrium',
        'RV': 'right ventricle', 'LV': 'left ventricle', 'PA': 'pulmonary artery',
        'RVSP': 'right ventricular systolic pressure', 'ERCP': 'endoscopic retrograde cholangiopancreatography',
        'MRCP': 'magnetic resonance cholangiopancreatography', 'MRSA': 'methicillin-resistant Staphylococcus aureus',
        'MR': 'magnetic resonance',	'EEG':'electroencephalogram','CSF':'cerebrospinal fluid',
        'AI': 'aortic insufficiency', 'TR': 'tricuspid regurgitation','ASM':'anti-seizure medication',
        'SPECT': 'single-photon emission computed tomography', 'ASD': 'atrial septal defect','ER':'emergency room',
        'LVH': 'left ventricular hypertrophy', 'MEDS': 'medication', 'ROS': 'review of system',
        'MOTOR': 'motor examination', 'SENSORY': 'sensory examination', 'STATION': 'station examination',
        'GAIT': 'gait examination', 'REFLEXES': 'reflex examination', 'AD': "alzheimer's disease",
        'ALS': 'amyotrophic lateral sclerosis', 'CNS': 'central nervous system', 'PD': "parkinson's disease",
        'PNS': 'peripheral nervous system', 'TIA': 'transient ischemic attack',	'HIV':'human immunodeficiency virus',
        'ASPECTS': 'alberta stroke program early ct score', 'CBF': 'cerebral blood flow','ED':'emergency department',
        'DWI': 'diffusion-weighted imaging', 'ICA': 'internal carotid artery','LP':'Lumbar Puncture',
        'IVH': 'intraventricular hemorrhage', 'PCA': 'posterior cerebral artery','NCS': 'nerve conduction studies',
        'SAH': 'subarachnoid hemorrhage', 'TBI': 'traumatic brain injury', 'VA': 'visual acuity',
        'DVT': 'deep vein thrombosis', 'Dx': 'diagnosis', 'C6': 'cervical vertebra 6','MCI':'mild cognitive impairment',
        'C5': 'cervical vertebra 5', 'C7': 'cervical vertebra 7', 'G': 'gravida', 'P': 'para', 'C': 'celsius',
        'yr': 'year', 'mo': 'month', 'mg': 'milligram', 'kg': 'kilogram', 'cm': 'centimeter',
        'EMS': 'emergency medical service', 'min': 'minute', 'yr/o': 'year-old', 'ENT': 'Ear Nose Throat',
        'Botox': 'botulinum toxin', 'ASAP': 'as soon as possible', 'BMI': 'body mass index', 'PE': 'physical examination',
        'ADL': 'activities of daily living', 'SC': 'subcutaneous', 'NSTEMI': 'non-st-elevation myocardial infarction',
        'STEMI': 'st-elevation myocardial infarction', 'COPD': 'chronic obstructive pulmonary disease',
        'DM': 'diabetes mellitus', 'HTN': 'hypertension', 'CHF': 'congestive heart failure', 'PCR': 'polymerase chain reaction',
        'CKD': 'chronic kidney disease', 'GERD': 'gastroesophageal reflux disease', 'RNA': 'ribonucleic acid',
        'IBD': 'inflammatory bowel disease', 'IBS': 'irritable bowel syndrome', 'UC': 'ulcerative colitis',
        'CD': "crohn's disease", 'EGD': 'Esophagogastroduodenoscopy', 'HIDA': 'hepatobiliary iminodiacetic acid',
        'SBS': 'short bowel syndrome', 'EUS': 'endoscopic ultrasound', 'GIB': 'gastrointestinal bleeding',
        'H.pylori': 'helicobacter pylori', 'LFT': 'liver function test','PET': 'positron emission tomography',
        'PPI': 'proton pump inhibitor', 'SBFT': 'small bowel follow through', 'TIPS': 'transjugular intrahepatic portosystemic shunt',
        'BE': 'barium enema', 'FMT': 'fecal microbiota transplantation', 'EMR': 'endoscopic mucosal resection',
        'HPV': 'human papillomavirus', 'SIBO': 'small intestinal bacterial overgrowth', 'dc': 'discharge',
        'po': 'by mouth', 'npo': 'nothing by mouth','DNA': 'deoxyribonucleic acid', 'AIDS': 'acquired immunodeficiency syndrome',
        'prn': 'as needed', 'q': 'every', 'qd': 'every day', 'bid': 'twice a day', 'tid': 'three times a day',
        'qid': 'four times a day', 'qhs': 'every night', 'q4h': 'every 4 hours', 'q6h': 'every 6 hours',
        'qod': 'every other day', 'bka': 'below knee amputation', 'aka': 'above knee amputation',
        'MI': 'myocardial infarction', 'PE': 'pulmonary embolism', 'UTI': 'urinary tract infection'}
        doc = nlp(document)
        expanded_text = []
        for token in doc:
            if token.text in medical_abbreviations:
                expanded_text.append(medical_abbreviations[token.text])
            else:
                expanded_text.append(token.text)
        expanded_text = " ".join(expanded_text)
        document=expanded_text

        return document

    def deidentify(self, document):
            text = document
            text = re.sub(r'\b(?:Mr\.|Mrs\.|Ms\.)\s[A-Z][a-z]*\b', ' ', text)
    
            # Remove doctor's name (after Dr.)
            text = re.sub(r'\bDr\.\s[A-Z][a-z]*\b', '', text)
            
            # Remove hospital name (before the word "hospital")
            text = re.sub(r'\b[A-Z][a-zA-Z]*\s+hospital\b', ' ', text)
            
            # Remove titles
            text = re.sub(r'\b(?:Mr|Ms|Mrs|Dr)\.?\b', ' ', text)

            return text

    def remove(self, document):
        text =re.sub(r'[^\w/\'-]', ' ', document)
        text = re.sub(r'\b(?:\d{1,2}|MM|mm)\s*[/-]\s*(?:\d{1,2}|DD|dd)\s*[/-]\s*(?:\d{4}|YYYY|yyyy)\b', ' ', text)
        text = re.sub(r'\d', ' ', text)
        text = re.sub(r'\n', '', text)
        text =re.sub(r'[^\w/\'-]', ' ', text)
        text = re.sub(r'\b(?![a-zA-Z]-)[a-zA-Z]\b', ' ', text)
        text = re.sub(r'\b(?:[0-9]|[1-9][0-9]|100)\b', ' ', text)
        text= re.sub(r'^b\s+','',text)
        text=nlp(text)
        text = ' '.join(token.lemma_ for token in text)

        return text



    def remove_stopwords(self, document):
        spacy_stop_words = nlp.Defaults.stop_words

        new_stop_words = ['one','two','three','four','addition','furthermore','at','in','on','by','with','and','or','but','so',
                        'yet','it','he','she','they','his','her','their','him','our','very','much','more','most','quite','always',
                        'never','to','when','while','the','of','from','do','does','make','take','get','have','say','call','go','come',
                        'see','seem','could','would','should','must','be','been','is','was','were','might','shall','has','had','this',
                        'these','that','those','all','difference','may','for','not','without','as','past','present',
                        'sometimes','if','thus','however','during','into','via','we','because','also','less','than','many','slowly',
                        'increasing','they','you','i','decreasing','become','frequently','good','bad','nice','beautiful',
                        'ugly','big','small','large','tiny','huge','old','new','young','high','low','deep','shallow','wide','narrow',
                        'long','short','tall','really','often','already','still','too','here','there','now','about','above','below',
                        'between','among','through','under','over','before','after','behind','beside','within','nor','who','what',
                        'which','whose','whom','mine','yours','hers','ours','theirs','five','six','seven','eight','nine','ten',
                        'additionally','moreover','besides','forward','backward','upward','downward','left','right','clockwise',
                        'counterclockwise','near','far','adjacent','distant','close','apart','few','several','much','little',
                        'fewest','least','numerous','later','earlier','subsequent','preceding','initial','final','alike','similar',
                        'different','distinct','equivalent','comparable','cause','effect','result','association','correlation',
                        'related','somewhat','neighboring','thereafter','further','ultimate','time','very','large','also','a','about',
                        'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any',
                        'are',"aren't",'as','at','be','because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 
                        'by','can',"can't",'cannot','could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't"
                        ,'down','during', 'each','few', 'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't"
                        ,'having', 'he',"he'd","he'll","he's",'her','here',"here's",'hers','herself','him','himself','his',
                        'how',"how's",'i',"i'd","i'll","i'm","i've",'if','in','into','is',"isn't",'it',"it's",'its',
                        'itself',"let's",'me','more','most',"mustn't",'my','myself','no','nor', 'day', 'week', 'month', 'year',
                        'today', 'tomorrow', 'yesterday', 'hour', 'minute', 'second', 'some', 'any', 'each', 'every', 'other','likelihood'
                        'several','identical','similar','alike','like','same','equal','matching','parallel','corresponding','equivalent'
                        ,'multiple','numerous','more','several','hope','wish','desire','aspiration','optimism','expectation','anticipation',
                        'confidence','trust','belief','faith','useful','beneficial','benefit','help','helpful','value','valuable','effect',
                        'effective','advantage','advantageous','practical','functional','handy','convenient','productive','immediate']

        medical_stop_words=['patient','doctor','hospital','clinic','health','accuracy',
                            'healthy','physician','healthcare','nurse','specialist','healthy',
                            'sick','diseased','stable','unstable','response','revision','severely',
                        'case','cases','participant','unit','department','findings','assessment','evaluation','analysis',
                        'conclusions','human','common','indication','sign','marker','manifestation',
                        'feature','criteria','presentation','index','pointer','cue','type','kind','version','document','documents',
                        'record','sheet','format','variety','emergency','critical','urgency','urgent','crisis','medical','services'
                        'service','men','women','man','woman','kid','abc','ab','abcd', 'abcg', 'abcmedical', 'abd','aca','af','ad',
                            'yes', 'no','yield', 'yo', 'yr', 'yrs', 'yy', 'yyyy','zzz','xyz','xx', 'xxx',
                        'et', 'etc','ii', 'iih', 'iii','xii','xl','xq', 'xr']
        #add new stopwords
        for word in new_stop_words:
            nlp.Defaults.stop_words.add(word)
            nlp.vocab[word].is_stop=True

        len(nlp.Defaults.stop_words)    
        for word in medical_stop_words:
            nlp.Defaults.stop_words.add(word)
            nlp.vocab[word].is_stop=True
        len(nlp.Defaults.stop_words)

        text=nlp(document)
        new_text=""
        for w in nlp(text):
            if w.text.lower() not in nlp.Defaults.stop_words:
                new_text += w.text.lower() + " " 

        return new_text

    def predict(self, document):
        preprocessed_input = self.preprocess_input(document)
        # Transform input using vectorizer
        vectorized_input = self.vectorizer.transform([preprocessed_input])
        # Make prediction using the model
        prediction = self.model.predict(vectorized_input)
        return prediction[0]

if __name__ == "__main__":
    model_path = "ml_artifacts/voting_classifier_model.joblib"
    vectorizer_path = "ml_artifacts/tfidf_vectorizer.joblib"
    ml_model = ClinicalDocumentClassifier(model_path, vectorizer_path)
    document = "This is a test document."
    prediction = ml_model.predict(document)
    print("Prediction:", prediction)