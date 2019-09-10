import pandas as pd

from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

import os

from model.train import _concatenate_features

DEFAULT_LE_PATH = './model/flair_encoding.pkl'
DEFAULT_MODEL_PATH = './model/model_svc.pkl'
DEFAULT_VECTORIZER_PATH = './model/model_tfidf.pkl'

def joblib_load(path):
    return joblib.load(path)

def joblib_dump(obj, path):
    joblib.dump(obj, path)

class SVCPredictor(object):
    def __init__(self, model_path=DEFAULT_MODEL_PATH, vectorizer_path=DEFAULT_VECTORIZER_PATH, *args, **kwargs):
        self.model = joblib_load(model_path)    
        vectorizer = joblib_load(vectorizer_path)
        self.vectorizer = vectorizer 
        self.le = self._load_labelencoder()
    
    @staticmethod
    def _load_labelencoder(path=DEFAULT_LE_PATH, *args, **kwargs):
        return joblib_load(path)
            
    def _predict(self, features):
        feature_arr = self.vectorizer.transform(features).toarray()
        return self.model.predict(feature_arr)        

predictor = SVCPredictor()

def _submission_to_df(scraped_submission):
    return pd.DataFrame(
        {
            'permalink': [scraped_submission['permalink']],
            'title': [scraped_submission['title']],
            'comments': [scraped_submission['comments']],
            'selftext': [scraped_submission['selftext']],
        }
    )

def _submission_df_to_features(df, *args, **kwargs):
    return _concatenate_features(df, *args, **kwargs)

def predict_flair(scraped_submission):
    df = _submission_to_df(scraped_submission)
    features = _submission_df_to_features(df, ["title", "permalink", "selftext", "comments"])
    print(features)
    return predictor.le.inverse_transform(predictor._predict(features))