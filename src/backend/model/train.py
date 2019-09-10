import argparse
import json
import logging
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import joblib
from sklearn.preprocessing import LabelEncoder

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline


logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)

DEFAULT_DATA_JSON = 'r_india_submissions.json'

""" 
    Data Representation
    
    1 submission
    | permalink | selfpost | selftext | upvotes | timestamp | title | flair | poster | comments |

    1 comment
    | author | body | created_utc | id | permalink | upvotes |  

    Problem -> Flair Classification
        (Multiclass Text Classification)
"""

"""
    json2df parses mongoexport json to dataframe.
"""
json2df = (lambda filename: pd.read_json(filename, lines=True))

def _extract_comment(comment_dict):
    return {
        "author": comment_dict["author"].rstrip().strip(),
        "body": comment_dict["body"].rstrip().strip(),
        "created_utc": comment_dict["created_utc"]["$numberInt"],
        "id": comment_dict["id"],
        "permalink": comment_dict["permalink"].rstrip().strip(),
        "upvotes": comment_dict["upvotes"]["$numberInt"]
    }

def _extract_comments(comment_list):
    return [_extract_comment(comment) for comment in comment_list]

def _extract_features(dataframe):
    feature_dict = {
        'permalink': [],
        'selfpost': [],
        'selftext': [],
        'upvotes': [],
        'timestamp': [],
        'title': [],
        'flair': [],
        'poster': [],
        'comments': [],
    }

    permalink_strat = (lambda row: row["permalink"].rstrip().strip())
    selfpost_strat = (lambda row: row["selfpost"])
    selftext_strat = (lambda row: row["selftext"].rstrip().strip())
    upvotes_strat = (lambda row: row["upvotes"]["$numberInt"])
    flair_strat = (lambda row: row["flair"])
    timestamp_strat = (lambda row: row["timestamp"])
    title_strat = (lambda row: row["title"].rstrip().strip())
    poster_strat = (lambda row: row["poster"])
    comments_strat = (lambda row: _extract_comments(row["comments"]))

    transform_strats = {
        'permalink': permalink_strat,
        'selfpost': selfpost_strat,
        'selftext': selftext_strat,
        'upvotes': upvotes_strat,
        'timestamp': timestamp_strat,
        'title': title_strat,
        'flair': flair_strat,
        'poster': poster_strat,
        'comments': comments_strat,
    }

    for index, row in dataframe.iterrows():
        for k in feature_dict:
            feature_dict[k].append(transform_strats[k](row))
    
    return pd.DataFrame(feature_dict)

def _get_cleaned_dataframe(path=DEFAULT_DATA_JSON):
    """ 
    Drop None rows
    Drop 4 classes 

    - Demonetization
    - Illegal Content
    - | Low-effort Self Post |
    - | Not Original/Relevant Title |

    """
    df = _extract_features(json2df(path).dropna())
    drop_classes = [
            "Demonetization",
            "Illegal Content",
            "| Low-effort Self Post |",
            "| Not Original/Relevant Title |"
        ]

    for cls in drop_classes:
        df = df[df["flair"] != cls]
    
    return df

def _concatenate_features(df, features):
    """ 
        Concatenate selected features from df.
    """
    feature_dict = {
        "feature" : []
    }
    for index, row in df.iterrows():
        data = ""
        for feature in features:
            if feature == "comments":
                data += "".join([comment["body"] for comment in row["comments"]])
            else:
                data += row[feature]
        feature_dict["feature"].append(data)
            
    return pd.DataFrame(feature_dict)

def _transform_tfidf(feature):
    tfidf = TfidfVectorizer(
        sublinear_tf=True, 
        min_df=5, 
        norm='l2', 
        encoding='latin-1', 
        ngram_range=(1, 2), 
        stop_words='english'
    )

    features = tfidf.fit_transform(feature).toarray()
    return features

def joblib_load(path):
    return joblib.load('rb')

def joblib_dump(obj, path):
    joblib.dump(obj, path)

def analyze_func(x):
    return x    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Reddit Thread Scraper")
    parser.add_argument("--json", type=str, default=DEFAULT_DATA_JSON, help="Dataset JSON, from mongoexport.")
    parser.add_argument("--output", type=str, default=DEFAULT_DATA_JSON, help="Model output file.")
    parser.add_argument("--debug", action='store_true', help="Debug logging.")
    
    args = parser.parse_args()

    """ Unique Classes """
    df = _get_cleaned_dataframe(args.json)
    classes = df.flair.unique()
    logging.info(f'Unique Classes: {classes}')
    
    """ Class Imbalance """
    class_imbalance = df.groupby('flair').title.count()
    logging.info('Class Imbalance')
    logging.info(class_imbalance)

    """ Feature Frame and Label Encoding """
    concat_df = _concatenate_features(df, ["title", "permalink", "comments", "selftext"])
    feature_df = concat_df
    feature_df = feature_df.assign(flair=df.flair)
    logging.info(feature_df)
    
    le = LabelEncoder()
    flair = feature_df.flair
    le = le.fit(df.flair)
    labels = le.transform(df.flair)
    feature_df.flair = labels

    flair_encoder_filename = "flair_encoding.pkl"
    joblib_dump(le, flair_encoder_filename)
    logging.info("Dumped flair encodings: {flair_encoder_filename}")

    # """ TF-IDF Features """
    # # https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f
    # features = _transform_tfidf(concat_df.feature)
    # logging.info(f'TF-IDF Feature Vector:  {features.shape}')

    """ Model Pipeline """
    svc_tfidf = Pipeline(
            [
                ("tfidf_vectorizer", TfidfVectorizer(
                    sublinear_tf=True, 
                    min_df=5, 
                    norm='l2', 
                    encoding='latin-1', 
                    ngram_range=(1, 2), 
                    stop_words='english'
                    )
                ), 
                ("multinomial nb", MultinomialNB())
            ]
        )
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
        feature_df["feature"], 
        feature_df["flair"], 
        df.index, 
        test_size=0.33,
        random_state=0
    )
    logging.info('Training SVC')
    svc_tfidf.fit(X_train, y_train)

    y_pred = svc_tfidf.predict(X_test)
    logging.info(classification_report(y_test, y_pred, target_names=classes))
    
    conf_mat = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(conf_mat, annot=True, fmt='d',
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    vectorizer = svc_tfidf.named_steps['tfidf_vectorizer'] 
    joblib_dump(svc_tfidf.named_steps['multinomial nb'], 'model_svc.pkl')
    joblib_dump(vectorizer, 'model_tfidf.pkl')
    logging.info('dumped model_svc.pkl')
    logging.info('dumped model_tfidf.pkl')