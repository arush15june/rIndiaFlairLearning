import numpy as np
import pandas as pd
import json

import logging
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
    return _extract_features(json2df(path).dropna())

if __name__ == '__main__':
    df = _get_cleaned_dataframe()
    
    logging.info(df)
    logging.info(df.columns)
    logging.info("Upvotes")
    logging.info(df['title'])
    logging.info("Comments")
    logging.info(df['comments'][0][0])
    logging.info("Flairs")
    logging.info(set(df['flair']))