"""
    Scrape reddit threads and subreddits via PRAW.
        - Collectable parameters
            - Submissions
                - permalink
                - upvotes at the time of scrape.
                - post timestamp
                - title
                - poster
                - subreddit
                - category flair
                - number of comments
                - post content
                - Every comment on the thread.
                    - comment poster
                    - comment content
                    - comment timestamp
                    - comment upvotes at time of collection
            - Subreddit
                - Scrape <Submissions> on the subreddit.

TODO: Push data to Mongo collection.
"""

import sys
import os
import logging
import argparse
from pprint import pprint

import praw
import pymongo

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)

# Flairs from r/India sidebar.
RINDIA_FLAIRS = [
    "AskIndia",
    "Non-Political",
    "[R]eddiqutte",
    "Scheduled",
    "Photography",
    "Science/Technology",
    "Sports",
    "Food"
]

DEFAULT_SUBREDDIT = "india"

DEFAULT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
DEFAULT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
DEFAULT_USER_AGENT = os.getenv("REDDIT_USER_AGENT") or "script/linux:"
DEFAULT_SUBMISSION_LIMIT = None

DEFAULT_MONGO_HOST = "localhost"
DEFAULT_MONGO_PORT = 27017

def pprint_exporter(submission):
    pprint(submission)

DEFAULT_SCRAPE_EXPORTER = pprint_exporter

class InvalidRedditAuthError(Exception):
    pass
    
class InvalidRedditDefaultAuthError(InvalidRedditAuthError):
    pass

class InvalidParameterAuthError(InvalidRedditAuthError):
    pass

class InvalidSubredditError(Exception):
    pass

class RedditScraper(object):
    """ 
    Abstract base to create praw.Reddit instances. 
    """
    def __init__(self, *args, **kwargs):
        """  
            Each instance will run its own Reddit instance.
        """
        
        client_id = kwargs.get("client_id", DEFAULT_CLIENT_ID)
        client_secret = kwargs.get("client_secret", DEFAULT_CLIENT_SECRET)
        user_agent = kwargs.get("user_agent", DEFAULT_USER_AGENT)
        if client_id is not None and client_secret is not None:
            try:
                self.reddit = praw.Reddit(
                    client_id=client_id,
                    client_secret=client_secret,
                    user_agent=user_agent
                )
            except:
                raise InvalidParameterAuthError()
        else:
            """ opt for praw.ini """
            try:
                self.reddit = praw.Reddit(
                    user_agent=user_agent
                )
            except:
                raise InvalidRedditDefaultAuthError()

class RedditSubmissionScraper(RedditScraper):
    """
    Scrape a submission off reddit 
        - by url
        - by id
        - by praw.Submission
    """
    def __init__(self, *args, **kwargs):
        super(RedditSubmissionScraper, self).__init__(*args, **kwargs)

        url = kwargs.get('url')
        if url is not None:
            self.submission = self.get_submission_by_url(url)

        submission_id = kwargs.get('submission_id')
        if submission_id is not None:
            self.submission = self.get_submission_by_id(submission_id)
                
        submission_instance = kwargs.get('submission')
        if submission_instance is not None:
            self.submission = submission_instance
        
        self.scraped = {}
        self.data_extractors = {
            "permalink": self._extract_permalink,
            "upvotes": self._extract_upvotes,
            "timestamp": self._extract_timestamp,
            "title": self._extract_title,
            "poster": self._extract_poster,
            "subreddit": self._extract_subreddit,
            "flair": self._extract_flair,
            "selfpost": self._extract_selfpost,
            "selftext": self._extract_selftext,
            "comments": self._extract_comments,
        }
        
        logging.info(f'Scraping Submission: {self.submission.title} {self.submission.permalink}')

    def get_submission_by_url(self, url):
        return self.reddit.submission(url=url)

    def get_submission_by_id(self, id):
        return self.reddit.submission(id=id)

    def _extract_permalink(self, *args, **kwargs):
        return self.submission.permalink

    def _extract_upvotes(self, *args, **kwargs):
        return self.submission.score

    def _extract_timestamp(self, *args, **kwargs):
        return self.submission.created_utc

    def _extract_title(self, *args, **kwargs):
        return self.submission.title

    def _extract_poster(self, *args, **kwargs):
        return self.submission.author.name

    def _extract_subreddit(self, *args, **kwargs):
        return self.submission.subreddit.name

    def _extract_flair(self, *args, **kwargs):
        return self.submission.link_flair_text

    def _extract_selfpost(self, *args, **kwargs):
        return self.submission.is_self

    def _extract_selftext(self, *args, **kwargs):
        return self.submission.selftext

    def _extract_comments(self, *args, **kwargs):
        self.submission.comments.replace_more(limit=0)
        self.submission.comment_sort = 'top'
        comments = self.submission.comments.list()
        return [ 
            { 
                "author": comment.author.name,
                "body": comment.body,
                "created_utc": comment.created_utc,
                "id": comment.id,
                "permalink": comment.permalink,
                "upvotes": comment.score
            }
            for comment in comments if comment.author
        ]

    def extract_data(self, *args, **kwargs):
        for k, v in self.data_extractors.items():
            data = v()
            self.scraped[k] = data
        
class RedditSubredditScraper(RedditScraper):
    """
    Scrape a reddit subreddit by sub name. 
        sub_scraper = RedditSubredditScraper(subreddit="india")
        sub_scraper.scrape()

    - Get subreddit
    - Scrape all data of the submissions.
    """
    def __init__(self, *args, **kwargs):
        """  
            Each instance will run its own Reddit instance.
        """
        super(RedditSubredditScraper, self).__init__(*args, **kwargs)

        subreddit = kwargs.get("subreddit", DEFAULT_SUBREDDIT)
        self._export = kwargs.get('export', False)
        self.exporter = kwargs.get('exporter', DEFAULT_SCRAPE_EXPORTER)

        try:
            self._subreddit = self.reddit.subreddit(subreddit)
        except:
            raise InvalidSubredditError()
    
    """ 
    _get_submission is a generator for top submissions in self._subreddit 

    :kwarg str limit: limit on submissions to extract.
    """
    def _get_submissions(self, *args, **kwargs):
        for submission in self._subreddit.hot(*args, **kwargs):
            yield submission
    
    def _scrape_submission(self, submission, *args, **kwargs):
        submission_scraper = RedditSubmissionScraper(submission=submission)
        submission_scraper.extract_data()
        logging.info(submission_scraper.scraped["flair"])
        if self._export:
            self.exporter(submission_scraper.scraped)
        
    def scrape(self, *args, **kwargs):
        logging.info(f'Subreddit: {self._subreddit}')
        for submission in self._get_submissions(*args, **kwargs):
            self._scrape_submission(submission)

def get_mongo_client(host, port, *args, **kwargs):
    client = pymongo.MongoClient(host, port)
    return client

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reddit Thread Scraper")
    parser.add_argument("--subreddit", type=str, default=DEFAULT_SUBREDDIT, help="Subreddit to scrape.")
    parser.add_argument("--limit", type=int, default=DEFAULT_SUBMISSION_LIMIT, help="Numbers of submission to scrape.")
    parser.add_argument("--export", action='store_true', help="Export scraped results.")
    parser.add_argument("--host", type=str, default=DEFAULT_MONGO_HOST, help="Mongo Host")
    parser.add_argument("--port", type=int, default=DEFAULT_MONGO_PORT, help="Mongo Port")

    args = parser.parse_args()


    client = get_mongo_client(args.host, args.port)
    logging.info(f'Mongo client: {client}')
    reddit_db = client.reddit
    submissions = reddit_db.submissions
    def mongo_submission_exporter(submission, *args, **kwargs):
        submissions.insert_one(submission)
        
    try:
        sub_scraper = RedditSubredditScraper(
            subreddit=args.subreddit,
            export=args.export,
            exporter=mongo_submission_exporter
        )
        sub_scraper.scrape(limit=args.limit)
    except InvalidRedditAuthError:
        logging.info("Authentication failure.")
        sys.exit(1)
