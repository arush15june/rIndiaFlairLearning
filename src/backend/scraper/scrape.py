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
"""

import os
import logging
import argparse

import praw

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.DEBUG)

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
        
        logging.info(f'Submission: {self.submission.title} {self.submission.url}')
                        
    def get_submission_by_url(self, url):
        return self.reddit.submission(url=url)

    def get_submission_by_id(self, id):
        return self.reddit.submission(id=id)
        

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

        try:
            self._subreddit = self.reddit.subreddit(subreddit)
        except:
            raise InvalidSubredditError()
    
    """ 
    _get_submission is a generator for top submissions in self._subreddit 

    :kwarg str limit: limit on submissions to extract.
    """
    def _get_submissions(self, *args, **kwargs):
        limit = kwargs.get("limit")
        for submission in self._subreddit.top(limit=limit):
            yield submission
    
    def _scrape_submission(self, submission, *args, **kwargs):
        submission_scraper = RedditSubmissionScraper(submission=submission)
        
    def scrape(self, *args, **kwargs):
        logging.info(f'Subreddit: {self._subreddit}')
        for submission in self._get_submissions():
            self._scrape_submission(submission)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Reddit Thread Scraper")
    # parser.add_argument("subreddit", type=str, default=DEFAULT_SUBREDDIT, help="Subreddit to scrape.")
    sub_scraper = RedditSubredditScraper()
    sub_scraper.scrape()