from marshmallow import Schema, fields, pprint
from flask import Flask
from flask_restplus import reqparse
from flask_restplus import Resource, Api

from scraper import scrape
import util

def analyze_func(x):
    return x    

api = Api()

app = Flask(__name__)
api.init_app(app)

""" Serialization Schemas """
class SubmissionResponseSchema(Schema):
    """ 
        SubmissionRequestSchema

        :param str submission: permalink
    """
    submission = fields.Str()

@api.route('/submission')
class SubmissionResource(Resource):    
    submission_parser = reqparse.RequestParser()
    submission_parser.add_argument('submission', type=str, help='Submission URL.')

    def post(self):
        submission_args = self.submission_parser.parse_args()
        submission_url = submission_args.get('submission', None)
        print(submission_url)
        if submission_url is None:
            return {
                "error": "URL not found."
            }, 400

        try:
            submission_scraper = scrape.RedditSubmissionScraper(url=submission_url)
            submission_scraper.extract_data()
        except:
            return {
                "error": "Submission not found."
            }, 404

        predicted_flair = util.predict_flair(submission_scraper.scraped)
        submission_scraper.scraped['comments'] = submission_scraper.scraped['comments'][:10] 
        submission_scraper.scraped['prediction'] = predicted_flair[0]

        return submission_scraper.scraped, 200

if __name__ == "__main__":
    app.run(debug=True)