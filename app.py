from flask import Flask, render_template, request
from werkzeug.datastructures import UpdateDictMixin
from flask_sqlalchemy import SQLAlchemy
import numpy as np
from sklearn.linear_model import LogisticRegression
from os import getenv
import tweepy
import spacy
# from .models import DB, User, Tweet
# from .twitter import add_or_update_user, get_all_usernames
# from .predict import predict_user

# Create a DB Object
DB = SQLAlchemy()

# Get API keys from .env
KEY = getenv('TWITTER_API_KEY')
SECRET = getenv('TWITTER_API_KEY_SECRET')

# Connect to the Twitter API
TWITTER_AUTH = tweepy.OAuthHandler(KEY, SECRET)
TWITTER = tweepy.API(TWITTER_AUTH)

# Load our pretrained SpaCy Word Embeddings model
nlp = spacy.load('my_model/')

def create_app():
    # initializes our app
    app = Flask(__name__)

    # Database configurations
    app.config["SQLALCHEMY_DATABASE_URI"] = 'sqlite:///db.sqlite3'
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    # Give our APP access to our database
    DB.init_app(app)

    # Listen to a "route"
    # '/' is the home page route
    @app.route('/')
    def root():
        # Query users to display on the home page
        return render_template('base.html', title="Home", users=User.query.all())

    @app.route('/update')
    def update():
        '''update all users'''
        usernames = get_all_usernames()
        for username in usernames:
            add_or_update_user(username)
        return "All users have been updated"
    
    @app.route('/reset')
    def reset():
        # remove everything from the database
        DB.drop_all()
        # Creates the database file initially.
        DB.create_all()
        return render_template('base.html', title='Reset Database')


    # API ENDPOINTS (Querying and manipulating data in a database)

    @app.route('/user', methods=['POST'])
    @app.route('/user/<name>', methods=['GET'])
    def user(name=None, message=''):
        # request.values is pulling data from the html
        # use the username from the URL (route)
        # or grab it from the dropdown menu
        name = name or request.values['user_name']

        #If the user exists in the db already, update it, and query for it
        
        try:
            if request.method == 'POST':
                add_or_update_user(name)
                message = f"User {name} Successfully Added!"

            #From the user that was just added / Updated 
            # get their tweets to display on the /user/<name> page
            tweets = User.query.filter(User.username == name).one().tweets

        except Exception as e:
            message = f"Error adding {name}: {e}"

            tweets = []

        return render_template('user.html', title=name, tweets=tweets, message=message)


    @app.route('/compare', methods=['POST'])
    def compare():
        user0 , user1 = sorted([request.values['user0'], request.values['user1']])

        if user0 == user1:
            message = "Cannot compare users to themselves!"

        else:
            prediction = predict_user(user0, user1, request.values['tweet_text'])
            if prediction == 0:
                predicted_user = user0
                non_predicted_user = user1
            else:
                predicted_user = user1
                non_predicted_user = user0
            message = f"'{request.values['tweet_text']}' is more likely to be said by '{predicted_user}' than by '{non_predicted_user}'"

        return render_template('prediction.html', title='Prediction', message=message)
        
    return app

# Make a User table by creating a User class
class User(DB.Model):
    '''Creates a User Table with SQLAlchemy'''
    # id column
    id = DB.Column(DB.BigInteger, primary_key=True)
    # username column
    username = DB.Column(DB.String, nullable=False)
    # keeps track of id for the newest tweet said by user
    newest_tweet_id = DB.Column(DB.BigInteger)
    # We don't need a tweets attribute because this is 
    # automatically being added by the backref in the Tweet model.
    # tweets = DB.column(DB.String)
    def __repr__(self):
        return f'<User: {self.username}>'

# Make a Tweet table by creating a Tweet class
class Tweet(DB.Model):
    '''Creates a Tweet Table with SQLAlchemy'''
    # id column
    id = DB.Column(DB.BigInteger, primary_key=True)
    # text column
    text = DB.Column(DB.Unicode(300)) # Unicode allows for both text and links and emojis, etc.
    # Create a relationship between a tweet and a user
    user_id = DB.Column(DB.BigInteger, DB.ForeignKey('user.id'), nullable=False)
    # Finalizing the relationship making sure it goes both ways. 
    user = DB.relationship('User', backref=DB.backref('tweets', lazy=True))
    # be able to include a word embedding on a tweet
    vect = DB.Column(DB.PickleType, nullable=False)

    def __repr__(self):
        return f'<Tweet: {self.text}>'
    
def predict_user(user0_name, user1_name, hypo_tweet_text):
    '''Take in two usernames, 
    query for the tweet vectorizations for those two users, 
    compile the vectorizations into an X matrix
    generate a numpy array of labels (y variable)
    fit a lostic regression using X and y
    vectorize the hypothetical tweet text
    generate and return a prediction'''

    # Query for our two users
    user0 = User.query.filter(User.username == user0_name).one()
    user1 = User.query.filter(User.username == user1_name).one()

    # Get the tweet vectorizations for the two users
    user0_vects = np.array([tweet.vect for tweet in user0.tweets])
    user1_vects = np.array([tweet.vect for tweet in user1.tweets])

    # Combine the vectors into an X Matrix
    X = np.vstack([user0_vects, user1_vects])
    # Generate labels and 0s and 1s for a y vecor
    y = np.concatenate([np.zeros(len(user0.tweets)), np.ones(len(user1.tweets))])

    # fit our Logistic regression model
    log_reg = LogisticRegression().fit(X,y)

    # vectorize our hypothetical tweet text
    hypo_tweet_vect = vectorize_tweet(hypo_tweet_text)

    # return the predicted label: (0 or 1)
    # reshaping to make a 2D NumPy array from a 1D NumPy array
    return log_reg.predict(hypo_tweet_vect.reshape(1,-1))

# Turn tweet text into word embeddings
def vectorize_tweet(tweet_text):
    return nlp(tweet_text).vector

# function to query the API for a user 
# and add the user to the DB.
def add_or_update_user(username):
    """
    Gets twitter user and tweets from twitter DB
    Gets user by "username" parameter.
    """
    try:
        # gets back twitter user object
        twitter_user = TWITTER.get_user(screen_name=username)
        # Either updates or adds user to our DB
        db_user = (User.query.get(twitter_user.id)) or User(
            id=twitter_user.id, username=username)
        DB.session.add(db_user)  # Add user if don't exist

        # Grabbing tweets from "twitter_user"
        tweets = twitter_user.timeline(
            count=200,
            exclude_replies=True,
            include_rts=False,
            tweet_mode="extended",
            since_id=db_user.newest_tweet_id
        )

        # check to see if the newest tweet in the DB is equal to the newest tweet from the Twitter API, if they're not equal then that means that the user has posted new tweets that we should add to our DB. 
        if tweets:
            db_user.newest_tweet_id = tweets[0].id

        # tweets is a list of tweet objects
        for tweet in tweets:
            # type(tweet) == object
            # Turn each tweet into a word embedding. (vectorization)
            tweet_vector = vectorize_tweet(tweet.full_text)
            db_tweet = Tweet(
                id=tweet.id,
                text=tweet.full_text,
                vect=tweet_vector
            )
            db_user.tweets.append(db_tweet)
            DB.session.add(db_tweet)

    except Exception as e:
        print("Error processing {}: {}".format(username, e))
        raise e

    else:
        DB.session.commit()


def get_all_usernames():
    '''get the usernames of all users that are already in the database'''
    usernames = []
    Users = User.query.all()
    for user in Users:
        usernames.append(user.username)
    
    return usernames