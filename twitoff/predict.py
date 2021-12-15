from .models import User
import numpy as np
from sklearn.linear_model import LogisticRegression
from .twitter import vectorize_tweet


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
