import nltk
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')


def preprocess_text(reviews):
  stop_words = set(stopwords.words('english'))
  processed_reviews = []

  for review in reviews:
    # lowercase and remove punctuation
    review = review.lower().translate(str.maketrans('', '', string.punctuation))
    # tokenize
    tokens = word_tokenize(review)
    # remove stopwords
    tokens = [token for token in tokens if token not in stop_words]
    processed_reviews.append(' '.join(tokens))

  return processed_reviews


def train_sentiment_model(reviews, labels):
  processed_reviews = preprocess_text(reviews)

  # convert text to features
  vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
  X = vectorizer.fit_transform(processed_reviews)
  y = np.array(labels)

  # model and hyperparameter grid
  model = LogisticRegression(max_iter=1000)
  param_grid = {
    'C': [0.1, 1.0, 10.0],
    'solver': ['liblinear', 'lbfgs']
  }

  # grid search for optimization
  grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
  grid_search.fit(X, y)

  print(f"Best parameters: {grid_search.best_params_}")
  print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

  return grid_search.best_estimator_, vectorizer


def predict_sentiment(model, vectorizer, test_reviews):
  processed_test = preprocess_text(test_reviews)
  X_test = vectorizer.transform(processed_test)
  predictions = model.predict(X_test)
  return predictions


if __name__ == "__main__":
  # sample training data
  train_reviews = [
    "Great movie, loved it!",
    "Really fun and exciting!",
    "A blast! I was on the edge of my seat!",
    "A riot, just amazing!",
    "I want to watch this movie again and again!",
    "Terrible, waste of time.",
    "Awful plot, boring.",
    "How could someone approve of this terrible movie?",
    "Honestly, I am not sure what was happening.",
    "The pacing was atrocious. Overall pretty dull."
  ]
  train_labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

  # train model
  model, vectorizer = train_sentiment_model(train_reviews, train_labels)

  # test data
  test_reviews = ["Amazing film, highly recommend!", "Not worth watching, dull."]
  test_labels = [1, 0]

  # predict and evaluate
  predictions = predict_sentiment(model, vectorizer, test_reviews)
  accuracy = accuracy_score(test_labels, predictions)

  print(f"Test predictions: {predictions}")
  print(f"Test accuracy: {accuracy:.4f}")
