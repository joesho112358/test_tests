import numpy as np
from src.ai.scripts.movie_reviews import preprocess_text, train_sentiment_model, predict_sentiment


def test_preprocess_text():
    reviews = [
        "Great movie, loved it!",
        "Really fun and exciting!",
        "A blast! I was on the edge of my seat!"
    ]
    expected_output = [
        "great movie loved",
        "really fun exciting",
        "blast edge seat"
    ]
    processed_reviews = preprocess_text(reviews)
    assert processed_reviews == expected_output


def test_train_sentiment_model():
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

    model, vectorizer = train_sentiment_model(train_reviews, train_labels)

    # check if the model and vectorizer are not None
    assert model is not None
    assert vectorizer is not None

    # check if the model has been fitted
    assert hasattr(model, 'coef_')


def test_predict_sentiment():
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

    model, vectorizer = train_sentiment_model(train_reviews, train_labels)

    test_reviews = ["Amazing film, highly recommend!", "Not worth watching, dull."]
    # expected predictions based on the training data
    expected_predictions = np.array([1, 0])

    predictions = predict_sentiment(model, vectorizer, test_reviews)

    assert len(predictions) == len(test_reviews)
    assert np.array_equal(predictions, expected_predictions)
