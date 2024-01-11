import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm


def train_model(id, essays, labels):
    """

    :param id: int
    :param essays: series of strings
    :param labels: series of int (0,1)
    :return: float (probability)
    """
    X_test = essays[id]
    y_test = labels[id]
    essays.pop(id)
    labels.pop(id)

    # Feature extraction
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=5000)
    X_train = vectorizer.fit_transform(essays)
    y_train = np.array(labels)
    X_test = vectorizer.transform([X_test])

    # Model selection
    rf_classifier = RandomForestClassifier(n_estimators=50, random_state=43)

    # Train the RF model
    rf_classifier.fit(X_train, y_train)
    # Predictions
    y_pred = rf_classifier.predict(X_test)
    y_prob = rf_classifier.predict_proba(X_test)[:, 1] # Probability of the positive class
    prob = y_prob[0]

    return prob


# Read the train essays .csv
essays_df = pd.read_csv('data/all_train_essays.csv')
essays_cols = essays_df.columns.tolist()

# Separate the essays and the labels
essays = essays_df[essays_cols[2]]
labels = essays_df[essays_cols[3]]


LLM_index = [i for i, essay in enumerate(essays_df[essays_cols[2]])
                if essays_df.loc[i, essays_cols[3]] == 1]

LLM_proba = []
with tqdm(total=len(LLM_index)) as pbar:
    for id in LLM_index:
        LLM_proba.append(train_model(id, essays, labels))
        pbar.update(1)  # Update the progress bar
