import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from nltk.tokenize import word_tokenize
from gensim.models import FastText

# Read the train essays .csv
train_essays_df = pd.read_csv('../data/all_train_essays.csv')
essays_cols = train_essays_df.columns.tolist()
# Separate the essays and the labels
essays_df = train_essays_df[essays_cols[2]]
labels = train_essays_df[essays_cols[3]]

# Test essays from the given test set
test_essays_df = pd.read_csv('../data/test_essays.csv')
test_essays = test_essays_df['text']

# Create a FastText model for the essays
essays = essays_df.apply(word_tokenize).values
test_essays = test_essays.apply(word_tokenize).values

ft_model = FastText(essays, epochs=3, vector_size=500, window=3, min_count=2, workers=10)

def get_FT_vector(doc):
    vectors = [ft_model.wv[word] for word in doc if word in ft_model.wv]
    return sum(vectors) / len(vectors) if vectors else [0] * 500

essays_df = pd.DataFrame({'essays': essays})
# Apply the function to get document vectors for each essay
X = essays_df['essays'].apply(get_FT_vector)
X = pd.DataFrame(X.tolist())
X = np.array(X)
y = np.array(labels)

test_df = pd.DataFrame({'test_essays': test_essays})
test_essays_X = test_df['test_essays'].apply(get_FT_vector)
test_essays_X = pd.DataFrame(test_essays_X.tolist())

# # Feature extraction
# vectorizer = TfidfVectorizer(ngram_range=(1, 4), max_features=5000)
# X = vectorizer.fit_transform(essays)
# test_essays_X = vectorizer.transform(test_essays)

# Model selection
rf_classifier = RandomForestClassifier(n_estimators=2000, random_state=42)

# Perform 10-fold cross-validation
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

for train_index, test_index in kf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Train the SVM model
    rf_classifier.fit(X_train, y_train)

    # Predictions
    y_pred = rf_classifier.predict(X_test)
    y_prob = rf_classifier.predict_proba(X_test)[:, 1]  # Probability of the positive class

    # Calculate metrics
    accuracy_scores.append(accuracy_score(y_test, y_pred))
    precision_scores.append(precision_score(y_test, y_pred))
    recall_scores.append(recall_score(y_test, y_pred))
    f1_scores.append(f1_score(y_test, y_pred))

# Print the final metrics
print(f"Accuracy: {np.mean(accuracy_scores):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Predict the probabilities for the given test set
predicted_test_set = rf_classifier.predict(test_essays_X)
results_df = pd.DataFrame({'id':test_essays_df['id'], 'generated': predicted_test_set})
results_df = results_df.set_index('id')
print(results_df)

# Save the results to a CSV file
results_df.to_csv('data/submission_RF.csv')