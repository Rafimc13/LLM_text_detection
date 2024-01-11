import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import tqdm as tqdm


train_essays_df = pd.read_csv('../data/best_train_essays.csv')
essays_cols = train_essays_df.columns.tolist()

# Separate the essays and the labels
essays = train_essays_df[essays_cols[2]]
labels = train_essays_df[essays_cols[3]]


# Feature extraction
vectorizer = TfidfVectorizer(ngram_range=(1, 4), max_features=5000)
X = vectorizer.fit_transform(essays)
y = np.array(labels)

# Model selection
rf_classifier = RandomForestClassifier(n_estimators=50, random_state=43)

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


