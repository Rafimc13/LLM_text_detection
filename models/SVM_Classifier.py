import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

train_essays_df = pd.read_csv('../data/all_train_essays.csv')
essays_cols = train_essays_df.columns.tolist()

# Separate the essays and the labels
essays = train_essays_df[essays_cols[2]]
labels = train_essays_df[essays_cols[3]]

# Test essays from the given test set
test_essays_df = pd.read_csv('../data/test_essays.csv')
test_essays = test_essays_df['text']

# Convert essays to a bag-of-words representation
vectorizer = CountVectorizer(max_features=1000, ngram_range=(1, 4))
X = vectorizer.fit_transform(essays)
y = np.array(labels)
test_essays_X = vectorizer.transform(test_essays)


# Define the SVM model
svm_model = SVC(kernel='linear', probability=True)

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
    svm_model.fit(X_train, y_train)

    # Predictions
    y_pred = svm_model.predict(X_test)
    y_prob = svm_model.predict_proba(X_test)[:, 1]  # Probability of the positive class

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
predicted_test_set = svm_model.predict(test_essays_X)
results_df = pd.DataFrame({'id':test_essays_df['id'], 'generated': predicted_test_set})
results_df = results_df.set_index('id')
print(results_df)

# Save the results to a CSV file
results_df.to_csv('data/submission_SVM.csv')