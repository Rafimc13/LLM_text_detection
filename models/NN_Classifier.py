import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, GlobalAveragePooling1D, Dense


train_essays_df = pd.read_csv('../data/all_train_essays.csv')
essays_cols = train_essays_df.columns.tolist()

# Separate the essays and the labels
essays = train_essays_df[essays_cols[2]]
labels = train_essays_df[essays_cols[3]]

# Test essays from the given test set
test_essays_df = pd.read_csv('../data/test_essays.csv')
test_essays = test_essays_df['text']

# Tokenization
max_words = 10000
max_len = 100

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(essays)
sequences = tokenizer.texts_to_sequences(essays)
X = pad_sequences(sequences, maxlen=max_len)
y = np.array(labels)


# Define the neural network model
def create_model():
    model = Sequential()
    model.add(Embedding(max_words, 16, input_length=max_len))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(32, activation='relu'))  # Adding a new dense layer with 32 units and ReLU activation
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Perform 10-fold cross-validation
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=43)

accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

for train_index, test_index in kf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = create_model()
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

    # Evaluate the model
    y_pred = model.predict(X_test)
    y_pred_binary = np.round(y_pred)

    # Calculate metrics
    accuracy_scores.append(accuracy_score(y_test, y_pred_binary))
    precision_scores.append(precision_score(y_test, y_pred_binary))
    recall_scores.append(recall_score(y_test, y_pred_binary))
    f1_scores.append(f1_score(y_test, y_pred_binary))

# Print the final metrics
print(f"Accuracy: {np.mean(accuracy_scores):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_binary))

# Tokenization for the given test set
tokenizer.fit_on_texts(test_essays)
sequences = tokenizer.texts_to_sequences(test_essays)
test_essays_X = pad_sequences(sequences, maxlen=max_len)

# Predict the probabilities for the given test set
predicted_test_set = model.predict(test_essays_X)
results_df = pd.DataFrame({'id':test_essays_df['id'], 'generated': predicted_test_set[:, 0]})
results_df = results_df.set_index('id')
print(results_df)

# Save the results to a CSV file
results_df.to_csv('data/submission_NN.csv')