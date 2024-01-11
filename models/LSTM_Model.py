import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import FastText
from gensim.utils import simple_preprocess
from keras.models import Sequential
from keras.layers import Embedding, Dense, LSTM, Dropout, BatchNormalization



train_essays_df = pd.read_csv('../data/all_train_essays.csv')
essays_cols = train_essays_df.columns.tolist()

# Separate the essays and the labels
essays = train_essays_df[essays_cols[2]]
labels = train_essays_df[essays_cols[3]]

# Test essays from the given test set
test_essays_df = pd.read_csv('../data/test_essays.csv')
test_essays = test_essays_df['text']


add01_essays = pd.read_csv("../data/train_drcat_01.csv")

# Load pre-trained FastText word vectors
fasttext_model_path = '../data/cc.en.300.bin.gz'
fasttext_model = FastText.load_fasttext_format(fasttext_model_path)

# Preprocess and tokenize the text in the DataFrame
tokenized_lines = [simple_preprocess(line) for line in essays.astype(str)]

# Vectorize the dataset
vectorized_dataset = []

for tokens in tokenized_lines:
    # Get the vector for each token and take the mean
    vectors = [fasttext_model.wv[word] for word in tokens if word in fasttext_model.wv]
    if vectors:
        # Take the mean of vectors to represent the whole line/document
        mean_vector = sum(vectors) / len(vectors)
        vectorized_dataset.append(mean_vector)
    else:
        # If none of the tokens are in the FastText model, add a placeholder vector
        vectorized_dataset.append([0.0] * fasttext_model.vector_size)


def create_model(max_words, max_len):
    """
    Create a deep network model with LTSM layers
    :param max_words: int,
    :param max_len: int,
    :return: class keras.src.engine
    """
    model = Sequential()
    model.add(Embedding(max_words, 64, input_length=max_len))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(64))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
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

    model = create_model(8000, 500)
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

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


# # Predict the probabilities for the given test set
# predicted_test_set = model.predict(test_essays_X)
# results_df = pd.DataFrame({'id':test_essays_df['id'], 'generated': predicted_test_set[:, 0]})
# results_df = results_df.set_index('id')
# print(results_df)
#
# # Save the results to a CSV file
# results_df.to_csv('data/submission_DNN.csv')