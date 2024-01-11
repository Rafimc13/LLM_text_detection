import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


# Add more essays for training based on regressor plot
add_essays_df = pd.read_csv('../data/train_drcat_01.csv')
add_essays_cols = add_essays_df.columns.tolist()

# Separate the essays and the labels
add_essays_students = [essay for i, essay in enumerate(add_essays_df[add_essays_cols[0]])
                       if add_essays_df.loc[i, add_essays_cols[1]] == 0]
add_essays_LLM = [essay for i, essay in enumerate(add_essays_df[add_essays_cols[0]])
                       if add_essays_df.loc[i, add_essays_cols[1]] == 1]

add_essays = add_essays_students[:2250] + add_essays_LLM[:2250]
add_labels = [0 for _ in range(2250)] + [1 for _ in range(2250)]

# Feature extraction
vectorizer = TfidfVectorizer(ngram_range=(1, 4), max_features=5000)
X = vectorizer.fit_transform(add_essays)
y = np.array(add_labels)