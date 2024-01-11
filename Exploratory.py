import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

train_essays = pd.read_csv('data/all_train_essays.csv')

# Calculate the length of each essay and create a new column
train_essays['essay_length'] = train_essays['text'].apply(len)

# Distribution of essay lengths for student essays
sns.histplot(train_essays[train_essays['generated'] == 0]['essay_length'], color="skyblue", label='Student Essays', kde=True)

# Distribution of essay lengths for LLM generated essays
sns.histplot(train_essays[train_essays['generated'] == 1]['essay_length'], color="red", label='LLM Generated Essays', kde=True)

plt.title('Distribution of Essay Lengths')
plt.xlabel('Essay Length (Number of Characters)')
plt.ylabel('Frequency')
plt.legend()
plt.show()

sns.boxplot(x='generated', y='essay_length', data=train_essays)
plt.title('Comparison of Essay Lengths by Source')
plt.xlabel('Essay Source')
plt.ylabel('Essay Length')
plt.xticks([0, 1], ['Student-written', 'LLM-generated'])
plt.show()


def most_common_words(text, num_words=30, title="Most Common Words"):
    """
     Plot the most common words for each class (student/LLM)
     Removing word with 1-4 chars in order to plot the most important
    :param text: string
    :param num_words: int
    :param title: string
    :return: barplot
    """
    all_text = ' '.join(text).lower()
    words = all_text.split()
    word_freq = Counter(words)
    pattern = re.compile(r'\b\w{1,4}\b')  # Pattern in order to remove the small words (e.g. 'the')
    words_removed = []
    for key, value in word_freq.items():
        if re.match(pattern, key):
            words_removed.append(key)
    for word in words_removed:
        del word_freq[word]
    common_words = word_freq.most_common(num_words)

    # Plot the most common words
    plt.figure(figsize=(12, 6))
    sns.barplot(x=[word for word, freq in common_words], y=[freq for word, freq in common_words])
    plt.title(title)
    plt.xticks(rotation=45)
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

title_student = "Most Common Words in Student Essays"
most_common_words(train_essays[train_essays['generated'] == 0]['text'], title=title_student)
title_LLM = "Most Common Words in LLM-generated Essays"
most_common_words(train_essays[train_essays['generated'] == 1]['text'], title=title_LLM)