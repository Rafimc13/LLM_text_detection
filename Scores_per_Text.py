import pandas as pd
from tqdm import tqdm
from nltk.metrics import jaccard_distance
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Read the train essays .csv
essays_df = pd.read_csv('data/all_train_essays.csv')
essays_cols = essays_df.columns.tolist()


def preprocess(text):
    """
    Preprocess the text in order to be clean and correctly formatted
    for calculations
    :param text: string
    :return: set
    """
    tokens = []
    for word in word_tokenize(text):
        if word.isalnum() and word.lower() not in stopwords.words('english'):
            tokens.append(word.lower())

    return set(tokens)


def calculate_similarity(generated_text, student_texts):
    """
    Calculation of similiraty between a text and a list of texts
    :param generated_text: string
    :param student_texts: array
    :return: float1, float2
    """
    generated_text = preprocess(generated_text)
    jaccard_distances = [jaccard_distance(generated_text, student_text) for student_text in student_texts]

    # Maximum and average similarity scores
    max_similarity = max(jaccard_distances)
    avg_similarity = sum(jaccard_distances) / len(jaccard_distances)

    return max_similarity, avg_similarity


student_essays = [essay for i, essay in enumerate(essays_df[essays_cols[2]])
                  if essays_df.loc[i, essays_cols[3]] == 0]

LLM_essays = [{essays_df.loc[i, essays_cols[0]]:essay} for i, essay in enumerate(essays_df[essays_cols[2]])
                if essays_df.loc[i, essays_cols[3]] == 1]

# Proprocess all the student essays iot avoid inside the iteration
student_texts = [preprocess(student_text) for student_text in student_essays]

with tqdm(total=len(LLM_essays)) as pbar:
    for essay in LLM_essays:
        similarity_scores = []
        id, text = list(essay.items())[0]
        max_similarity, avg_similarity = calculate_similarity(text, student_texts)
        similarity_scores.append({'id': id, 'max_similarity': max_similarity, 'avg_similarity': avg_similarity})

        # print(f"The Maximum Similarity for generated essay with id {key}: {max_similarity}")
        # print(f"The Average Similarity for generated essay with id {key}: {avg_similarity}")
        pbar.update(1)  # Update the progress bar


scores_df = pd.DataFrame(similarity_scores)
