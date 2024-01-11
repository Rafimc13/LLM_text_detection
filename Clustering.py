import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF


# Read the train essays .csv
essays_df = pd.read_csv('data/best_train_essays.csv')
essays_cols = essays_df.columns.tolist()
# Separate the essays and the labels
essays = essays_df[essays_cols[2]]


def clustering_essays(dataset, optimal_k=2):
    """
    Make a clustering of k clusters for a dataset
    that contains essays.
    :param dataset: list of essays
    :param optimal_k: int
    :return: array of ints (the labels)
    :return: two arrays (1st array contains the labels and the 2nd the number of
    essays on each cluster)
    """
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    kmeans.fit(dataset)
    cluster_labels = kmeans.labels_
    unique_clusters, cluster_counts = np.unique(cluster_labels, return_counts=True)

    return cluster_labels, unique_clusters, cluster_counts


def bar_plot(cluster_labels, cluster_counts, labels):
    """

    :param cluster_labels: array of ints (the labels of each cluster)
    :param cluster_counts: array of ints (the number of essays for each cluster)
    :param labels: arrays of strings (labels in order to present better the bars)
    :return: None (show of bar plot)
    """
    colors = ['blue', 'green']
    plt.figure(figsize=(10,6))
    plt.bar(cluster_labels, cluster_counts, align='center', label=labels, alpha=0.7, color = colors)
    plt.xlabel('Cluster Number')
    plt.ylabel('Number of Essays')
    plt.xlim(-1,2)
    plt.ylim(0, 1000)
    plt.title('Distribution of Essays in Clusters')
    plt.legend()
    plt.show()


student_essays = [essay for i, essay in enumerate(essays_df[essays_cols[2]])
                  if essays_df.loc[i, essays_cols[3]] == 0]

LLM_essays = [essay for i, essay in enumerate(essays_df[essays_cols[2]])
                if essays_df.loc[i, essays_cols[3]] == 1]

vectorizer = TfidfVectorizer(stop_words='english')
X_LLM = vectorizer.fit_transform(LLM_essays)
X_students = vectorizer.fit_transform(student_essays)

# Apply K-Means for LLM essays
cluster_labels_LLM, clusters_LLM, cluster_count_LLM  = clustering_essays(X_LLM)
labels = ['LLM cluster 1', 'LLM cluster 2']
# bar_plot(clusters_LLM, cluster_count_LLM, labels)

# Apply K-Means for student essays
cluster_labels_students, clusters_students, cluster_count_students = clustering_essays(X_students)
labels = ['students cluster 1', 'students cluster 2']
# bar_plot(clusters_students, cluster_count_students, labels)

cluster1_LLM = [essay for i, essay in enumerate(LLM_essays)
                  if cluster_labels_LLM[i] == 0]

cluster2_LLM = [essay for i, essay in enumerate(LLM_essays)
                  if cluster_labels_LLM[i] == 1]

cluster1_students = [essay for i, essay in enumerate(student_essays)
                  if cluster_labels_students[i] == 0]

cluster2_students = [essay for i, essay in enumerate(student_essays)
                  if cluster_labels_students[i] == 1]

# Combine essays within each cluster into a single document for topic analysis
cluster1_LLM_text = ' '.join(cluster1_LLM)
cluster2_LLM_text = ' '.join(cluster2_LLM)
cluster1_students_text = ' '.join(cluster1_students)
cluster2_students_text = ' '.join(cluster2_students)


def most_common_words(cluster1, cluster2, n_topic=2, n=-7):
    """
    Caclulation of the most common words for each cluster.
    By changing the parameter 'n' we can adjust the returned
    number of most common words
    :param cluster1: list of texts (strings),
    :param cluster2: list of texts (strings),
    :param n_topic: int (topics of clusters)
    :param n: int (influence the number of most common words that will be outputed)
    :return: list of strings (most common words)
    """


    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform([cluster1, cluster2])

    # Fit an NMF model
    num_topics = n_topic
    nmf_model = NMF(n_components=num_topics, random_state=42)
    nmf_model.fit(X)

    # Get the top words for each topic
    feature_names = np.array(vectorizer.get_feature_names_out())
    top_words = []
    for topic_idx, topic in enumerate(nmf_model.components_):
        top_words.append([feature_names[i] for i in topic.argsort()[:n:-1]])

    return top_words

# Assign titles based on the top words in each cluster
LLM_top_words = most_common_words(cluster1_LLM_text, cluster2_LLM_text)
title_cluster1_LLM = f"Cluster 1 (LLM): {' the '.join(LLM_top_words[0])}"
title_cluster2_LLM = f"Cluster 2 (LLM): {' the '.join(LLM_top_words[1])}"

print(title_cluster1_LLM)
print(title_cluster2_LLM)


students_top_words = most_common_words(cluster1_students_text, cluster2_students_text)
title_cluster1_students = f"Cluster 1 (LLM): {' the '.join(students_top_words[0])}"
title_cluster2_students = f"Cluster 2 (LLM): {' the '.join(students_top_words[1])}"

print(title_cluster1_students)
print(title_cluster2_students)

LLM_most_common_words = most_common_words(cluster1_LLM_text, cluster2_LLM_text, n=-300)
students_most_common_words = most_common_words(cluster1_students_text, cluster2_students_text, n=-300)

LLM_most_common_words1 = set(LLM_most_common_words[0])
LLM_most_common_words2 = set(LLM_most_common_words[1])

students_most_common_words1 = set(students_most_common_words[0])
students_most_common_words2 = set(students_most_common_words[1])

count_words_cluster11_LLM = len(LLM_most_common_words1.intersection(students_most_common_words1))
count_words_cluster12_LLM = len(LLM_most_common_words1.intersection(students_most_common_words2))

count_words_cluster21_LLM = len(LLM_most_common_words2.intersection(students_most_common_words1))
count_words_cluster22_LLM = len(LLM_most_common_words2.intersection(students_most_common_words2))


LLM_cluster1 = [count_words_cluster11_LLM, count_words_cluster12_LLM]
LLM_cluster2 = [count_words_cluster21_LLM, count_words_cluster22_LLM]
labels1 = ['similarity between LLM cluster 1 with students cluster 1',
          'similarity between LLM cluster 1 with students cluster 2']
bar_plot([0,1], LLM_cluster1, labels1)
labels2 = ['similarity between LLM cluster 2 with students cluster 1',
          'similarity between LLM cluster 2 with students cluster 2']
bar_plot([0,1], LLM_cluster2, labels2)