import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
import seaborn as sns
from sklearn.linear_model import LinearRegression


def train_rf_model(X_train, y_train, X_test, y_test):
    """
    Train a Random Forest model with portion of training
    dataset. Test it with the hidden test set and/or the same
    portion of training set and calculate accuracy and F1 score
    :param X_train: array of strings (essays)
    :param y_train: array of ints (labels 0,1)
    :param X_test: array of strings (essays)
    :param y_test: array of ints (labels 0,1)
    :return: floats (accuracy and F1 score)
    """
    # Feature extraction
    vectorizer = TfidfVectorizer(ngram_range=(1, 4), max_features=6000)
    Xtrain = vectorizer.fit_transform(X_train)
    ytrain = np.array(y_train)
    Xtest = vectorizer.fit_transform(X_test)
    ytest = np.array(y_test)

    # Model selection
    rf_classifier = RandomForestClassifier(n_estimators=200, random_state=43)

    # Train the SVM model
    rf_classifier.fit(Xtrain, ytrain)

    # Predictions
    y_pred = rf_classifier.predict(Xtest)

    # Evaluate the model
    accuracy = accuracy_score(ytest, y_pred)
    # report = classification_report(ytest, y_pred)
    f1 = f1_score(ytest, y_pred)
    accuracy_rounded = round(accuracy, 3)
    f1_rounded = round(f1, 3)
    return accuracy_rounded, f1_rounded


def plotting_curves(test_accuracies, train_accuracies, F1_test, F1_trains):
    """
    Plotting curves for the accuracies and F1 scores which originated by
    the above function.
    :param test_accuracies: array of floats (accuracies from test set)
    :param train_accuracies: array of floats (accuracies from train set)
    :param F1_test: array of floats (F1 scores from test set)
    :param F1_trains: array of floats (F1 scores from train set)
    :return: None. Just shows the plot
    """
    percentages = np.array([i * 10 for i in range(1, 11)])

    sns.set(style="whitegrid")
    viridis = plt.get_cmap('viridis')

    plt.figure(figsize=(12, 6))
    plt.plot(percentages, test_accuracies, color=viridis(0.1), label='Accuracy of test set', linewidth=1.5)
    plt.plot(percentages, train_accuracies, color=viridis(0.4), label='Accuracy of training sets', linewidth=1.5)
    plt.plot(percentages, F1_test, color=viridis(0.7), linestyle='--', label='F1 score of test set', linewidth=1.5)
    plt.plot(percentages, F1_trains, color=viridis(0.9),linestyle='--', label='F1 score of training sets', linewidth=1.5)
    plt.ylim(0.1, 1.05)
    plt.xlabel('% of training data used for training the classifier')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve for the Dataset')
    plt.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)
    plt.show()


# Read the train essays .csv
train_essays_df = pd.read_csv('data/best_train_essays.csv')
essays_cols = train_essays_df.columns.tolist()
# Separate the essays and the labels
essays = train_essays_df[essays_cols[2]]
labels = train_essays_df[essays_cols[3]]

X_train, X_test, y_train, y_test = train_test_split(essays, labels, test_size=0.2,
                                                    random_state=43, shuffle=True)

n_split = len(X_train)
accuracies_test = []
f1_test = []
accuracies_train = []
f1_train = []

for i in range(1, 11):
    Xtrain = X_train[:int(i*0.1 * n_split)]
    ytrain = y_train[:int(i*0.1 * n_split)]

    acc_portion_test, f1_portion_test = train_rf_model(Xtrain, ytrain, X_test, y_test)
    accuracies_test.append(acc_portion_test)
    f1_test.append(f1_portion_test)
    acc_portion_train, f1_portion_train = train_rf_model(Xtrain, ytrain, Xtrain, ytrain)
    accuracies_train.append(acc_portion_train)
    f1_train.append(f1_portion_train)

plotting_curves(accuracies_test, accuracies_train, f1_test, f1_train)

y = np.array(accuracies_test + f1_test)
y.reshape(1, -1)
X = np.array([i * 10 for i in range(1, 11)] + [i * 10 for i in range(1, 11)])

def calc_theta(X, Y):
    """
    Calculate the theta of datapoints accuracies & f1_scores using
    linear regression (Least Squares method)
    :param X: np.array of floats (dataset X),
    :param Y: np.array of ints (dataset y),
    :return: np.array of floats (theta)
    """
    aces = [1 for i in range(len(X))]
    X = np.column_stack((aces, X))
    XtX = np.dot(np.transpose(X), X)
    XtX_inverse = np.linalg.inv(XtX)
    XtY = np.dot(np.transpose(X), Y)
    my_theta = np.dot(XtX_inverse, XtY)
    return my_theta


def f(a, b, y):
    """
    Estimate the percentage of required dataset in order the
    accuracy-f1_score to become 100%
    :param a: float (θ0)
    :param b: float (θ1)
    :param y: int (1)
    :return: float (percentage of required dataset)
    """
    p = (y-a)/b
    print(f'The required percentage in order the accuracy and/or f1 score to become 1.00 is: {p*100:.1f}%')
    return int(p/100)

theta = calc_theta(X, y)
print(f'theta paraemeters are: θ0: {theta[0]}, θ1:{theta[1]}')
y_essays = f(theta[0], theta[1], 1)
required_essays = y_essays*len(train_essays_df)
