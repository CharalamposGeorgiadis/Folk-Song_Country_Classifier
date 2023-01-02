from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np


def svm(x_train, y_train, x_test, y_test, c=10, kernel='linear', degree=3, gamma=0.1):
    """
    Trains and evaluates a Support Vector Machine
    :param x_train: Training sample features
    :param y_train: Training sample labels
    :param x_test: Test sample features
    :param y_test: Test sample labels
    :param c: C parameter (default = 10)
    :param kernel: Kernel type (default = 'linear')
    :param degree: Degree parameter (default = 3, is only used in polynomial kernels)
    :param gamma: Gamma parameter (default = 0.1)
    :returns Accuracy score (in percentage)
    """
    clf = SVC(C=c, kernel=kernel, degree=degree, gamma=gamma, random_state=42)
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)
    accuracy = np.sum(predictions == y_test) / len(y_test)
    return accuracy * 100


def logistic_regression(x_train, y_train, x_test, y_test, penalty='l2', c=10, solver='saga'):
    """
    Trains and evaluates a Logistic Regression classifier
    :param x_train: Training sample features
    :param y_train: Training sample labels
    :param x_test: Test sample features
    :param y_test: Test sample labels
    :param penalty: Penalty term (default = 'l2')
    :param c: C parameter (default = 10)
    :param solver: Solver optimization algorithm (default = 'saga')
    :returns Accuracy score (in percentage)
    """
    clf = LogisticRegression(penalty=penalty, C=c, solver=solver, tol=1, random_state=42, n_jobs=-1)
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)
    accuracy = np.sum(predictions == y_test) / len(y_test)
    return accuracy * 100
