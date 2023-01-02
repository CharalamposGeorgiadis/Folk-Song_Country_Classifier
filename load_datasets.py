import pandas as pd
from sklearn.utils import shuffle


def dataset_loader():
    """
    Loads various versions of training and test datasets
    :return:
        x_train: Contains all the real training sample features
        y_train: Contains all the real training sample labels
        x_test: Contains all the AI generated sample features
        y_test: Contains all the AI generated sample labels
        x_train_both: Contains both the real training sample and 25 Dutch and 25 Irish AI generated sample features
        y_train_both: Contains both the real training sample and 25 Dutch and 25 Irish AI generated sample labels
        x_test_half: Contains the remaining 25 Dutch and Irish AI generated sample features
        y_test_half: Contains the remaining 25 Dutch and Irish AI generated sample labels
        x_train_split: Contains all but 100 training sample features
        y_train_split: Contains all but 100 training sample labels
        x_test_split: Contains the remaining 100 training sample features
        y_test_split: Contains the remaining 100 training sample labels
        x_train_both_split: Contains all but 100 training sample as well as all AI generated sample features
        y_train_both_split: Contains all but 100 training sample as well as all AI generated sample labels
    """
    dutch_train = pd.read_csv("Datasets/Dutch_train.csv")
    irish_train = pd.read_csv("Datasets/Irish_train.csv")
    dutch_test = pd.read_csv("Datasets/Dutch_test.csv")
    irish_test = pd.read_csv("Datasets/Irish_test.csv")

    # Dropping song name columns from the test sets
    dutch_test = dutch_test.drop(dutch_test.columns[[0]], axis=1)
    irish_test = irish_test.drop(irish_test.columns[[0]], axis=1)

    # Adding the 'Country' labels to the datasets
    dutch_train["Country"] = ['Dutch' for _ in range(len(dutch_train))]
    irish_train["Country"] = ['Irish' for _ in range(len(irish_train))]
    dutch_test["Country"] = ['Dutch' for _ in range(50)]
    irish_test["Country"] = ['Irish' for _ in range(50)]

    # Combining the Irish and Dutch training sets into one training set
    train_set = pd.concat([dutch_train, irish_train])
    train_set = shuffle(train_set, random_state=42)

    # Combining the Irish and Dutch test sets into one test set
    test_set = pd.concat([dutch_test, irish_test])
    test_set = shuffle(test_set, random_state=42)

    # Creating a training set containing all the training samples as well as 25 Dutch and 25 Irish AI generated songs
    both_train_set = pd.concat([train_set, pd.concat([dutch_test[25:], irish_test[25:]])])
    both_train_set = shuffle(both_train_set, random_state=42)

    # Creating a test set containing the remaining 25 Dutch and 25 Irish AI generated songs
    half_test_set = pd.concat([dutch_test[:25], irish_test[:25]])
    half_test_set = shuffle(half_test_set, random_state=42)

    # Creating a training set containing all but 100 real training samples
    split_train_set = pd.concat([dutch_train[50:], irish_train[50:]])
    split_train_set = shuffle(split_train_set, random_state=42)

    # Creating a training set containing all but 100 real training samples as well as all the AI generated songs
    split_train_set_both = pd.concat([pd.concat([dutch_train[50:], irish_train[50:]]), test_set])
    split_train_set_both = shuffle(split_train_set_both, random_state=42)

    # Creating a test set containing the remaining 100 training samples
    split_test_set = pd.concat([dutch_train[:50], irish_train[:50]])
    split_test_set = shuffle(split_test_set, random_state=42)

    # Removing the all-zero features from all the datasets
    train_set, test_set = remove_zero_columns(train_set=train_set, test_set=test_set)
    both_train_set, half_test_set = remove_zero_columns(train_set=both_train_set, test_set=half_test_set)
    split_train_set, split_test_set = remove_zero_columns(train_set=split_train_set, test_set=split_test_set)
    split_train_set_both, _ = remove_zero_columns(train_set=split_train_set_both,
                                                                    test_set=split_test_set)

    # Splitting the datasets into feature and label numpy arrays
    x_train = train_set.drop('Country', axis=1).values
    y_train = train_set['Country'].values

    x_test = test_set.drop('Country', axis=1).values
    y_test = test_set['Country'].values

    x_train_both = both_train_set.drop('Country', axis=1).values
    y_train_both = both_train_set['Country'].values

    x_test_half = half_test_set.drop('Country', axis=1).values
    y_test_half = half_test_set['Country'].values

    x_train_split = split_train_set.drop('Country', axis=1).values
    y_train_split = split_train_set['Country'].values

    x_test_split = split_test_set.drop('Country', axis=1).values
    y_test_split = split_test_set['Country'].values

    x_train_both_split = split_train_set_both.drop('Country', axis=1).values
    y_train_both_split = split_train_set_both['Country'].values

    return x_train, y_train, \
           x_test, y_test, \
           x_train_both, y_train_both, \
           x_test_half, y_test_half, \
           x_train_split, y_train_split, \
           x_test_split, y_test_split, \
           x_train_both_split, y_train_both_split


def remove_zero_columns(train_set, test_set):
    """
    Removes all the features (columns) that are zero for all samples in both the training and test datasets
    :param train_set: Training set
    :param test_set: Test set
    :return:
        train_set: Cleaned version of the training set
        test_set: Cleaned version of the test set
    """
    train_zero_cols = [col for col in train_set if train_set[col].nunique() != 1]
    test_zero_cols = [col for col in test_set if test_set[col].nunique() != 1]

    columns_to_keep = []
    for train_column in train_zero_cols:
        for test_column in test_zero_cols:
            if train_column == test_column:
                columns_to_keep.append(test_column)
                break

    train_set = train_set[columns_to_keep]
    test_set = test_set[columns_to_keep]

    return train_set, test_set
