from load_datasets import dataset_loader
from classifiers import svm, logistic_regression
from tabulate import tabulate

# Loading Datasets
print("Loading Datasets")
print("----------------")
x_train, y_train, \
x_test, y_test, \
x_train_both, y_train_both, \
x_test_half, y_test_half, \
x_train_split, y_train_split, \
x_test_split, y_test_split, \
x_train_both_split, y_train_both_split = dataset_loader()

print("Training and Evaluating Classifiers")
print("-----------------------------------")
# Training and Evaluating Support Vector Machine on 100 Real Folk Songs
svm_real_acc = svm(x_train=x_train_split, y_train=y_train_split, x_test=x_test_split, y_test=y_test_split)

# Training Support Vector Machine on Real Folk Songs and Evaluating it on 100 AI Generated Songs
svm_ai_acc = svm(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

# Training and Evaluating Logistic Regression Classifier on 100 Real Folk Songs
log_real_acc = logistic_regression(x_train=x_train_split, y_train=y_train_split,
                                   x_test=x_test_split, y_test=y_test_split)

# Training Logistic Regression Classifier on Real Folk Songs and Evaluating it on 100 AI Generated Songs
log_ai_acc = logistic_regression(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

# Training Support Vector Machine on Both Real and AI Generated Folk Songs and Evaluating it on Real Songs
svm_both_real_acc = svm(x_train=x_train_both_split, y_train=y_train_both_split,
                        x_test=x_test_split, y_test=y_test_split)

# Training Support Vector Machine on Both Real and AI Generated Folk Songs and Evaluating it on AI Songs
svm_both_ai_acc = svm(x_train=x_train_both, y_train=y_train_both, x_test=x_test_half, y_test=y_test_half)

# Training Logistic Regression Classifier on Both Real and AI Generated Folk Songs and Evaluating it on Real Songs
log_both_real_acc = logistic_regression(x_train=x_train_both_split, y_train=y_train_both_split,
                                        x_test=x_test_split, y_test=y_test_split)

# Training Logistic Regression Classifier on Both Real and AI Generated Folk Songs and Evaluating it on AI Songs
log_both_ai_acc = logistic_regression(x_train=x_train_both, y_train=y_train_both,
                                      x_test=x_test_half, y_test=y_test_half)

# Printing the accuracy scores for each model
print()
print(tabulate([['SVM (Trained and evaluated on real songs)', str(svm_real_acc) + "%"],
                ['LRC (Trained and evaluated on real songs)', str(log_real_acc) + "%"],
                ['SVM (Trained on real songs and evaluated on AI generated songs)', str(svm_ai_acc) + "%"],
                ['LRC (Trained on real songs and evaluated on AI generated songs)', str(log_ai_acc) + "%"],
                ['SVM (Trained on real and AI generated songs and evaluated on real songs)', str(svm_both_real_acc) + "%"],
                ['LRC (Trained on real and AI generated songs and evaluated on real songs)', str(log_both_real_acc) + "%"],
                ['SVM (Trained on real and AI generated songs and evaluated on AI generated songs)', str(svm_both_ai_acc) + "%"],
                ['LRC (Trained on real and AI generated songs and evaluated on AI generated songs)', str(log_both_ai_acc) + "%"]],
               headers=['Model', 'Accuracy'],
               tablefmt="orgtbl"))
