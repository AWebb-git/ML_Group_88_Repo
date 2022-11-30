import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


def get_data(filename):
    with open(filename, 'r') as f:
        data = pd.read_csv(f)
        Y = data.loc[:, "user_rating"]
        X = data.loc[:, "review_count":"price"]
    return X, Y


if __name__ == "__main__":
    X, Y = get_data("formatted_data.csv")
    trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=5)

    mean_error = []
    std_error = []
    C_vals = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    for c in C_vals:
        # for kernel in ['rbf', 'sigmoid', 'poly']:
        #     d = [3]  # junk default value for degree when not using poly kernel
        #     if kernel == 'poly':
        #         d = [1, 2, 3, 5, 9]
        #     for deg in d:
        svm_model = SVC(C=c, kernel='rbf', degree=3, decision_function_shape='ovr')
        # svm_model.fit(trainX, trainY)
        # preds = svm_model.predict(testX)
        # f1_val = f1_score(testY, preds, average='micro')
        # acc = accuracy_score(testY, preds)
        # scores.append([acc, c, kernel, deg])
        kf = KFold(n_splits=5, shuffle=True, random_state=5)
        scores = []
        for train, test in kf.split(X):
            svm_model.fit(X.loc[train], Y[train])
            preds = svm_model.predict(X.loc[test])
            scores.append(f1_score(Y[test], preds, average="micro"))
            print(confusion_matrix(Y[train], svm_model.predict(X.loc[train])))
            print(confusion_matrix(Y[test], preds))
        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())
plt.figure(num=3)
plt.rcParams['figure.constrained_layout.use'] = True
plt.errorbar(C_vals, mean_error, yerr=std_error, linewidth=3)
plt.xlabel('C')
plt.ylabel('F1 Score')
#plt.savefig(f'C:/Users/andrew/PycharmProjects/ML_Project/images/svm_cv_reg.png')
plt.show()