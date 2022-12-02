import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.svm import SVC


def get_data(filename):
    with open(filename, 'r') as f:
        data = pd.read_csv(f)
        Y = data.loc[:, "user_rating"]
        X = data.loc[:, "review_count":]
    return X, Y


def get_micro_metrics(confus_mat):
    tp = []
    fp = []
    fn = []
    tn = []
    ind = [0, 1, 2, 3, 4]
    for i in ind:
        ind_i = ind.copy()
        ind_i.remove(i)
        tp.append(confus_mat[i, i])
        fp.append(sum(confus_mat[ind_i, i]))
        fn.append(sum(confus_mat[i, ind_i]))
        tn.append(sum(confus_mat[ind_i, ind_i]))
    micro_acc = (sum(tp) + sum(tn)) / (sum(tp) + sum(tn) + sum(fp) + sum(fn))
    micro_prec = sum(tp) / (sum(tp) + sum(fp))
    micro_f1 = 2 * sum(tp) / (2 * sum(tp) + sum(fp) + sum(fn))
    return micro_acc, micro_prec, micro_f1


if __name__ == "__main__":
    X, Y = get_data("formatted_data.csv")
    trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=5)

    # hyperparameter tuning
    hypertuning = False
    if hypertuning:
        mean_error = []
        std_error = []
        C_vals = [100]  # [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
        k_vals = ['rbf', 'sigmoid', 'poly d = 1', 'poly d = 3', 'poly d = 5']
        for c in C_vals:
            for kernel in ['rbf']:  # ['rbf', 'sigmoid', 'poly']:
                d = [3]  # junk default value for degree when not using poly kernel
                if kernel == 'poly':
                    d = [1, 3, 5]
                for deg in d:
                    print(f'{c} {kernel} {deg}')
                    svm_model = SVC(C=c, kernel=kernel, degree=deg, decision_function_shape='ovr')
                    kf = KFold(n_splits=5, shuffle=True, random_state=5)
                    scores = []
                    first = True
                    all_preds = []
                    all_truths = []
                    for train, test in kf.split(X):
                        print('hi')
                        svm_model.fit(X.loc[train], Y[train])
                        preds = svm_model.predict(X.loc[test])
                        scores.append(f1_score(Y[test], preds, average="micro"))
                    mean_error.append(np.array(scores).mean())
                    std_error.append(np.array(scores).std())

        # plot for cross validation
        plt.figure(num=3)
        plt.rcParams['figure.constrained_layout.use'] = True
        x = np.array([1, 2, 3, 4, 5])
        plt.xticks(x, k_vals)
        plt.errorbar(x, mean_error, yerr=std_error, linewidth=3)
        plt.xlabel('C')
        plt.ylabel('F1 Score')
        plt.savefig(f'C:/Users/andrew/PycharmProjects/ML_Project/images/svm_cv_ker.png')
        plt.show()
    else:
        svm_model = SVC(C=100, kernel='rbf', decision_function_shape='ovr')
        svm_model.fit(trainX, trainY)
        preds = svm_model.predict(testX)
        con_mat = confusion_matrix(testY, preds)
        print(con_mat)
        acc, prec, f1 = get_micro_metrics(con_mat)
        print(f'acc: {acc} prec: {prec} f1: {f1}')

        con_mat = np.array([[3, 1, 1, 2, 2],
                            [1, 1, 3, 4, 4],
                            [0, 0, 7, 9, 13],
                            [2, 1, 6, 34, 39],
                            [1, 0, 5, 23, 117]])
        acc, prec, f1 = get_micro_metrics(con_mat)
        print(f'acc: {acc} prec: {prec} f1: {f1}')

        svm_model = OneVsRestClassifier(SVC(C=100, kernel='rbf', decision_function_shape='ovr'))
        Y = label_binarize(Y, classes=[1, 2, 3, 4, 5])
        trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=5)
        prob = svm_model.fit(trainX, trainY).decision_function(testX)
        fpr, tpr, _ = roc_curve(testY.ravel(), prob.ravel())
        auc_micro = auc(fpr, tpr)
        print(auc_micro)
