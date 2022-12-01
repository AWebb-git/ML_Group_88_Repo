import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, mean_squared_error, roc_curve, auc, confusion_matrix
from sklearn.preprocessing import PolynomialFeatures
import random
import glob
import numpy as np
import matplotlib.pyplot as plt


class ContentBasedComparer:
    def __init__(self):
        self.model_predictions_train = list()
        self.model_probabilities_train = list()
        self.knn_predictions_train = list()
        
        self.business_avg_train = list()
        self.user_avg_train = list()
        
        self.train_targets = list()

        self.model_predictions_test = list()
        self.model_probabilities_test = list()
        self.knn_predictions_test = list()

        self.business_avg_test = list()
        self.user_avg_test = list()
        
        self.test_targets = list()

        self.test_targets_in_folds = list()
        self.knn_cross_val_predictions = dict()
        self.poly_order_cross_val_predicitons = dict()
    
    def run(self):
        user_csvs = glob.glob("user_csvs/*")
        all_users_targets = list()
        for csv in user_csvs:
            df = pd.read_csv(csv)
            df = df.loc[:, (df != 0).any(axis=0)]
            kf = KFold(n_splits=5, random_state=1, shuffle=True)
            curr_user_targets = list()
            for train, test in kf.split(df):
                train_df = df.loc[train, :].reset_index(drop=True)
                test_df = df.loc[test, :].reset_index(drop=True)
                self.knn_cross_val(train_df, test_df)

                train_user_mean_rating = train_df["user_rating"].mean()

                train_content_features = train_df.iloc[:, 5::]
                knn_model = self.get_knn_model(train_content_features, train_df["user_rating"])
                
                train_knn_values = self.get_knn_training_values(train_content_features, train_df["user_rating"])
                train_df["knn_value"] = train_knn_values

                classification_model = self.get_classification_model(
                    train_df.loc[:, ["review_count", "rating", "price", "knn_value"]],
                    train_df["user_rating"])

                train_predictions = classification_model.predict(train_df.loc[:, ["review_count", "rating", "price", "knn_value"]])
                probabilities = self.get_probabilites(classification_model, train_df.loc[:, ["review_count", "rating", "price", "knn_value"]])
                self.update_train_predictions(train_predictions, train_knn_values, train_user_mean_rating, train_df, probabilities)

                test_content_features = test_df.iloc[:, 5::]
                knn_values = knn_model.predict(test_content_features)
                knn_values = knn_values.tolist()
                test_df["knn_value"] = knn_values
                predictions = classification_model.predict(
                    test_df.loc[:, ["review_count", "rating", "price", "knn_value"]])
                self.poly_order_cross_val(train_df.loc[:, ["review_count", "rating", "price", "knn_value", "user_rating"]],
                                          test_df.loc[:, ["review_count", "rating", "price", "knn_value", "user_rating"]])
                probabilities = self.get_probabilites(classification_model,
                                                      test_df.loc[:, ["review_count", "rating", "price", "knn_value"]])
                self.update_test_predictions(predictions, knn_values, train_user_mean_rating, test_df, probabilities)
                curr_user_targets += test_df["user_rating"].tolist()
            all_users_targets += curr_user_targets

        self.show_performance_summary()
        self.plot_roc_curve()
        self.plot_predictions_vs_targets()
        self.plot_knn_cross_val()
        self.plot_poly_cross_val()

    def poly_order_cross_val(self, train_df, test_df):
        poly_order_values = [1, 2, 3]
        for poly_order in poly_order_values:
            if poly_order not in self.poly_order_cross_val_predicitons.keys():
                self.poly_order_cross_val_predicitons[poly_order] = list()

            train_features = PolynomialFeatures(poly_order).fit_transform(train_df.loc[:, ["review_count", "rating", "price", "knn_value"]])
            train_features = [features[1:] for features in train_features]

            model = LogisticRegression(max_iter=2e31)
            model.fit(train_features, train_df["user_rating"])

            test_features = PolynomialFeatures(poly_order).fit_transform(test_df.loc[:, ["review_count", "rating", "price", "knn_value"]])
            test_features = [features[1:] for features in test_features]

            predictions = model.predict(test_features)
            self.poly_order_cross_val_predicitons[poly_order].append(predictions.tolist())

    def knn_cross_val(self, train_df, test_df):
        k_values = [1, 3, 5, 7, 9, 13, 19]
        for k in k_values:
            if k not in self.knn_cross_val_predictions.keys():
                self.knn_cross_val_predictions[k] = list()
            knn_model = KNeighborsRegressor(n_neighbors=k)
            knn_model.fit(train_df.iloc[:, 5::], train_df["user_rating"])

            predictions = knn_model.predict(test_df.iloc[:, 5::])
            self.knn_cross_val_predictions[k].append(predictions.tolist())

    def plot_knn_cross_val(self):
        knn_cross_val_mse = dict()
        knn_cross_val_std = dict()
        for k_val, fold_predictions in self.knn_cross_val_predictions.items():
            square_errors = list()
            for predictions, targets in zip(fold_predictions, self.test_targets_in_folds):
                square_errors.append(mean_squared_error(targets, predictions))
            knn_cross_val_mse[k_val] = np.mean(square_errors)
            knn_cross_val_std[k_val] = np.std(square_errors)
        plt.rcParams["figure.figsize"] = (9, 9)
        plt.title("K vs MSE at knn step", size=20)
        subplot = plt.subplot(111)
        box = subplot.get_position()
        subplot.set_position([box.x0, box.y0, box.width, box.height])
        subplot.set_xlabel("k")
        subplot.set_ylabel("MSE")
        plt.errorbar(list(knn_cross_val_mse.keys()), list(knn_cross_val_mse.values()),
                     yerr=list(knn_cross_val_std.values()), color="#0000ff", linewidth=3)
        plt.legend(["points"], bbox_to_anchor=(0.9, 0.1), prop={'size': 12})
        plt.show()

    def plot_poly_cross_val(self):
        poly_order_cross_val_f1 = dict()
        poly_order_cross_val_f1_std = dict()
        for poly_order, fold_predictions in self.poly_order_cross_val_predicitons.items():
            f1_scores = list()
            for predictions, targets in zip(fold_predictions, self.test_targets_in_folds):
                f1_scores.append(f1_score(targets, predictions, average="micro"))
            poly_order_cross_val_f1[poly_order] = np.mean(f1_scores)
            poly_order_cross_val_f1_std[poly_order] = np.std(f1_scores)
        plt.rcParams["figure.figsize"] = (9, 9)
        plt.title("poly order vs f1 score", size=20)
        subplot = plt.subplot(111)
        box = subplot.get_position()
        subplot.set_position([box.x0, box.y0, box.width, box.height])
        subplot.set_xlabel("poly order")
        subplot.set_ylabel("f1 score")
        plt.errorbar(list(poly_order_cross_val_f1.keys()), list(poly_order_cross_val_f1.values()),
                     yerr=list(poly_order_cross_val_f1_std.values()), color="#0000ff", linewidth=3)
        plt.legend(["points"], bbox_to_anchor=(0.9, 0.1), prop={'size': 12})
        plt.show()


    def show_performance_summary(self):
        print("\nTRAIN AVERAGE")
        print(f"MSE final output: {mean_squared_error(self.train_targets, self.model_predictions_train)}")
        print(f"MSE knn step: {mean_squared_error(self.train_targets, self.knn_predictions_train)}")
        print(f"MSE user average rating: {mean_squared_error(self.train_targets, self.user_avg_train)}")
        print(f"MSE business average rating: {mean_squared_error(self.train_targets, self.business_avg_train)}")

        print("\nTEST AVERAGE")
        print(f"MSE final output: {mean_squared_error(self.test_targets, self.model_predictions_test)}")
        print(f"MSE knn step: {mean_squared_error(self.test_targets, self.knn_predictions_test)}")
        print(f"MSE user average rating: {mean_squared_error(self.test_targets, self.user_avg_test)}")
        print(f"MSE business average rating: {mean_squared_error(self.test_targets, self.business_avg_test)}")

        f1_micro = f1_score([round(prediction) for prediction in self.model_predictions_test], self.test_targets,
                      average="micro")
        f1_macro = f1_score([round(prediction) for prediction in self.model_predictions_test], self.test_targets,
                      average="macro")
        print(f"\nmicro avg F1 final output: {f1_micro}")
        print(f"macro avg F1 final output: {f1_macro}")

        f1_micro = f1_score([round(prediction) for prediction in self.knn_predictions_test], self.test_targets,
                      average="micro")
        f1_macro = f1_score([round(prediction) for prediction in self.knn_predictions_test], self.test_targets,
                      average="macro")
        print(f"\nmicro avg F1 knn step: {f1_micro}")
        print(f"macro avg F1 knn step: {f1_macro}")

        f1_micro = f1_score([round(prediction) for prediction in self.user_avg_test], self.test_targets,
                            average="micro")
        f1_macro = f1_score([round(prediction) for prediction in self.user_avg_test], self.test_targets,
                            average="macro")
        print(f"\nmicro avg F1 user average rating: {f1_micro}")
        print(f"macro avg F1 user average rating: {f1_macro}")

        f1_micro = f1_score([round(prediction) for prediction in self.business_avg_test], self.test_targets,
                            average="micro")
        f1_macro = f1_score([round(prediction) for prediction in self.business_avg_test], self.test_targets,
                            average="macro")
        print(f"\nmicro avg F1 business average rating: {f1_micro}")
        print(f"macro avg F1 business average rating: {f1_macro}")

        get_micro_metrics(confusion_matrix(self.test_targets, self.model_predictions_test))
    
    def update_train_predictions(self, predictions, knn_values, user_mean, df, probabilities):
        self.model_predictions_train += predictions.tolist()
        self.knn_predictions_train += knn_values
        self.user_avg_train += [user_mean] * len(predictions)
        self.business_avg_train += df["rating"].tolist()
        self.model_probabilities_train += probabilities
        
        self.train_targets += df["user_rating"].tolist()

    def update_test_predictions(self, predictions, knn_values, user_mean, df, probabilites):
        self.model_predictions_test += predictions.tolist()
        self.knn_predictions_test += knn_values
        self.user_avg_test += [user_mean] * len(predictions)
        self.business_avg_test += df["rating"].tolist()

        self.model_probabilities_test += probabilites

        self.test_targets += df["user_rating"].tolist()
        self.test_targets_in_folds.append(df["user_rating"].tolist())

    def get_train_test_df(self, csv, test_proportion=0.2):
        df = pd.read_csv(csv)
        test = random.sample(range(len(df)), int(len(df)*test_proportion))
        train = list(set(range(len(df))) - set(test))
        df = df.loc[:, (df != 0).any(axis=0)]
        train_df = df.loc[train, :].reset_index(drop=True)
        test_df = df.loc[test, :].reset_index(drop=True)
        return train_df, test_df

    def get_knn_model(self, content_features, targets):
        knn_model = KNeighborsRegressor(n_neighbors=7, weights="uniform", metric="cosine")
        knn_model.fit(content_features, targets)
        return knn_model

    def get_classification_model(self, features, targets):
        model = LogisticRegression(max_iter=2e31)
        model.fit(features, targets)
        return model

    def get_probabilites(self, model, features):
        return_val = list()
        probabilities = model.predict_proba(features)
        for probability_list in probabilities:
            row = [0] * 5
            for classification, probability in zip(model.classes_, probability_list):
                row[classification - 1] = probability
            return_val.append(row)
        return return_val

    def get_knn_training_values(self, features, targets):
        knn_values = list()
        for index in range(len(features)):
            knn_model = self.get_knn_model(features.drop(index), targets.drop(index))
            knn_value = knn_model.predict(features.loc[index:index, :])
            knn_values.append(knn_value[0])
        return knn_values

    def plot_roc_curve(self):
        fp_rate_for_models = list()
        tp_rate_for_models = list()
        fp_rate_for_models.append([0, 1])
        tp_rate_for_models.append([0, 1])
        decision_boundaries = np.linspace(1.01, -0.01, 201)
        all_classes = sorted(list(set(self.test_targets)))
        colours = ["#ff0000", "#00ff00", "#0000ff", "#f0f000", "#f000f0", "#00f0f0"]
        for label in all_classes:
            tp_rates = list()
            fp_rates = list()
            for decision_boundary in decision_boundaries:
                predictions = [-1 if probability[label - 1] <= decision_boundary else 1 for probability in self.model_probabilities_test]
                targets = [-1 if target != label else 1 for target in self.test_targets]
                tp = 0
                tn = 0
                fp = 0
                fn = 0
                for index, prediction in enumerate(predictions):
                    if prediction == targets[index]:
                        if prediction == 1:
                            tp += 1
                        else:
                            tn += 1
                    else:
                        if prediction == 1:
                            fp += 1
                        else:
                            fn += 1
                tp_rate = tp / (tp + fn)
                fp_rate = fp / (fp + tn)
                tp_rates.append(tp_rate)
                fp_rates.append(fp_rate)
            tp_rate_for_models.append(tp_rates)
            fp_rate_for_models.append(fp_rates)
        plt.rcParams["figure.figsize"] = (9, 9)
        plt.title("ROC Curves", size=20)
        subplot = plt.subplot(111)
        box = subplot.get_position()
        subplot.set_position([box.x0, box.y0, box.width, box.height])
        subplot.set_xlabel("fp rate")
        subplot.set_ylabel("tp rate")

        for index in range(0, len(tp_rate_for_models)):
            subplot.plot(fp_rate_for_models[index], tp_rate_for_models[index], color=colours[index], linewidth=3)

        #init fpr and tpr lists
        all_fpr = np.unique(np.concatenate([fp_rate_for_models[i] for i in range(len(all_classes))]))
        macro_tpr = np.zeros_like(all_fpr)
        micro_tpr = np.zeros_like(all_fpr)

        #create averaged tprs
        for i in range(len(all_classes)):
            print(np.all(np.diff(fp_rate_for_models[i]) >= 0))
            tpr_macro = [tpr / 5 for tpr in tp_rate_for_models[i]]
            n_targets = len(self.test_targets)
            n_targets_equal_class = self.test_targets.count(all_classes[i])
            tpr_micro = [tpr * (n_targets_equal_class / n_targets) for tpr in tp_rate_for_models[i]]

            macro_tpr += np.interp(all_fpr, fp_rate_for_models[i], tpr_macro)
            micro_tpr += np.interp(all_fpr, fp_rate_for_models[i], tpr_micro)

        plt.plot(all_fpr, macro_tpr, color="#000080", linestyle=":", linewidth=3)
        plt.plot(all_fpr, micro_tpr, color="#800000", linestyle=":", linewidth=3)
        plt.legend(["baseline"] + all_classes + ["macro avg"] + ["micro avg"], bbox_to_anchor=(0.9, 0.3), prop={'size': 12})
        plt.show()

    def plot_predictions_vs_targets(self):
        plt.rcParams["figure.figsize"] = (9, 9)
        plt.title("Targets vs Predictions", size=20)
        subplot = plt.subplot(111)
        box = subplot.get_position()
        subplot.set_position([box.x0, box.y0, box.width, box.height])
        subplot.set_xlabel("Prediction")
        subplot.set_ylabel("Target")
        plt.scatter(self.model_predictions_test, self.test_targets, color="#0000ff")
        line = np.polyfit(self.model_predictions_test, self.test_targets, 1)
        plt.plot(self.model_predictions_test, [line[0]*x+line[1] for x in self.model_predictions_test], color="#ff0000", linewidth=3)
        plt.legend(["points", "line fit"], bbox_to_anchor=(0.9, 0.1), prop={'size': 12})
        plt.show()


def get_micro_metrics(confusion_matrix):
    tp = []
    fp = []
    fn = []
    tn = []
    ind = [0, 1, 2, 3, 4]
    for i in ind:
        ind_i = ind.copy()
        ind_i.remove(i)
        tp.append(confusion_matrix[i, i])
        fp.append(sum(confusion_matrix[ind_i, i]))
        fn.append(sum(confusion_matrix[i, ind_i]))
        tn.append(sum(confusion_matrix[ind_i, ind_i]))
    micro_acc = (sum(tp) + sum(tn))/(sum(tp) + sum(tn) + sum(fp) + sum(fn))
    micro_prec = sum(tp)/(sum(tp) + sum(fp))
    micro_tpr = sum(tp)/(sum(tp) + sum(fn))
    micro_fpr = sum(fp)/(sum(fp) + sum(tn))
    print(f"micro_acc: {micro_acc}")
    print(f"micro_prec: {micro_prec}")
    print(f"micro_tpr: {micro_tpr}")
    print(f"micro_fpr: {micro_fpr}")

def main():
    content_based_what = ContentBasedComparer()
    content_based_what.run()


if __name__ == "__main__":
    main()
