import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, mean_squared_error, roc_curve, auc
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
                probabilities = self.get_probabilites(classification_model,
                                                      test_df.loc[:, ["review_count", "rating", "price", "knn_value"]])
                self.update_test_predictions(predictions, knn_values, train_user_mean_rating, test_df, probabilities)
                curr_user_targets += test_df["user_rating"].tolist()
            all_users_targets += curr_user_targets

        self.show_performance_summary()
        self.plot_roc_curve()
        self.plot_predictions_vs_targets()

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

    def get_train_test_df(self, csv, test_proportion=0.2):
        df = pd.read_csv(csv)
        test = random.sample(range(len(df)), int(len(df)*test_proportion))
        train = list(set(range(len(df))) - set(test))
        df = df.loc[:, (df != 0).any(axis=0)]
        train_df = df.loc[train, :].reset_index(drop=True)
        test_df = df.loc[test, :].reset_index(drop=True)
        return train_df, test_df

    def get_knn_model(self, content_features, targets):
        knn_model = KNeighborsRegressor(n_neighbors=5, weights="uniform", metric="cosine")
        knn_model.fit(content_features, targets)
        return knn_model

    def get_classification_model(self, features, targets):
        model = LogisticRegression(max_iter=2e16)
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


def main():
    content_based_what = ContentBasedComparer()
    content_based_what.run()


if __name__ == "__main__":
    main()
