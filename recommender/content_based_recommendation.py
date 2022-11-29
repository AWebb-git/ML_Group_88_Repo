import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import random
import glob


class ContentBasedComparer:
    def __init__(self):
        self.mse_values_model_all_users_train = list()
        self.mse_values_knn_all_users_train = list()
        self.mse_values_user_avg_all_users_train = list()
        self.mse_values_business_avg_all_users_train = list()

        self.mse_values_model_all_users_test = list()
        self.mse_values_knn_all_users_test = list()
        self.mse_values_user_avg_all_users_test = list()
        self.mse_values_business_avg_all_users_test = list()

        self.curr_user_predictions_model = list()
        
        self.mse_values_model_curr_user_train = list()
        self.mse_values_knn_curr_user_train = list()
        self.mse_values_user_avg_curr_user_train = list()
        self.mse_values_business_avg_curr_user_train = list()

        self.mse_values_model_curr_user_test = list()
        self.mse_values_knn_curr_user_test = list()
        self.mse_values_user_avg_curr_user_test = list()
        self.mse_values_business_avg_curr_user_test = list()

        self.all_users_predictions_model = list()
    
    def run(self):
        user_csvs = glob.glob("user_csvs/*")
        all_users_targets = list()
        for csv in user_csvs:
            print(f"\n\n---{csv}---")
            self.reset_curr_user_performance()
            df = pd.read_csv(csv)
            df = df.loc[:, (df != 0).any(axis=0)]
            kf = KFold(n_splits=len(df), random_state=1, shuffle=True)
            curr_user_targets = list()
            for train, test in kf.split(df):
                train_df = df.loc[train, :].reset_index(drop=True)
                test_df = df.loc[test, :].reset_index(drop=True)

                train_user_mean_rating = train_df["user_rating"].mean()

                train_content_features = train_df.iloc[:, 5::]
                knn_model = self.get_knn_model(train_content_features, train_df["user_rating"])
                
                train_knn_values = self.get_knn_training_values(train_content_features, train_df["user_rating"])
                train_df["knn_value"] = train_knn_values

                regression_model = self.get_regression_model(
                    train_df.loc[:, ["review_count", "rating", "price", "knn_value"]],
                    train_df["user_rating"])

                train_predictions = self.get_bound_predictions(regression_model, train_df.loc[:, ["review_count", "rating", "price", "knn_value"]])
                self.update_curr_user_train_performance(train_predictions, train_knn_values, train_user_mean_rating, train_df)

                test_content_features = test_df.iloc[:, 5::]
                knn_values = knn_model.predict(test_content_features)
                test_df["knn_value"] = knn_values
                predictions = self.get_bound_predictions(regression_model, test_df.loc[:, ["review_count", "rating", "price", "knn_value"]])
                self.update_curr_user_test_performance(predictions, knn_values, train_user_mean_rating, test_df)
                curr_user_targets += test_df["user_rating"].tolist()

            self.print_curr_user_performance(curr_user_targets)
            self.update_all_user_performance()
            all_users_targets += curr_user_targets

        self.print_all_user_performance(all_users_targets)

    def reset_curr_user_performance(self):
        self.mse_values_model_curr_user_train = list()
        self.mse_values_knn_curr_user_train = list()
        self.mse_values_user_avg_curr_user_train = list()
        self.mse_values_business_avg_curr_user_train = list()

        self.mse_values_model_curr_user_test = list()
        self.mse_values_knn_curr_user_test = list()
        self.mse_values_user_avg_curr_user_test = list()
        self.mse_values_business_avg_curr_user_test = list()

        self.curr_user_predictions_model = list()

    def update_all_user_performance(self):
        self.mse_values_model_all_users_train += self.mse_values_model_curr_user_train
        self.mse_values_knn_all_users_train += self.mse_values_knn_curr_user_train
        self.mse_values_user_avg_all_users_train += self.mse_values_user_avg_curr_user_train
        self.mse_values_business_avg_all_users_train += self.mse_values_business_avg_curr_user_train
        
        self.mse_values_model_all_users_test += self.mse_values_model_curr_user_test
        self.mse_values_knn_all_users_test += self.mse_values_knn_curr_user_test
        self.mse_values_user_avg_all_users_test += self.mse_values_user_avg_curr_user_test
        self.mse_values_business_avg_all_users_test += self.mse_values_business_avg_curr_user_test

        self.all_users_predictions_model += self.curr_user_predictions_model

    def print_curr_user_performance(self, targets_in_order):
        print("\nTRAIN PERFORMANCE")
        print(f"MSE model: {np.mean(self.mse_values_model_curr_user_train)}")
        print(f"MSE knn: {np.mean(self.mse_values_knn_curr_user_train)}")
        print(f"MSE user: {np.mean(self.mse_values_user_avg_curr_user_train)}")
        print(f"MSE business: {np.mean(self.mse_values_business_avg_curr_user_train)}")

        print("\nTEST PERFORMANCE")
        print(f"MSE model: {np.mean(self.mse_values_model_curr_user_test)}")
        print(f"MSE knn: {np.mean(self.mse_values_knn_curr_user_test)}")
        print(f"MSE user: {np.mean(self.mse_values_user_avg_curr_user_test)}")
        print(f"MSE business: {np.mean(self.mse_values_business_avg_curr_user_test)}")

        self.print_f1_curr_user(targets_in_order)
        
    def print_all_user_performance(self, targets_in_order):
        print("\nTRAIN AVERAGE")
        print(f"MSE model: {np.mean(self.mse_values_model_all_users_train)}")
        print(f"MSE knn: {np.mean(self.mse_values_knn_all_users_train)}")
        print(f"MSE user: {np.mean(self.mse_values_user_avg_all_users_train)}")
        print(f"MSE business: {np.mean(self.mse_values_business_avg_all_users_train)}")

        print("\nTEST AVERAGE")
        print(f"MSE model: {np.mean(self.mse_values_model_all_users_test)}")
        print(f"MSE knn: {np.mean(self.mse_values_knn_all_users_test)}")
        print(f"MSE user: {np.mean(self.mse_values_user_avg_all_users_test)}")
        print(f"MSE business: {np.mean(self.mse_values_business_avg_all_users_test)}")

        self.print_f1_all_users(targets_in_order)
        
    def get_square_errors(self, estimates, targets):
        if type(estimates) in [float, np.float64]:
            square_error_values = [(target - estimates) ** 2 for target in targets]
        else:
            square_error_values = [(target - estimate) ** 2 for estimate, target in zip(estimates, targets)]
        return square_error_values
    
    def update_curr_user_train_performance(self, predictions, knn_values, user_mean, df):
        mse_model = self.get_square_errors(predictions, df["user_rating"])
        mse_knn = self.get_square_errors(knn_values, df["user_rating"])
        mse_mean_user_rating = self.get_square_errors(user_mean, df["user_rating"])
        mse_mean_business_rating = self.get_square_errors(df["rating"], df["user_rating"])
        
        self.mse_values_model_curr_user_train += mse_model
        self.mse_values_knn_curr_user_train += mse_knn
        self.mse_values_user_avg_curr_user_train += mse_mean_user_rating
        self.mse_values_business_avg_curr_user_train += mse_mean_business_rating

    def update_curr_user_test_performance(self, predictions, knn_values, user_mean, df):
        mse_model = self.get_square_errors(predictions, df["user_rating"])
        mse_knn = self.get_square_errors(knn_values, df["user_rating"])
        mse_mean_user_rating = self.get_square_errors(user_mean, df["user_rating"])
        mse_mean_business_rating = self.get_square_errors(df["rating"], df["user_rating"])

        self.mse_values_model_curr_user_test += mse_model
        self.mse_values_knn_curr_user_test += mse_knn
        self.mse_values_user_avg_curr_user_test += mse_mean_user_rating
        self.mse_values_business_avg_curr_user_test += mse_mean_business_rating

        self.curr_user_predictions_model += predictions

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

    def get_regression_model(self, features, targets):
        model = LinearRegression(copy_X=True)
        model.fit(features, targets)
        return model

    def get_bound_predictions(self, model, features):
        predictions = model.predict(features)
        predictions = [min(max(prediction, 0), 5) for prediction in predictions]
        return predictions

    def get_knn_training_values(self, features, targets):
        knn_values = list()
        for index in range(len(features)):
            knn_model = self.get_knn_model(features.drop(index), targets.drop(index))
            knn_value = knn_model.predict(features.loc[index:index, :])
            knn_values.append(knn_value)
        return knn_values

    def print_f1_curr_user(self, targets):
        f1 = f1_score([round(prediction) for prediction in self.curr_user_predictions_model], targets, average="micro")
        print(f"micro avg F1: {f1}")
        f1 = f1_score([round(prediction) for prediction in self.curr_user_predictions_model], targets, average="macro")
        print(f"macro avg F1: {f1}")

    def print_f1_all_users(self, targets):
        f1 = f1_score([round(prediction) for prediction in self.all_users_predictions_model], targets,
                      average="micro")
        print(f"micro avg F1: {f1}")
        f1 = f1_score([round(prediction) for prediction in self.all_users_predictions_model], targets,
                      average="macro")
        print(f"macro avg F1: {f1}")


def main():
    content_based_what = ContentBasedComparer()
    content_based_what.run()


if __name__ == "__main__":
    main()
