import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, mean_squared_error
import random
import glob


class ContentBasedComparer:
    def __init__(self):
        self.model_predictions_train = list()
        self.knn_predictions_train = list()
        
        self.business_avg_train = list()
        self.user_avg_train = list()
        
        self.train_targets = list()

        self.model_predictions_test = list()
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

                regression_model = self.get_regression_model(
                    train_df.loc[:, ["review_count", "rating", "price", "knn_value"]],
                    train_df["user_rating"])

                train_predictions = self.get_bound_predictions(regression_model, train_df.loc[:, ["review_count", "rating", "price", "knn_value"]])
                self.update_train_predictions(train_predictions, train_knn_values, train_user_mean_rating, train_df)

                test_content_features = test_df.iloc[:, 5::]
                knn_values = knn_model.predict(test_content_features)
                knn_values = knn_values.tolist()
                test_df["knn_value"] = knn_values
                predictions = self.get_bound_predictions(regression_model, test_df.loc[:, ["review_count", "rating", "price", "knn_value"]])
                self.update_test_predictions(predictions, knn_values, train_user_mean_rating, test_df)
                curr_user_targets += test_df["user_rating"].tolist()
            all_users_targets += curr_user_targets

        self.show_performance_summary()

        
    def show_performance_summary(self):
        print("\nTRAIN AVERAGE")
        print(f"MSE model: {mean_squared_error(self.train_targets, self.model_predictions_train)}")
        print(f"MSE knn: {mean_squared_error(self.train_targets, self.knn_predictions_train)}")
        print(f"MSE user: {mean_squared_error(self.train_targets, self.user_avg_train)}")
        print(f"MSE business: {mean_squared_error(self.train_targets, self.business_avg_train)}")

        print("\nTEST AVERAGE")
        print(f"MSE model: {mean_squared_error(self.test_targets, self.model_predictions_test)}")
        print(f"MSE knn: {mean_squared_error(self.test_targets, self.knn_predictions_test)}")
        print(f"MSE user: {mean_squared_error(self.test_targets, self.user_avg_test)}")
        print(f"MSE business: {mean_squared_error(self.test_targets, self.business_avg_test)}")

        f1_micro = f1_score([round(prediction) for prediction in self.model_predictions_test], self.test_targets,
                      average="micro")
        f1_macro = f1_score([round(prediction) for prediction in self.model_predictions_test], self.test_targets,
                      average="macro")
        print(f"\nmicro avg F1: {f1_micro}")
        print(f"macro avg F1: {f1_macro}")
    
    def update_train_predictions(self, predictions, knn_values, user_mean, df):
        self.model_predictions_train += predictions
        self.knn_predictions_train += knn_values
        self.user_avg_train += [user_mean] * len(predictions)
        self.business_avg_train += df["rating"].tolist()
        
        self.train_targets += df["user_rating"].tolist()

    def update_test_predictions(self, predictions, knn_values, user_mean, df):
        self.model_predictions_test += predictions
        self.knn_predictions_test += knn_values
        self.user_avg_test += [user_mean] * len(predictions)
        self.business_avg_test += df["rating"].tolist()
        
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
            knn_values.append(knn_value[0])
        return knn_values



def main():
    content_based_what = ContentBasedComparer()
    content_based_what.run()


if __name__ == "__main__":
    main()
