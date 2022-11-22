import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import random
import glob


def mean_square_error(estimates, targets):
    square_error_values = list()
    for index, target in enumerate(targets):
        if type(estimates) in [float, np.float64]:
            square_error_values.append((target - estimates) ** 2)
        else:
            square_error_values.append((target - estimates[index]) ** 2)
    return np.mean(square_error_values)


def compare_performance(predictions, user_mean, df):
    mse_model = mean_square_error(predictions, df["user_rating"])
    print(f"mean square error model: {mse_model}")
    mse_mean_user_rating = mean_square_error(user_mean, df["user_rating"])
    print(f"mean square error dummy user rating: {mse_mean_user_rating}")
    mse_mean_business_rating = mean_square_error(df["rating"], df["user_rating"])
    print(f"mean square error dummy business rating: {mse_mean_business_rating}")


def get_train_test_df(csv, test_proportion=0.4):
    df = pd.read_csv(csv)
    test = random.sample(range(len(df)), int(len(df)*test_proportion))
    train = list(set(range(len(df))) - set(test))
    df = df.loc[:, (df != 0).any(axis=0)]
    train_df = df.loc[train, :].reset_index(drop=True)
    test_df = df.loc[test, :].reset_index(drop=True)
    return train_df, test_df


def get_knn_model(content_features, targets):
    knn_model = KNeighborsRegressor(n_neighbors=5, weights="uniform", metric="cosine")
    knn_model.fit(content_features, targets)
    return knn_model


def get_regression_model(features, targets):
    model = LinearRegression(copy_X=True)
    model.fit(features, targets)
    return model


def get_bound_predictions(model, features):
    predictions = model.predict(features)
    predictions = [min(max(prediction, 0), 5) for prediction in predictions]
    return predictions


def main():
    random.seed(2e8 + 2e12 + 54056869)
    user_csvs = glob.glob("user_csvs/*")
    for csv in user_csvs:
        print(f"\n\n{csv}\n")
        train_df, test_df = get_train_test_df(csv)
        train_user_mean_rating = train_df["user_rating"].mean()
        train_content_features = train_df.iloc[:, 5::]
        knn_model = get_knn_model(train_content_features, train_df["user_rating"])

        knn_values = knn_model.predict(train_content_features)
        train_df["knn_value"] = knn_values

        regression_model = get_regression_model(train_df.loc[:, ["review_count", "rating", "price", "knn_value"]],
                                                train_df["user_rating"])

        predictions = get_bound_predictions(regression_model,
                                            train_df.loc[:, ["review_count", "rating", "price", "knn_value"]])

        print("\n\nTRAINING PERFORMANCE\n")
        compare_performance(predictions, train_user_mean_rating, train_df)
        print("\n\nTEST PERFORMANCE\n")
        compare_performance(predictions, train_user_mean_rating, test_df)


if __name__ == "__main__":
    main()
