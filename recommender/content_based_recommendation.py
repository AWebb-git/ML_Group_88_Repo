import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
import random
import glob


def mean_square_error(estimates, targets):
    if type(estimates) in [float, np.float64]:
        square_error_values = [(target - estimates) ** 2 for target in targets]
    else:
        square_error_values = [(target - estimate) ** 2 for estimate, target in zip(estimates, targets)]
    return np.mean(square_error_values)


def compare_performance(predictions, knn_values, user_mean, df):
    mse_model = mean_square_error(predictions, df["user_rating"])
    print(f"MSE model: {mse_model}")
    mse_knn = mean_square_error(knn_values, df["user_rating"])
    print(f"MSE knn: {mse_knn}")
    mse_mean_user_rating = mean_square_error(user_mean, df["user_rating"])
    print(f"MSE dummy user rating: {mse_mean_user_rating}")
    mse_mean_business_rating = mean_square_error(df["rating"], df["user_rating"])
    print(f"MSE dummy business rating: {mse_mean_business_rating}")
    return mse_model, mse_knn, mse_mean_user_rating, mse_mean_business_rating


def get_train_test_df(csv, test_proportion=0.2):
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


def get_knn_training_values(features, targets):
    knn_values = list()
    for index in range(len(features)):
        knn_model = get_knn_model(features.drop(index), targets.drop(index))
        knn_value = knn_model.predict(features.loc[index:index, :])
        knn_values.append(knn_value)
    return knn_values


def main():
    random.seed(2e8 + 2e12 + 54056869)
    user_csvs = glob.glob("user_csvs/*")

    mse_models_train = list()
    mse_knns_train = list()
    mse_user_train = list()
    mse_business_train = list()

    mse_models_test = list()
    mse_knns_test = list()
    mse_user_test = list()
    mse_business_test = list()

    for csv in user_csvs:
        print(f"\n\n{csv}")
        train_df, test_df = get_train_test_df(csv)
        train_user_mean_rating = train_df["user_rating"].mean()
        train_content_features = train_df.iloc[:, 5::]
        knn_model = get_knn_model(train_content_features, train_df["user_rating"])

        knn_values = get_knn_training_values(train_content_features, train_df["user_rating"])
        train_df["knn_value"] = knn_values

        regression_model = get_regression_model(train_df.loc[:, ["review_count", "rating", "price", "knn_value"]],
                                                train_df["user_rating"])

        predictions = get_bound_predictions(regression_model,
                                            train_df.loc[:, ["review_count", "rating", "price", "knn_value"]])

        print("\nTRAINING PERFORMANCE\n")
        mse_model, mse_knn, mse_mean_user_rating, mse_mean_business_rating = compare_performance(predictions,
                                                                                                 knn_values,
                                                                                                 train_user_mean_rating,
                                                                                                 train_df)
        mse_models_train.append(mse_model)
        mse_knns_train.append(mse_knn)
        mse_user_train.append(mse_mean_user_rating)
        mse_business_train.append(mse_mean_business_rating)

        test_content_features = test_df.iloc[:, 5::]
        knn_values = knn_model.predict(test_content_features)
        test_df["knn_value"] = knn_values
        predictions = get_bound_predictions(regression_model,
                                            test_df.loc[:, ["review_count", "rating", "price", "knn_value"]])
        print("\nTEST PERFORMANCE")

        mse_model, mse_knn, mse_mean_user_rating, mse_mean_business_rating = compare_performance(predictions, knn_values, train_user_mean_rating, test_df)
        mse_models_test.append(mse_model)
        mse_knns_test.append(mse_knn)
        mse_user_test.append(mse_mean_user_rating)
        mse_business_test.append(mse_mean_business_rating)

    print("\nTRAIN AVERAGE")
    print(f"avg MSE model: {np.mean(mse_models_train)}")
    print(f"avg MSE knn: {np.mean(mse_knns_train)}")
    print(f"avg MSE user: {np.mean(mse_user_train)}")
    print(f"avg MSE business: {np.mean(mse_business_train)}")

    print("\nTEST AVERAGE")
    print(f"avg MSE model: {np.mean(mse_models_test)}")
    print(f"avg MSE knn: {np.mean(mse_knns_test)}")
    print(f"avg MSE user: {np.mean(mse_user_test)}")
    print(f"avg MSE business: {np.mean(mse_business_test)}")


if __name__ == "__main__":
    main()
