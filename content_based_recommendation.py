import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor


def hamming_distance(a, b):
    return sum(abs(e1 - e2) for e1, e2 in zip(a, b)) / len(a)


def get_knn(test_business: pd.DataFrame, training_businesses: pd.DataFrame, k: int = 5):
    distances = list()
    training_businesses = training_businesses.copy()
    test_business = test_business.copy()
    training_businesses.drop(["uid", "user_rating", "review_count", "rating", "price"], axis=1, inplace=True)
    test_business.drop(["uid", "user_rating", "review_count", "rating", "price"], inplace=True)

    for index, row in training_businesses.iterrows():
        distances.append(hamming_distance(test_business, row.tolist()))

    knn_indices = np.argsort(np.array(distances))[:k]
    return knn_indices


def compare_performance(model1, model2, features1, features2, targets, user_mean):
    predictions1 = model1.predict(features1)

    predictions2 = model2.predict(features2)

    square_error_values_model = list()
    square_error_values_model_2 = list()
    square_error_values_user_avg = list()

    for index, target in enumerate(targets):
        square_error_values_model.append((target - predictions1[index]) ** 2)
        square_error_values_user_avg.append((target - user_mean) ** 2)
        square_error_values_model_2.append((target - predictions2[index]) ** 2)

    print(f"targets     : {targets.tolist()}")
    print(f"model1 preds: {predictions1}")
    print(f"model2 preds: {predictions2}")
    print(f"mse model: {np.mean(square_error_values_model)}")
    print(f"mse model 2: {np.mean(square_error_values_model_2)}")
    print(f"mse user avg: {np.mean(square_error_values_user_avg)}")

def main():
    df = pd.read_csv("one_user.csv")

    df = df.loc[:, (df != 0).any(axis=0)]

    train_df = df.loc[6:].reset_index(drop=True)
    test_df = df.loc[:5].reset_index(drop=True)

    mean_rating = train_df["user_rating"].mean()
    print(f"user avg: {mean_rating}")

    knn_avg_scores = list()
    for index, row in train_df.iterrows():
        knn_indices = get_knn(row, train_df.drop(index), 5)
        knn_scores = list()
        for i in knn_indices:
            knn_scores.append(train_df.at[i, "user_rating"])
        knn_avg_scores.append(np.mean(knn_scores))
    train_df["knn_ratings"] = knn_avg_scores

    standard_feature_df = train_df.drop(["uid", "user_rating", "knn_ratings"], axis=1)
    knn_feature_df = train_df[["rating", "review_count", "price", "knn_ratings"]]
    content_based_model = LogisticRegression(max_iter=2e16)
    print("im here babyy")
    print(train_df["user_rating"].values.tolist())
    print(knn_feature_df.values.tolist())

    knn_features = [[2.0, 8.0, 0.0, 4.4], [4.5, 6.0, 2.0, 5.0], [5.0, 26.0, 0.0, 5.0], [4.5, 23.0, 2.0, 5.0], [4.5, 188.0, 0.0, 5.0]]
    knn_targets = [2, 5, 5, 5, 4]
    content_based_model.fit(knn_features, knn_targets)
    print(f"content based: {content_based_model.coef_}")

    model = LogisticRegression(max_iter=2e16)
    model.fit(standard_feature_df.values.tolist(), train_df["user_rating"].values.tolist())

    print(f"std: {model.coef_}")

    print("\n\n--training performance--")
    compare_performance(content_based_model, model, knn_feature_df.to_numpy(), standard_feature_df.to_numpy(), train_df["user_rating"],
                        np.mean(train_df["user_rating"]))

    knn_avg_scores = list()
    for index, row in test_df.iterrows():
        knn_indices = get_knn(row, train_df, 5)
        knn_scores = list()
        for i in knn_indices:
            knn_scores.append(train_df.at[i, "user_rating"])
        knn_avg_scores.append(np.mean(knn_scores))

    test_df["knn_ratings"] = knn_avg_scores

    standard_feature_df = test_df.drop(["uid", "user_rating", "knn_ratings"], axis=1).to_numpy()
    knn_feature_df = test_df[["rating", "review_count", "price", "knn_ratings"]].to_numpy()

    print()
    print("\n\n--test performance--")
    compare_performance(content_based_model, model, knn_feature_df, standard_feature_df, test_df["user_rating"],
                        np.mean(train_df["user_rating"]))


if __name__ == "__main__":
    main()
