import csv
import json
import pandas


def split_by_user(df: pandas.DataFrame):
    uids = df["uid"].unique()

    # create a data frame dictionary to store your data frames
    uid_to_df = dict()

    for uid in uids:
        uid_to_df[uid] = df.where(df["uid"] == uid)

    for uid, user_df in uid_to_df.items():
        user_df.dropna(inplace=True)
        user_df.to_csv(f"recommender/user_csvs/{uid}.csv", index=False)

def main():
    #add more here? cant remember what we need
    standard_values = ["review_count", "rating", "price"]
    output_df = pandas.DataFrame()
    with open('yelp_data.csv', 'r', encoding='UTF8', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            row_index = len(output_df)

            output_df.at[row_index, "uid"] = row[0]
            output_df.at[row_index, "user_rating"] = row[1]

            business_info = json.loads(row[3])
            for key in standard_values:
                if key in business_info.keys():
                    output_df.at[row_index, key] = business_info[key]
                else:
                    output_df.at[row_index, key] = ""

            #replace unicode euro and pound with dollars
            output_df.at[row_index, "price"] = output_df.at[row_index, "price"].replace("\u20ac", "$")
            output_df.at[row_index, "price"] = output_df.at[row_index, "price"].replace("\u00a3", "$")
            output_df.at[row_index, "price"] = len(output_df.at[row_index, "price"])
            for category in business_info["categories"]:
                category_name = category["alias"]
                output_df.at[row_index, category_name] = 1
    output_df.fillna(0, inplace=True)
    print(output_df.at[1, "icecream"])
    output_df.to_csv("formatted_data.csv", index=False)

    split_by_user(output_df)


if __name__ == "__main__":
    main()
