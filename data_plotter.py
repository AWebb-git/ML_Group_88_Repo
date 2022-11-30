import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import csv

with open('formatted_data.csv', 'r') as f:
    reader = csv.reader(f)
    data = [row for row in reader]
    data.pop(0)

ratings_data = []
ratings = ['1.0', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0', '4.5', '5.0']
for rating in ratings:
    rating_data = []
    for i in range(1, 6):
        count = len([x for x in data if x[1] == f'{i}' and x[3] == rating])
        rating_data.append(count)
    ratings_data.append(rating_data)

prices_data = []
for price in range(0, 5):
    price_data = []
    for i in range(1, 6):
        count = len([x for x in data if x[1] == f'{i}' and x[4] == f'{price}'])
        price_data.append(count)
    prices_data.append(price_data)

custom_points = [Line2D([0], [0], color='blue', lw=4), Line2D([0], [0], color='magenta', lw=4),
                 Line2D([0], [0], color='green', lw=4), Line2D([0], [0], color='yellow', lw=4),
                 Line2D([0], [0], color='orange', lw=4), Line2D([0], [0], color='red', lw=4)]

plt.rcParams['figure.constrained_layout.use'] = True
plt.figure(num=1)
ax = plt.subplot(111)
ax.set_xlabel('Business Rating')
ax.set_ylabel('User Rating')
ax.set_title('Frequency of User Scores w/ varying Business Ratings')
for rating in ratings_data:
    for user_scores in enumerate(rating):
        if 0 < user_scores[1] <= 5:
            ax.scatter(float(ratings[ratings_data.index(rating)]), user_scores[0] + 1, color='blue', s=25)
        elif 5 < user_scores[1] <= 10:
            ax.scatter(float(ratings[ratings_data.index(rating)]), user_scores[0] + 1, color='magenta', s=50)
        elif 10 < user_scores[1] <= 20:
            ax.scatter(float(ratings[ratings_data.index(rating)]), user_scores[0] + 1, color='green', s=75)
        elif 20 < user_scores[1] <= 50:
            ax.scatter(float(ratings[ratings_data.index(rating)]), user_scores[0] + 1, color='yellow', s=100)
        elif 50 < user_scores[1] <= 100:
            ax.scatter(float(ratings[ratings_data.index(rating)]), user_scores[0] + 1, color='orange', s=125)
        elif 100 < user_scores[1]:
            ax.scatter(float(ratings[ratings_data.index(rating)]), user_scores[0] + 1, color='red', s=150)
plt.legend(custom_points, ['1-5', '5-10', '10-20', '20-50', '50-100', '100+'], loc='lower left', bbox_to_anchor=(1,0.65))
plt.show()

plt.figure(num=2)
plt.rcParams['figure.constrained_layout.use'] = True
plt.xlabel('Price Rating')
plt.ylabel('User Rating')
plt.title('Frequency of User Scores w/ varying Price Ratings')
for price in enumerate(prices_data):
    for user_scores in enumerate(price[1]):
        if 0 < user_scores[1] <= 5:
            plt.scatter(price[0], user_scores[0] + 1, color='blue', s=25)
        elif 5 < user_scores[1] <= 10:
            plt.scatter(price[0], user_scores[0] + 1, color='magenta', s=50)
        elif 10 < user_scores[1] <= 20:
            plt.scatter(price[0], user_scores[0] + 1, color='green', s=75)
        elif 20 < user_scores[1] <= 50:
            plt.scatter(price[0], user_scores[0] + 1, color='yellow', s=100)
        elif 50 < user_scores[1] <= 100:
            plt.scatter(price[0], user_scores[0] + 1, color='orange', s=125)
        elif 100 < user_scores[1]:
            plt.scatter(price[0], user_scores[0] + 1, color='red', s=150)
plt.legend(custom_points, ['1-5', '5-10', '10-20', '20-50', '50-100', '100+'], loc='lower left', bbox_to_anchor=(1,0.65))
plt.show()
