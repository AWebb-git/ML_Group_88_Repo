import re
from urllib.request import urlopen
import urllib
import requests
import json
from math import trunc
from bs4 import BeautifulSoup
import csv

review_threshold = 30


# returns price of a lego set
def get_reviews(user_id, page_start=0):
    try:
        req = urllib.request.Request(
            f'https://www.yelp.ie/user_details_reviews_self?userid={user_id}&rec_pagestart={page_start}',
            headers={'User-Agent': "Magic Browser"})
        web_page = urlopen(req)
        html = web_page.read().decode('utf-8')
        soup = BeautifulSoup(html, "html.parser")

        review_count = soup.find_all(attrs={"class": "review-count"})
        review_count = re.findall(r'\d+', review_count[0].text)[0]
        if int(review_count) >= review_threshold + 10:
            reviews = soup.find_all(attrs={"class": "ylist ylist-bordered reviews"})
            reviews = [x for x in reviews[0].contents if x.name == 'li']
            if page_start + 10 <= review_threshold:
                reviews += get_reviews(user_id, page_start + 10)
            return reviews
        print('Not Enough Reviews')
    except:
        return None


def get_business_id(biz_name, biz_address, alias):
    try:
        headers = {
            "Authorization": "Bearer vdWdSEyzRS8WJwudQqEcsZCETQx5aV_IetF23gswwc3BHiPoQu6hoC0iIibGCEAEuuGDNg80t8EoeWe5d47VRx6njyiF9OWAeyXGcCIFkPVbn1cu2Yr0x9ExpzFxY3Yx"}
        response = requests.get(
            f'https://api.yelp.com/v3/businesses/search?term={biz_name}&location={biz_address}', headers=headers)
        response.raise_for_status()
        resp_content = json.loads(response.content.decode('utf-8'))
        id = [x for x in resp_content['businesses'] if x['alias'] == alias]
        if len(id) != 0:
            id = id[0]['id']
        # if exact match not found take 'best match' from api
        elif len(resp_content['businesses']) != 0:
            id = resp_content['businesses'][0]['id']
        else:
            id = None
        return id
    except requests.exceptions.HTTPError as error:
        print(error)


def get_business_details(biz_id):
    try:
        headers = {
            "Authorization": "Bearer vdWdSEyzRS8WJwudQqEcsZCETQx5aV_IetF23gswwc3BHiPoQu6hoC0iIibGCEAEuuGDNg80t8EoeWe5d47VRx6njyiF9OWAeyXGcCIFkPVbn1cu2Yr0x9ExpzFxY3Yx"}
        response = requests.get(
            f'https://api.yelp.com/v3/businesses/{biz_id}', headers=headers)
        response.raise_for_status()
        resp_content = response.content.decode('utf-8')
        return resp_content
    except requests.exceptions.HTTPError as error:
        print(error)


user = input('Enter UserID: ')
user_reviews = get_reviews(user)
file_data = []

i = 0
for review in user_reviews:
    if i < review_threshold:
        biz_name = review.find_all(attrs={"class": "biz-name js-analytics-click"})[0].text
        biz_loc = review.find_all("address")[0]
        if len(biz_loc) > 2:
            biz_loc = biz_loc.contents[2]
        else:
            biz_loc = biz_loc.contents[0]
        review_score = re.findall(r'\d+', review.find_all(attrs={"class": "i-stars"})[0].attrs['title'])[0]
        biz_alias = review.find_all(attrs={"class": "biz-name js-analytics-click"})[0].attrs['href'].replace('/biz/',
                                                                                                             "")
        biz_id = get_business_id(biz_name, biz_loc, biz_alias)
        if biz_id is not None:
            biz_details = get_business_details(biz_id)
            file_data.append([user, review_score, biz_name, biz_details])
            i += 1
        else:
            pass
    else:
        break

with open('yelp_data.csv', 'a', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(file_data)
