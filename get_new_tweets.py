import json
import time
import tweepy
from datetime import datetime
from pathlib import Path
import pandas as pd
import glob

from get_tweets import DateTimeEncoder


def get_new_tweets():
    # Create the OAuth 1.0a Bearer token
    api_keys = json.load(open("api_keys.json"))
    bearer_token = json.load(open("bearer_tokens.json"))[2]["token"]

    # Create the client object
    client = tweepy.Client(
        consumer_key=api_keys["api_key"],
        consumer_secret=api_keys["api_key_secret"],
        access_token=api_keys["access_token"],
        access_token_secret=api_keys["access_token_secret"],
        bearer_token=bearer_token,
    )

    # Define the search parameters
    query = "climate change OR global warming"
    start_date = datetime(2022, 10, 1)
    end_date = datetime(2022, 10, 28)
    # Save as YYYY-MM-DD
    str_start_date = start_date.strftime("%Y-%m-%d")
    str_end_date = end_date.strftime("%Y-%m-%d")

    tweets_per_page = 100
    pages = 250

    folder_name = f"data_collection/{str_start_date}_{str_end_date}_num_tweets_{pages * tweets_per_page}_with_loc_for_real"
    Path(folder_name).mkdir(parents=True, exist_ok=True)

    next_token = None

    # Execute the search
    for page in range(1, pages + 1):
        search_results = client.search_all_tweets(
            query,
            next_token=next_token,
            max_results=tweets_per_page,
            start_time=start_date,
            end_time=end_date,
            expansions=["author_id", "referenced_tweets.id", "geo.place_id"],
            user_fields=["description", "created_at", "location"],
            tweet_fields=["created_at", "public_metrics", "text", "geo"],
            place_fields=["country", "country_code", "geo", "name", "place_type"],
        )
        # Save the results in a json file for this page
        # in data_collection/{str_start_date}_{str_end_date}/tweets_page_{page}.json"
        # For each tweet, save the following fields:
        # - id
        # - text
        # - author_id
        # - referenced_tweets.id
        # - created_at
        # - author_data: the data for the author of the tweet, taken from the includes

        tweets_data = []

        for tweet in search_results.data:
            tweet_data = {
                "id": tweet.id,
                "text": tweet.text,
                "author_id": tweet.author_id,
                "geo": tweet.geo,
                "referenced_tweets": [rt.data for rt in tweet.referenced_tweets]
                if tweet.referenced_tweets is not None
                else [],
                "created_at": tweet.created_at,
                "author_data": next(
                    u.data
                    for u in search_results.includes["users"]
                    if u.id == tweet.author_id
                ),
                "referenced_tweets_data": [
                    t.data
                    for t in search_results.includes["tweets"]
                    if t.id in [rt.id for rt in tweet.referenced_tweets]
                ]
                if tweet.referenced_tweets is not None
                else [],
                "place_data": next(
                    p.data
                    for p in search_results.includes["places"]
                    if p.id == tweet.geo["place_id"]
                )
                if tweet.geo is not None
                else None,
            }
            tweets_data.append(tweet_data)

        # Write to file
        with open(f"{folder_name}/tweets_page_{page}.json", "w") as f:
            json.dump(tweets_data, f, indent=4, cls=DateTimeEncoder)

        next_token = search_results.meta["next_token"]
        time.sleep(1)


def assemble_jsons_in_csv():
    # Import all the jsons done above, and export them in a single csv file
    # The columns should be:
    # id,created_at,text,author_id,referenced_tweets,author
    # where author is a json string with the following keys:
    # id, created_at, description, name, username.
    # The content of the csv file should look the same as:
    # id,created_at,text,author_id,referenced_tweets,author
    # 1102721226638712832,2019-03-05 00:03:22+00:00,RT @DocsEnvAus: Exciting to see @JulianBurnside running in the Kooyong seat on action on #climatechange; care for people seeking asylum and…,456416267,"[{""tweet_id"": 1102707641783017473, ""type"": ""retweeted"", ""text"": ""Exciting to see @JulianBurnside running in the Kooyong seat on action on #climatechange; care for people seeking asylum and influence big corporates out of politics. We need this level of educated debate and representation""}]","{""id"": 456416267, ""created_at"": ""2012-01-06T06:51:35+00:00"", ""description"": ""Grandma on Githabel country. Nature, climate action, politics, great outdoors, art, social justice etc. #EnoughIsEnough\nMoving to Mastodon - Maggie_May"", ""name"": ""Sarah Moles"", ""username"": ""molessarah""}"
    # 1102723437066301440,2019-03-05 00:12:09+00:00,"RT @piecetocamera: @KetanJ0 @Brandwatch News Corp, where climate change is fake &amp; paedophiles are defended (if they’re white, rich &amp; write…",1315921093,"[{""tweet_id"": 1102460497671086080, ""type"": ""retweeted"", ""text"": ""@KetanJ0 @Brandwatch News Corp, where climate change is fake &amp; paedophiles are defended (if they\u2019re white, rich &amp; write opinion pieces for them)""}]","{""id"": 1315921093, ""created_at"": ""2013-03-30T06:48:17+00:00"", ""description"": ""Fact checking is everybody's business.\n\nPARODY (just kidding)\n\n\ud83c\udf32\ud83c\udf0f\ud83c\udf32"", ""name"": ""NOT a Canberra Bubbler \ud83c\udf32\ud83c\udf0f\ud83c\udf32"", ""username"": ""MSMWatchdog2013""}"
    folder_name = (
        "data_collection/2022-10-01_2022-10-28_num_tweets_25000_with_loc_for_real"
    )
    json_files = glob.glob(f"{folder_name}/tweets_page_*.json")
    tweets_list = []

    for json_file in json_files:
        with open(json_file, "r") as f:
            tweets_data = json.load(f)

        for tweet in tweets_data:
            tweet_dict = {
                "id": tweet["id"],
                "created_at": tweet["created_at"],
                "text": tweet["text"],
                "author_id": tweet["author_id"],
                "referenced_tweets": json.dumps(tweet["referenced_tweets_data"]),
                "author": json.dumps(tweet["author_data"]),
            }
            tweets_list.append(tweet_dict)

    df = pd.DataFrame(tweets_list)
    csv_file_name = "combined_tweets_data.csv"
    df.to_csv(csv_file_name, index=False)


if __name__ == "__main__":
    assemble_jsons_in_csv()
