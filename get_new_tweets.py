import json
import time
import tweepy
from datetime import datetime
from pathlib import Path

from get_tweets import DateTimeEncoder


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
pages = 200

folder_name = f"data_collection/{str_start_date}_{str_end_date}_num_tweets_{pages * tweets_per_page}_with_loc"
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
        expansions=["author_id", "referenced_tweets.id"],
        user_fields=["description", "created_at", "location"],
        tweet_fields=["created_at", "public_metrics", "text", "geo"],
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
        }
        tweets_data.append(tweet_data)

    # Write to file
    with open(f"{folder_name}/tweets_page_{page}.json", "w") as f:
        json.dump(tweets_data, f, indent=4, cls=DateTimeEncoder)

    next_token = search_results.meta["next_token"]
    time.sleep(1)
