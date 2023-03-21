import json
import time
import tweepy
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import glob
import pytz

from get_tweets import DateTimeEncoder


def generate_date_ranges(start_date, end_date, interval_days=1):
    current_date = start_date
    date_ranges = []

    while current_date < end_date:
        next_date = current_date + timedelta(days=interval_days)
        date_ranges.append((current_date, min(next_date, end_date)))
        current_date = next_date

    return date_ranges


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
    query = '("climate change" OR "global warming") -is:retweet -is:reply'
    start_date = datetime(2022, 10, 1)
    end_date = datetime(2022, 10, 28)
    interval_days = 28  # Change this value to adjust the interval length

    tweets_per_page = 500
    pages = 250

    date_ranges = generate_date_ranges(start_date, end_date, interval_days)
    folder_name = f"data_collection/{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}_num_tweets_{pages * tweets_per_page}_full"
    Path(folder_name).mkdir(parents=True, exist_ok=True)

    def process_search_results(search_results):
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
                if tweet.geo is not None and "place_id" in tweet.geo
                else None,
            }
            tweets_data.append(tweet_data)

        return tweets_data

    # Execute the search
    for date_range in date_ranges:
        date_range = (
            date_range[0].replace(tzinfo=pytz.UTC),
            date_range[1].replace(tzinfo=pytz.UTC),
        )
        next_token = None
        page = 1

        while True:
            print(f"Page {page}")
            make_request = lambda: client.search_all_tweets(
                query,
                next_token=next_token,
                max_results=tweets_per_page,
                start_time=date_range[0],
                end_time=date_range[1],
                expansions=["author_id", "referenced_tweets.id", "geo.place_id"],
                user_fields=["description", "created_at", "location"],
                tweet_fields=["created_at", "public_metrics", "text", "geo"],
                place_fields=["country", "country_code", "geo", "name", "place_type"],
            )

            try:
                search_results = make_request()
            except tweepy.TooManyRequests:
                print("Too many requests, waiting 15.5 minutes...")
                time.sleep(60 * 15.5)
                search_results = make_request()

            tweets_data = process_search_results(search_results)

            if len(tweets_data) > 0:
                last_tweet_date = tweets_data[-1]["created_at"]
                time_elapsed = (date_range[1] - last_tweet_date).total_seconds()
                total_time_range = (date_range[1] - date_range[0]).total_seconds()
                proportion_covered = time_elapsed / total_time_range
                total_pages_estimate = int(page / proportion_covered)
                print(f"Estimated total pages: {total_pages_estimate}")

            if "next_token" not in search_results.meta:
                next_token = "end"
            else:
                next_token = search_results.meta["next_token"]

            # Write to file
            with open(
                f"{folder_name}/tweets_page_{page}_from_{date_range[0].strftime('%Y-%m-%d')}_to_{date_range[1].strftime('%Y-%m-%d')}.json",
                "w",
            ) as f:
                json.dump(
                    {"data": tweets_data, "next_token": next_token},
                    f,
                    indent=4,
                    cls=DateTimeEncoder,
                )

            if next_token == "end":
                break

            page += 1
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
    folder_name = "data_collection/2022-10-01_2022-10-28_num_tweets_125000_full"
    json_files = glob.glob(f"{folder_name}/tweets_page_*.json")
    tweets_list = []

    for json_file in json_files:
        with open(json_file, "r") as f:
            tweets_data = json.load(f)["data"]

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


def sample_N_tweets_from_csv(N):
    # Sample N tweets from the csv file created above
    # The sample should not be stratified by the number of referenced tweets
    df = pd.read_csv("combined_tweets_data.csv")

    print(f"Sampling {N} tweets out of {len(df)} tweets")
    print(f"That is {N / len(df) * 100:.2f}% of the total tweets")
    print(f"Per day, that is in average {N / 28:.2f} tweets")

    cost_openai_api_for_1000_tokens = 0.002
    batch_size = 2500  # 2500 tokens for a batch of 30 tweets
    tweets_per_batch = 30
    batch_cost = cost_openai_api_for_1000_tokens * batch_size / 1000
    num_batches = int(N / tweets_per_batch)
    total_cost = num_batches * batch_cost

    print(
        f"Cost of classifying {N} tweets: ${total_cost:.2f} (assuming {batch_size} tokens per batch)"
    )

    time_to_classify_one_batch = 30  # seconds
    total_classify_time = num_batches * time_to_classify_one_batch
    print(f"Time to classify {N} tweets: {total_classify_time / 60:.2f} minutes")

    df = df.sample(n=N)
    df.to_csv("subsets_v1/sampled_tweets_data.csv", index=False)


# Now we want to sample in particular from users who have been tweeting before, on the day of the soup
# (2022-10-14) and after.
def sample_tweets_for_regular_posters():
    df = pd.read_csv("combined_tweets_data.csv")

    print(
        f"Sampling tweets from users who have been tweeting before, on the day of the soup (2022-10-14) and after."
    )
    print(f"Total number of tweets: {len(df)}")

    # Convert 'created_at' to datetime objects
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

    soup_day = pd.Timestamp("2022-10-14", tz="UTC")

    # Filter tweets based on the specified date range
    before_date = soup_day - pd.Timedelta(days=1)
    after_date = soup_day + pd.Timedelta(days=1)

    before_tweets = df[df["created_at"] <= before_date]
    on_day_tweets = df[
        (df["created_at"] > before_date) & (df["created_at"] <= after_date)
    ]
    after_tweets = df[df["created_at"] > after_date]

    # Get the users who have tweeted before, on the day, and after the specified date
    users_before = set(before_tweets["author_id"])
    users_on_day = set(on_day_tweets["author_id"])
    users_after = set(after_tweets["author_id"])

    regular_users = users_before.intersection(users_on_day).intersection(users_after)

    # Filter the DataFrame to only include tweets from regular users
    regular_user_tweets = df[df["author_id"].isin(regular_users)]

    print(f"Total number of tweets from regular users: {len(regular_user_tweets)}")

    user_counts = regular_user_tweets["author_id"].value_counts()
    print(f"Min tweets per regular user: {user_counts.min()}")
    print(f"Max tweets per regular user: {user_counts.max()}")
    print(f"Mean tweets per regular user: {user_counts.mean()}")
    print(f"Median tweets per regular user: {user_counts.median()}")
    print(f"Std tweets per regular user: {user_counts.std()}")
    print(f"25% quantile: {user_counts.quantile(0.25)}")
    print(f"75% quantile: {user_counts.quantile(0.75)}")

    # Get the user IDs of regular users that post below the 75% quartile of posts
    quartile_75 = user_counts.quantile(0.75)
    regular_users_below_quartile_75 = user_counts[
        user_counts <= quartile_75
    ].index.tolist()

    print(
        f"Number of regular users below the 75% quartile: {len(regular_users_below_quartile_75)}"
    )

    regular_users_75_tweets = df[df["author_id"].isin(regular_users_below_quartile_75)]

    print(
        f"Number of tweets from regular users below the 75% quartile: {len(regular_users_75_tweets)}"
    )

    regular_users_75_tweets.to_csv(
        "subsets_v1/regular_users_75_tweets.csv", index=False
    )

    return regular_users_75_tweets


if __name__ == "__main__":
    # get_new_tweets()
    # assemble_jsons_in_csv()
    # sample_N_tweets_from_csv(12500)
    sample_tweets_for_regular_posters()
