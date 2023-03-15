from datetime import datetime
import time
import requests
import tweepy
import json
import pandas as pd
from typing import List, Tuple


api_key = json.load(open("bearer_tokens.json"))[1]["token"]
client = tweepy.Client(bearer_token=api_key)

THROTTLE_GET_TWEETS_DATA_FROM_API = 100
TWEET_FIELDS = ["author_id", "created_at"]
EXPANSIONS = ["referenced_tweets.id"]

make_relevant_df_name = (
    lambda dates: f"subsets/relevant_tweets_{dates[0]}_{dates[1]}.csv"
)

RELEVANT_TIMESPANS = [
    ("2019-03-05", "2019-03-25"),
    ("2019-09-10", "2019-09-30"),
]


def get_subset_for_dates(df, start_date, end_date):
    df = df[df["created_at"] >= start_date]
    df = df[df["created_at"] <= end_date]
    return df


def generate_relevant_subsets():
    df = pd.read_csv("The Climate Change Twitter Dataset.csv")

    dfs_relevant = [
        get_subset_for_dates(df, start_date, end_date)
        for start_date, end_date in RELEVANT_TIMESPANS
    ]

    for dfr, dates in zip(dfs_relevant, RELEVANT_TIMESPANS):
        dfr.to_csv(make_relevant_df_name(dates), index=False)


def get_tweets_metrics(tweet_ids: List[int]) -> List[dict]:
    # If the list is bigger than 100 elements, split it into chunks of 100 elements.
    if len(tweet_ids) > 100:
        tweets_data = []
        for i in range(0, len(tweet_ids), 100):
            print(f"Getting tweets {i} to {i + 100} of {len(tweet_ids)}")
            tweets_data += get_tweets_metrics(tweet_ids[i : i + 100])

            if len(tweet_ids) - i >= 100:
                # If we're not on the last pull, throttle the call a bit
                time.sleep(THROTTLE_GET_TWEETS_DATA_FROM_API / 1000)

        return tweets_data

    else:
        make_call = lambda ids: client.get_tweets(
            ids, tweet_fields=TWEET_FIELDS, expansions=EXPANSIONS
        )
        try:
            response = make_call(tweet_ids)
        except requests.exceptions.ConnectionError:
            print("Connection error, retrying after a little bit of throttle!")
            time.sleep(THROTTLE_GET_TWEETS_DATA_FROM_API / 1000)

            response = make_call(tweet_ids)

        return [
            {
                "id": tweet.id,
                "created_at": tweet.created_at,
                "text": tweet.text,
                "author_id": tweet.author_id,
                "referenced_tweets": json.dumps(
                    [
                        {"tweet_id": r_tweet.id, "type": r_tweet.type}
                        for r_tweet in tweet.referenced_tweets
                    ]
                )
                if tweet.referenced_tweets
                else None,
            }
            for tweet in response.data
        ]


def hydrate_relevant_subsets():
    # For each relevant subset, hydrate, using the tweet ids, the tweets data, including:

    for dates in RELEVANT_TIMESPANS:
        df = pd.read_csv(make_relevant_df_name(dates))
        tweet_ids = [str(x) for x in df["id"].values]

        tweets_data = get_tweets_metrics(tweet_ids)

        df = pd.DataFrame(tweets_data)

        df.to_csv(f"subsets/hydrated_tweets_{dates[0]}_{dates[1]}.csv", index=False)


def get_texts_for_tweets(dates: Tuple[str, str], date: str):
    # In this df, pick the hydrated tweets of the day indicated in date "YYYY-MM-DD"
    # Output this as a list of strings, where each string is a tweet text

    df = pd.read_csv(f"subsets/hydrated_tweets_{dates[0]}_{dates[1]}.csv")
    df = df[df["created_at"].str.startswith(date)]
    return df["text"].values.tolist()


def analysis_of_hydrated_tweets(dates: Tuple[str, str]):
    # For the dates chosen, first join the hydrated tweets with the original subset of tweets
    df = pd.read_csv(f"subsets/hydrated_tweets_{dates[0]}_{dates[1]}.csv")
    df_original = pd.read_csv(make_relevant_df_name(dates))

    df = df_original.merge(df, on="id", how="inner")

    # Count and print the occurences of author_id
    author_occurences = df["author_id"].value_counts()
    print(author_occurences[author_occurences >= 2])
    authors_multiple_tweets = author_occurences[author_occurences >= 2].index

    # Count and print the occurences of referenced_tweets
    rt_occurences = df["referenced_tweets"].value_counts()
    print(rt_occurences[rt_occurences >= 5])

    # Display the tweets of the authors with multiple tweets
    for author in authors_multiple_tweets:
        print(df[df["author_id"] == author][["stance", "created_at_x"]].values.tolist())
        print()


analysis_of_hydrated_tweets(("2019-03-05", "2019-03-25"))
