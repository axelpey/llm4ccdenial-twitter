import time
from datetime import datetime, date
import requests
import tweepy
import json
import pandas as pd
from typing import List, Tuple

from constants import CLASSIFIER_VERSION, RELEVANT_TIMESPANS, make_relevant_df_name


api_key = json.load(open("bearer_tokens.json"))[1]["token"]
client = tweepy.Client(bearer_token=api_key)

THROTTLE_GET_TWEETS_DATA_FROM_API = 100
TWEET_FIELDS = ["author_id", "created_at"]
EXPANSIONS = ["referenced_tweets.id", "author_id"]
USER_FIELDS = ["description", "created_at"]


# Necessary to serialize datetime objects that are part of the tweet object created by tweepy
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return json.JSONEncoder.default(self, obj)


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


def get_tweets_data(tweet_ids: List[int]) -> List[dict]:
    # If the list is bigger than 100 elements, split it into chunks of 100 elements.
    if len(tweet_ids) > 100:
        tweets_data = []
        for i in range(0, len(tweet_ids), 100):
            print(f"Getting tweets {i} to {i + 100} of {len(tweet_ids)}")
            tweets_data += get_tweets_data(tweet_ids[i : i + 100])

            if len(tweet_ids) - i >= 100:
                # If we're not on the last pull, throttle the call a bit
                time.sleep(THROTTLE_GET_TWEETS_DATA_FROM_API / 1000)

        return tweets_data

    else:
        make_call = lambda ids: client.get_tweets(
            ids,
            tweet_fields=TWEET_FIELDS,
            expansions=EXPANSIONS,
            user_fields=USER_FIELDS,
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
                        {
                            "tweet_id": r_tweet.id,
                            "type": r_tweet.type,
                            "text": next(
                                (
                                    rt.text
                                    for rt in response.includes["tweets"]
                                    if rt.id == r_tweet.id
                                ),
                                None,
                            ),
                        }
                        for r_tweet in tweet.referenced_tweets
                    ]
                )
                if tweet.referenced_tweets
                else None,
                "author": next(
                    (
                        json.dumps(
                            {
                                "id": user.id,
                                "created_at": user.created_at,
                                "description": user.description,
                                "name": user.name,
                                "username": user.username,
                            },
                            cls=DateTimeEncoder,
                        )
                        for user in response.includes["users"]
                        if user.id == tweet.author_id
                    ),
                    None,
                ),
            }
            for tweet in response.data
        ]


def hydrate_relevant_subsets():
    # For each relevant subset, hydrate, using the tweet ids, the tweets data, including:

    for dates in RELEVANT_TIMESPANS[:1]:
        df = pd.read_csv(make_relevant_df_name(dates))
        tweet_ids = [str(x) for x in df["id"].values]

        tweets_data = get_tweets_data(tweet_ids[:1])

        df = pd.DataFrame(tweets_data)

        df.to_csv(
            f"subsets_v{CLASSIFIER_VERSION}/hydrated_tweets_{dates[0]}_{dates[1]}.csv",
            index=False,
        )


def analysis_of_hydrated_tweets(dates: Tuple[str, str]):
    # For the dates chosen, first join the hydrated tweets with the original subset of tweets
    df = pd.read_csv(
        f"subsets_v{CLASSIFIER_VERSION}/hydrated_tweets_{dates[0]}_{dates[1]}.csv"
    )
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


if __name__ == "__main__":
    # generate_relevant_subsets()
    hydrate_relevant_subsets()
