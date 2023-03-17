RELEVANT_TIMESPANS = [
    ("2019-03-05", "2019-03-25"),
    ("2019-09-10", "2019-09-30"),
    ("2022-10-01", "2022-10-28"),
]
make_relevant_df_name = (
    lambda dates: f"subsets_v{CLASSIFIER_VERSION}/relevant_tweets_{dates[0]}_{dates[1]}.csv"
)
make_hydrated_df_name = (
    lambda dates: f"subsets_v{CLASSIFIER_VERSION}/hydrated_tweets_{dates[0]}_{dates[1]}.csv"
)

CLASSIFIER_VERSION = 1
