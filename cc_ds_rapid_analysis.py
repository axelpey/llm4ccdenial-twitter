# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%

df = pd.read_csv("The Climate Change Twitter Dataset.csv")

rows_with_coords = df[~np.isnan(df["lng"])]

# %%

# Compute correlation between aggressiveness of a tweet and its stance
df_to_corr = rows_with_coords[["aggressiveness", "stance"]]
df_to_corr["aggressiveness"] = df_to_corr.applymap(
    lambda x: 1 if x == "aggressive" else 0
)["aggressiveness"]
df_to_corr["stance"] = df_to_corr.applymap(
    lambda x: 1 if x == "believer" else (0.5 if x == "neutral" else 0)
)["stance"]

print(df_to_corr.corr())

# %%

df_politics = rows_with_coords[rows_with_coords["topic"] == "Politics"]
print(f"Number of tweets about politics: {len(df_politics)}")
print(f"Proportion of tweets about politics: {len(df_politics) / len(df)}")

df_to_corr = df_politics[["aggressiveness", "stance"]]
df_to_corr["aggressiveness"] = df_to_corr.applymap(
    lambda x: 1 if x == "aggressive" else 0
)["aggressiveness"]
df_to_corr["stance"] = df_to_corr.applymap(
    lambda x: 1 if x == "believer" else (0.5 if x == "neutral" else 0)
)["stance"]

print(df_to_corr.corr())

# %%

# Plot number of monthly tweets over time

df["created_at"] = pd.to_datetime(df["created_at"])
df["month"] = df["created_at"].apply(lambda x: x.month)
df["year"] = df["created_at"].apply(lambda x: x.year)
df["year_month"] = df["created_at"].apply(lambda x: f"{x.year}-{x.month}")
df_year_month = df.groupby("year_month").count()
df_year = df.groupby("year").count()

plt.plot(df_year.index, df_year["created_at"])
plt.title("Number of tweets over time")
plt.xlabel("Date")
plt.ylabel("Number of tweets")

plt.savefig("tweets_per_year.png")


# %%


def plot_daily_tweets(df, start_date, end_date, title, filename):
    df_subset = df[(df["created_at"] >= start_date) & (df["created_at"] <= end_date)]

    df_subset["created_at"] = pd.to_datetime(df_subset["created_at"])
    df_subset["day"] = df_subset["created_at"].apply(lambda x: x.day)
    df_subset["month"] = df_subset["created_at"].apply(lambda x: x.month)
    df_subset["year"] = df_subset["created_at"].apply(lambda x: x.year)
    df_subset["year_month"] = df_subset["created_at"].apply(
        lambda x: f"{x.year}-{x.month}"
    )
    df_subset["year_month_day"] = df_subset["created_at"].apply(
        lambda x: f"{x.year}-{x.month:02d}-{x.day:02d}"
    )
    df_year_month_day = df_subset.groupby("year_month_day").count()

    plt.figure(figsize=(8, 3))
    plt.plot(df_year_month_day.index, df_year_month_day["created_at"])
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Number of tweets")

    # Make the x-axis labels readable and diagonal
    plt.xticks(rotation=45)

    plt.savefig(filename)
    plt.close()


# %%

plot_daily_tweets(
    df,
    "2019-09-10",
    "2019-09-30",
    "Number of daily tweets around the climate strike of 2019-09-20",
    "tweets_per_day_sept_2019.png",
)


# plot_daily_tweets(
#     df_politics,
#     "2019-09-10",
#     "2019-09-30",
#     "Number of daily tweets around the climate strike of 2019-09-20",
#     "tweets_politics_per_day_sept_2019.png",
# )


# plot_daily_tweets(
#     df_politics,
#     "2019-03-05",
#     "2019-03-25",
#     "Number of daily tweets around the climate strike of 2019-03-15",
#     "tweets_politics_per_day_march_2019.png",
# )


plot_daily_tweets(
    df,
    "2019-03-05",
    "2019-03-25",
    "Number of daily tweets around the climate strike of 2019-03-15",
    "tweets_per_day_march_2019.png",
)

# %%

import numpy as np


def plot_daily_metric(df, start_date, end_date, title, filename, metric):
    df_subset = df[(df["created_at"] >= start_date) & (df["created_at"] <= end_date)]
    df_subset["aggressiveness"] = df_subset.applymap(
        lambda x: 1 if x == "aggressive" else 0
    )["aggressiveness"]
    df_subset["stance"] = df_subset.applymap(
        lambda x: 1 if x == "believer" else (0.5 if x == "neutral" else 0)
    )["stance"]

    df_subset["created_at"] = pd.to_datetime(df_subset["created_at"])
    df_subset["year_month_day"] = df_subset["created_at"].apply(
        lambda x: f"{x.year}-{x.month:02d}-{x.day:02d}"
    )

    # Calculate the mean and standard deviation of the metric
    df_year_month_day_mean = df_subset.groupby("year_month_day").mean()
    df_year_month_day_std = df_subset.groupby("year_month_day").std()

    # Plot the mean line
    plt.figure(figsize=(8, 3))
    plt.plot(df_year_month_day_mean.index, df_year_month_day_mean[metric], label="Mean")

    # Plot the standard deviation as a shaded area around the mean line
    plt.fill_between(
        df_year_month_day_mean.index,
        df_year_month_day_mean[metric] - df_year_month_day_std[metric],
        df_year_month_day_mean[metric] + df_year_month_day_std[metric],
        alpha=0.3,
        label="Standard Deviation",
    )

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(f"Average {metric}")

    # Make the x-axis labels readable and diagonal
    plt.xticks(rotation=45)

    # Show the legend
    plt.legend()

    plt.savefig(filename)
    plt.close()


# %%

plot_daily_metric(
    df,
    "2019-03-05",
    "2019-03-25",
    "Mean climate sentiment around 2019-03-15",
    "tweet_sentiment_03-15.png",
    "sentiment",
)


# plot_daily_metric(
#     df,
#     "2019-03-05",
#     "2019-03-25",
#     "Mean aggressiveness around 2019-03-15",
#     "tweet_aggressiveness_03-15.png",
#     "aggressiveness",
# )

plot_daily_metric(
    df,
    "2019-03-05",
    "2019-03-25",
    "Mean stance around 2019-03-15",
    "tweet_stance_03-15.png",
    "stance",
)

# Same but for the 2019-09-20 climate strike

plot_daily_metric(
    df,
    "2019-09-10",
    "2019-09-30",
    "Mean climate sentiment around 2019-09-20",
    "tweet_sentiment_09-20.png",
    "sentiment",
)


# plot_daily_metric(
#     df,
#     "2019-09-10",
#     "2019-09-30",
#     "Mean aggressiveness around 2019-09-20",
#     "tweet_aggressiveness_09-20.png",
#     "aggressiveness",
# )


plot_daily_metric(
    df,
    "2019-09-10",
    "2019-09-30",
    "Mean stance around 2019-09-20",
    "tweet_stance_09-20.png",
    "stance",
)


# %%
