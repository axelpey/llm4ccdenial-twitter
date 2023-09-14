# %%
from math import trunc
import re
import pandas as pd
import glob
import matplotlib.pyplot as plt


# %%
def assemble_files():
    files = glob.glob("gpt_calls_v1/res_('2022-10-01', '2022-10-28')_*.txt")
    df_original = pd.read_csv("subsets_v1/hydrated_tweets_2022-10-01_2022-10-28.csv")

    res_array = []

    for file in files:
        res = open(file).read()
        try:
            # remove the line jumps at the beginning of res until the first character
            res = re.sub(r"^\s+", "", res)
            res = res.split("\n")
            for l in range(len(res)):
                t_id, stance, sentiment, agg = res[l].split(",")
                res[l] = [
                    int(t_id),
                    float(stance),
                    float(sentiment),
                    int(trunc(float(agg))),
                ]

            res_array.extend(
                [
                    {
                        "id": x[0],
                        "stance": x[1],
                        "sentiment": x[2],
                        "aggressiveness": x[3],
                        "text": df_original[df_original["id"] == x[0]]["text"].values[
                            0
                        ],
                        "created_at": df_original[df_original["id"] == x[0]][
                            "created_at"
                        ].values[0],
                        "author_id": df_original[df_original["id"] == x[0]][
                            "author_id"
                        ].values[0],
                    }
                    for x in res
                ]
            )
        except Exception as e:
            print(e)
            print(f"Error with files {file}. Continuing...")

    return pd.DataFrame(res_array)


# %%

df = assemble_files()
# df.loc[df["stance"] == -1, "stance"] = 0.5
df = df[df["stance"] != -1]
# df[df["stance"] == -1]["stance"] = 0.5

# %%


def plot_daily_tweets(df, title, filename):
    df_subset = df

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
    df, "Number of tweets over time", "tweets_usuals_soupgate_per_day.png"
)

# %%


def plot_daily_metric(df, title, filename, metric, std=False):
    df_subset = df
    # df_subset["aggressiveness"] = df_subset.applymap(
    #     lambda x: 1 if x == "aggressive" else 0
    # )["aggressiveness"]

    # Drop the rows where stance is -1
    df_subset = df_subset[df_subset["stance"] != -1]

    # Drop the rows where stance is 0.5
    df_subset = df_subset[df_subset["stance"] != 0.5]

    # df_subset["stance"] = df_subset.applymap(
    #     lambda x: 1 if x == "believer" else (0.5 if x == "neutral" else 0)
    # )["stance"]

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
    if std:
        df_year_month_day = df_subset.groupby("year_month_day").std()
    else:
        df_year_month_day = df_subset.groupby("year_month_day").mean()

    plt.figure(figsize=(8, 3))
    plt.plot(df_year_month_day.index, df_year_month_day[metric])
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(f"Average {metric}")

    # Make the x-axis labels readable and diagonal
    plt.xticks(rotation=45)

    plt.savefig(filename)
    plt.close()


def plot_daily_metric_filled(df, title, filename, metric, std=False):
    df_subset = df
    # df_subset["aggressiveness"] = df_subset.applymap(
    #     lambda x: 1 if x == "aggressive" else 0
    # )["aggressiveness"]

    # Drop the rows where stance is -1
    df_subset = df_subset[df_subset["stance"] != -1]

    # Drop the rows where stance is 0.5
    df_subset = df_subset[df_subset["stance"] != 0.5]

    # df_subset["stance"] = df_subset.applymap(
    #     lambda x: 1 if x == "believer" else (0.5 if x == "neutral" else 0)
    # )["stance"]

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
    # Calculate the mean and standard deviation of the metric
    df_year_month_day_mean = df_subset.groupby("year_month_day").mean()
    df_year_month_day_std = df_subset.groupby("year_month_day").std()

    # Plot the mean line
    plt.figure(figsize=(8, 5))
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
    df, "Average stance over time", "stance_usuals_soupgate_per_day.png", "stance"
)
plot_daily_metric(
    df,
    "Std stance over time",
    "stance_usuals_soupgate_per_day_std.png",
    "stance",
    std=True,
)

plot_daily_metric_filled(
    df, "Stance over time", "filled_stance_usuals_soupgate_per_day.png", "stance"
)

# %%

soup_day = pd.Timestamp("2022-10-14", tz="UTC")

import seaborn as sns

TWO_WEEKS_STUDY = False

if TWO_WEEKS_STUDY:
    # Add a new column 'event' to label tweets as 'Before' or 'After' the soup event
    df["event"] = df["created_at"].apply(
        lambda x: "Before"
        if x < soup_day
        else ("After" if x > soup_day + pd.Timedelta(days=1) else "Soup Day")
    )
else:
    # Add a new column 'event' to label tweets as 'Before' or 'After' the soup event
    df["event"] = df["created_at"].apply(
        lambda x: "Before"
        if (x <= soup_day and x > soup_day - pd.Timedelta(days=7))
        else (
            "After"
            if (
                x > soup_day + pd.Timedelta(days=1)
                and x <= soup_day + pd.Timedelta(days=8)
            )
            else (
                "Long before"
                if (x <= soup_day - pd.Timedelta(days=7))
                else (
                    "Long after"
                    if x >= soup_day + pd.Timedelta(days=7)
                    else (
                        "Soup Day"
                        if (x > soup_day and x < soup_day + pd.Timedelta(days=1))
                        else "Error"
                    )
                )
            )
        )
    )

# Create a box plot of the stance values before and after the soup event
plt.figure(figsize=(12, 6))
sns.violinplot(
    x="event",
    y="stance",
    data=df,
    order=["Before", "Soup Day", "After"],
)
plt.title("Stance Distribution Before and After Soup Event")
plt.xlabel("Event")
plt.ylabel("Stance")
plt.savefig("stance_usuals_distribution_before_after_soup.png")
plt.show()

# %%

# %%

# Compute summary statistics for the stance values before and after the event
before_stats = df[df["event"] == "Before"]["stance"].describe()
after_stats = df[df["event"] == "After"]["stance"].describe()
soup_day_stats = df[df["event"] == "Soup Day"]["stance"].describe()

print("Before Soup Event Stance Statistics:")
print(before_stats)
print("\nAfter Soup Event Stance Statistics:")
print(after_stats)
print("\nSoup Day Stance Statistics:")
print(soup_day_stats)

# %%

# Create a line plot showing the daily average stance values
df["day"] = df["created_at"].apply(lambda x: x.date())
daily_avg_stance = df.groupby(["day", "event"])["stance"].mean().reset_index()

plt.figure(figsize=(10, 6))
plt.ylim(0.5, 0.95)
sns.lineplot(x="day", y="stance", hue="event", data=daily_avg_stance)
plt.title("Daily Average Stance Values Before and After Soup Event")
plt.xlabel("Date")
plt.ylabel("Average Stance")
plt.xticks(rotation=45)
plt.savefig("daily_avg_stance_before_after_soup.png")
plt.show()

# %%

# Extract the stance values before and after the event
before_stance_values = df[df["event"] == "Before"]["stance"]
after_stance_values = df[df["event"] == "After"]["stance"]

from scipy.stats import chi2_contingency

# Create contingency table of before and after stance values
before_counts = before_stance_values.value_counts().sort_index()
after_counts = after_stance_values.value_counts().sort_index()
contingency_table = pd.concat(
    [before_counts, after_counts], axis=1, keys=["Before", "After"]
).fillna(0)

# Perform chi-square test
chi2_statistic, p_value, dof, expected = chi2_contingency(contingency_table)

# Print the results
print("Chi-square test results:")
print("Comparing between before and after day:")
print(f"Chi-square statistic: {chi2_statistic:.4f}")
print(f"P-value: {p_value:.4f}")

# Check if the p-value is below a certain significance level, e.g., 0.05
if p_value < 0.05:
    print(
        "\nThere is a significant difference in the stance values before and after the event (p < 0.05)."
    )
else:
    print(
        "\nThere is no significant difference in the stance values before and after the event (p >= 0.05)."
    )


# Same but during and before the soup day
soupday_stance_values = df[df["event"] == "Soup Day"]["stance"]
soupday_counts = soupday_stance_values.value_counts().sort_index()

contingency_table = pd.concat(
    [before_counts, soupday_counts], axis=1, keys=["Before", "Soup day"]
).fillna(0)

print("Comparing between before and soup day:")

chi2_statistic, p_value, dof, expected = chi2_contingency(contingency_table)

# Print the results
print("Chi-square test results:")
print("Comparing between before and after day:")
print(f"Chi-square statistic: {chi2_statistic:.4f}")
print(f"P-value: {p_value:.4f}")

# Check if the p-value is below a certain significance level, e.g., 0.05
if p_value < 0.05:
    print(
        "\nThere is a significant difference in the stance values before and after the event (p < 0.05)."
    )
else:
    print(
        "\nThere is no significant difference in the stance values before and after the event (p >= 0.05)."
    )


# %%

# Now we want to have one person count at most once per day
# When it's more than once, we want to take the mean of the stance, sentiment and aggressiveness
# for that user on that day.


# %%
def aggregate_daily_user_metrics(df):
    # First, convert the created_at column to a date (without time) and store it as a new column
    df["day"] = df["created_at"].apply(lambda x: x.date())

    # Group by the day, author_id, and event, and then aggregate the metrics by taking the mean
    df_agg = (
        df.groupby(["day", "author_id", "event"])
        .agg(
            {
                "stance": "mean",
                "sentiment": "mean",
                "aggressiveness": "mean",
            }
        )
        .reset_index()
    )

    return df_agg


# %%
# Aggregate the daily user metrics
df_agg = aggregate_daily_user_metrics(df)

# %%
# Now, plot the daily average stance values for the aggregated data
daily_avg_stance_agg = df_agg.groupby(["day", "event"])["stance"].mean().reset_index()

plt.figure(figsize=(12, 6))
sns.lineplot(x="day", y="stance", hue="event", data=daily_avg_stance_agg)
plt.title("Daily Average Stance Values (Aggregated) Before and After Soup Event")
plt.xlabel("Date")
plt.ylabel("Average Stance")
plt.xticks(rotation=45)
plt.savefig("daily_avg_stance_agg_before_after_soup.png")
plt.show()


# %%

before_stats = df_agg[df_agg["event"] == "Before"]["stance"].describe()
after_stats = df_agg[df_agg["event"] == "After"]["stance"].describe()
soup_day_stats = df_agg[df_agg["event"] == "Soup Day"]["stance"].describe()

print("Before Soup Event Stance Statistics:")
print(before_stats)
print("\nAfter Soup Event Stance Statistics:")
print(after_stats)
print("\nSoup Day Stance Statistics:")
print(soup_day_stats)

# %%

# Perform the two-sample t-test on the aggregated data
before_stance_values_agg = df_agg[df_agg["event"] == "Before"]["stance"]
after_stance_values_agg = df_agg[df_agg["event"] == "After"]["stance"]
from scipy.stats import chi2_contingency

# Create contingency table of before and after stance values
before_counts = before_stance_values_agg.value_counts().sort_index()
after_counts = after_stance_values_agg.value_counts().sort_index()
contingency_table = pd.concat(
    [before_counts, after_counts], axis=1, keys=["Before", "After"]
).fillna(0)

# Perform chi-square test
chi2_statistic, p_value, dof, expected = chi2_contingency(contingency_table)

# Print the results
print("Chi-square test results:")
print("Comparing between before and after day:")
print(f"Chi-square statistic: {chi2_statistic:.4f}")
print(f"P-value: {p_value:.4f}")

# Check if the p-value is below a certain significance level, e.g., 0.05
if p_value < 0.05:
    print(
        "\nThere is a significant difference in the stance values before and after the event (p < 0.05)."
    )
else:
    print(
        "\nThere is no significant difference in the stance values before and after the event (p >= 0.05)."
    )


# Same but during and before the soup day
soupday_stance_values = df_agg[df_agg["event"] == "Soup Day"]["stance"]
soupday_counts = soupday_stance_values.value_counts().sort_index()

contingency_table = pd.concat(
    [before_counts, soupday_counts], axis=1, keys=["Before", "Soup day"]
).fillna(0)

print("Comparing between before and soup day:")

chi2_statistic, p_value, dof, expected = chi2_contingency(contingency_table)

# Print the results
print("Chi-square test results:")
print("Comparing between before and after day:")
print(f"Chi-square statistic: {chi2_statistic:.4f}")
print(f"P-value: {p_value:.4f}")

# Check if the p-value is below a certain significance level, e.g., 0.05
if p_value < 0.05:
    print(
        "\nThere is a significant difference in the stance values before and after the event (p < 0.05)."
    )
else:
    print(
        "\nThere is no significant difference in the stance values before and after the event (p >= 0.05)."
    )

# %%
# Now we also want to see how aggressive the users are before, durign and after the soup event

plot_daily_metric(
    df,
    "Mean Aggressiveness",
    "aggressiveness_usuals_soupgate_per_day.png",
    "aggressiveness",
)
plot_daily_metric(
    df,
    "Std Aggressiveness",
    "aggressiveness_usuals_soupgate_per_day_std.png",
    "aggressiveness",
    std=True,
)


# %%
# Now we also want to see the sentiment of the users before, durign and after the soup event

plot_daily_metric(
    df,
    "Mean Sentiment",
    "sentiment_usuals_soupgate_per_day.png",
    "sentiment",
)

plot_daily_metric(
    df,
    "Std Sentiment",
    "sentiment_usuals_soupgate_per_day_std.png",
    "sentiment",
    std=True,
)

# %%

# Next thing to do is see what happens in the last 3 days before the 28, as it
# seems that the stance values are higher in the last 3 days before the 28

# How can I perform a quick analysis of the last 3 days before the 28?
# Check what's in the tweets of these days?
# Maybe extract their texts and feed it to an LLM so that it tells me what are common topics?

from datetime import datetime, timedelta

# Set the end date and number of days to look back
end_date = datetime.strptime("2022-10-28", "%Y-%m-%d").date()
days_to_look_back = 3
tweet_limit = 70

# Calculate the start date
start_date = end_date - timedelta(days=days_to_look_back)

# Filter the dataframe to only include tweets between the start and end dates
filtered_df = df[(df["day"] >= start_date) & (df["day"] < end_date)]

# If the number of tweets exceeds the limit, sample a subset of them
if len(filtered_df) > tweet_limit:
    filtered_df = filtered_df.sample(n=tweet_limit, random_state=42)

# Extract the text of the tweets
tweet_texts = filtered_df["text"].tolist()

# Concatenate the tweet texts into a single string, separated by newline characters
tweet_texts_str = "\n".join(tweet_texts)

# You can now use `tweet_texts_str` as input to ChatGPT.

# %%
