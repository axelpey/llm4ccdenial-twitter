# %%
import json
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
            try:
                res = json.loads(res.replace("'", '"'))
            except:
                res = json.loads(res)

            res_array.extend(
                [
                    {
                        "id": x[0],
                        "stance": x[1],
                        "reason": x[2],
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
        except:
            print(f"Error with files {file}. Continuing...")

    return pd.DataFrame(res_array)


# %%

if __name__ == "__main__":
    df = assemble_files()
    df.to_csv("stance_analysis_v1.csv")

# %%

df = assemble_files()

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

plot_daily_tweets(df, "Number of tweets over time", "tweets_soupgate_per_day.png")

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


# %%
plot_daily_metric(
    df, "Average stance over time", "stance_soupgate_per_day.png", "stance"
)
plot_daily_metric(
    df,
    "Std stance over time",
    "stance_soupgate_per_day_std.png",
    "stance",
    std=True,
)

# %%

# Print the user ids that are present more than once in the dataframe
user_ids_multiple_tweets = df["author_id"].value_counts()[
    df["author_id"].value_counts() > 1
]

df_multiple_tweets = df[df["author_id"].isin(user_ids_multiple_tweets.index)]

plot_daily_tweets(
    df_multiple_tweets,
    "Number of tweets over time",
    "tweets_soupgate_per_day_multiple_tweets.png",
)

plot_daily_metric(
    df_multiple_tweets,
    "Average stance over time",
    "stance_soupgate_per_day_multiple_tweets.png",
    "stance",
)

plot_daily_metric(
    df_multiple_tweets,
    "Std stance over time",
    "stance_soupgate_per_day_multiple_tweets_std.png",
    "stance",
    True,
)

# %%

# Subset to include only people who have tweeted before and after the soup on 2022-10-14

# Convert the created_at column to datetime
df["created_at"] = pd.to_datetime(df["created_at"])

# Remove the rows with -1 stance
df = df[df["stance"] != -1]

# Create timezone-aware comparison timestamps
soup_day = pd.Timestamp("2022-10-14", tz="UTC")

# Group by author_id and filter authors based on their minimum and maximum tweet dates
filtered_authors = df.groupby("author_id").filter(
    lambda x: x["created_at"].min() < soup_day and x["created_at"].max() > soup_day
)

# Create a subset DataFrame using the filtered author_ids
df_before_after_soup = df[df["author_id"].isin(filtered_authors["author_id"].unique())]

len(df_before_after_soup)

# %%

import seaborn as sns

# Add a new column 'event' to label tweets as 'Before' or 'After' the soup event
df_before_after_soup["event"] = df_before_after_soup["created_at"].apply(
    lambda x: "Before"
    if x < soup_day
    else ("After" if x > soup_day + pd.Timedelta(days=1) else "Soup Day")
)

# Create a box plot of the stance values before and after the soup event
plt.figure(figsize=(8, 2))
sns.violinplot(
    x="event",
    y="stance",
    data=df_before_after_soup,
    order=["Before", "Soup Day", "After"],
)
plt.title("Stance Distribution Before and After Soup Event")
plt.xlabel("Event")
plt.ylabel("Stance")
plt.savefig("stance_distribution_before_after_soup.png")
plt.show()

# %%

# %%

# Compute summary statistics for the stance values before and after the event
before_stats = df_before_after_soup[df_before_after_soup["event"] == "Before"][
    "stance"
].describe()
after_stats = df_before_after_soup[df_before_after_soup["event"] == "After"][
    "stance"
].describe()

print("Before Soup Event Stance Statistics:")
print(before_stats)
print("\nAfter Soup Event Stance Statistics:")
print(after_stats)

# %%

# Create a line plot showing the daily average stance values
df_before_after_soup["day"] = df_before_after_soup["created_at"].apply(
    lambda x: x.date()
)
daily_avg_stance = (
    df_before_after_soup.groupby(["day", "event"])["stance"].mean().reset_index()
)

plt.figure(figsize=(12, 6))
sns.lineplot(x="day", y="stance", hue="event", data=daily_avg_stance)
plt.title("Daily Average Stance Values Before and After Soup Event")
plt.xlabel("Date")
plt.ylabel("Average Stance")
plt.xticks(rotation=45)
plt.savefig("daily_avg_stance_before_after_soup.png")
plt.show()

# %%

from scipy.stats import ttest_ind

# Extract the stance values before and after the event
before_stance_values = df_before_after_soup[df_before_after_soup["event"] == "Before"][
    "stance"
]
after_stance_values = df_before_after_soup[df_before_after_soup["event"] == "After"][
    "stance"
]

# Perform the two-sample t-test
t_statistic, p_value = ttest_ind(before_stance_values, after_stance_values)

# Print the results
print("Two-sample t-test results:")
print(f"T-statistic: {t_statistic:.4f}")
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
