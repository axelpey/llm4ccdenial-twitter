import tweepy
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("The Climate Change Twitter Dataset.csv")

api_key = json.load(open("bearer_tokens.json"))[0]["token"]

client = tweepy.Client(bearer_token=api_key)
