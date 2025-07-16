import pandas as pd
import re
from textblob import TextBlob
import matplotlib.pyplot as plt

# Step 1: Read CSV file
df = pd.read_csv('sample_tweets.csv')

# Step 2: Clean the tweets
def clean_tweet(tweet):
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet)  # Remove links
    tweet = re.sub(r'\@\w+|\#','', tweet)  # Remove mentions and hashtags
    tweet = re.sub(r'[^A-Za-z\s]', '', tweet)  # Remove special characters
    return tweet.lower()

df['cleaned'] = df['tweet'].apply(clean_tweet)

# Step 3: Sentiment Analysis Function
def get_sentiment(tweet):
    analysis = TextBlob(tweet)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity == 0:
        return "Neutral"
    else:
        return "Negative"

df['sentiment'] = df['cleaned'].apply(get_sentiment)

# Step 4: Show the result
print(df)

# Step 5: Visualize the sentiment counts
df['sentiment'].value_counts().plot(kind='bar', title='Sentiment Analysis Result', color=['green', 'gray', 'red'])
plt.xlabel("Sentiment")
plt.ylabel("Number of Texts")
plt.show()

# Step 6: Save output
df.to_csv("sentiment_output.csv", index=False)
