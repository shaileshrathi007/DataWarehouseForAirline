# -*- coding: utf-8 -*-
"""
Created on Wed May 23 21:20:02 2018

@author: Shailesh
"""

import tweepy
import pandas as pd
import numpy as np
from textblob import TextBlob
# For plotting and visualization:
#from IPython.display import display
#import matplotlib.pyplot as plt
#import seaborn as sns
# import unicodedecode

''' Twitter Credentials'''
from textblob import TextBlob
import re
from unidecode import unidecode
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
# os.chdir('/media/viraj/New Volume/Work/DWBI/uni')

os.chdir('/New Volume/Work/DWBI/uni/Tweeter')
CONSUMER_KEY = 'Ali0tFk6wkmVrEJC0CgG9DUJb'
CONSUMER_SECRET = 'h8ZvtSvvzOTnfqFohzX6jJnfkcdatqo5ef1Nu7KGy7Y95GQwlQ'
ACCESS_TOKEN = '871472195108843522-7Xvdm0cqzlH8mcEsg4QO94w6oJCM6rx'
ACCESS_SECRET = 'D7Dia2Y4MmSF3dnz5yOjcYpUVlgdaIEWTNsaWxKg67zLl'


# dep = df['tweet_id'].tolist()

# print(dep[0])


def TwitterSetup():
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)
    api = tweepy.API(auth)
    return api


def clean_tweet(tweet):

    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())


def analize_sentiment(tweet):
    '''
    Utility function to classify the polarity of a tweet
    using textblob.
    '''
    analysis = TextBlob(clean_tweet(tweet))
    if analysis.sentiment.polarity > 0:
        return 1
    elif analysis.sentiment.polarity == 0:
        return 0
    else:
        return -1


def tweets(dept, filename, tweet_id):
    extractor = TwitterSetup()
    analyzer = SentimentIntensityAnalyzer()
    tweets = extractor.user_timeline(screen_name=dept, count=200)
    for tweet in tweets[:200]:
        print(tweet.text)
    data = pd.DataFrame(data=[tweet.text for tweet in tweets[:200]], columns=['Tweets'])
    #data['len'] = np.array([len(tweet.text) for tweet in tweets])
    data['ID'] = np.array([tweet.id for tweet in tweets])
    data['Date'] = np.array([tweet.created_at for tweet in tweets])
    data['Source'] = np.array([tweet.source for tweet in tweets])
    data['Likes'] = np.array([tweet.favorite_count for tweet in tweets])
    #data['RTs'] = np.array([tweet.retweet_count for tweet in tweets])
    data['Sentiments'] = np.array([analize_sentiment(tweet) for tweet in data['Tweets']])
    # data['sents'] ='neg' lambda x: for sents in data['Sentiments'] == -1
    data['university'] = np.array(filename)
    data['tweet_id'] = np.array(tweet_id)
    

    # data = pd.concat([data['Tweets'], data['len'], data['ID'],
    #	data['Date'], data['Source'], data['Likes'],data['RTs'], data['Sentiments']])
    #data = pd.DataFrame(data, columns=['Tweets', 'len','ID','Date','Source','Likes','RTs','Sentiments'])
    #filename = filename + '.csv'
    return data

    #clean_tweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())
    #tweet = unidecode[data['Tweets']]
    #vs = analyzer.polarity_scores(tweet)
    #analysis = TextBlob(tweet)
    #sentiment = vs['compound']
    '''if analysis.sentiment.polarity > 0:
		return 1
	elif analysis.sentiment.polarity == 0:
		return 0
	else:
	    return -1
	data['sentiments'] = np.array([analize_sentiment(tweet) for tweet in data['Tweets']])'''
    # print(data.head(5))
    # print(sentiment)


df = pd.read_csv('tweetid.csv', sep=',')

tweetid = df['TweetId'].unique().tolist()
uni = df['uni'].unique().tolist()
print(tweetid[2])

df1 = tweets(tweetid[0], uni[0], tweetid[0])
data.to_csv('twitter_sentiments.csv', index=False, encoding='utf-8')
df2 = tweets(tweetid[1], uni[1], tweetid[1])
df3 = tweets(tweetid[2], uni[2], tweetid[2])
df4 = tweets(tweetid[3], uni[3], tweetid[3])
df6 = tweets(tweetid[5], uni[5], tweetid[5])
df5 = tweets(tweetid[4], uni[4], tweetid[4])
df7 = tweets(tweetid[6], uni[6], tweetid[6])
df8 = tweets(tweetid[7], uni[7], tweetid[7])
df9 = tweets(tweetid[8], uni[8], tweetid[8])
df10 = tweets(tweetid[9], uni[9], tweetid[9])
df11 = tweets(tweetid[10], uni[10], tweetid[10])
df12 = tweets(tweetid[11], uni[11], tweetid[11])
df13 = tweets(tweetid[12], uni[12], tweetid[12])
df14 = tweets(tweetid[13], uni[13], tweetid[13])
df15 = tweets(tweetid[14], uni[14], tweetid[14])
# df15 = tweets(tweetid[15], uni[15], tweetid[15])


# df9 = tweets(tweetid[9], uni[9], tweetid[9])
# df9 = tweets(tweetid[9], uni[9], tweetid[9])
# df9 = tweets(tweetid[9], uni[9], tweetid[9])
# df9 = tweets(tweetid[9], uni[9], tweetid[9])
# df9 = tweets(tweetid[9], uni[9], tweetid[9])
# df9 = tweets(tweetid[9], uni[9], tweetid[9])
# df9 = tweets(tweetid[9], uni[9], tweetid[9])
# df9 = tweets(tweetid[9], uni[9], tweetid[9])


frame = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15]
data = pd.concat(frame)
data.to_csv('twitter_sentiments.csv', index=False, encoding='utf-8')