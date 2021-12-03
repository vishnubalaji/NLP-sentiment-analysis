from transformers import pipeline
import tweepy as tw
import praw
import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import random
import requests
#import pandas_datareader as pdr
from pandas import json_normalize
from alpha_vantage.timeseries import TimeSeries

# Utils
import joblib 
pipe_lr = joblib.load(open("model/sentiment_classifier.pkl","rb"))
from operator import add
import altair as alt

# Fxn
def predict_emotions(docx):
	results = pipe_lr.predict([docx])
	return results[0]

def get_prediction_proba(docx):
	results = pipe_lr.predict_proba([docx])
	return results

emotions_emoji_dict = {"anger":"😠","disgust":"🤮", "fear":"😨😱", "happy":"🤗", "joy":"😂", "neutral":"😐", "sad":"😔", "sadness":"😔", "shame":"😳", "surprise":"😮"}


REDDITCLIENTID = os.environ['REDDIT_CLIENT_ID']
REDDITCLIENTSECRET = os.environ['REDDIT_CLIENT_SECRET']
USERAGENT = os.environ['USER_AGENT']
USERNAME = os.environ['USERNAME']
PASSWORD = os.environ['PASSWORD']

auth = tw.OAuthHandler(os.environ['API_KEY'], os.environ['API_KEY_SECRET'])
auth.set_access_token(os.environ['ACCESS_TOKEN'], os.environ['ACCESS_TOKEN_SECRET'])
api = tw.API(auth, wait_on_rate_limit=True)

# By default downloads the distilbert-base-uncased-finetuned-sst-2-english model
# Uses the DistilBERT architecture 
classifier = pipeline('sentiment-analysis')

def home():
    home_page = st.sidebar.radio('Welcome to our project!', ['Twitter', 'Reddit','AlphaVantage'])

    if home_page == 'Twitter':
        twitter()
    elif home_page == 'Reddit':
        reddit()
    elif home_page == 'AlphaVantage':
        alpha()

def alpha():
    st.title('AlphaVantage API stock data analysis')
    st.markdown('Fill the below details')
    with st.form(key='form_input'):
        st.write('Welcome to AlphaVantage API stock data analysis')
        keyword=st.text_input('Please enter the name of the company you wish to get financial stock data for:')
        date = st.date_input('Enter the date for which you would like the stock change analysis')
        #date = st.date_input('Enter the date until when to fetch')
        submit_button = st.form_submit_button(label = 'Fetch')
        # aux = 'https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords='+keyword+'&apikey=ZNJRZJFE5TTR9JT7'
        # if submit_button:
        #     av=requests.get(aux)
        #     data=av.json()
        #     st.write(data)
        #confirm from
        # ts = pdr.av.time_series.AVTimeSeriesReader(keyword, api_key='ZNJRZJFE5TTR9JT7')
        # df = ts.read()
        # df.index = pd.to_datetime(df.index, format='%Y-%m-%d')
        date=str(date)
        st.write(f'On Date {date}')
        api_key = 'ZNJRZJFE5TTR9JT7'
        #date = '2021-11-22'
        #confirm to
        ts = TimeSeries(key = api_key, output_format= 'pandas')
        data = ts.get_daily(keyword)
        df = data[0]
        df1 = df.loc[date]
        o = df1.iloc[0]['1. open']
        c = df1.iloc[0]['4. close']
        percent_change = 100*(c - o)/o
        st.write(f'Change in opening and closing value, in terms of percentage : {percent_change} ')
        #dataframe[dataframe['Percentage'] >80]
        
def reddit():
    st.title('Reddit Sentiment Analysis')
    st.markdown('Fill the form')
    with st.form(key='form_input'):
        st.write('Welcome to Reddit Sentiment Analyser. Please click the Fetch button to analyse comments of the top post as of now')
        #number_of_posts=st.number_input('Enter the number of latest posts(Maximum 10 posts)', min_value = 0, max_value = 10, value = 1)
        submit_button = st.form_submit_button(label = 'Fetch')
        #st.write('Nothing here to show. Mind your business -_-')
        if submit_button:
            reddit = praw.Reddit(client_id = REDDITCLIENTID, client_secret = REDDITCLIENTSECRET, user_agent = USERAGENT, username = USERNAME, password = PASSWORD)
            subreddit=reddit.subreddit('wallstreetbets').hot(limit=1)
            #subroutine to get the comment id
            id_list=[]
            for i in subreddit:
              id_list.append(i.id)
            post_id=id_list[0]
            submission = reddit.submission(post_id)
            post_title=submission.title
            submission.comments.replace_more(limit=0)
            comments_list=[]
            for top_level_comments in submission.comments:
              comments_list.append(top_level_comments.body)
            #comments_list
            #comments_list has 50+ comments, limiting to 25 for easy training of model.
            comment_list=comments_list[1:25] #0th index is metadata, we dont want to confuse poor distilbert

            emotion_list = [emotion for emotion in classifier(comment_list)]

            emotion_label = [emotion['label'] for emotion in emotion_list]
            emotion_score = [emotion['score'] for emotion in emotion_list]

            label_list = [emotion_list[i]['label'] for i in range(len(emotion_list))]
            df = pd.DataFrame(
                list(zip(comment_list, emotion_label, emotion_score)),
                columns =['Latest post on '+post_title, 'Sentiment', 'Score']
            )
            df
            negative_count = (df['Sentiment'] == 'NEGATIVE').sum()
            positive_count = (df['Sentiment'] == 'POSITIVE').sum()

            st.write(f'Negative count : {negative_count}    Positive count : {positive_count}')
            count = [i for i in range(0,51,10)]
            fig = plt.figure(figsize=(10,7))
            sns.barplot(x='Sentiment', y='Score', data=df, order=['NEGATIVE','POSITIVE'])
            st.pyplot(fig)

def twitter():
    st.title('Twitter Sentiment Analysis')
    st.markdown('Fill the form')
    with st.form(key='form_input'):
        search_word = st.text_input('Enter the word')
        number_of_tweets = st.number_input('Enter the number of latest tweets(Maximum 50 tweets)', min_value = 0, max_value = 50, value = 1)
        date_since = st.date_input('Enter the date until when to fetch')
        submit_button = st.form_submit_button(label = 'Fetch')
        
        if submit_button:
            tweets = api.search_tweets(q=search_word, count = number_of_tweets, result_type='mixed', until = date_since, lang='en')
            tweet_list = [tweets[i].text for i in range(number_of_tweets)]
            tweet_location = [tweets[i].user.location for i in range(number_of_tweets)]
            emotion_list = [emotion for emotion in classifier(tweet_list)]
            sentiment_list = [predict_emotions(tweet) for tweet in tweet_list]
            sentiment_proba_list = [get_prediction_proba(tweet) for tweet in tweet_list]
            total_sentiment = sentiment_proba_list[0]
            for i in range(1,len(sentiment_proba_list)):
                total_sentiment[0] = list( map(add, total_sentiment[0], sentiment_proba_list[i][0]) )
            avg_sentiment = [[sentiment/len(sentiment_proba_list) for sentiment in total_sentiment[0]]]

            emotion_label = [emotion['label'] for emotion in emotion_list]
            emotion_score = [emotion['score'] for emotion in emotion_list]

            label_list = [emotion_list[i]['label'] for i in range(len(emotion_list))]
            df = pd.DataFrame(
                list(zip(tweet_list, emotion_label, emotion_score,sentiment_list)),
                columns =['Latest '+str(number_of_tweets)+ ' tweets'+' on '+search_word, 'Emotion', 'Score', "Sentiment"]
            )
            df
            negative_count = (df['Sentiment'] == 'NEGATIVE').sum()
            positive_count = (df['Sentiment'] == 'POSITIVE').sum()

            st.write(f'Negative count : {negative_count}    Positive count : {positive_count}')
            count = [i for i in range(0,51,10)]
            fig = plt.figure(figsize=(10,7))
            sns.barplot(x='Sentiment', y='Score', data=df, order=['NEGATIVE','POSITIVE'])
            st.pyplot(fig)

            st.success("Prediction Probability")
            # st.write(probability)
            proba_df = pd.DataFrame(avg_sentiment,columns=pipe_lr.classes_)
            # st.write(proba_df.T)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["emotions","probability"]

            fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions',y='probability',color='emotions')
            st.altair_chart(fig,use_container_width=True)

    
if __name__=='__main__':
    home()