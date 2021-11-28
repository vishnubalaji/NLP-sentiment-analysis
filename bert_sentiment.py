from transformers import pipeline
import tweepy as tw
import praw
import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

REDDIT_CLIENT_ID = os.environ['REDDIT_CLIENT_ID']
REDDIT_CLIENT_SECRET = os.environ['REDDIT_CLIENT_SECRET']
USER_AGENT = os.environ['USER_AGENT']
USERNAME = os.environ['USERNAME']
PASSWORD = os.environ['PASSWORD']

reddit = praw.Reddit(client_id = REDDIT_CLIENT_ID, client_secret = REDDIT_CLIENT_SECRET, user_agent = USER_AGENT, username = USERNAME, password = PASSWORD)

auth = tw.OAuthHandler(os.environ['API_KEY'], os.environ['API_KEY_SECRET'])
auth.set_access_token(os.environ['ACCESS_TOKEN'], os.environ['ACCESS_TOKEN_SECRET'])
api = tw.API(auth, wait_on_rate_limit=True)

# By default downloads the distilbert-base-uncased-finetuned-sst-2-english model
# Uses the DistilBERT architecture 
classifier = pipeline('sentiment-analysis')

def home():
    home_page = st.sidebar.radio('Welcome to our project!', ['Twitter', 'Reddit'])

    if home_page == 'Twitter':
        twitter()
    elif home_page == 'Reddit':
        reddit()

def reddit():
    subreddit=reddit.subreddit('wallstreetbets')
    st.write('Nothing here to show. Mind your business -_-')

def twitter():
    st.title('Twitter Sentiment Analysis')
    st.markdown('Fill the form')
    with st.form(key='form_input'):

        type_of_word = st.radio('Choose the type of word',['Trading','Universal'])
        search_word = st.text_input('Enter the word')
        number_of_tweets = st.number_input('Enter the number of latest tweets(Maximum 50 tweets)', min_value = 0, max_value = 50, value = 1)
        date_since = st.date_input('Enter the date until when to fetch')
        submit_button = st.form_submit_button(label = 'Fetch')
        
        if submit_button:
            tweets = api.search_tweets(q=search_word, count = number_of_tweets, result_type='mixed', until = date_since, lang='en')
            tweet_list = [tweets[i].text for i in range(number_of_tweets)]
            tweet_location = [tweets[i].user.location for i in range(number_of_tweets)]
            emotion_list = [emotion for emotion in classifier(tweet_list)]

            emotion_label = [emotion['label'] for emotion in emotion_list]
            emotion_score = [emotion['score'] for emotion in emotion_list]

            label_list = [emotion_list[i]['label'] for i in range(len(emotion_list))]
            df = pd.DataFrame(
                list(zip(tweet_list, emotion_label, emotion_score)),
                columns =['Latest '+str(number_of_tweets)+ ' tweets'+' on '+search_word, 'Sentiment', 'Score']
            )
            df
            negative_count = (df['Sentiment'] == 'NEGATIVE').sum()
            positive_count = (df['Sentiment'] == 'POSITIVE').sum()

            st.write(f'Negative count : {negative_count}    Positive count : {positive_count}')
            count = [i for i in range(0,51,10)]
            fig = plt.figure(figsize=(10,7))
            sns.barplot(x='Sentiment', y='Score', data=df, order=['POSITIVE','NEGATIVE'])
            st.pyplot(fig)

if __name__=='__main__':
    home()