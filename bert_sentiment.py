from transformers import pipeline
import tweepy as tw
import streamlit as st
import pandas as pd
import os

auth = tw.OAuthHandler(os.environ['API_KEY'], os.environ['API_KEY_SECRET'])
auth.set_access_token(os.environ['ACCESS_TOKEN'], os.environ['ACCESS_TOKEN_SECRET'])
api = tw.API(auth, wait_on_rate_limit=True)

# By default downloads the distilbert-base-uncased-finetuned-sst-2-english model
# Uses the DistilBERT architecture 
classifier = pipeline('sentiment-analysis')

st.title('Live Twitter Sentiment Analysis with Tweepy and HuggingFace Transformers')
# st.markdown('This app uses tweepy to get tweets from twitter based on the input name/phrase. It then processes the tweets through HuggingFace transformers pipeline function for sentiment analysis. The resulting sentiments and corresponding tweets are then put in a dataframe for display which is what you see as result.')

def run():
    with st.form(key='input'):
        search_words = st.text_input('Enter the name for which you want to know the sentiment')
        number_of_tweets = st.number_input('Enter the number of latest tweets for which you want to know the sentiment(Maximum 50 tweets)', 0,50,10)
        submit_button = st.form_submit_button(label = 'Submit')
        
        if submit_button:
            tweets = tw.Cursor(api.search_tweets,q=search_words,lang='en').items(number_of_tweets)
            tweet_list = [i.text for i in tweets]
            p = [i for i in classifier(tweet_list)]
            q = [p[i]['label'] for i in range(len(p))]
            df = pd.DataFrame(list(zip(tweet_list, q)),columns =['Latest '+str(number_of_tweets)+ 'Tweets'+' on '+search_words, 'sentiment'])
            df
 
if __name__=='__main__':
    run()