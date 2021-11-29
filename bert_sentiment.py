from transformers import pipeline
import tweepy as tw
import praw
import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

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
    home_page = st.sidebar.radio('Welcome to our project!', ['Twitter', 'Reddit'])

    if home_page == 'Twitter':
        twitter()
    elif home_page == 'Reddit':
        reddit()

def reddit():
    reddit = praw.Reddit(client_id = REDDITCLIENTID, client_secret = REDDITCLIENTSECRET, user_agent = USERAGENT, username = USERNAME, password = PASSWORD)
    number_of_posts = st.number_input('Enter the number of latest posts(Maximum 50 posts)', min_value = 0, max_value = 50, value = 1)
    subreddit=reddit.subreddit('wallstreetbets').hot(limit=number_of_posts)
    #submission.comments.replace_more(limit=0)
    comments_list=[]
    for top_level_comments in submission.comments:
      comments_list.append(top_level_comments.body)
    #comments_list
    submit_button = st.form_submit_button(label = 'Fetch')
    # st.write('Nothing here to show. Mind your business -_-')
    if submit_button:
        #subroutine to get the comment id
        id_list=[]
        for i in subreddit:
          id_list.append(i.id)
        submission = reddit.submission(random.choice(id_list))
        submission.comments.replace_more(limit=0)
        comments_list=[]
        for top_level_comments in submission.comments:
          comments_list.append(top_level_comments.body)
        #comments_list
        #comments_list has 50+ comments, limiting to 15 for easy training of model.
        comment_list=comments_list[1:21] #0th index is metadata, we dont want to confuse poor distilbert
        
        emotion_list = [emotion for emotion in classifier(comment_list)]
        
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