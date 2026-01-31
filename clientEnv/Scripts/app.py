# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

if "results" not in st.session_state:
    st.session_state.results = None
    
st.set_page_config(
    page_title="IMDB Movie Recommendation System",
    page_icon="C:\MAMP\htdocs\imdbMovieRecommendationSystem\clientEnv\Scripts\IMDB_Logo_2016.svg",
    layout="wide"
) 


col1, col2, col3 = st.columns([5, 2, 5])
with col2:
    st.image("C:/MAMP/htdocs/imdbMovieRecommendationSystem/clientEnv/Scripts/IMDB_Logo_2016.svg",width=120)

st.markdown(
    "<h2 style='text-align:center;'>IMDB Movie Recommendation System</h2>",
    unsafe_allow_html=True
)

imdb_df = pd.read_csv('C:\MAMP\htdocs\imdbMovieRecommendationSystem\clientEnv\Scripts\imdb_movies_2024.csv')

imdb_df['Movie Name'] = imdb_df['Movie Name'].str.replace(r'^\d+\.\s*', '', regex=True)


# Attempt to download stopwords
try:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
except Exception as e:
    print(f"NLTK download failed: {e}")
    # Fallback to a basic list of English stop words
    stop_words = set(["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"])

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers/unnecessary characters (optional, but requested "unnecessary characters")
    text = re.sub(r'\d+', '', text)
    # Remove stop words
    words = text.split()
    words = [w for w in words if w not in stop_words]
    # Join and strip
    return " ".join(words).strip()

# Apply cleaning
imdb_df['Cleaned_Storyline'] = imdb_df['Storyline'].apply(clean_text)
imdb_df['Tokens'] = imdb_df['Cleaned_Storyline'].fillna('').apply(lambda x: x.split())

cleaned_imdb_df = pd.read_csv('C:\MAMP\htdocs\imdbMovieRecommendationSystem\clientEnv\Scripts\cleaned_imdb_movies_2024.csv')
cleaned_imdb_df['Cleaned_Storyline'] = cleaned_imdb_df['Cleaned_Storyline'].fillna('')

if "show_title" not in st.session_state:
    st.session_state.show_title = True
    st.session_state.run_reco = False

def run_recommendation():
    st.session_state.run_reco = True
    
col1, col2 = st.columns([6, 6])
with col1:
    st.markdown("<br>", unsafe_allow_html=True)
    
    form = st.form(key="userStoryLine_form")
    userStoryLine = form.text_area("Story Line")
    
    if userStoryLine:
        st.session_state.show_title = True
        
    submitted = form.form_submit_button("Get Recommendations", on_click=run_recommendation)

with col2:
    if st.session_state.run_reco:
        st.markdown(
            "<h4 style='text-align:center;'>Top 5 🎬 Recommended Movies</h4>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<h4 style='text-align:center;'></h4>",
            unsafe_allow_html=True
        )

    
    if submitted:
        if userStoryLine == '':
                st.error('Enter the Story Line')
        else:
            st.session_state.show_title = False
            st.session_state.run_reco = True
            top_n = 5
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(cleaned_imdb_df['Cleaned_Storyline'])

            user_input = userStoryLine
            cleaned_user_input = clean_text(user_input)
            user_vector = vectorizer.transform([cleaned_user_input])

            cosine_sim_user = cosine_similarity(user_vector, tfidf_matrix)
            top_indices = cosine_sim_user.argsort()[0][-top_n:][::-1]

            top_5_imdb_user_movie_recommendations = cleaned_imdb_df.iloc[top_indices]
            
            display_df = top_5_imdb_user_movie_recommendations.drop(
                columns=["Cleaned_Storyline"],
                errors="ignore",
            )

            with col2:
                if st.session_state.run_reco:
                    st.dataframe(display_df, hide_index=True)
