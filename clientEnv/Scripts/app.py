# app.py
import streamlit as st
import pandas as pd
from collections import Counter
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.sentiment import SentimentIntensityAnalyzer

if "results" not in st.session_state:
    st.session_state.results = None
    
st.set_page_config(
    page_title="IMDB Movie Recommendation System",
    page_icon="C:\MAMP\htdocs\imdbMovieRecommendationSystem\clientEnv\Scripts\IMDB_Logo_2016.svg",
    layout="wide"
) 

sia = SentimentIntensityAnalyzer()

left, center, right = st.columns([1, 6, 1])

with center:
    col_logo, col_title = st.columns([1, 6])

    with col_logo:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/6/69/IMDB_Logo_2016.svg",
            width=70
        )

    with col_title:
        st.markdown(
            "<h1 style='color:white; margin-top:-35px;'>Movie Recommendation System</h1>",
            unsafe_allow_html=True
        )
    
userStoryLine = st.text_area(
    "📝 Story Line",
    placeholder="Enter movie story, theme ...",
    height=20
)


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

def extract_keywords(text, stop_words):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)  # remove punctuation & numbers
    words = text.split()
    keywords = [w for w in words if w not in stop_words and len(w) > 2]
    return keywords


def get_sentiment(text):
    score = sia.polarity_scores(text)["compound"]

    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def getMovieRecommendationsList():
        if userStoryLine == '':
                st.error('Enter the Story Line')
        else:
            st.session_state.show_title = False
            st.session_state.run_reco = True
            top_n = 5
            
            cleaned_imdb_df = pd.read_csv('C:\MAMP\htdocs\imdbMovieRecommendationSystem\clientEnv\Scripts\cleaned_imdb_movies_2024.csv')
            cleaned_imdb_df['Cleaned_Storyline'] = cleaned_imdb_df['Cleaned_Storyline'].fillna('')

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
            
            top_5_imdb_user_movie_recommendations["Similarity_Score"] = top_indices

            top_5_imdb_user_movie_recommendations["Sentiment"] = (
                top_5_imdb_user_movie_recommendations["Storyline"]
                .fillna("")
                .apply(get_sentiment)
            )

            viz_df = (
                top_5_imdb_user_movie_recommendations[
                    ["Movie Name", "Similarity_Score"]
                ]
                .sort_values(by="Similarity_Score", ascending=True)
            )

            if st.session_state.run_reco:
                col1, col2 = st.columns([6, 6])
                with col1:
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
                        
                    st.dataframe(display_df, hide_index=True, use_container_width=True, height=250)
                
                all_keywords = []

                for storyline in top_5_imdb_user_movie_recommendations["Storyline"].dropna():
                    all_keywords.extend(extract_keywords(storyline, stop_words))


                keyword_freq = Counter(all_keywords)
                top_keywords = keyword_freq.most_common(10)

                keywords_df = pd.DataFrame(
                    top_keywords,
                    columns=["Keyword", "Frequency"]
                )

                with col2:
                    st.subheader("🔑 Top Storyline Keywords")

                    st.bar_chart(
                        keywords_df.set_index("Keyword"),
                        height=250
                    )
                    
                col1, col2 = st.columns([6, 6])
                with col1:
                    st.subheader("🎯 Similarity Score vs User Storyline")

                    st.bar_chart(
                        viz_df.set_index("Movie Name")
                    )
                
                with col2:
                    sentiment_count = (
                        top_5_imdb_user_movie_recommendations["Sentiment"]
                        .value_counts()
                        .reset_index()
                    )

                    sentiment_count.columns = ["Sentiment", "Count"]
                    
                    st.subheader("😊 Sentiment Distribution of Top 5 Movies")

                    st.bar_chart(
                        sentiment_count.set_index("Sentiment")
                    )


if st.button("Get Recommendations"):
    getMovieRecommendationsList()


imdb_df = pd.read_csv('C:\MAMP\htdocs\imdbMovieRecommendationSystem\clientEnv\Scripts\imdb_movies_2024.csv')

imdb_df['Movie Name'] = imdb_df['Movie Name'].str.replace(r'^\d+\.\s*', '', regex=True)

viz_df = []

# Apply cleaning
imdb_df['Cleaned_Storyline'] = imdb_df['Storyline'].apply(clean_text)
imdb_df['Tokens'] = imdb_df['Cleaned_Storyline'].fillna('').apply(lambda x: x.split())



if "show_title" not in st.session_state:
    st.session_state.show_title = True
    st.session_state.run_reco = False

    
st.markdown("""
<style>
/* Main background */
.stApp {
    background: linear-gradient(to right, #4b6493, #1f4d83);
    color: white;
}

.stAppHeader {
    background: linear-gradient(to right, #4b6493, #1f4d83);
    color: white;
}

/* Card style */
.card {
    background-color: #1f2933;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.3);
}

/* Text area */
textarea {
    border-radius: 10px !important;
    font-size: 16px !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(90deg, #f5c518, #ffdd00);
    color: black;
    font-weight: bold;
    border-radius: 25px;
    padding: 10px 30px;
    border: none;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #ffdd00, #f5c518);
    transform: scale(1.05);
}

p {
    color: white !important;
}
textarea {
    background-color: white !important;
}
</style>
""", unsafe_allow_html=True)