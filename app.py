import streamlit as st
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px
import os

# --- Page Setup ---
st.set_page_config(page_title="Airline Sentiment Dashboard", layout="wide")


# --- Topic/Domain Definitions ---
TOPIC_KEYWORDS = {
    "Staff": ["staff", "crew", "service", "rude", "friendly", "helpful", "unprofessional", "attendant"],
    "Timings": ["late", "delay", "on time", "schedule", "early", "delayed", "timing"],
    "Food": ["food", "meal", "drink", "beverage", "snack", "tasty", "awful meal"],
    "Baggage": ["bag", "baggage", "luggage", "lost", "damaged", "suitcase"],
    "Price": ["price", "cost", "cheap", "expensive", "value", "fare", "ticket price"],
    "Comfort": ["seat", "comfort", "legroom", "clean", "dirty", "comfortable"],
}


# --- Model and Functions (Cached for speed) ---

@st.cache_resource
def get_vader_model():
    """Loads the VADER model once."""
    return SentimentIntensityAnalyzer()

def predict_sentiment(text, analyzer):
    """Predicts 'Positive' or 'Negative' (no neutral)."""
    scores = analyzer.polarity_scores(text)
    return "Positive" if scores['compound'] > 0 else "Negative"

def find_topic(review):
    """Finds the first matching topic for a review."""
    if not isinstance(review, str):
        return "Other"
        
    review_lower = review.lower()
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(keyword in review_lower for keyword in keywords):
            return topic
    return "Other"

@st.cache_data
def analyze_dataframe(_df, review_column):
    """Runs full analysis on the uploaded dataframe."""
    analyzer = get_vader_model()
    
    # Create a new copy to avoid cache warnings
    df = _df.copy()
    
    df['review_text'] = df[review_column].astype(str)
    df['Sentiment'] = df['review_text'].apply(lambda x: predict_sentiment(x, analyzer))
    df['Topic'] = df['review_text'].apply(find_topic)
    return df


# --- Sidebar (All Controls) ---

st.sidebar.title("Airline Sentiment Dashboard")

st.sidebar.header("Real-Time Review Analysis")
real_time_review = st.sidebar.text_area(
    "Enter a single review to analyze:", 
    "The crew was very friendly, but the flight was delayed."
)
real_time_button = st.sidebar.button("Analyze Review")

st.sidebar.header("Batch CSV Analysis")
uploaded_file = st.sidebar.file_uploader("Upload your review CSV file", type=["csv"])

review_column = None
if uploaded_file is not None:
    try:
        # Read only the first 5 rows to get column names
        df_preview = pd.read_csv(uploaded_file, nrows=5)
        available_columns = df_preview.columns.tolist()
        review_column = st.sidebar.selectbox(
            "Which column has the review text?",
            available_columns,
            index=0 
        )
    except Exception as e:
        st.sidebar.error(f"Error reading CSV header: {e}")
        uploaded_file = None
        

# --- Main Page (All Results) ---

if real_time_button:
    st.header("Real-Time Analysis")
    analyzer = get_vader_model()
    sentiment = predict_sentiment(real_time_review, analyzer)
    topic = find_topic(real_time_review)
    
    st.subheader("Sentiment")
    if sentiment == "Positive":
        st.success(f"**{sentiment}**")
    else:
        st.error(f"**{sentiment}**")
        
    st.subheader("Detected Topic")
    st.info(f"**{topic}**")

elif uploaded_file is not None and review_column is not None:
    st.header("Batch Analysis Dashboard")
    
    try:
        # Now, load and analyze the *full* CSV
        df = pd.read_csv(uploaded_file)
        df_analyzed = analyze_dataframe(df, review_column)

        # 2. Show Sentiment Pie Chart
        st.subheader("Overall Sentiment Breakdown")
        sentiment_counts = df_analyzed['Sentiment'].value_counts()
        fig_pie = px.pie(
            sentiment_counts,
            names=sentiment_counts.index,
            values=sentiment_counts.values,
            title="Overall Sentiment"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        # 3. Show Topic Bar Charts
        st.subheader("Topic Analysis")
        col1, col2 = st.columns(2)
        
        neg_df = df_analyzed[df_analyzed['Sentiment'] == 'Negative']
        neg_topics = neg_df['Topic'].value_counts()
        fig_neg_bar = px.bar(
            neg_topics,
            x=neg_topics.index,
            y=neg_topics.values,
            title="Top Negative Topics",
            labels={'x': 'Topic', 'y': 'Number of Reviews'}
        )
        col1.plotly_chart(fig_neg_bar, use_container_width=True)

        pos_df = df_analyzed[df_analyzed['Sentiment'] == 'Positive']
        pos_topics = pos_df['Topic'].value_counts()
        fig_pos_bar = px.bar(
            pos_topics,
            x=pos_topics.index,
            y=pos_topics.values,
            title="Top Positive Topics",
            labels={'x': 'Topic', 'y': 'Number of Reviews'}
        )
        col2.plotly_chart(fig_pos_bar, use_container_width=True)

        # 4. Show the full analyzed data
        st.subheader("Analyzed Data")
        st.write("Here is your uploaded data with the new 'Sentiment' and 'Topic' columns.")
        st.dataframe(df_analyzed)
        
    except Exception as e:
        st.error(f"An error occurred during analysis: {e}")

else:
    # This is the default welcome page
    st.header("Welcome to the Airline Sentiment Dashboard")
    st.info("Use the controls in the sidebar to analyze a single review or upload a full CSV.")
