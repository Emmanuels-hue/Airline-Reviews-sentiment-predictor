import streamlit as st
import pandas as pd
import plotly.express as px
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

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

# --- Load Model Once ---
@st.cache_resource
def get_vader_model():
    return SentimentIntensityAnalyzer()

def predict_sentiment(text, analyzer):
    scores = analyzer.polarity_scores(text)
    return "Positive" if scores['compound'] > 0 else "Negative"

def find_topic(review):
    review_lower = str(review).lower()
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(keyword in review_lower for keyword in keywords):
            return topic
    return "Other"

@st.cache_data
def analyze_dataframe(df, review_column):
    analyzer = get_vader_model()
    df = df.copy()
    df['review_text'] = df[review_column].apply(str)
    df['Sentiment'] = df['review_text'].apply(lambda x: predict_sentiment(x, analyzer))
    df['Topic'] = df['review_text'].apply(find_topic)
    return df

# --- Sidebar UI ---
st.sidebar.title("Airline Sentiment Dashboard")

st.sidebar.header("Real-Time Review Analysis")
real_time_review = st.sidebar.text_area("Enter a review:", "The crew was friendly but the flight was delayed.")
real_time_button = st.sidebar.button("Analyze Review")

st.sidebar.header("Batch CSV Analysis")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

review_column = None

if uploaded_file is not None:
    try:
        preview_df = pd.read_csv(uploaded_file, nrows=5)
        review_column = st.sidebar.selectbox("Select Review Column", preview_df.columns.tolist())
    except Exception as e:
        st.sidebar.error(f"Failed to read CSV: {e}")
        uploaded_file = None

# --- Main Outputs ---
if real_time_button:
    st.header("Real-Time Analysis")
    analyzer = get_vader_model()
    sentiment = predict_sentiment(real_time_review, analyzer)
    topic = find_topic(real_time_review)

    st.subheader("Sentiment Result")
    if sentiment == "Positive":
        st.success(f"✅ {sentiment}")
    else:
        st.error(f"❌ {sentiment}")

    st.subheader("Detected Topic")
    st.info(f"📌 {topic}")

elif uploaded_file is not None and review_column is not None:
    st.header("📊 Batch Analysis Dashboard")

    try:
        uploaded_file.seek(0)  # ✅ Important fix!
        df = pd.read_csv(uploaded_file)

        if df.empty:
            st.error("Uploaded file has no data!")
            st.stop()

        df_analyzed = analyze_dataframe(df, review_column)

        # Pie chart
        st.subheader("Overall Sentiment Breakdown")
        sentiment_counts = df_analyzed['Sentiment'].value_counts()
        fig_sentiment = px.pie(sentiment_counts, names=sentiment_counts.index, values=sentiment_counts.values)
        st.plotly_chart(fig_sentiment, use_container_width=True)

        # Topic analysis
        st.subheader("Topic Breakdown")
        col1, col2 = st.columns(2)

        neg_df = df_analyzed[df_analyzed['Sentiment'] == "Negative"]
        fig_neg = px.bar(neg_df['Topic'].value_counts(), title="Top Negative Topics")
        col1.plotly_chart(fig_neg, use_container_width=True)

        pos_df = df_analyzed[df_analyzed['Sentiment'] == "Positive"]
        fig_pos = px.bar(pos_df['Topic'].value_counts(), title="Top Positive Topics")
        col2.plotly_chart(fig_pos, use_container_width=True)

        # Show table
        st.subheader("Analyzed Data Table")
        st.dataframe(df_analyzed)

    except Exception as e:
        st.error(f"An error occurred during analysis: {e}")

else:
    st.header("Welcome 👋")
    st.info("Use the sidebar to analyze a review or upload a CSV.")
