import streamlit as st
import pandas as pd
import plotly.express as px
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re

# --- Page Setup ---
st.set_page_config(page_title="Airline Sentiment Dashboard", layout="wide")

# --- Topic/Domain Definitions ---
TOPIC_KEYWORDS = {
    "Staff": ["staff", "crew", "service", "attendant", "agent", "customer service",
              "boarding crew", "flight attendants", "steward", "stewardess",
              "helpful", "rude", "friendly", "support"],
    "Timings": ["delay", "delayed", "late", "on time", "schedule", "timing",
                "earlier", "landed earlier", "flight landed",
                "terrible flight delays", "flight cancelled"],
    "Food": ["food", "meal", "drink", "beverage", "snack", "tasty", "water",
             "water bottle", "water bottles", "food tasted", "meal service"],
    "Baggage": ["baggage", "bag", "luggage", "lost", "damaged", "suitcase",
                "lost my baggage", "baggage arrived", "baggage damaged"],
    "Price": ["price", "cost", "cheap", "expensive", "fare", "ticket price",
             "charged extra", "value"],
    "Comfort": ["seat", "seats", "legroom", "comfort", "comfortable",
                "uncomfortable", "dirty", "clean", "tray tables",
                "restroom", "toilet", "cabin", "air conditioning"],
    "Boarding / Checkin": ["boarding", "check in", "check-in", "checkin",
                           "boarding was", "fast boarding", "slow boarding",
                           "check in took forever", "friendly check in"],
    "Wifi / Entertainment": ["wifi", "internet", "entertainment",
                             "no entertainment", "inflight entertainment",
                             "wifi did not work", "wifi worked"],
    "Security": ["security", "security check", "security staff", "polite", "chaotic"],
    "Announcements / Pilot": ["pilot", "announcements", "pilot announcements",
                              "gave updates", "no updates"]
}

COMPILED_KEYWORDS = {
    topic: [re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)
            for kw in keywords]
    for topic, keywords in TOPIC_KEYWORDS.items()
}

@st.cache_resource
def get_vader_model():
    return SentimentIntensityAnalyzer()

def predict_sentiment_with_score(text, analyzer):
    if not isinstance(text, str):
        text = str(text)
    scores = analyzer.polarity_scores(text)
    compound = scores.get("compound", 0.0)
    if compound >= 0.05:
        return "Positive", compound
    elif compound <= -0.05:
        return "Negative", compound
    return "Neutral", compound

def find_topic_by_count(review_text):
    if not isinstance(review_text, str):
        review_text = str(review_text)
    text = review_text.lower()
    match_counts = {topic: sum(len(p.findall(text)) for p in patterns)
                    for topic, patterns in COMPILED_KEYWORDS.items()}
    best_topic = max(match_counts, key=match_counts.get)
    return best_topic if match_counts[best_topic] > 0 else "Other"

@st.cache_data
def analyze_dataframe(df, review_column):
    analyzer = get_vader_model()
    df = df.copy()
    df['review_text'] = df[review_column].fillna("").astype(str).str.strip()
    results = df['review_text'].apply(lambda x: predict_sentiment_with_score(x, analyzer))
    df['Sentiment'] = results.str[0]
    df['SentimentScore'] = results.str[1]
    df['Topic'] = df['review_text'].apply(find_topic_by_count)
    return df

# --- Sidebar ---
st.sidebar.title("Airline Sentiment Dashboard")
st.sidebar.header("Real-Time Review")
review = st.sidebar.text_area("Enter a review:", "The crew was friendly but the flight was delayed.")
analyze_button = st.sidebar.button("Analyze Review")

st.sidebar.header("Batch CSV Analysis")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
review_column = None
if uploaded_file is not None:
    try:
        preview_df = pd.read_csv(uploaded_file, nrows=5)
        review_column = st.sidebar.selectbox("Select review column", preview_df.columns.tolist())
    except:
        st.sidebar.error("Error reading CSV")
        uploaded_file = None

# --- Single Analysis ---
if analyze_button:
    st.header("Real-Time Analysis")
    analyzer = get_vader_model()
    label, score = predict_sentiment_with_score(review, analyzer)
    topic = find_topic_by_count(review)
    st.subheader("Sentiment")
    st.write(f"{label} (score={score:.3f})")
    st.subheader("Topic")
    st.write(topic)

# --- Batch Analysis ---
elif uploaded_file is not None and review_column:
    st.header("Batch Analysis Dashboard")
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)

        if df.empty:
            st.error("CSV is empty")
            st.stop()

        df_analyzed = analyze_dataframe(df, review_column)

        st.subheader("Sample of analyzed data")
        st.dataframe(df_analyzed.head(20))

        # Charts
        st.subheader("Sentiment Distribution")
        sentiment_counts = df_analyzed['Sentiment'].value_counts().reindex(
            ['Positive', 'Neutral', 'Negative']).fillna(0)
        fig = px.pie(names=sentiment_counts.index,
                     values=sentiment_counts.values)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Model Accuracy (if ground truth exists)")
        possible_truth_cols = ['Label', 'TrueSentiment', 'ActualSentiment', 'GroundTruth']
        truth_col = next((col for col in possible_truth_cols if col in df.columns), None)

        if truth_col:
            truth = df[truth_col].astype(str).str.strip().str.capitalize()
            pred = df_analyzed['Sentiment'].astype(str).str.strip().str.capitalize()
            accuracy = (truth == pred).mean() * 100
            st.metric("Accuracy", f"{accuracy:.2f}%")
            st.caption(f"Compared against `{truth_col}` column")
        else:
            st.info("⚠️ No ground-truth sentiment column found.")
    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.header("Welcome!")
    st.info("Analyze a review or upload a CSV using the sidebar!")
