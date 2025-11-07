import streamlit as st
import pandas as pd
import plotly.express as px
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re

# --- Page Setup ---
st.set_page_config(page_title="Airline Sentiment Dashboard", layout="wide")

# --- Topic/Domain Definitions (expand/modify as needed) ---
TOPIC_KEYWORDS = {
    "Staff": [
        "staff", "crew", "service", "attendant", "agent", "customer service",
        "boarding crew", "flight attendants", "steward", "stewardess", "helpful",
        "rude", "friendly", "support"
    ],
    "Timings": [
        "delay", "delayed", "late", "on time", "schedule", "timing", "earlier",
        "landed earlier", "flight landed", "terrible flight delays", "flight cancelled"
    ],
    "Food": [
        "food", "meal", "drink", "beverage", "snack", "tasty", "water", "water bottle",
        "water bottles", "food tasted", "meal service"
    ],
    "Baggage": [
        "baggage", "bag", "luggage", "lost", "damaged", "suitcase", "lost my baggage",
        "baggage arrived", "baggage damaged"
    ],
    "Price": [
        "price", "cost", "cheap", "expensive", "fare", "ticket price", "charged extra",
        "value"
    ],
    "Comfort": [
        "seat", "seats", "legroom", "comfort", "comfortable", "uncomfortable", "dirty",
        "clean", "tray tables", "restroom", "toilet", "cabin", "air conditioning"
    ],
    "Boarding / Checkin": [
        "boarding", "check in", "check-in", "checkin", "boarding was", "fast boarding",
        "slow boarding", "check in took forever", "friendly check in"
    ],
    "Wifi / Entertainment": [
        "wifi", "internet", "entertainment", "no entertainment", "inflight entertainment",
        "wifi did not work", "wifi worked"
    ],
    "Security": [
        "security", "security check", "security staff", "polite", "chaotic"
    ],
    "Announcements / Pilot": [
        "pilot", "announcements", "pilot announcements", "gave updates", "no updates"
    ]
}

# Precompile keyword regexes for speed (create a dict topic -> list of compiled patterns)
COMPILED_KEYWORDS = {}
for topic, keywords in TOPIC_KEYWORDS.items():
    patterns = []
    for kw in keywords:
        # create word-boundary insensitive regex for multi-word too
        # escape special chars in keyword
        esc = re.escape(kw)
        patterns.append(re.compile(r"\b" + esc + r"\b", flags=re.IGNORECASE))
    COMPILED_KEYWORDS[topic] = patterns

# --- VADER model (cached) ---
@st.cache_resource
def get_vader_model():
    return SentimentIntensityAnalyzer()

def predict_sentiment_with_score(text, analyzer):
    """
    Returns tuple (label, compound_score). Use VADER recommended thresholds:
     - compound >= 0.05 => Positive
     - compound <= -0.05 => Negative
     - otherwise => Neutral
    """
    if not isinstance(text, str):
        text = str(text)
    scores = analyzer.polarity_scores(text)
    compound = scores.get("compound", 0.0)
    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"
    return label, compound

def find_topic_by_count(review_text):
    """
    Count keyword matches across all topics and return topic with highest count.
    Fall back to "Other" when no matches.
    """
    if not isinstance(review_text, str):
        review_text = str(review_text)
    text = review_text.lower()
    match_counts = {}
    for topic, patterns in COMPILED_KEYWORDS.items():
        count = 0
        for pat in patterns:
            # count occurrences (findall)
            found = pat.findall(text)
            if found:
                count += len(found)
        match_counts[topic] = count

    # pick the topic with the maximum matches
    best_topic = max(match_counts, key=lambda t: match_counts[t])
    if match_counts[best_topic] == 0:
        return "Other"
    return best_topic

# cached analyzer for DataFrame
@st.cache_data
def analyze_dataframe(df, review_column):
    analyzer = get_vader_model()
    df = df.copy()

    # ensure text column
    df['review_text'] = df[review_column].apply(lambda x: "" if pd.isna(x) else str(x).strip())

    # sentiment label + score
    results = df['review_text'].apply(lambda txt: predict_sentiment_with_score(txt, analyzer))
    df['Sentiment'] = results.apply(lambda t: t[0])
    df['SentimentScore'] = results.apply(lambda t: t[1])

    # topic detection
    df['Topic'] = df['review_text'].apply(find_topic_by_count)

    return df

# --- Sidebar (UI) ---
st.sidebar.title("Airline Sentiment Dashboard")
st.sidebar.header("Real-Time Review")
real_time_review = st.sidebar.text_area("Enter a review:", "The crew was friendly but the flight was delayed.")
real_time_button = st.sidebar.button("Analyze Review")

st.sidebar.header("Batch CSV Analysis")
uploaded_file = st.sidebar.file_uploader("Upload your reviews CSV", type=["csv"])

review_column = None
if uploaded_file is not None:
    try:
        # preview to get columns
        preview_df = pd.read_csv(uploaded_file, nrows=5)
        review_column = st.sidebar.selectbox("Which column contains the review text?", preview_df.columns.tolist())
    except Exception as e:
        st.sidebar.error(f"Error reading CSV header: {e}")
        uploaded_file = None

# --- Main page ---
if real_time_button:
    st.header("Real-Time Analysis")
    analyzer = get_vader_model()
    label, score = predict_sentiment_with_score(real_time_review, analyzer)
    topic = find_topic_by_count(real_time_review)

    st.subheader("Sentiment")
    if label == "Positive":
        st.success(f"{label} (score={score:.3f})")
    elif label == "Negative":
        st.error(f"{label} (score={score:.3f})")
    else:
        st.info(f"{label} (score={score:.3f})")

    st.subheader("Topic")
    st.write(topic)

elif uploaded_file is not None and review_column is not None:
    st.header("Batch Analysis Dashboard")
    try:
        uploaded_file.seek(0)  # important: reset pointer after preview
        df = pd.read_csv(uploaded_file)

        if df.empty:
            st.error("Uploaded CSV is empty.")
            st.stop()

        # run analysis
        df_analyzed = analyze_dataframe(df, review_column)

        # show sample and stats
        st.subheader("Sample of analyzed data")
        st.dataframe(df_analyzed.head(20))

        # overall sentiment counts (include Neutral)
        st.subheader("Sentiment distribution")
        sentiment_counts = df_analyzed['Sentiment'].value_counts().reindex(['Positive','Neutral','Negative']).fillna(0)
        fig_pie = px.pie(names=sentiment_counts.index, values=sentiment_counts.values, title="Sentiment breakdown")
        st.plotly_chart(fig_pie, use_container_width=True)

        # topic counts for Negative and Positive separately
        st.subheader("Top topics by sentiment")
        col1, col2 = st.columns(2)

        neg_topics = df_analyzed[df_analyzed['Sentiment'] == 'Negative']['Topic'].value_counts()
        pos_topics = df_analyzed[df_analyzed['Sentiment'] == 'Positive']['Topic'].value_counts()

        fig_neg = px.bar(x=neg_topics.index, y=neg_topics.values, labels={'x': 'Topic', 'y': 'Count'}, title="Top Negative Topics")
        fig_pos = px.bar(x=pos_topics.index, y=pos_topics.values, labels={'x': 'Topic', 'y': 'Count'}, title="Top Positive Topics")

        col1.plotly_chart(fig_neg, use_container_width=True)
        col2.plotly_chart(fig_pos, use_container_width=True)

        # show the full table if user likes
        with st.expander("Show full analyzed table"):
            st.dataframe(df_analyzed)

        # show a small debug table of borderline cases
        st.subheader("Borderline / neutral cases (inspect manually)")
        borderline = df_analyzed[df_analyzed['Sentiment'] == 'Neutral'][['review_text','SentimentScore','Topic']].head(30)
        st.table(borderline)

    except Exception as e:
        st.error(f"An error occurred during analysis: {e}")

else:
    st.header("Welcome to the Airline Sentiment Dashboard")
    st.info("Use the sidebar to analyze a single review or upload a CSV file.")
# After df_analyzed is created and before st.dataframe(df_analyzed)

# --- Accuracy Evaluation (only if ground truth exists) ---
possible_truth_cols = ['Sentiment', 'Label', 'TrueSentiment', 'ActualSentiment', 'GroundTruth']
truth_col = next((col for col in possible_truth_cols if col in df.columns), None)

if truth_col:
    truth = df[truth_col].str.capitalize().str.strip()
    predicted = df_analyzed['Sentiment'].str.capitalize().str.strip()

    correct = (truth == predicted).sum()
    total = len(df_analyzed)
    accuracy = correct / total if total > 0 else 0

    st.metric("📊 Model Accuracy", f"{accuracy*100:.2f}%",
              help="Based on sentiment labels in your uploaded dataset")
else:
    st.info("⚠️ No labeled sentiment column found → Accuracy can’t be calculated.")
