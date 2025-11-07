import streamlit as st
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sentiment Analyzer", layout="centered")

# ---------------- Helper Functions ---------------- #
def analyze_sentiment(review):
    polarity = TextBlob(review).sentiment.polarity
    if polarity > 0:
        sentiment = "Positive"
    elif polarity == 0:
        sentiment = "Neutral"
    else:
        sentiment = "Negative"
    return sentiment, polarity

def analyze_dataframe(df, column):
    df["Sentiment"] = df[column].apply(lambda x: analyze_sentiment(str(x))[0])
    df["Polarity Score"] = df[column].apply(lambda x: analyze_sentiment(str(x))[1])
    return df

# ---------------- UI Layout ---------------- #
st.title("✈ Airline Review Sentiment Predictor")
st.write("Analyze text or upload batch CSV for sentiment insights!")

tabs = st.tabs(["Single Review", "Batch CSV Analysis"])

# ---------------- SINGLE REVIEW TAB ---------------- #
with tabs[0]:
    user_review = st.text_area("Enter a review:")
    
    if st.button("Analyze Review"):
        if user_review.strip():
            sentiment, polarity = analyze_sentiment(user_review)
            st.success(f"Predicted Sentiment: **{sentiment}**")
            st.write(f"Polarity Score: `{polarity:.2f}`")
        else:
            st.error("Please enter a review.")

# ---------------- BATCH CSV TAB ---------------- #
with tabs[1]:
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    review_column = st.text_input("Enter column name containing reviews:", placeholder="e.g., Review")

    if uploaded_file is not None:
        uploaded_file.seek(0)
        try:
            df = pd.read_csv(uploaded_file)
            st.success("✅ File loaded successfully!")
            st.write("📌 Available columns:", df.columns.tolist())
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            df = None
    else:
        df = None

    if st.button("Start Batch Analysis"):
        if df is None:
            st.error("Upload a CSV first.")
        elif review_column not in df.columns:
            st.error("Review column not found!")
        else:
            st.header("📊 Batch Analysis Dashboard")
            df_analyzed = analyze_dataframe(df.copy(), review_column)
            st.dataframe(df_analyzed.head())

            # ---------- Accuracy Evaluation (if ground truth exists) ---------- #
            possible_truth_cols = [
                'Sentiment', 'Label', 'TrueSentiment', 
                'ActualSentiment', 'GroundTruth'
            ]
            truth_col = next((col for col in possible_truth_cols if col in df.columns and col != review_column and col != "Sentiment"), None)

            if truth_col:
                truth = df[truth_col].astype(str).str.capitalize().str.strip()
                predicted = df_analyzed['Sentiment'].astype(str).str.capitalize().str.strip()

                total = len(df_analyzed)
                correct = (truth == predicted).sum()
                accuracy = correct / total if total > 0 else 0

                st.metric("✅ Model Accuracy", f"{accuracy*100:.2f}%")
                st.caption(f"Compared against column: `{truth_col}`")
            else:
                st.info("⚠️ No labelled sentiment column detected → Cannot compute accuracy.")

            # ---------- Visualization ---------- #
            sent_counts = df_analyzed["Sentiment"].value_counts()
            st.subheader("Sentiment Distribution")
            fig, ax = plt.subplots()
            sent_counts.plot(kind="bar")
            plt.title("Sentiment Count")
            plt.xlabel("Sentiment")
            plt.ylabel("Count")
            st.pyplot(fig)

            st.download_button(
                label="📥 Download Analyzed Results",
                data=df_analyzed.to_csv(index=False),
                file_name="sentiment_results.csv",
                mime="text/csv"
            )
