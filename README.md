Airline Sentiment Dashboard
The Airline Sentiment Dashboard is an interactive web app that makes it easy to analyze airline customer reviews using natural language processing. Instead of manually reading through thousands of comments, the dashboard instantly processes them to uncover overall sentiment and key discussion themes.

What It Does
Runs VADER sentiment analysis to detect whether reviews are positive, negative, or neutral.

Uses a custom topic detection system to identify what customers are talking about — things like staff behavior, flight delays, pricing, comfort, food quality, and more.

Lets users analyze a single review in real time or upload an entire CSV file for bulk sentiment analysis.

Visual Insights
Automatically generates interactive charts such as pie charts and topic-wise bar plots.

Displays a review table with detailed sentiment and topic annotations for quick reference.

Tech Stack
Frontend & Framework: Streamlit

Core Language: Python

Libraries: Pandas, Plotly, VADER

Logic: Regex-based keyword models for topic detection

Deployment: Hosted on Hugging Face Spaces for easy browser access

Why I Built It
This project combines my interests in NLP, data visualization, and real-world problem-solving. The goal was to create a simple and efficient tool that turns messy text reviews into clean insights—something airlines and businesses can actually use to make better decisions.
