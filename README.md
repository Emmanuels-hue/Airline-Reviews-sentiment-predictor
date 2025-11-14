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
The goal was to create a simple and efficient tool that turns messy text reviews into clean insights—something airlines and businesses can actually use to know the ground reality of their product/service and make data driven decisions.

Live Demo: https://huggingface.co/spaces/Emmanuels-hue/airline-sentiment-dashboard

<img width="1452" height="537" alt="Screenshot 2025-11-14 114936" src="https://github.com/user-attachments/assets/c4062a38-6e16-4e9c-89a7-d0123b58e2a7" />
<img width="1484" height="660" alt="Screenshot 2025-11-14 114925" src="https://github.com/user-attachments/assets/dcd69b86-9b80-40b9-bd13-e10dcf66f3a0" />
<img width="1445" height="609" alt="Screenshot 2025-11-14 114917" src="https://github.com/user-attachments/assets/d7446eed-969e-4068-95bf-0773a7f235d2" />
<img width="1445" height="683" alt="Screenshot 2025-11-14 114910" src="https://github.com/user-attachments/assets/8474c9fa-fb31-4f05-b8e2-95f470dee3ae" />
<img width="1807" height="452" alt="Screenshot 2025-11-14 114823" src="https://github.com/user-attachments/assets/6bddc205-c538-4681-ab9d-f22b6ed553f0" />
