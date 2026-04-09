import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from services import fetch_reviews, filter_english_reviews
from summarizer import summarize_reviews
from sentiment import add_sentiment

st.set_page_config(page_title="Play Store Analytics + AI", layout="wide")
st.title("Play Store Reviews Analytics Dashboard with AI Insights")

# Sidebar Filters
st.sidebar.header("Filters")
app_id = st.sidebar.text_input("Application ID", "com.instagram.android")
review_count = st.sidebar.slider("Number of Reviews", 500, 5000, 1000)
country = st.sidebar.selectbox("Country", ["United States of America", "India", "United Kingdom", "Canada"])
#country = st.sidebar.selectbox("Country", ["us", "in", "gb", "ca"])

if country == "United States of America":
    country_code = "us"
elif country == "India":
    country_code = "us"
elif country == "United Kingdom":
    country_code = "gb"
elif country == "Canada":
    country_code = "ca"

if st.sidebar.button("Get Reviews"):

    with st.spinner("Fetching reviews..."):
        df = fetch_reviews(app_id, language="en", country=country_code, review_count=review_count)
        # Keep only English reviews
        #df = filter_english_reviews(df)

    if not df.empty:

        # Add sentiment
        df = add_sentiment(df)

        # KPI Metrics
        avg_rating = round(df["Rating"].mean(), 2)
        total_reviews = len(df)
        positive_percent = round((df["Rating"] >= 4).mean() * 100, 1)
        positive_sentiment = round((df["Sentiment"]=="Positive").mean() * 100, 1)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("⭐ Average Rating", avg_rating)
        col2.metric("📝 Total Reviews", total_reviews)
        col3.metric("😊 Positive Reviews (%)", f"{positive_percent}%")
        col4.metric("😃 Positive Sentiment (%)", f"{positive_sentiment}%")

        st.divider()

        # Rating Distribution
        rating_counts = df["Rating"].value_counts().sort_index()
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("⭐ Rating Distribution")
            st.bar_chart(rating_counts)
        with col2:
            st.subheader("⭐ Rating Share (Pie)")
            fig, ax = plt.subplots()
            ax.pie(rating_counts, labels=rating_counts.index, autopct="%1.1f%%")
            st.pyplot(fig)

        # Reviews Over Time
        reviews_over_time = df.groupby(df["Review Date"].dt.date).size()
        st.subheader("📅 Reviews Over Time")
        st.line_chart(reviews_over_time)

        # Sentiment Distribution
        st.subheader("🧠 Sentiment Distribution")
        sentiment_counts = df["Sentiment"].value_counts()
        st.bar_chart(sentiment_counts)

        # Executive AI Summary
        st.subheader("📋 Executive Summary (AI Generated)")
        with st.spinner("Generating AI summary..."):
            summary_text = summarize_reviews(df)
            st.write(summary_text)

        st.divider()

        # Review Table + Download
        st.subheader("📝 Application Reviews Table")
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV", csv, file_name="reviews.csv", mime="text/csv")