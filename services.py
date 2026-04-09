from google_play_scraper import reviews, Sort
import pandas as pd
from langdetect import detect, LangDetectException

def fetch_reviews(app_id, language="en", country="us", review_count=100):
    """
    Fetch latest reviews from Play Store for given app_id.
    """
    result, _ = reviews(
        app_id,
        lang=language,
        country=country,
        sort=Sort.NEWEST,
        count=review_count
    )

    df = pd.DataFrame(result)
    df = df[["userName", "score", "content", "at"]]
    df = df.rename(columns={
        "userName": "User Name",
        "score": "Rating",
        "content": "Review Text",
        "at": "Review Date",
        #"thumbsUpCount": "Helpful Votes"
    })
    df["Review Date"] = pd.to_datetime(df["Review Date"])
    return df

def filter_english_reviews(df, text_column="Review Text"):
    """
    Keep only English reviews to avoid summarizer language issues.
    Handles empty or invalid text safely.
    """
    df = df.copy()

    def safe_detect(text):
        try:
            if not text.strip():
                return "en"
            return detect(text)
        except LangDetectException:
            return "en"

    df["lang"] = df[text_column].apply(safe_detect)
    return df[df["lang"] == "en"]