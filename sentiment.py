# Mapping for cardiffnlp/twitter-roberta-base-sentiment
LABEL_MAP = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}

def add_sentiment(df, text_column="Review Text"):
    from transformers import pipeline
    sentiment_analyzer = pipeline(
        "text-classification",
        model="cardiffnlp/twitter-roberta-base-sentiment"
    )

    sentiments = []
    for text in df[text_column].astype(str):
        try:
            result = sentiment_analyzer(text[:512])
            label = result[0]["label"]
            sentiments.append(LABEL_MAP.get(label, "Neutral"))
        except Exception:
            sentiments.append("Neutral")
    df["Sentiment"] = sentiments
    return df