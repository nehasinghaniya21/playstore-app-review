# summarizer.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load BART
model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def summarize_reviews(df, text_column="Review Text", max_chars=4000):
    # 1. Cleaner joining with bullet points helps the model distinguish reviews
    reviews_list = df[text_column].astype(str).tolist()
    reviews_text = "Review: " + " Review: ".join(reviews_list)
    
    if len(reviews_text) > max_chars:
        reviews_text = reviews_text[:max_chars].rsplit('.', 1)[0] + '.'

    inputs = tokenizer(
        reviews_text, 
        return_tensors="pt", 
        max_length=1024, 
        truncation=True
    ).to(model.device) # Ensure it runs on the right device (GPU/CPU)

    # 2. Better generation parameters
    summary_ids = model.generate(
        inputs["input_ids"],
        max_new_tokens=500,      # Modern replacement for max_length
        min_length=100,
        length_penalty=2.0,      # More balanced length
        num_beams=5,             # Slightly higher beam search for quality
        no_repeat_ngram_size=3,  # Prevents repetitive phrases
        early_stopping=True
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
