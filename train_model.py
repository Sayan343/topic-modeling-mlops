"""
train_model.py

Trains an LDA (Latent Dirichlet Allocation) topic model, with several
quality improvements over a bare-bones LDA setup:

  1. Extended stopword list -- filters out informal filler words
     (contractions like "don't", "just", "like") that standard English
     stopword lists miss, which otherwise form a meaningless "junk topic".
  2. Bigrams (two-word phrases) -- lets the model capture phrases like
     "climate change" or "interest rate" as single meaningful units,
     instead of just scattered single words.
  3. Corpus-size-aware filtering -- min_df/max_df thresholds scale to
     how much data you actually have, instead of using fixed values
     tuned for a much larger dataset.
  4. Optimized Learning Method -- uses 'batch' instead of 'online' for 
     superior convergence and coherence on small-to-medium datasets.

Produces:
  - lda_model.pkl
  - vectorizer.pkl
  - topic_labels.json
"""

import os
import pickle
import json
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import LatentDirichletAllocation

N_TOP_WORDS = 8
MODERN_CORPUS_PATH = "modern_corpus.txt"

def choose_n_topics(n_docs):
    """
    LDA needs enough documents per topic to find stable patterns. Too many
    topics for too little data produces muddled, overlapping topics (as
    seen when testing with a 16-document corpus). Scale topic count to
    roughly one topic per 20-30 documents, within a sane range.
    """
    if n_docs < 60:
        return 3
    elif n_docs < 150:
        return 5
    elif n_docs < 400:
        return 8
    else:
        return 10

# Standard English stopwords miss informal contractions and filler words
# common in casual writing (news comments, social text, RSS summaries).
# These are exactly what caused a "junk topic" in earlier testing.
EXTRA_STOPWORDS = {
    "don", "just", "know", "like", "said", "says", "ve", "ll", "re", "doesn",
    "didn", "wasn", "isn", "gonna", "gotta", "get", "got", "going", "really",
    "would", "could", "also", "one", "us", "new", "year", "years", "time",
    "day", "week", "make", "made", "according", "told", "including", "mr"
}
CUSTOM_STOPWORDS = list(ENGLISH_STOP_WORDS.union(EXTRA_STOPWORDS))

def load_texts():
    """Loads text from the modern corpus, falling back to 20 Newsgroups if missing."""
    if os.path.exists(MODERN_CORPUS_PATH):
        print(f"Found {MODERN_CORPUS_PATH} -- training on modern data.")
        with open(MODERN_CORPUS_PATH, encoding="utf-8") as f:
            # Added a truthiness check to ensure no purely empty strings are loaded
            texts = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(texts)} modern documents.")
        return texts
    else:
        print(f"No {MODERN_CORPUS_PATH} found -- falling back to 20 Newsgroups.")
        from sklearn.datasets import fetch_20newsgroups
        data = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
        print(f"Loaded {len(data.data)} documents.")
        return data.data

def main():
    texts = load_texts()
    n_docs = len(texts)

    if n_docs == 0:
        print("Corpus is empty. Aborting.")
        return

    # Scale filtering thresholds to corpus size instead of using fixed
    # values that assume a large dataset like the original 20 Newsgroups.
    min_df = 2 if n_docs < 1000 else 5
    max_df = 0.85

    print("Vectorizing text (with bigrams + extended stopwords)...")
    vectorizer = CountVectorizer(
        max_df=max_df,
        min_df=min_df,
        stop_words=CUSTOM_STOPWORDS,
        max_features=5000,
        ngram_range=(1, 2),  # unigrams AND bigrams, e.g. "climate_change"
    )
    
    doc_term_matrix = vectorizer.fit_transform(texts)
    print(f"Vocabulary size: {len(vectorizer.get_feature_names_out())}")

    n_topics = choose_n_topics(n_docs)
    print(f"Training LDA with {n_topics} topics (auto-scaled for {n_docs} documents)...")
    
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        learning_method="batch", # Switched to batch for better small-corpus coherence
        max_iter=50,             # More iterations than default for better convergence
        n_jobs=-1                # Utilize all CPU cores for faster training
    )
    lda.fit(doc_term_matrix)

    feature_names = vectorizer.get_feature_names_out()
    topic_labels = {}
    
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-N_TOP_WORDS - 1:-1]]
        topic_labels[str(topic_idx)] = {
            "keywords": top_words,
            "label": ", ".join(top_words[:3]),
        }
        print(f"Topic {topic_idx}: {top_words}")

    # Save artifacts
    with open("lda_model.pkl", "wb") as f:
        pickle.dump(lda, f)
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open("topic_labels.json", "w") as f:
        json.dump(topic_labels, f, indent=2)

    print("\nSaved lda_model.pkl, vectorizer.pkl, topic_labels.json")
    print("You can now run app.py or streamlit_app.py to serve predictions.")

if __name__ == "__main__":
    main()