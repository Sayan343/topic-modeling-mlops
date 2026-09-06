"""
train_model.py
Trains an LDA (Latent Dirichlet Allocation) topic model on the
20 Newsgroups dataset and saves the model + vectorizer to disk.

Run this once, locally. It produces:
  - lda_model.pkl
  - vectorizer.pkl
  - topic_labels.json   (human-friendly names for each topic)
"""

import pickle
import json
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

N_TOPICS = 10
N_TOP_WORDS = 8

def main():
    print("Downloading 20 Newsgroups dataset (first run only, cached after)...")
    data = fetch_20newsgroups(
        subset="train",
        remove=("headers", "footers", "quotes"),
    )
    texts = data.data
    print(f"Loaded {len(texts)} documents.")

    # Convert text -> word-count vectors (LDA works on counts, not TF-IDF)
    print("Vectorizing text...")
    vectorizer = CountVectorizer(
        max_df=0.95,       # ignore words in >95% of docs (too common)
        min_df=5,          # ignore words in <5 docs (too rare)
        stop_words="english",
        max_features=5000,
    )
    doc_term_matrix = vectorizer.fit_transform(texts)

    # Train LDA
    print(f"Training LDA with {N_TOPICS} topics (this takes a minute or two)...")
    lda = LatentDirichletAllocation(
        n_components=N_TOPICS,
        random_state=42,
        learning_method="online",
        max_iter=10,
    )
    lda.fit(doc_term_matrix)

    # Build a human-readable label for each topic from its top words
    feature_names = vectorizer.get_feature_names_out()
    topic_labels = {}
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-N_TOP_WORDS - 1:-1]]
        topic_labels[str(topic_idx)] = {
            "keywords": top_words,
            "label": ", ".join(top_words[:3])  # simple auto-label; rename manually if you like
        }
        print(f"Topic {topic_idx}: {top_words}")

    # Save everything needed to serve predictions later
    with open("lda_model.pkl", "wb") as f:
        pickle.dump(lda, f)
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open("topic_labels.json", "w") as f:
        json.dump(topic_labels, f, indent=2)

    print("\nSaved lda_model.pkl, vectorizer.pkl, topic_labels.json")
    print("You can now run app.py to serve predictions.")

if __name__ == "__main__":
    main()
