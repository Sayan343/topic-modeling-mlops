"""
streamlit_app.py
Visual frontend for the LDA topic model.

Run locally with: streamlit run streamlit_app.py
Deploy for free at: https://share.streamlit.io
"""

import pickle
import json
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Topic Classifier", page_icon="🧠", layout="centered")


@st.cache_resource
def load_model():
    with open("lda_model.pkl", "rb") as f:
        lda = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("topic_labels.json") as f:
        topic_labels = json.load(f)
    return lda, vectorizer, topic_labels


lda, vectorizer, topic_labels = load_model()

st.title("🧠 Text Topic Classifier")
st.caption(
    "An LDA topic model trained on current news headlines (bigrams + "
    "extended stopword filtering for cleaner topics). Paste in any text "
    "and see which topic it's predicted to belong to."
)
with st.expander("ℹ️ Known limitations"):
    st.write(
        "- Training data comes from a UK/international-leaning set of RSS feeds, "
        "so the 'sports' topic is anchored on soccer/football vocabulary and may "
        "miss American football, baseball, etc.\n"
        "- With ~270 training documents, some topics may be dominated by a single "
        "trending headline rather than a broad recurring theme.\n"
        "- Confidence naturally drops on text that blends multiple topics at "
        "once (e.g. an article about both AI and energy policy) -- this is "
        "expected behavior, not a bug."
    )

st.write("**Try an example:**")
example_cols = st.columns(3)
examples = [
    "The basketball team won the championship after a last-second shot.",
    "New government policy on tax reform sparked debate in parliament.",
    "Scientists announced a breakthrough in renewable energy storage.",
]
if "text_input" not in st.session_state:
    st.session_state.text_input = ""

for col, example in zip(example_cols, examples):
    if col.button(example[:28] + "...", use_container_width=True):
        st.session_state.text_input = example

text = st.text_area(
    "Or type your own text:",
    value=st.session_state.text_input,
    height=120,
    placeholder="Type or paste a sentence or paragraph here...",
)

if st.button("Classify Topic", type="primary", use_container_width=True):
    if not text.strip():
        st.warning("Please enter some text first.")
    else:
        vec = vectorizer.transform([text])
        topic_distribution = lda.transform(vec)[0]
        best_idx = int(topic_distribution.argmax())
        confidence = float(topic_distribution[best_idx])

        st.success(f"**Predicted topic:** {topic_labels[str(best_idx)]['label']}")
        st.metric("Confidence", f"{confidence:.1%}")

        st.write("**Top keywords for this topic:**")
        st.write(", ".join(topic_labels[str(best_idx)]["keywords"]))

        st.write("**Full topic probability distribution:**")
        dist_df = pd.DataFrame({
            "Topic": [topic_labels[str(i)]["label"] for i in range(len(topic_distribution))],
            "Probability": topic_distribution,
        }).sort_values("Probability", ascending=False)
        st.bar_chart(dist_df.set_index("Topic"))

st.divider()
st.caption(
    "Built as part of an MLOps portfolio project: LDA topic modeling with "
    "bigram + stopword tuning, trained on live RSS news data, containerized "
    "with Docker, deployable to AWS. [View the code on GitHub](#)"
)
