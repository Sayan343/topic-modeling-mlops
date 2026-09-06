"""
fetch_modern_data.py
Collects modern text data from free public RSS feeds across news,
business, tech, health, and sports, and saves them as a corpus for
retraining the topic model.

Run this once to produce: modern_corpus.txt (one document per line)
Then run train_model.py to train on this file.
"""

import feedparser

RSS_FEEDS = {
    "BBC News": "https://feeds.bbci.co.uk/news/rss.xml",
    "BBC Business": "https://feeds.bbci.co.uk/news/business/rss.xml",
    "BBC Technology": "https://feeds.bbci.co.uk/news/technology/rss.xml",
    "BBC Health": "https://feeds.bbci.co.uk/news/health/rss.xml",
    "TechCrunch": "https://techcrunch.com/feed/",
    "ESPN": "https://www.espn.com/espn/rss/news",
    "NPR News": "https://feeds.npr.org/1001/rss.xml",
    "Al Jazeera": "https://www.aljazeera.com/xml/rss/all.xml",
    "CNBC": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
}


def fetch_rss_articles():
    print("Fetching current headlines from RSS feeds...")
    documents = []
    for name, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            count = 0
            for entry in feed.entries:
                title = getattr(entry, "title", "")
                summary = getattr(entry, "summary", "")
                text = f"{title}. {summary}".strip()
                if len(text) > 20:  # skip near-empty entries
                    documents.append(text)
                    count += 1
            print(f"  {name}: {count} articles")
        except Exception as e:
            print(f"  {name}: failed ({e})")
    return documents


def main():
    docs = fetch_rss_articles()
    print(f"\nTotal articles collected: {len(docs)}")

    if len(docs) < 50:
        print("\nWarning: fewer than 50 articles collected. Topic modeling works")
        print("best with more documents. Consider re-running later (feeds refresh")
        print("over time) or adding more feeds to RSS_FEEDS.")

    with open("modern_corpus.txt", "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(doc.replace("\n", " ").replace("\r", " ") + "\n")

    print("\nSaved modern_corpus.txt")
    print("Next: run train_model.py to train on this data.")


if __name__ == "__main__":
    main()
