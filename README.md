# wsbPicker

```python
#!/usr/bin/env python3
"""
WSB Stock Picker
----------------
Scrapes r/wallstreetbets, extracts stock tickers, runs sentiment analysis,
and outputs a ranked list with ratings.

Setup:
    pip install praw vaderSentiment requests

Reddit API credentials:
    1. Go to https://www.reddit.com/prefs/apps
    2. Create a new app (type: script)
    3. Fill in CLIENT_ID, CLIENT_SECRET, USER_AGENT below
"""

import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

import praw
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ─── CONFIG ──────────────────────────────────────────────────────────────────

CLIENT_ID     = "YOUR_CLIENT_ID"       # from reddit app page
CLIENT_SECRET = "YOUR_CLIENT_SECRET"   # from reddit app page
USER_AGENT    = "wsb_stock_picker/1.0 by YOUR_REDDIT_USERNAME"

SUBREDDIT     = "wallstreetbets"
POST_LIMIT    = 200          # number of hot/new posts to scan
COMMENT_DEPTH = 2            # comment tree depth (0 = top-level only)
MIN_MENTIONS  = 3            # ignore tickers mentioned fewer times than this
TOP_N         = 20           # how many tickers to show in final output

# ─── TICKER FILTERING ────────────────────────────────────────────────────────

# Common English words that look like tickers — filtered out to reduce noise
FALSE_POSITIVES = {
    "A", "I", "AM", "AN", "ARE", "AS", "AT", "BE", "BY", "DO", "FOR",
    "GO", "HE", "IF", "IN", "IS", "IT", "ME", "MY", "NO", "OF", "ON",
    "OR", "RE", "SO", "TO", "UP", "US", "WE", "CEO", "CFO", "CTO",
    "IPO", "ETF", "EPS", "GDP", "IRS", "SEC", "NYSE", "NASDAQ",
    "ATH", "ATL", "DD", "OP", "OG", "RH", "TD", "TD", "PM", "DM",
    "AH", "IMO", "TBH", "FUD", "WSB", "YOLO", "APE", "GUH",
    "AND", "THE", "BUT", "NOT", "ALL", "BUY", "PUT", "CALL",
    "NOW", "NEW", "OLD", "LOW", "HIGH", "LOSS", "WIN", "GAIN",
    "EDIT", "UPDATE", "TLDR", "TL", "DR", "EOD", "EOW", "EOY",
}

# Regex: 1–5 uppercase letters, optionally followed by a dot + 1-2 letters (e.g. BRK.B)
TICKER_RE = re.compile(r'\b([A-Z]{1,5}(?:\.[AB])?)\b')

# ─── DATA MODEL ──────────────────────────────────────────────────────────────

@dataclass
class TickerData:
    symbol: str
    mentions: int = 0
    sentiment_scores: list = field(default_factory=list)

    @property
    def avg_sentiment(self) -> float:
        if not self.sentiment_scores:
            return 0.0
        return sum(self.sentiment_scores) / len(self.sentiment_scores)

    @property
    def positive_ratio(self) -> float:
        if not self.sentiment_scores:
            return 0.0
        positive = sum(1 for s in self.sentiment_scores if s > 0.05)
        return positive / len(self.sentiment_scores)

    @property
    def rating(self) -> float:
        """
        Composite rating (0–10):
          - 50% weight: average compound sentiment  (-1 to +1 → normalised)
          - 30% weight: positive mention ratio       (0 to 1)
          - 20% weight: mention volume score         (log-scaled, capped at 50)
        """
        import math
        sentiment_score  = (self.avg_sentiment + 1) / 2          # 0–1
        volume_score     = min(math.log1p(self.mentions) / math.log1p(50), 1.0)
        composite        = (0.50 * sentiment_score +
                            0.30 * self.positive_ratio +
                            0.20 * volume_score)
        return round(composite * 10, 2)

    @property
    def sentiment_label(self) -> str:
        s = self.avg_sentiment
        if s >= 0.35:  return "🚀 Very Bullish"
        if s >= 0.10:  return "📈 Bullish"
        if s >= -0.10: return "😐 Neutral"
        if s >= -0.35: return "📉 Bearish"
        return                "💀 Very Bearish"

# ─── SCRAPER ─────────────────────────────────────────────────────────────────

def build_reddit_client() -> praw.Reddit:
    return praw.Reddit(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        user_agent=USER_AGENT,
    )


def extract_tickers(text: str) -> list[str]:
    """Return all plausible ticker symbols found in a string."""
    return [t for t in TICKER_RE.findall(text) if t not in FALSE_POSITIVES]


def score_text(analyzer: SentimentIntensityAnalyzer, text: str) -> float:
    """Return the compound VADER sentiment score for a piece of text."""
    return analyzer.polarity_scores(text)["compound"]


def scrape_wsb(reddit: praw.Reddit,
               analyzer: SentimentIntensityAnalyzer) -> dict[str, TickerData]:
    """
    Iterate over hot + new posts on the subreddit.
    For each post and its comments, extract tickers and score sentiment.
    """
    ticker_map: dict[str, TickerData] = defaultdict(lambda: TickerData(""))
    subreddit = reddit.subreddit(SUBREDDIT)
    posts_seen = 0

    def process_text(text: str, tickers_found: list[str]) -> None:
        """Attach sentiment of `text` to each ticker found nearby."""
        sentiment = score_text(analyzer, text)
        for ticker in tickers_found:
            if ticker_map[ticker].symbol == "":
                ticker_map[ticker].symbol = ticker
            ticker_map[ticker].mentions += 1
            ticker_map[ticker].sentiment_scores.append(sentiment)

    feeds = [subreddit.hot(limit=POST_LIMIT // 2),
             subreddit.new(limit=POST_LIMIT // 2)]

    seen_ids: set[str] = set()

    for feed in feeds:
        for post in feed:
            if post.id in seen_ids:
                continue
            seen_ids.add(post.id)
            posts_seen += 1

            # ── Title + selftext ──────────────────────────────────────────
            full_text = post.title + " " + (post.selftext or "")
            tickers   = extract_tickers(full_text)
            if tickers:
                process_text(full_text, tickers)

            # ── Comments ─────────────────────────────────────────────────
            try:
                post.comments.replace_more(limit=COMMENT_DEPTH)
                for comment in post.comments.list():
                    if not comment.body or comment.body in ("[deleted]", "[removed]"):
                        continue
                    ctickers = extract_tickers(comment.body)
                    if ctickers:
                        process_text(comment.body, ctickers)
            except Exception:
                pass  # skip comment errors gracefully

            if posts_seen % 25 == 0:
                print(f"  … scanned {posts_seen} posts", flush=True)

    print(f"\n✅  Scanned {posts_seen} posts from r/{SUBREDDIT}")
    return ticker_map

# ─── OUTPUT ──────────────────────────────────────────────────────────────────

def print_results(ticker_map: dict[str, TickerData]) -> None:
    # Filter by minimum mentions
    filtered = [td for td in ticker_map.values()
                if td.mentions >= MIN_MENTIONS]

    # Sort: only show tickers with net positive sentiment, then by rating desc
    bullish = [td for td in filtered if td.avg_sentiment > 0]
    bullish.sort(key=lambda td: td.rating, reverse=True)
    top     = bullish[:TOP_N]

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"\n{'─'*66}")
    print(f"  🏆  WSB Stock Picker  ·  r/{SUBREDDIT}  ·  {now}")
    print(f"{'─'*66}")
    print(f"  {'#':<4} {'TICKER':<8} {'RATING':>6}  {'MENTIONS':>8}  {'SENTIMENT':<20}  AVG SCORE")
    print(f"{'─'*66}")

    for i, td in enumerate(top, 1):
        print(f"  {i:<4} {td.symbol:<8} {td.rating:>6.1f}  "
              f"{td.mentions:>8}  {td.sentiment_label:<20}  {td.avg_sentiment:+.3f}")

    print(f"{'─'*66}")
    print(f"  Showing top {len(top)} bullish tickers  "
          f"(min {MIN_MENTIONS} mentions, {len(filtered)} total found)")
    print(f"{'─'*66}\n")

    # Also dump all tickers to a simple text file
    out_path = "wsb_results.txt"
    with open(out_path, "w") as f:
        f.write(f"WSB Stock Picker results — {now}\n\n")
        f.write(f"{'RANK':<5} {'TICKER':<8} {'RATING':>6}  {'MENTIONS':>8}  "
                f"{'POS%':>6}  AVG_SENTIMENT\n")
        f.write("-" * 55 + "\n")
        for i, td in enumerate(top, 1):
            f.write(f"{i:<5} {td.symbol:<8} {td.rating:>6.1f}  "
                    f"{td.mentions:>8}  {td.positive_ratio:>5.0%}  "
                    f"{td.avg_sentiment:+.4f}\n")
    print(f"📄  Results saved to {out_path}\n")

# ─── ENTRY POINT ─────────────────────────────────────────────────────────────

def main() -> None:
    if CLIENT_ID == "YOUR_CLIENT_ID":
        print("⚠️  Please fill in CLIENT_ID, CLIENT_SECRET and USER_AGENT before running.")
        print("   See https://www.reddit.com/prefs/apps to create a free Reddit app.\n")
        sys.exit(1)

    print(f"🔍  Connecting to Reddit …")
    reddit   = build_reddit_client()
    analyzer = SentimentIntensityAnalyzer()

    print(f"📡  Scraping r/{SUBREDDIT} (up to {POST_LIMIT} posts) …\n")
    t0         = time.time()
    ticker_map = scrape_wsb(reddit, analyzer)
    elapsed    = time.time() - t0
    print(f"⏱️   Scraping took {elapsed:.1f}s")

    print_results(ticker_map)


if __name__ == "__main__":
    main()
```
