
import os
import time
from typing import List, Dict, Optional

import pandas as pd
import praw
from dotenv import load_dotenv
from urllib.parse import urlparse


REQUIRED_COLUMNS = [
    "title",
    "score",
    "upvote_ratio",
    "num_comments",
    "author",
    "subreddit",
    "url",
    "permalink",
    "created_utc",
    "is_self",
    "selftext",
    "flair",
    "domain",
    "search_query",
]


def load_reddit_from_env(env_path: str = "reddit.env") -> praw.Reddit:
    """
    Load credentials from reddit.env and return an authenticated PRAW client.

    Your reddit.env should contain:
      REDDIT_CLIENT_ID="..."
      REDDIT_CLIENT_SECRET="..."
      REDDIT_USER_AGENT="..."

    We do NOT hard-code secrets here. Keep reddit.env out of GitHub.
    """
    load_dotenv(env_path)

    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT")

    missing = [k for k, v in {
        "REDDIT_CLIENT_ID": client_id,
        "REDDIT_CLIENT_SECRET": client_secret,
        "REDDIT_USER_AGENT": user_agent,
    }.items() if not v]

    if missing:
        raise ValueError(
            f"Missing credentials in {env_path}: {', '.join(missing)}. "
            "Open your reddit.env and fill them in."
        )

# Creating the PRAW client (polite rate-limiting) !!!!
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        ratelimit_seconds=5,
    )
    return reddit


def safe_get_domain(url_value: Optional[str]) -> Optional[str]:
    """Extract domain from a URL safely (or return None)."""
    if not url_value:
        return None
    try:
        return urlparse(url_value).netloc
    except Exception:
        return None


def submission_to_row(submission, search_query: Optional[str] = None) -> Dict:
    """
    Convert a PRAW submission into a clean dictionary row matching REQUIRED_COLUMNS.
    We gracefully handle missing fields and truncate long text.
    """
    author_name = getattr(submission.author, "name", None) if submission.author else None
    body = getattr(submission, "selftext", None)
    if body:
        body = body[:500]  # keep at most 500 characters (per assignment)

    return {
        "title": getattr(submission, "title", None),
        "score": getattr(submission, "score", None),
        "upvote_ratio": getattr(submission, "upvote_ratio", None),
        "num_comments": getattr(submission, "num_comments", None),
        "author": author_name,
        "subreddit": str(getattr(submission, "subreddit", "")),
        "url": getattr(submission, "url", None),
        "permalink": f"https://www.reddit.com{getattr(submission, 'permalink', '')}",
        "created_utc": int(getattr(submission, "created_utc", 0)) if getattr(submission, "created_utc", None) else None,
        "is_self": getattr(submission, "is_self", None),
        "selftext": body,
        "flair": getattr(submission, "link_flair_text", None),
        "domain": safe_get_domain(getattr(submission, "url", None)),
        "search_query": search_query,  # provenance of how we found it
    }


def collect_hot_posts(reddit: praw.Reddit, subreddits: List[str], limit_per_sub: int) -> List[Dict]:
    """
    Pull 'hot' posts from each subreddit (limit_per_sub each).
    Return a list of dictionary rows ready for a DataFrame.
    """
    rows: List[Dict] = []
    total = 0

    for sr in subreddits:
        print(f" Collecting hot posts from r/{sr} …")
        try:
            for sub in reddit.subreddit(sr).hot(limit=limit_per_sub):
                rows.append(submission_to_row(sub))
                total += 1
            print(f"    Done r/{sr}: {limit_per_sub} requested")
            time.sleep(1)  # be nice to the API
        except Exception as e:
            print(f"    Skipping r/{sr} due to error: {e}")

    print(f" Summary (hot): collected {total} posts total.\n")
    return rows


def collect_search_posts(reddit: praw.Reddit, subreddits: List[str], query: str, limit_per_sub: int) -> List[Dict]:
    """
    Search for a keyword in each subreddit.
    Adds a 'search_query' value to each row for provenance.
    """
    rows: List[Dict] = []
    total = 0

    for sr in subreddits:
        print(f" Searching '{query}' in r/{sr} …")
        try:
            for sub in reddit.subreddit(sr).search(query, limit=limit_per_sub):
                rows.append(submission_to_row(sub, search_query=query))
                total += 1
            print(f"    Done r/{sr}: {limit_per_sub} requested")
            time.sleep(1)
        except Exception as e:
            print(f"    Search failed for r/{sr}: {e}")

    print(f" Summary (search): collected {total} posts total.\n")
    return rows


def save_clean_csv(rows: List[Dict], out_path: str = "reddit_data.csv") -> None:
    """
    Turn collected rows into a DataFrame, drop duplicates (by permalink), and save to CSV.
    """

# Ensuring DataFrame has all expected columns (and in a consistent order) !!!!
    df = pd.DataFrame(rows)
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[REQUIRED_COLUMNS]

# Deduplicating !!!!
    before = len(df)
    df = df.drop_duplicates(subset=["permalink"]).reset_index(drop=True)
    after = len(df)
    removed = before - after

    print(f" Removed {removed} duplicates. {after} unique rows remain.")
    df.to_csv(out_path, index=False)
    print(f" Saved clean data to {out_path}\n")

# Picking a topic and filling in 3 related subreddits !!!!
TOPIC_SUBREDDITS = ["ArtificialInteligence", "WearableTech", "HomeAutomation"]

# Adding 2–4 search keywords about what i want to collect posts about !!!!
SEARCH_KEYWORDS = ["AI agent", "smart home", "wearable", "automation hack"]

POST_LIMIT = 50
ENV_FILE = "reddit.env"                                            # leave unless your file name differs



def main():
    print(" Starting Reddit data collection …")

# Loading credentials and connecting it to Reddit !!!!
    reddit = load_reddit_from_env(ENV_FILE)

# Collecting Hot posts !!!!
    hot_rows = collect_hot_posts(reddit, TOPIC_SUBREDDITS, POST_LIMIT)

# Searching posts for multiple keywords !!!!
    search_rows = []
    for keyword in SEARCH_KEYWORDS:
        search_rows.extend(collect_search_posts(reddit, TOPIC_SUBREDDITS, keyword, POST_LIMIT))

# Combining everything and saving it to CSV !!!!
    all_rows = hot_rows + search_rows
    save_clean_csv(all_rows, out_path="reddit_data.csv")


if __name__ == "__main__":
    main()
