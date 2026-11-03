import base64
import datetime as dt
import os
import sqlite3
from contextlib import closing

import requests

DB = "data/prices.db"
QUERIES = ["thinkpad laptop", "macbook air", "dell xps", "hp elitebook"]


def get_token():
    client_id = os.environ.get("EBAY_ID")
    client_secret = os.environ.get("EBAY_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError("Set EBAY_ID and EBAY_SECRET before collecting listings.")
    creds = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    response = requests.post(
        "https://api.ebay.com/identity/v1/oauth2/token",
        headers={
            "Authorization": f"Basic {creds}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data={
            "grant_type": "client_credentials",
            "scope": "https://api.ebay.com/oauth/api_scope",
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["access_token"]


def request_headers(token):
    return {
        "Authorization": f"Bearer {token}",
        "X-EBAY-C-MARKETPLACE-ID": "EBAY_US",
    }


def fetch(token, query, pages=5):
    for page in range(pages):
        response = requests.get(
            "https://api.ebay.com/buy/browse/v1/item_summary/search",
            headers=request_headers(token),
            params={
                "q": query,
                "limit": 200,
                "offset": page * 200,
                "filter": "conditions:{USED},buyingOptions:{FIXED_PRICE}",
            },
            timeout=30,
        )
        response.raise_for_status()
        items = response.json().get("itemSummaries", [])
        if not items:
            break
        yield from items


def main():
    os.makedirs("data", exist_ok=True)
    with closing(sqlite3.connect(DB)) as con, con:
        con.execute(
            """CREATE TABLE IF NOT EXISTS raw_listings(
                snapshot_date TEXT, item_id TEXT, title TEXT, price REAL,
                currency TEXT, condition TEXT, seller_feedback REAL,
                location TEXT, url TEXT,
                PRIMARY KEY (snapshot_date, item_id))"""
        )
        token = get_token()
        today = dt.date.today().isoformat()
        for query in QUERIES:
            for item in fetch(token, query):
                con.execute(
                    "INSERT OR IGNORE INTO raw_listings VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        today,
                        item["itemId"],
                        item["title"],
                        float(item["price"]["value"]),
                        item["price"]["currency"],
                        item.get("condition"),
                        float(item.get("seller", {}).get("feedbackPercentage", 0) or 0),
                        item.get("itemLocation", {}).get("country"),
                        item["itemWebUrl"],
                    ),
                )


if __name__ == "__main__":
    main()
