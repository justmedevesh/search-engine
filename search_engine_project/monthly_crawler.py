"""
Monthly crawler script for Coventry University
Research Centre for Computational Science and Mathematical Modelling

This script is intended to be executed automatically
once per month using an OS scheduler (cron / Task Scheduler).
"""

import json
import os
from datetime import datetime

from core.crawler import ImprovedSeleniumCrawler
from core.index1 import AdvancedInvertedIndex

# ---------------- CONFIG ----------------
BASE_URL = (
    "https://pureportal.coventry.ac.uk/en/organisations/"
    "ics-research-centre-for-computational-science-and-mathematical-mo"
)

MAX_AUTHORS = 50

DATA_DIR = "data"
DATA_FILE = os.path.join(DATA_DIR, "publications.json")
INDEX_FILE = os.path.join(DATA_DIR, "search_index.pkl")
LOG_FILE = os.path.join(DATA_DIR, "crawl.log")

os.makedirs(DATA_DIR, exist_ok=True)

# ---------------- LOGGING ----------------
def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{timestamp}] {msg}"
    print(full_msg)
    with open(LOG_FILE, "a") as f:
        f.write(full_msg + "\n")

# ---------------- MAIN TASK ----------------
def run_monthly_crawl():
    log("=== MONTHLY CRAWL STARTED ===")

    crawler = ImprovedSeleniumCrawler(callback=log)
    publications = crawler.crawl_department(BASE_URL, MAX_AUTHORS)

    log(f"Extracted {len(publications)} publications")

    with open(DATA_FILE, "w") as f:
        json.dump(publications, f, indent=2, default=str)

    index = AdvancedInvertedIndex()
    for i, pub in enumerate(publications):
        index.add_document(i, pub)

    index.save(INDEX_FILE)

    log("Index updated successfully")
    log("=== MONTHLY CRAWL COMPLETED ===\n")

# ---------------- ENTRY POINT ----------------
if __name__ == "__main__":
    run_monthly_crawl()