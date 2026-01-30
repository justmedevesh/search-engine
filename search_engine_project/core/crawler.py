import time
import re
import json
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup


class ImprovedSeleniumCrawler:
    def __init__(self, callback=None):
        self.callback = callback
        self.seed_file = Path(__file__).parent / "ics_authors.json"
        self.driver = None

    def log(self, msg):
        if self.callback:
            self.callback(msg)

    def init_driver(self):
        options = Options()
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--start-maximized")
        self.driver = webdriver.Chrome(options=options)

    def close_driver(self):
        if self.driver:
            self.driver.quit()

    def load_author_seeds(self):
        if not self.seed_file.exists():
            raise FileNotFoundError("ics_authors.json not found")
        with open(self.seed_file, "r") as f:
            return json.load(f)

    def crawl_department(self, base_url, max_authors):
        self.init_driver()
        publications = []

        try:
            authors = self.load_author_seeds()
            self.log(f"Loaded {len(authors)} ICS author profiles")

            for i, author_url in enumerate(authors[:max_authors], 1):
                self.log(f"[{i}/{len(authors)}] Crawling author profile")
                self.log(author_url)

                pubs = self.crawl_author(author_url)
                self.log(f"  → {len(pubs)} publications found")
                publications.extend(pubs)

            self.log(f"✓ Crawling finished. Total publications collected: {len(publications)}")
            return publications

        finally:
            self.close_driver()

    def crawl_author(self, profile_url):
        self.driver.get(profile_url)
        time.sleep(3)

        soup = BeautifulSoup(self.driver.page_source, "html.parser")

        pub_links = set()
        for a in soup.find_all("a", href=True):
            if "/en/publications/" in a["href"]:
                pub_links.add(urljoin(profile_url, a["href"].split("?")[0]))

        publications = []
        for link in pub_links:
            pub = self.parse_publication_page(link, profile_url)
            if pub:
                publications.append(pub)

        return publications

    def parse_publication_page(self, pub_url, profile_url):
        try:
            self.driver.get(pub_url)
            time.sleep(2)
            soup = BeautifulSoup(self.driver.page_source, "html.parser")

            # ---------- TITLE ----------
            title_tag = soup.find("h1")
            title = title_tag.get_text(strip=True) if title_tag else "No title"

            # ---------- AUTHORS ----------
            authors = []
            for a in soup.find_all("a", href=True):
                if "/en/persons/" in a["href"]:
                    name = a.get_text(strip=True)
                    if name:
                        authors.append(name)
            if not authors:
                authors = ["Unknown"]

            # ---------- DATE EXTRACTION ----------
            published_date = "N/A"
            online_date = "N/A"
            year = "N/A"

            for row in soup.find_all("tr"):
                cells = row.find_all("td")
                if len(cells) != 2:
                    continue

                label = cells[0].get_text(strip=True).lower()
                value = cells[1].get_text(strip=True)

                if "publication status" in label:
                    published_date = value
                    m = re.search(r"(19|20)\d{2}", value)
                    if m:
                        year = m.group()

                if "early online date" in label:
                    online_date = value
                    if year == "N/A":
                        m = re.search(r"(19|20)\d{2}", value)
                        if m:
                            year = m.group()

            if year == "N/A":
                m = re.search(r"(19|20)\d{2}", soup.get_text(" "))
                if m:
                    year = m.group()

            return {
                "title": title,
                "authors": authors,
                "year": year,
                "published_date": published_date,
                "online_date": online_date,
                "publication_link": pub_url,
                "profile_link": profile_url,
                "author_profile_name": authors[0],
                "crawled_at": datetime.now().isoformat()
            }

        except Exception:
            self.log(f"    ✗ Failed publication page: {pub_url}")
            return None
