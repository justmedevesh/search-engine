import streamlit as st
from core.crawler import ImprovedSeleniumCrawler
from core.index import AdvancedInvertedIndex
import time, json, pickle, os, math
import numpy as np

# ==================================================
# CONFIG
# ==================================================
st.set_page_config(
    page_title="Coventry University Research Search Engine",
    layout="wide"
)

DATA_DIR = "data"
PUB_FILE = f"{DATA_DIR}/publications.json"
INDEX_FILE = f"{DATA_DIR}/index.pkl"
LOG_FILE = f"{DATA_DIR}/crawl_logs.txt"

RESULTS_PER_PAGE = 5
TOTAL_ICS_AUTHORS = 42

os.makedirs(DATA_DIR, exist_ok=True)

# ==================================================
# SESSION STATE (SAFE LOADER)
# ==================================================
def load_index_safely():
    if os.path.exists(INDEX_FILE):
        try:
            with open(INDEX_FILE, "rb") as f:
                idx = pickle.load(f)

            test = idx.search("test")
            if test and len(test[0]) != 4:
                raise ValueError

            return idx
        except Exception:
            return AdvancedInvertedIndex()
    return AdvancedInvertedIndex()

if "index" not in st.session_state:
    st.session_state.index = load_index_safely()

if "crawl_logs" not in st.session_state:
    st.session_state.crawl_logs = (
        open(LOG_FILE).read().splitlines()
        if os.path.exists(LOG_FILE) else []
    )

if "results" not in st.session_state:
    st.session_state.results = []

if "last_query" not in st.session_state:
    st.session_state.last_query = None

if "page" not in st.session_state:
    st.session_state.page = 1

if "is_crawling" not in st.session_state:
    st.session_state.is_crawling = False

tabs = st.tabs(["Crawler", "Search", "Statistics"])

# ==================================================
# ================= CRAWLER TAB ====================
# ==================================================
with tabs[0]:
    st.subheader("Web Crawler")

    url = st.text_input(
        "Base URL",
        "httpqqqqs://pureportal.coventry.ac.uk/en/organisations/"
        "ics-research-centre-for-computational-science-and-mathematical-mo"
    )

    crawl_mode = st.radio(
        "Author Crawling Mode",
        [f"Crawl all authors ({TOTAL_ICS_AUTHORS})", "Crawl a specific number"],
        index=0
    )

    if crawl_mode == "Crawl a specific number":
        max_authors = st.number_input(
            "Number of authors",
            min_value=1,
            max_value=TOTAL_ICS_AUTHORS,
            value=10
        )
    else:
        max_authors = TOTAL_ICS_AUTHORS

    col1, col2 = st.columns(2)
    start_crawl = col1.button("Start Crawl")
    clear_data = col2.button("Clear Crawl Data")

    progress = st.progress(0)
    log_box = st.empty()
    log_box.text("\n".join(st.session_state.crawl_logs))

    def log_callback(msg):
        st.session_state.crawl_logs.append(msg)
        log_box.text("\n".join(st.session_state.crawl_logs))
        with open(LOG_FILE, "w") as f:
            f.write("\n".join(st.session_state.crawl_logs))
        time.sleep(0.05)

    if clear_data:
        st.session_state.index = AdvancedInvertedIndex()
        st.session_state.results = []
        st.session_state.crawl_logs = []
        for f in [PUB_FILE, INDEX_FILE, LOG_FILE]:
            if os.path.exists(f):
                os.remove(f)
        progress.progress(0)
        st.success("All crawl data cleared")

    if start_crawl and not st.session_state.is_crawling:
        st.session_state.is_crawling = True
        st.session_state.crawl_logs = []
        progress.progress(5)

        crawler = ImprovedSeleniumCrawler(callback=log_callback)
        publications = crawler.crawl_department(url, max_authors)

        progress.progress(70)

        index = AdvancedInvertedIndex()
        for i, pub in enumerate(publications):
            index.add_document(i, pub)

        # 🔥 REQUIRED FOR COSINE SIMILARITY
        index.build_tfidf_vectors()

        st.session_state.index = index

        with open(PUB_FILE, "w") as f:
            json.dump(publications, f, indent=2)

        with open(INDEX_FILE, "wb") as f:
            pickle.dump(index, f)

        progress.progress(100)
        st.session_state.is_crawling = False
        st.success(f"Indexed {len(publications)} publications")

# ==================================================
# ================= SEARCH TAB =====================
# ==================================================
with tabs[1]:
    st.subheader("Search Publications")

    query = st.text_input(
        "Search",
        placeholder="Search by title, author, year, keyword",
        label_visibility="collapsed"
    )

    # Run search
    if query:
        st.session_state.page = 1
        st.session_state.last_query = query
        st.session_state.results = st.session_state.index.search(query)

    # ------------------------------------------------
    # NORMALIZE RESULTS
    # ------------------------------------------------
    raw_results = st.session_state.results
    results = []

    for item in raw_results:
        if isinstance(item, tuple):
            if len(item) == 4:
                # New format: (doc_id, doc, tfidf, cosine)
                results.append(item)
            elif len(item) == 3:
                # Old format: (doc_id, doc, score)
                doc_id, doc, score = item
                results.append((doc_id, doc, score, 0.0))

    # ------------------------------------------------
    # DISPLAY RESULTS
    # ------------------------------------------------
    if results:
        start_idx = (st.session_state.page - 1) * RESULTS_PER_PAGE
        end_idx = start_idx + RESULTS_PER_PAGE
        page_results = results[start_idx:end_idx]

        for _, d, tfidf_score, cosine_score in page_results:
            st.markdown(f"### [{d['title']}]({d['publication_link']})")
            st.write(", ".join(d["authors"]), "·", d["year"])
            st.write(f"Cosine Similarity: {cosine_score:.4f}")
            st.write(f"TF-IDF Score: {tfidf_score:.4f}")
            st.markdown("---")

        # ------------------------------------------------
        # PAGINATION
        # ------------------------------------------------
        total_pages = math.ceil(len(results) / RESULTS_PER_PAGE)
        if total_pages > 1:
            st.session_state.page = st.number_input(
                "Page",
                min_value=1,
                max_value=total_pages,
                value=st.session_state.page,
                step=1,
                format="%d"
            )
    else:
        st.info("No results found. Try a different query.")
# ==================================================
# ================= STATISTICS TAB =================
# ==================================================
with tabs[2]:
    st.subheader("Statistics")

    index = st.session_state.index
    docs = index.documents

    # ------------------------------------------------
    # NORMALIZE RESULTS (CRITICAL – DO NOT REMOVE)
    # ------------------------------------------------
    raw_results = st.session_state.results
    results = []

    for item in raw_results:
        if isinstance(item, tuple):
            if len(item) == 4:
                # (doc_id, doc, tfidf, cosine)
                results.append(item)
            elif len(item) == 3:
                # (doc_id, doc, score) → old format
                doc_id, doc, score = item
                results.append((doc_id, doc, score, 0.0))

    # =================================================
    # 📊 Collection Overview
    # =================================================
    st.markdown("### 📊 Collection Overview")
    st.write(f"Total Documents: {len(docs)}")

    unique_authors = {
        a for d in docs.values() for a in d.get("authors", [])
    }
    st.write(f"Unique Authors: {len(unique_authors)}")

    # =================================================
    # 📂 Index Statistics
    # =================================================
    st.markdown("### 📂 Index Statistics")
    vocab_size = len(index.index)
    posting_lengths = [len(v) for v in index.index.values()]

    st.write(f"Vocabulary Size: {vocab_size}")

    if vocab_size > 0:
        st.write(f"Average Posting List Length: {np.mean(posting_lengths):.2f}")
        st.write(f"Maximum Posting List Length: {max(posting_lengths)}")
    else:
        st.write("Average Posting List Length: 0.00")

    # =================================================
    # 🔍 Query Statistics
    # =================================================
    st.markdown("### 🔍 Query Statistics")
    if st.session_state.last_query:
        st.write(f"Query Terms: {len(st.session_state.last_query.split())}")
        st.write(f"Retrieved Documents: {len(results)}")
    else:
        st.write("No query executed yet")

    # =================================================
    # 🏆 Ranking Summary
    # =================================================
    st.markdown("### 🏆 Ranking Summary")
    if results:
        top_doc = results[0][1]
        top_tfidf = results[0][2]
        top_cosine = results[0][3]

        st.write(f"Top Ranked Document: {top_doc['title']}")
        st.write(f"Top TF-IDF Score: {top_tfidf:.4f}")
        st.write(f"Top Cosine Similarity: {top_cosine:.4f}")
    else:
        st.write("No ranking available")

    # =================================================
    # 📐 Evaluation Metrics
    # =================================================
    st.markdown("### 📐 Evaluation Metrics")

    if not st.session_state.last_query:
        st.info("Run a search query to compute evaluation metrics.")
    else:
        query_terms = set(st.session_state.last_query.lower().split())

        relevant_docs = set()
        for doc_id, d in docs.items():
            text = (
                d["title"].lower()
                + " "
                + " ".join(d["authors"]).lower()
                + " "
                + str(d["year"])
            )
            if any(t in text for t in query_terms):
                relevant_docs.add(doc_id)

        retrieved_docs = {doc_id for doc_id, *_ in results}

        TP = len(retrieved_docs & relevant_docs)
        FP = len(retrieved_docs - relevant_docs)
        FN = len(relevant_docs - retrieved_docs)
        TN = len(docs) - TP - FP - FN

        accuracy = (TP + TN) / max((TP + TN + FP + FN), 1)
        precision = TP / max((TP + FP), 1)
        recall = TP / max((TP + FN), 1)
        f1 = 2 * precision * recall / max((precision + recall), 1)

        st.metric("Accuracy", f"{accuracy:.4f}")
        st.metric("Precision", f"{precision:.4f}")
        st.metric("Recall", f"{recall:.4f}")
        st.metric("F1-score", f"{f1:.4f}")