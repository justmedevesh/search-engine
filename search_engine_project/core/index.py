import math
from collections import defaultdict
import re

STOP_WORDS = {
    "a","an","and","are","as","at","be","by","for","from","has","he",
    "in","is","it","its","of","on","or","that","the","to","was","will",
    "with","this","but","they","have","had","what","when","where",
    "who","why","how"
}

# --------------------------------------------------
# TEXT PREPROCESSOR
# --------------------------------------------------
class TextPreprocessor:
    @staticmethod
    def preprocess(text):
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        return text

    @staticmethod
    def tokenize(text):
        return text.split()

    @staticmethod
    def remove_stopwords(tokens):
        return [t for t in tokens if t not in STOP_WORDS and len(t) > 2]


# --------------------------------------------------
# ADVANCED INVERTED INDEX (COSINE + TF-IDF)
# --------------------------------------------------
class AdvancedInvertedIndex:
    def __init__(self):
        self.index = defaultdict(list)        # term → [(doc_id, tf)]
        self.documents = {}                   # doc_id → document
        self.doc_vectors = {}                 # doc_id → tf-idf vector
        self.doc_count = 0

    # --------------------------------------------------
    # ADD DOCUMENT
    # --------------------------------------------------
    def add_document(self, doc_id, doc):
        self.documents[doc_id] = doc
        self.doc_count += 1

        text = (
            doc.get("title", "") + " " +
            " ".join(doc.get("authors", [])) + " " +
            str(doc.get("year", ""))
        )

        processed = TextPreprocessor.preprocess(text)
        tokens = TextPreprocessor.tokenize(processed)
        tokens = TextPreprocessor.remove_stopwords(tokens)

        tf = defaultdict(int)
        for t in tokens:
            tf[t] += 1

        for term, freq in tf.items():
            self.index[term].append((doc_id, freq))

    # --------------------------------------------------
    # BUILD TF-IDF VECTORS (REQUIRED)
    # --------------------------------------------------
    def build_tfidf_vectors(self):
        self.doc_vectors = {}

        for doc_id in self.documents:
            vec = {}
            for term, postings in self.index.items():
                df = len(postings)
                idf = math.log((self.doc_count + 1) / (df + 1))

                for d_id, tf in postings:
                    if d_id == doc_id:
                        vec[term] = tf * idf

            self.doc_vectors[doc_id] = vec

    # --------------------------------------------------
    # SEARCH (COSINE PRIMARY, TF-IDF SECONDARY)
    # RETURNS: (doc_id, doc, tfidf_score, cosine_score)
    # --------------------------------------------------
    def search(self, query):
        processed = TextPreprocessor.preprocess(query)
        tokens = TextPreprocessor.tokenize(processed)
        tokens = TextPreprocessor.remove_stopwords(tokens)

        if not tokens:
            return []

        # ---------- QUERY VECTOR ----------
        q_vec = defaultdict(float)
        for term in tokens:
            if term in self.index:
                df = len(self.index[term])
                idf = math.log((self.doc_count + 1) / (df + 1))
                q_vec[term] += idf

        q_norm = math.sqrt(sum(v * v for v in q_vec.values()))
        if q_norm == 0:
            return []

        results = []

        # ---------- DOCUMENT COMPARISON ----------
        for doc_id, d_vec in self.doc_vectors.items():
            dot = sum(q_vec[t] * d_vec.get(t, 0) for t in q_vec)
            d_norm = math.sqrt(sum(v * v for v in d_vec.values()))

            cosine = dot / (q_norm * d_norm) if d_norm != 0 else 0.0
            tfidf_score = sum(d_vec.get(t, 0) for t in q_vec)

            if cosine > 0:
                results.append(
                    (doc_id, self.documents[doc_id], tfidf_score, cosine)
                )

        # 🔥 COSINE PRIMARY SORT
        results.sort(key=lambda x: (x[3], x[2]), reverse=True)
        return results