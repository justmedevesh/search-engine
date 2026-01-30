import re

STOP_WORDS = {
    'a','an','and','are','as','at','be','by','for','from','has','he','in',
    'is','it','its','of','on','or','that','the','to','was','will','with',
    'this','but','they','have','had','what','when','where','who','why',
    'how','all','each','every','both','few','more','most','other','some',
    'such','no','nor','not','only','same','so','than','too','very','can',
    'just','should','now'
}

class TextPreprocessor:
    @staticmethod
    def preprocess(text):
        text = text.lower()
        return re.sub(r'[^\w\s]', '', text)

    @staticmethod
    def tokenize(text):
        return text.split()

    @staticmethod
    def remove_stopwords(tokens):
        return [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
