import pandas as pd
import numpy as np
import re
from collections import Counter
import nltk
nltk.download('punkt_tab')
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings

warnings.filterwarnings('ignore')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class AdNLPAnalyzer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2)
        )

    def clean_text(self, text):
        if pd.isna(text):
            return ""

        text_clean = re.sub(r'[^\w\s]', ' ', str(text))
        text_clean = re.sub(r'\s+', ' ', text_clean).strip().lower()
        return text_clean

    def extract_emojis(self, text):
        if pd.isna(text):
            return []

        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols and pictographs
            "\U0001F680-\U0001F6FF"  # transport and map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE
        )
        return emoji_pattern.findall(str(text))

    def get_sentiment_score(self, text):
        if pd.isna(text):
            return 0.0

        blob = TextBlob(str(text))
        return (blob.sentiment.polarity + 1) / 2

    def extract_keywords(self, text):
        clean_text = self.clean_text(text)
        if not clean_text:
            return []

        tokens = word_tokenize(clean_text)
        keywords = [word for word in tokens
                    if word not in self.stop_words
                    and len(word) > 2
                    and word.isalpha()]
        return keywords

    def get_urgency_score(self, text):
        if pd.isna(text):
            return 0.0

        urgency_words = [
            'now', 'today', 'limited', 'hurry', 'fast', 'quick', 'urgent',
            'immediate', 'instant', 'asap', 'deadline', 'expires', 'ending',
            'last chance', 'final', 'rush', 'flash', 'lightning'
        ]

        text_lower = str(text).lower()
        score = sum(1 for word in urgency_words if word in text_lower)
        return min(score / 3, 1.0)

    def get_cta_strength(self, text):
        if pd.isna(text):
            return 0.0

        cta_words = [
            'buy', 'shop', 'get', 'download', 'subscribe', 'join', 'start',
            'try', 'book', 'call', 'click', 'order', 'purchase', 'apply',
            'register', 'sign up', 'learn more', 'discover', 'explore'
        ]

        text_lower = str(text).lower()
        score = sum(1 for word in cta_words if word in text_lower)
        return min(score / 2, 1.0)

    def count_features(self, text):
        if pd.isna(text):
            return {
                'word_count': 0,
                'char_count': 0,
                'emoji_count': 0,
                'exclamation_count': 0,
                'question_count': 0,
                'number_count': 0
            }

        text_str = str(text)

        return {
            'word_count': len(text_str.split()),
            'char_count': len(text_str),
            'emoji_count': len(self.extract_emojis(text_str)),
            'exclamation_count': text_str.count('!'),
            'question_count': text_str.count('?'),
            'number_count': len(re.findall(r'\d+', text_str))
        }

    def analyze_text_batch(self, texts):
        results = []

        for text in texts:
            features = self.count_features(text)
            features.update({
                'sentiment_score': self.get_sentiment_score(text),
                'urgency_score': self.get_urgency_score(text),
                'cta_strength': self.get_cta_strength(text),
                'keywords': self.extract_keywords(text),
                'emojis': self.extract_emojis(text)
            })
            results.append(features)

        return results

    def get_top_keywords(self, texts, top_n=20):
        all_keywords = []
        for text in texts:
            all_keywords.extend(self.extract_keywords(text))

        return Counter(all_keywords).most_common(top_n)

    def analyze_dataset(self, df, text_column='ad_text'):
        print("🧠 Starting NLP analysis...")

        nlp_results = self.analyze_text_batch(df[text_column].fillna(''))

        features_df = pd.DataFrame(nlp_results)

        for col in ['sentiment_score', 'urgency_score', 'cta_strength',
                    'word_count', 'char_count', 'emoji_count',
                    'exclamation_count', 'question_count', 'number_count']:
            df[f'nlp_{col}'] = features_df[col]

        top_keywords = self.get_top_keywords(df[text_column])

        print(f"✅ NLP analysis complete!")
        print(f"📊 Top 10 keywords: {[word for word, count in top_keywords[:10]]}")

        return df, top_keywords


def main():
    print("📂 Loading ad performance data...")
    df = pd.read_csv('../data/raw/ad_performance_data.csv')

    analyzer = AdNLPAnalyzer()

    df_analyzed, top_keywords = analyzer.analyze_dataset(df)

    df_analyzed.to_csv('../data/processed/ad_data_with_nlp.csv', index=False)

    keywords_df = pd.DataFrame(top_keywords, columns=['keyword', 'frequency'])
    keywords_df.to_csv('../data/processed/top_keywords.csv', index=False)

    print(f" NLP Insights:")
    print(f"Average sentiment score: {df_analyzed['nlp_sentiment_score'].mean():.3f}")
    print(f"Average urgency score: {df_analyzed['nlp_urgency_score'].mean():.3f}")
    print(f"Average CTA strength: {df_analyzed['nlp_cta_strength'].mean():.3f}")

    numeric_cols = ['nlp_sentiment_score', 'nlp_urgency_score', 'nlp_cta_strength',
                    'nlp_word_count', 'nlp_emoji_count']

    print(f" Correlation with CTR:")
    for col in numeric_cols:
        corr = df_analyzed[col].corr(df_analyzed['ctr'])
        print(f"{col}: {corr:.3f}")

    return df_analyzed, top_keywords


if __name__ == "__main__":
    df_analyzed, keywords = main()