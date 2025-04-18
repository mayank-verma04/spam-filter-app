# model-training/train_model.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk

nltk.download('stopwords')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Keep important symbols and numbers
    text = re.sub(r'[^\w\s$%!?]', '', text)
    text = re.sub(r'\d+', ' NUM ', text)
    
    # More aggressive stemming
    stemmer = SnowballStemmer('english')
    words = [stemmer.stem(word) for word in text.split()]
    
    # Remove unhelpful stopwords
    stops = set(stopwords.words('english')) - {'you', 'your', 'free', 'win', 'claim'}
    words = [word for word in words if word not in stops]
    
    return ' '.join(words)

def train_model():
    # Load and prepare data
    df = pd.read_csv('data/spam.csv', encoding='latin-1')
    df = df[['v1', 'v2']].rename(columns={'v1':'label', 'v2':'text'})
    
    # Balance classes by oversampling spam
    spam_df = df[df['label'] == 'spam']
    df = pd.concat([df, spam_df])  # Simple duplicate spam samples
    
    # Preprocess
    df['processed'] = df['text'].apply(preprocess_text)
    
    # Optimized pipeline
    model = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 3),  # Check 1-3 word combinations
            stop_words='english')),
        ('clf', LogisticRegression(
            class_weight='balanced',
            C=0.5,  # Regularization strength
            solver='liblinear'))
    ])
    
    # Train and save
    model.fit(df['processed'], df['label'])
    joblib.dump(model, 'models/spam_model.pkl')
    print("Enhanced model trained successfully!")

if __name__ == '__main__':
    train_model()