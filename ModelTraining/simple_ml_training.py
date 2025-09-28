import re
import time
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix


stop_words = set(stopwords.words('english')) # stopwords to discard ("and", "the", etc.) that don't hold much value for the model
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text) # Remove punctuation and numbers
    uncleaned_tokens = text.split(' ')

    cleaned_tokens = [lemmatizer.lemmatize(token) for token in uncleaned_tokens if token not in stop_words]

    return " ".join(cleaned_tokens)



def createProcessedData():
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))

    original_data = pd.read_parquet('../data/parquet/labeled_sentiments.parquet', engine='fastparquet')
    texts = original_data['text'].apply(preprocess)

    # Vectorize
    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(texts)

    joblib.dump(vectorizer, "../models/tfidf_vectorizer.pkl")
    # Create processed DataFrame
    processed_data = pd.DataFrame(embeddings.toarray(), columns=vectorizer.get_feature_names_out())
    processed_data['sentiment_value'] = original_data['sentiment'].map({'positive': 1, 'negative': -1, 'neutral': 0})
    print(processed_data.head())
    processed_data.to_parquet('../data/parquet/processed_sentiments.parquet', engine='fastparquet')



def train_model():
    data = pd.read_parquet('../data/parquet/processed_sentiments.parquet', engine='fastparquet')
    # Separate data into train x, train y, 80% train split
    
    num_columns = len(data.columns)
    num_rows = int(len(data) * 0.8)
    train_data = np.array(data)

    X = train_data[:, :-1]
    y = train_data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    

    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    
    start = time.time()
    model.fit(X_train, y_train)

    time_elapsed = time.time() - start

    joblib.dump(model, '../models/logreg_model.pkl')
    y_pred = model.predict(X_test)


    with open('../logs/simple_model_training.log', 'w') as log:
        log.write("Logistic Regression Model Training results\n")
        log.write(f"Training took {time_elapsed:.2f} seconds\n")
        log.write(classification_report(y_test, y_pred))
     
    print(confusion_matrix(y_test, y_pred))

    


createProcessedData()
train_model()