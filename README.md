# Sentiment Based Financial Predictions

This repository performs risk prediction and analysis based on previous financial data, as well as current sentiment. From scratch, this model builds a framework to analyze financial and sentiment data using ML.

## 1. Data Cleaning and Exploration
This module imports data from yfinance on the 'GOOG', 'TSLA', 'NVDA', and 'AAPL' tickers, cleans the data, and performs standard data analyses on them (i.e. Max drawdown, rolling volatility, distribution stats)

## 2. Sentiment Model Training
This module uses the FinancialPhraseBankv1.0 dataset to train 2 different models to recognize positive, negative, and neutral sentiment finance based sentences. The first is a logistic regression model, trained from scratch using lemmatizing and TF-IDF to vectorize sentences. The second is FinBERT, a state-of-the-art pretrained transformer trained on financial sentiment data.

### Results
Logistic Regression - 91% Accuracy on test dataset
FinBERT - 96% Accuracy on test dataset
Given a larger amount of labeled data, I would have likely attempted to train a deep learning model as well, but 96% is good enough for my purposes

## 3. **Upcoming** Sentiment Based Risk Analysis Model
