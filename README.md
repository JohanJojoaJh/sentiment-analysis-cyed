# sentiment-analysis-cyed

Introduction
Sentiment analysis, a vital aspect of natural language processing, aims to discern sentiments in textual data. This project focuses on creating sentiment analysis models using two neural network architectures: vanilla Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks. Sentiment analysis is integral for extracting emotions and opinions from text, finding applications across various domains.

Objectives
The primary goals of this project include developing and evaluating sentiment analysis models using RNNs and LSTMs. The dataset is sourced from Amazon, IMDb, and Yelp, with binary labels indicating positive or negative sentiments. Objectives encompass constructing robust models, comprehensive performance evaluation, and hyperparameter optimization.

Methodology
The project involves data collection, preprocessing, and model development. RNNs and LSTMs are implemented on the Sentiment Labelled Sentences dataset. Hyperparameter tuning is performed through grid searches, and model performance is evaluated using metrics like accuracy, precision, recall, and F1-score.

Dataset Details
The dataset comprises labeled sentences from Amazon, IMDb, and Yelp, contributing 500 positive and 500 negative sentences each, resulting in a balanced dataset.

Preprocessing Steps
Data is loaded, concatenated, and shuffled. NLTK is used for tokenization, lowercasing, and stopword removal, resulting in a DataFrame with original and processed sentences.

Models
Dummy Classifier:

Description: A baseline model assigning the most frequent class.
Architecture: Simple, no complex layers or training.
Training: Minimal or nonexistent, as it doesn't learn from data.
Purpose: Establishes a baseline for comparison with more advanced models.
RNN Sentiment Analysis Model:

Architecture: Embedding, SimpleRNN, Dropout, Dense layers.
Training: Tokenization, padding, fitting with a 20% validation split.
Results: Balanced performance with equal precision, recall, and F1-score.
LSTM Sentiment Analysis Model:

Architecture: Embedding, LSTM, Dropout, Dense layers.
Training: Similar to RNN, but with LSTM layers for long-term dependencies.
Results: Improved performance compared to RNN, demonstrating LSTM's ability to capture long-term patterns.
Performance Evaluation
Models are run 100 times for robust evaluation. Metrics include accuracy, precision, recall, F1-score, and Kappa-score. LSTM outperforms Dummy Classifier and RNN in precision and overall effectiveness.

Comparative Analysis
Dummy Classifier: Low accuracy and precision, perfect recall.
RNN: Balanced performance with consistent metrics.
LSTM: Improved precision, Kappa score, and overall performance.
Conclusion
The LSTM model significantly outperforms both the Dummy Classifier and RNN, demonstrating enhanced precision and predictive capability. The choice of LSTM is justified by its ability to handle long-term dependencies in sequential data, capturing more complex patterns for improved accuracy. This project provides insights into effective sentiment analysis models and their comparative performance.
