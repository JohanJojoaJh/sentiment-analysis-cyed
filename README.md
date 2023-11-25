# Sentiment Analysis Project Resume

## Introduction

Sentiment analysis, a crucial facet of natural language processing, endeavors to discern sentiments in textual data. This project delves into the creation of sentiment analysis models utilizing two powerful neural network architectures: vanilla Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks. Sentiment analysis plays a pivotal role in extracting emotions and opinions from text, finding applications across diverse domain. This project also enables to observe the improvent of performance between models like RNN and more powerful models like LSTM. 

## Objectives

The primary objectives of this project revolve around the development and evaluation of sentiment analysis models using RNNs and LSTMs. The dataset, sourced from Amazon, IMDb, and Yelp, features binary labels indicating positive or negative sentiments. The objectives encompass constructing robust models, conducting comprehensive performance evaluations, and optimizing hyperparameters.

## Methodology

The project unfolds through phases of data collection, preprocessing, and model development. RNNs and LSTMs are implemented on the Sentiment Labelled Sentences dataset, with hyperparameter tuning executed through grid searches. Model performance is assessed using metrics such as accuracy, precision, recall, and F1-score.

## Dataset Details

The dataset comprises labeled sentences from Amazon, IMDb, and Yelp, contributing 500 positive and 500 negative sentences each, resulting in a meticulously balanced dataset.

## Preprocessing Steps

Data undergoes loading, concatenation, and shuffling. NLTK is employed for tokenization, lowercasing, and stopword removal, culminating in a DataFrame featuring both original and processed sentences.

## Models

### Dummy Classifier:

- **Description:** A baseline model assigning the most frequent class.
- **Architecture:** Simple, with no complex layers or training.
- **Training:** Minimal or nonexistent, serving as a baseline for comparison.
- **Purpose:** Establishes a baseline for comparison with more advanced models.

### RNN Sentiment Analysis Model:

- **Architecture:** Embedding, SimpleRNN, Dropout, Dense layers.
- **Training:** Tokenization, padding, fitting with a 20% validation split.
- **Results:** Balanced performance with equal precision, recall, and F1-score.

### LSTM Sentiment Analysis Model:

- **Architecture:** Embedding, LSTM, Dropout, Dense layers.
- **Training:** Similar to RNN, but with LSTM layers for long-term dependencies.
- **Results:** Improved performance compared to RNN, demonstrating LSTM's ability to capture long-term patterns.

## Performance Evaluation

Models are iteratively run 100 times for robust evaluation. Metrics include accuracy, precision, recall, F1-score, and Kappa-score. LSTM consistently outperforms the Dummy Classifier and RNN in precision and overall effectiveness.

## Comparative Analysis

- **Dummy Classifier:** Low accuracy and precision, perfect recall.
- **RNN:** Balanced performance with consistent metrics.
- **LSTM:** Improved precision, Kappa score, and overall performance.

## Conclusion

The LSTM model significantly outperforms both the Dummy Classifier and RNN, showcasing enhanced precision and predictive capability. The selection of LSTM is justified by its proficiency in handling long-term dependencies in sequential data, capturing more complex patterns for improved accuracy. This project offers valuable insights into effective sentiment analysis models and their comparative performance.
