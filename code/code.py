import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,cohen_kappa_score, precision_score, f1_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Task 1: Gather the dataset
# Assuming the dataset is in a folder named "sentiment labelled sentences"
folder_path = "sentiment labelled sentences"
file_names = ['amazon_cells_labelled.txt', 'imdb_labelled.txt', 'yelp_labelled.txt']

# Load data from multiple files
dfs = []
for file_name in file_names:
    file_path = f"{folder_path}/{file_name}"
    df = pd.read_csv(file_path, sep='\t', header=None, names=['sentence', 'label'])
    dfs.append(df)

# Concatenate data from different files
df = pd.concat(dfs, ignore_index=True)

# Task 2: Preprocess the text data
# Tokenization, lowercasing, and removing stopwords using NLTK
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    words = word_tokenize(text.lower())
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
    return ' '.join(filtered_words)

df['processed_sentence'] = df['sentence'].apply(preprocess_text)

# Task 3: Implement a DummyClassifier
X_train, X_test, y_train, y_test = train_test_split(df['processed_sentence'], df['label'], test_size=0.2, random_state=42)

dummy_clf = make_pipeline(CountVectorizer(), DummyClassifier(strategy='most_frequent'))
dummy_clf.fit(X_train, y_train)
y_pred_dummy = dummy_clf.predict(X_test)

# Evaluate DummyClassifier performance
accuracy_dummy = accuracy_score(y_test, y_pred_dummy)
precision_dummy = precision_score(y_test, y_pred_dummy)
recall_dummy = recall_score(y_test, y_pred_dummy)
f1_dummy = f1_score(y_test, y_pred_dummy)

print("DummyClassifier Performance:")
print(f"Accuracy: {accuracy_dummy}")
print(f"Precision: {precision_dummy}")
print(f"Recall: {recall_dummy}")
print(f"F1-score: {f1_dummy}")