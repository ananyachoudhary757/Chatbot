# -*- coding: utf-8 -*-
"""
Chatbot Implementation
Refactored from Colab Notebook for local execution.
"""

import json
import os
import sys

# Fix for TensorFlow deadlock/mutex issues on Mac M1/M2
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pickle
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

import tensorflow as tf

# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration
INTENTS_FILE = "intents.json"
MODEL_SAVE_PATH = "intent_classification_model.h5"

def check_dependencies():
    """Download necessary NLTK data."""
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading NLTK stopwords...")
        nltk.download("stopwords")

def load_data(filepath):
    """Load intents from JSON file."""
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        print("Please ensure 'intents.json' is in the current directory.")
        # Create a dummy file for demonstration if missing? 
        # Better to just fail gracefully or ask user.
        return None
    
    with open(filepath, "r") as file:
        intents = json.load(file)
    return intents

def preprocess_for_sklearn(intents):
    """
    Preprocess data for Sklearn models:
    - Flatten JSON to DataFrame
    - Clean text
    - TF-IDF Vectorization
    """
    print("\n--- Preprocessing for Sklearn ---")
    stop_words = set(stopwords.words("english"))
    
    data = []
    for intent in intents:
        for pattern in intent.get("patterns", []):
            data.append({"pattern": pattern, "tag": intent["tag"]})
    
    df = pd.DataFrame(data)
    
    # Remove duplicates and clean
    df.drop_duplicates(inplace=True)
    df["pattern"] = df["pattern"].str.lower()
    df["pattern"] = df["pattern"].str.replace(r"[^a-zA-Z\s]", "", regex=True)
    # df["pattern"] = df["pattern"].apply(lambda x: " ".join([word for word in x.split() if word not in stop_words]))
    
    # Encode tags
    label_encoder = LabelEncoder()
    df["tag_encoded"] = label_encoder.fit_transform(df["tag"])
    
    # Visualization (optional, can be commented out for pure script usage)
    # intent_counts = df["tag"].value_counts()
    # plt.figure(figsize=(12, 6))
    # intent_counts.plot(kind="bar")
    # plt.title("Distribution of Intents")
    # plt.show()
    
    return df, label_encoder

def train_eval_sklearn(df, label_encoder):
    """Train and evaluate Sklearn models."""
    print("\n--- Training & Evaluating Sklearn Models ---")
    tfidf_vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), stop_words=None)
    X = tfidf_vectorizer.fit_transform(df["pattern"]).toarray()
    y = df["tag_encoded"]
    
    # Split
    test_size = max(0.2, len(df['tag'].unique()) / len(df))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"Random Forest - Accuracy: {accuracy_score(y_test, y_pred):.4f}, F1: {f1:.4f}")
    
    best_model = model
    
    # Save Sklearn Artifacts
    print("Saving Sklearn Artifacts...")
    with open('sklearn_model.pickle', 'wb') as f:
        pickle.dump(best_model, f)
    with open('tfidf_vectorizer.pickle', 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    
    return best_model

def preprocess_for_keras(intents):
    """Preprocess data for Keras/TensorFlow models."""
    print("\n--- Preprocessing for Keras ---")
    tags = []
    inputs = []
    for intent in intents:
        for pattern in intent.get('patterns', []):
            inputs.append(pattern)
            tags.append(intent['tag'])
            
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(tags)
    
    tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(inputs)
    sequences = tokenizer.texts_to_sequences(inputs)
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post')
    
    vocab_size = len(tokenizer.word_index) + 1
    num_classes = len(label_encoder.classes_)
    
    return padded_sequences, labels, vocab_size, num_classes, tokenizer, label_encoder

def build_keras_model(vocab_size, input_length, num_classes, embedding_dim=16, dense_units=16, dropout_rate=0.2):
    """Build a Sequential Keras model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=input_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(dense_units, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_keras_pipeline(intents):
    """Run the Keras training pipeline."""
    X, y, vocab_size, num_classes, tokenizer, label_encoder = preprocess_for_keras(intents)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Vocab Size: {vocab_size}, Classes: {num_classes}")
    
    # Hyperparameter tuning (simplified)
    best_accuracy = 0
    best_model = None
    
    embedding_dims = [16, 32]
    dense_units_options = [16, 32]
    
    print("Training Keras Model...")
    # Simplified training (no hyperparameter tuning for speed)
    model = build_keras_model(vocab_size, X.shape[1], num_classes, embedding_dim=16, dense_units=16)
    model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), verbose=1)
    
    best_model = model
    best_accuracy = model.evaluate(X_test, y_test, verbose=0)[1]
                
    print(f"Best Keras Validation Accuracy: {best_accuracy:.4f}")
    
    if best_model:
        best_model.save(MODEL_SAVE_PATH)
        print(f"Model saved to {MODEL_SAVE_PATH}")
        
        # Save artifacts
        with open('tokenizer.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Tokenizer saved to tokenizer.pickle")
        
        with open('label_encoder.pickle', 'wb') as handle:
            pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Label Encoder saved to label_encoder.pickle")
        
    return best_model

def main():
    print("Initializing Chatbot Training Script...")
    check_dependencies()
    
    intents = load_data(INTENTS_FILE)
    if intents is None:
        return

    # 1. Sklearn Pipeline (Primary for Mac stability)
    df, label_encoder = preprocess_for_sklearn(intents)
    train_eval_sklearn(df, label_encoder)
    
    # Save label encoder separate
    with open('label_encoder.pickle', 'wb') as handle:
        pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Label Encoder saved.")

    # 2. Keras Pipeline (for .h5 model)
    train_keras_pipeline(intents)
        
    print("\nAll training tasks completed successfully.")

if __name__ == "__main__":
    main()