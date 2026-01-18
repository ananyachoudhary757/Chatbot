# 🤖 AI Intent Recognition Chatbot

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![Scikit-learn](https://img.shields.io/badge/Sklearn-ML-orange)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DL-FF6F00)

## 1. Project Overview

This project is an advanced AI-powered chatbot capable of understanding user intents and providing context-aware responses. Built with a robust hybrid machine learning pipeline, it leverages both **Scikit-learn (Random Forest)** and **TensorFlow/Keras** to ensure high accuracy in intent detection.

The application features a modern, responsive user interface built with **Streamlit**, offering:
- **Real-time Intent Classification**: Instantly categorizes user queries.
- **Confidence Scoring**: Displays the model's certainty for each prediction.
- **Neural Dashboard**: A side panel visualizing live tracking statistics and system status.
- **Interactive UI**: Dark-themed, glassmorphism-inspired design for a premium user experience.

This tool is designed to demonstrate proficiency in Natural Language Processing (NLP), Full-Stack Data Science application development, and UI/UX design in Python.

---

## 2. Step-by-Step Installation Guide

Follow these instructions to set up the project on your local machine.

### Prerequisites
- **Python 3.8** or higher installed. [Download Python](https://www.python.org/downloads/)
- **Git** (optional, for cloning).

### Installation Steps

1.  **Clone the Repository**
    Open your terminal or command prompt and run:
    ```bash
    git clone <your-repo-url>
    cd Chatbot
    ```

2.  **Create a Virtual Environment (Recommended)**
    It's best practice to use a virtual environment to manage dependencies.
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # Mac/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    Install the required Python packages using `pip`.
    ```bash
    pip install -r requirements.txt
    ```
    
    > **Note**: This project also requires TensorFlow. If it's not installed automatically, run:
    ```bash
    pip install tensorflow
    ```

4.  **Download NLTK Data**
    The chatbot relies on NLTK for text processing. The script handles this automatically, but you can also run:
    ```python
    import nltk
    nltk.download('stopwords')
    ```

---

## 3. Usage Guide

### Training the Model
Before running the chat interface, ensure the model is trained on the latest `intents.json` data.

```bash
python chatbot.py
```
*This command will process the data, train the Random Forest and Neural Network models, and save the artifacts (`sklearn_model.pickle`, `intent_classification_model.h5`, etc.) to your directory.*

### Running the Application
Launch the web interface using Streamlit:

```bash
streamlit run app.py
```
*A new tab will open in your default browser displaying the chatbot interface.*

---

## 4. Project Structure

- **`app.py`**: The main Streamlit application script containing the UI and inference logic.
- **`chatbot.py`**: The backend script for data preprocessing, model training, and evaluation.
- **`intents.json`**: The knowledge base containing training patterns and responses.
- **`requirements.txt`**: List of Python dependencies.

---

### Author
Developed by [Ananya Choudhary]
