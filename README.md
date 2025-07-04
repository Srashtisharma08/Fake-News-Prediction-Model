# 📰 Fake News Prediction Model 🚦

## Overview
Welcome to the **Fake News Prediction Model** project! This repository provides a robust, production-ready solution for detecting fake news using advanced Natural Language Processing (NLP) and Machine Learning techniques. Built with Python, Flask, and scikit-learn, this project empowers users to verify the authenticity of news articles in real time.

---

## ✨ Features
- ⚡ **Real-Time Fake News Detection**
- 🤖 **AI-Powered Model** (Logistic Regression / SVM)
- 📊 **High Accuracy** with detailed evaluation metrics
- 🌐 **Web Interface** for easy user interaction
- 🏷️ **Batch Training** for large datasets
- 🧹 **Advanced Text Preprocessing** (Stemming, Stopword Removal)
- 📝 **Jupyter Notebook** for experimentation
- 🗃️ **Multiple Dataset Support** (train.csv, archive/fake.csv, archive/true.csv)
- 🛡️ **Robust Logging & Error Handling**

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/fake-news-prediction-model.git
cd fake-news-prediction-model
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare Data
- Place your datasets in the correct locations:
  - `train.csv` in the root directory
  - `archive/fake.csv` and `archive/true.csv` in the `archive/` folder

### 4. Train the Model
```bash
python train_model.py
```
- Training logs will be saved to `training.log`
- The trained model and vectorizer will be saved in the `models/` directory

### 5. Run the Web App
```bash
python app.py
```
- Open your browser and go to [http://localhost:5000](http://localhost:5000)

---

## 🖥️ Project Structure
```
Fake-News-Prediction-Model/
├── app.py                # Flask web application
├── train_model.py        # Model training script
├── Model.ipynb           # Jupyter notebook for experiments
├── analyze_datasets.py   # Dataset analysis utilities
├── static/               # CSS & JS files
│   ├── style.css
│   └── script.js
├── templates/            # HTML templates
│   └── index1.html
├── models/               # Saved models
│   ├── fake_news_model.pkl
│   └── vectorizer.pkl
├── archive/              # Additional datasets
│   ├── fake.csv
│   └── true.csv
├── train.csv             # Main training dataset
├── test.csv              # (Optional) Test dataset
├── logo.png              # Project logo
└── README.md             # This file
```

---

## 🧠 Model Details
- **Text Preprocessing:**
  - Lowercasing, punctuation removal, stopword removal, stemming
- **Vectorization:**
  - TF-IDF (Term Frequency-Inverse Document Frequency)
- **Model:**
  - Logistic Regression (default) or Linear SVM
- **Evaluation:**
  - Accuracy, Precision, Recall, F1-Score, Classification Report

---

## 🌟 Example Usage
### Web App
- Paste a news article URL or text into the web interface
- Click **Verify** to get instant prediction: `Real News` or `Fake News`

### Programmatic Prediction
```python
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re

def preprocess(text):
    port_stem = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    text = re.sub('[^a-zA-Z]', ' ', text).lower().split()
    text = [port_stem.stem(word) for word in text if word not in stop_words]
    return ' '.join(text)

with open('models/fake_news_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

sample = "Your news article text here."
processed = preprocess(sample)
vector = vectorizer.transform([processed])
pred = model.predict(vector)
print('Fake News' if pred[0] == 0 else 'Real News')
```

---

## 📈 Results
- **Training Accuracy:** _See `training.log` for details_
- **Test Accuracy:** _See `training.log` for details_
- **Classification Report:** _See `training.log` for precision, recall, F1-score_

---

## 🛠️ Customization
- You can easily swap out the model (e.g., use SVM or other classifiers)
- Add more datasets to the `archive/` folder for improved performance
- Tweak preprocessing or vectorization in `train_model.py` or `Model.ipynb`

---

## 🤝 Contributing
Pull requests, issues, and suggestions are welcome! Please open an issue or submit a PR.

---

## 📄 License
This project is licensed under the MIT License.

---

## 🙏 Acknowledgements
- [scikit-learn](https://scikit-learn.org/)
- [NLTK](https://www.nltk.org/)
- [Flask](https://flask.palletsprojects.com/)
- [Pandas](https://pandas.pydata.org/)

---

> **Built with ❤️ by [Your Name/Team] — April 2025**
=======
This is a Fake News Prediction Model that uses a Logistic Regression model to predict whether a given news article is fake or not. The model is trained on a dataset of 5000 news articles, and the accuracy of the model is 95.8%. 
>>>>>>> 3c15b84fcbaa79c8fc4a81d2ec052a0a7a25d1d2

