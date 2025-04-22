from flask import Flask, render_template, request, url_for, send_from_directory
import requests
from bs4 import BeautifulSoup
import pickle
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import nltk
import os

# Download required NLTK data
nltk.download('stopwords')

# Initialize Flask app
app = Flask(__name__)

# Configure static folder explicitly
app.static_folder = 'static'

# Add debug route for static files
@app.route('/static/<path:filename>')
def serve_static(filename):
    try:
        return send_from_directory(app.static_folder, filename)
    except Exception as e:
        app.logger.error(f"Error serving static file {filename}: {str(e)}")
        return str(e), 404

# Initialize PorterStemmer
port_stem = PorterStemmer()

# Load the saved model and vectorizer
try:
    with open('fake_news_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)
    print("Model and vectorizer loaded successfully")
except Exception as e:
    print(f"Error loading model or vectorizer: {e}")
    model = None
    vectorizer = None

def stemming(content):
    """Process and stem the text content."""
    # Remove all characters that are not letters
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    
    # Convert to lowercase
    stemmed_content = stemmed_content.lower()
    
    # Split into words
    stemmed_content = stemmed_content.split()
    
    # Apply stemming and remove stopwords
    stemmed_content = [port_stem.stem(word) for word in stemmed_content 
                      if not word in stopwords.words('english')]
    
    # Join words back together
    return ' '.join(stemmed_content)

def predict_news(news_text):
    """Predict if news is real or fake."""
    try:
        if not model or not vectorizer:
            raise RuntimeError("Model or vectorizer not loaded")
        
        # Preprocess the text
        processed_text = stemming(news_text)
        
        if not processed_text:
            raise ValueError("Text preprocessing failed")
        
        # Transform using the saved vectorizer
        text_vector = vectorizer.transform([processed_text])
        
        # Make prediction
        prediction = model.predict(text_vector)
        
        return "Real News" if prediction[0] == 0 else "Fake News"
    
    except Exception as e:
        raise Exception(f"Error in prediction: {str(e)}")

def extract_text_from_url(url):
    """Extract text content from a URL."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # Send request
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract text from paragraphs
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs])
        
        if not text:
            raise ValueError("No text content found in the URL")
        
        return text
    
    except Exception as e:
        raise Exception(f"Error extracting text from URL: {str(e)}")

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        url = request.form.get('url')
        if not url:
            return render_template('index1.html', error="Please enter a URL")

        # Extract text from URL
        news_text = extract_text_from_url(url)
        
        # Make prediction
        result = predict_news(news_text)
        
        return render_template('index1.html', url=url, result=result, text_preview=news_text[:500] + "...")
    
    except Exception as e:
        return render_template('index1.html', url=url, error=str(e))

if __name__ == '__main__':
    if model and vectorizer:
        app.run(debug=True)
    else:
        print("Application cannot start: Model or vectorizer not loaded")