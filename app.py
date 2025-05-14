from flask import Flask, render_template, request, url_for, send_from_directory, send_file
import requests
from bs4 import BeautifulSoup
import pickle
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import nltk
import os
import pandas as pd

# Download required NLTK data
nltk.download('stopwords')

# Initialize Flask app
app = Flask(__name__)

# Configure static folder explicitly
app.static_folder = 'static'

# Define trusted domains
trusted_domains = [
    # International News Agencies
    'reuters.com', 'apnews.com', 'afp.com', 'bloomberg.com', 
    'dpa-international.com', 'efe.com', 'kyodonews.jp', 'xinhuanet.com',
    
    # North American
    'nytimes.com', 'washingtonpost.com', 'wsj.com', 'npr.org',
    'cbc.ca', 'theglobeandmail.com', 'mexiconewsdaily.com',
    
    # European
    'bbc.com', 'bbc.co.uk', 'theguardian.com', 'dw.com', 
    'lemonde.fr', 'elpais.com', 'ansa.it', 'tass.com',
    
    # Asia-Pacific
    'scmp.com', 'japantimes.co.jp', 'koreatimes.co.kr', 'straitstimes.com',
    'abc.net.au', 'rnz.co.nz',
    
    # Indian Major English News
    'timesofindia.indiatimes.com', 'thehindu.com', 'indianexpress.com',
    'hindustantimes.com', 'ndtv.com', 'news18.com', 'indiatoday.in',
    'theprint.in', 'business-standard.com', 'livemint.com',
    'deccanherald.com', 'thewire.in', 'scroll.in', 'firstpost.com',
    'thequint.com', 'outlookindia.com', 'tribuneindia.com',
    'telegraphindia.com', 'asianage.com', 'newslaundry.com',
    'rediff.com', 'dnaindia.com', 'theweek.in',
    
    # Indian Regional News
    'aajtak.in', 'amarujala.com', 'bhaskar.com', 'jagran.com',
    'patrika.com', 'navbharattimes.indiatimes.com', 'zeenews.india.com',
    'prabhatkhabar.com', 'loksatta.com', 'maharashtratimes.com',
    'eenadu.net', 'manoramaonline.com', 'mathrubhumi.com',
    'dinamalar.com', 'dinamani.com', 'kannadaprabha.com',
    'anandabazar.com', 'gujaratsamachar.com', 'divyabhaskar.co.in',
    
    # News Agencies and Public Broadcasters
    'pti.in', 'ani.in', 'ians.in', 'uniindia.com',
    'newsonair.gov.in', 'doordarshan.gov.in', 'ddnews.gov.in',
    'prasarbharati.gov.in',
    
    # Fact-Checking
    'altnews.in', 'boomlive.in', 'factchecker.in', 'smhoaxslayer.com',
    'newschecker.in', 'indiacheck.org',
    
    # Business News
    'moneycontrol.com', 'economictimes.indiatimes.com', 
    'financialexpress.com', 'cnbctv18.com'
]

# Add route for logo.jpg
@app.route('/logo.jpg')
def serve_logo():
    return send_file('static/logo.jpg', mimetype='image/jpeg')

# Add debug route for static files
@app.route('/static/<path:filename>')
def serve_static(filename):
    try:
        return send_from_directory(app.static_folder, filename)
    except Exception as e:
        app.logger.error(f"Error serving static file {filename}: {str(e)}")
        return str(e), 404

def extract_text_from_url(url):
    """Extract text content from a URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract all paragraphs
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs])
        
        return text
    
    except Exception as e:
        raise Exception(f"Error extracting text from URL: {str(e)}")

def stemming(content):
    """Process text while preserving more meaningful content"""
    try:
        port_stem = PorterStemmer()
        
        # Convert to string and lowercase
        content = str(content).lower()
        
        # Remove URLs but preserve important markers
        content = re.sub(r'http\S+|www\S+|https\S+', '', content, flags=re.MULTILINE)
        
        # Remove special characters but keep sentence structure and quotes
        content = re.sub(r'[^\w\s.,!?"\']', ' ', content)
        
        # Split into words
        words = content.split()
        
        # Apply selective stemming
        processed_words = []
        for word in words:
            # Keep important structural words and numbers
            if len(word) <= 3 or word.isdigit():
                processed_words.append(word)
                continue
            
            # Skip common words
            if word in stopwords.words('english'):
                continue
            
            # Apply stemming to longer words
            if len(word) > 3:
                word = port_stem.stem(word)
            
            processed_words.append(word)
        
        return ' '.join(processed_words)
    except Exception as e:
        print(f"Error in stemming: {str(e)}")
        return ""

def calculate_credibility_score(text, url):
    """Calculate a credibility score with enhanced precision"""
    score = 0
    
    # 1. Domain Credibility (max 3 points)
    if any(domain in url.lower() for domain in trusted_domains):
        score += 3
    
    # 2. Professional Writing Indicators (max 4 points)
    professional_indicators = {
        r'\b\d{4}\b': 0.5,  # Years
        r'\b\d+%\b': 0.5,   # Percentages
        r'\baccording to\b': 0.5,
        r'\bsaid\b.*\b(analyst|expert|official|spokesperson|professor)\b': 0.5,
        r'\b(study|research|survey|report)\b.*\bshows\b': 0.5,
        r'\bcited\b': 0.5,
        r'\bsources\b.*\bsaid\b': 0.5,
        r'\bin a statement\b': 0.25,
        r'\bconfirmed\b': 0.25
    }
    
    for pattern, points in professional_indicators.items():
        if re.search(pattern, text.lower()):
            score += points
    
    # 3. Balanced Reporting (max 2 points)
    balanced_indicators = {
        r'\bhowever\b': 0.4,
        r'\bon the other hand\b': 0.4,
        r'\bin contrast\b': 0.4,
        r'\bwhile\b': 0.4,
        r'\bdespite\b': 0.4
    }
    
    for pattern, points in balanced_indicators.items():
        if re.search(pattern, text.lower()):
            score += points
    
    # 4. Article Structure (max 3 points)
    # Length check
    word_count = len(text.split())
    if word_count > 300:
        score += 1.5
    elif word_count > 150:
        score += 0.75
    
    # Paragraph structure
    paragraphs = text.split('\n\n')
    if len(paragraphs) >= 4:
        score += 0.75
    
    # Sentence complexity
    sentences = text.split('.')
    avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
    if 10 <= avg_sentence_length <= 25:  # Ideal range for professional writing
        score += 0.75
    
    # 5. Quote Analysis (max 2 points)
    quotes = re.findall(r'"([^"]*)"', text)
    if len(quotes) >= 3:
        score += 1
    elif len(quotes) >= 1:
        score += 0.5
    
    # Additional quote validation
    valid_quotes = [q for q in quotes if len(q.split()) > 3]  # Meaningful quotes
    if len(valid_quotes) >= 2:
        score += 0.5
    
    return min(score, 10)  # Cap at 10 points

def assess_content_quality(text):
    """Assess content quality with enhanced precision"""
    quality_score = 0
    
    # 1. Professional Writing Style (max 2 points)
    sentences = re.split(r'[.!?]+', text)
    properly_capitalized = sum(1 for s in sentences if s.strip() and s.strip()[0].isupper())
    capitalization_ratio = properly_capitalized / max(len(sentences), 1)
    
    if capitalization_ratio > 0.9:
        quality_score += 2
    elif capitalization_ratio > 0.7:
        quality_score += 1
    
    # 2. Data and Facts (max 2 points)
    fact_patterns = {
        r'\b\d+\b': 0.5,  # Numbers
        r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\b': 0.5,  # Months
        r'\b\d{1,2}(st|nd|rd|th)\b': 0.5,  # Ordinal numbers
        r'\$\d+(\.\d{2})?\b': 0.5,  # Money amounts
    }
    
    for pattern, points in fact_patterns.items():
        if re.search(pattern, text):
            quality_score += points
    
    # 3. Named Entity Usage (max 2 points)
    words = text.split()
    capitalized_words = [w for w in words if w and w[0].isupper()]
    unique_capitalized = len(set(capitalized_words))
    
    if unique_capitalized >= 10:
        quality_score += 2
    elif unique_capitalized >= 5:
        quality_score += 1
    
    return min(quality_score, 6)  # Cap at 6 points

def predict_news(news_text, url=""):
    """Predict if news is real or fake using enhanced precision"""
    try:
        if not model or not vectorizer:
            raise RuntimeError("Model or vectorizer not loaded")
        
        # 1. Get base model prediction
        processed_text = stemming(news_text)
        if not processed_text:
            raise ValueError("Text preprocessing failed")
        
        text_vector = vectorizer.transform([processed_text])
        model_prediction = model.predict(text_vector)[0]
        model_confidence = 1  # Default confidence
        
        # 2. Calculate credibility and quality scores
        credibility_score = calculate_credibility_score(news_text, url)
        quality_score = assess_content_quality(news_text)
        
        # 3. Advanced Decision Logic
        total_score = credibility_score + quality_score
        
        # Definite Real News conditions
        if credibility_score >= 7:
            return "Real News"
        
        # Definite Fake News conditions
        if credibility_score <= 2 and quality_score <= 1:
            return "Fake News"
        
        # Model-based decision with score adjustment
        if model_prediction == 0:  # Model predicts Real
            if total_score >= 8:
                return "Real News"
            elif total_score >= 6:
                return "Real News"
            else:
                return "Fake News"  # Low quality despite model prediction
        else:  # Model predicts Fake
            if total_score >= 12:  # Very high quality can override
                return "Real News"
            else:
                return "Fake News"
    
    except Exception as e:
        raise Exception(f"Error in prediction: {str(e)}")

# Initialize model and vectorizer
try:
    with open('models/fake_news_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('models/vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)
    print("Model and vectorizer loaded successfully")
except Exception as e:
    print(f"Error loading model or vectorizer: {e}")
    model = None
    vectorizer = None

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        url = request.form.get('url')
        if not url:
            return render_template('index1.html', error="Please enter a URL")

        # Extract text from URL with improved extraction
        news_text = extract_text_from_url(url)
        
        # Make prediction with detailed analysis
        result = predict_news(news_text, url)
        
        # Calculate confidence scores for display
        credibility_score = calculate_credibility_score(news_text, url)
        quality_score = assess_content_quality(news_text)
        
        # Prepare analysis summary
        analysis = {
            'credibility_score': f"{credibility_score:.1f}/10",
            'quality_score': f"{quality_score:.1f}/6",
            'prediction': result,
            'confidence': 'High' if credibility_score + quality_score >= 10 else 'Medium' if credibility_score + quality_score >= 7 else 'Low'
        }
        
        return render_template('index1.html', 
                             url=url, 
                             result=result, 
                             analysis=analysis,
                             text_preview=news_text[:500] + "...")
    
    except Exception as e:
        return render_template('index1.html', url=url, error=str(e))

if __name__ == '__main__':
    print("DEBUG: Starting Flask application")
    app.run(debug=True, port=5000)