import numpy as np
import pandas as pd
import pickle
import re
import logging
import gc
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import shuffle
from scipy import sparse
import os
from concurrent.futures import ThreadPoolExecutor
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize NLTK and download required data
def setup_nltk():
    """Download and set up NLTK data"""
    try:
        logger.info("Setting up NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        
        # Verify stopwords are accessible
        stop_words = set(stopwords.words('english'))
        if not stop_words:
            raise RuntimeError("Failed to load stopwords")
        
        logger.info("NLTK setup completed successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to set up NLTK: {str(e)}")
        return False

# Initialize NLTK at startup
if not setup_nltk():
    raise RuntimeError("Failed to initialize NLTK. Cannot proceed with training.")

# Cache stopwords
STOP_WORDS = set(stopwords.words('english'))

def create_models_directory():
    """Create models directory if it doesn't exist"""
    if not os.path.exists('models'):
        os.makedirs('models')
        logger.info("Created models directory")

def stemming(content):
    """Process and stem the text content."""
    if pd.isna(content):
        return ""
    
    try:
        # Initialize stemmer
        port_stem = PorterStemmer()
        
        # Convert content to string if it isn't already
        content = str(content)
        
        # Remove non-alphabetic characters
        stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
        
        # Convert to lowercase and split
        stemmed_content = stemmed_content.lower().split()
        
        # Apply stemming and remove stopwords using cached STOP_WORDS
        stemmed_content = [port_stem.stem(word) for word in stemmed_content 
                          if word not in STOP_WORDS]
        
        return ' '.join(stemmed_content)
    except Exception as e:
        logger.error(f"Error in stemming function: {str(e)}")
        return ""

def apply_stemming_batch(texts, batch_size=1000):
    """Process and stem text content in batches with parallel processing"""
    processed_texts = []
    total_batches = math.ceil(len(texts) / batch_size)
    
    def process_batch(batch):
        return [stemming(text) for text in batch]
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_processed = list(executor.map(stemming, batch))
            processed_texts.extend(batch_processed)
            batch_num = (i // batch_size) + 1
            logger.info(f"Processed batch {batch_num}/{total_batches} ({(i+len(batch))/len(texts)*100:.1f}% complete)")
            
            # Force garbage collection every 5 batches
            if batch_num % 5 == 0:
                gc.collect()
                logger.info("Performed garbage collection")
    
    return processed_texts

def load_and_preprocess_data():
    """Load and preprocess news dataset"""
    logger.info("Starting to load dataset...")
    
    chunk_size = 10000  # Process files in chunks
    
    try:
        logger.info("Loading train dataset...")
        # Specify datatypes explicitly during loading
        dtype_dict = {
            'title': str,
            'author': str,
            'text': str,
            'label': int  # Ensure label is loaded as integer
        }
        
        chunks = []
        chunk_count = 0
        for chunk in pd.read_csv('train.csv', chunksize=chunk_size, dtype=dtype_dict):
            chunk_count += 1
            logger.info(f"Processing chunk {chunk_count} of train.csv with shape: {chunk.shape}")
            chunks.append(chunk)
        
        if not chunks:
            raise ValueError("No data was loaded from train.csv")
            
        df = pd.concat(chunks, ignore_index=True)
        logger.info(f"Combined dataset shape: {df.shape}")
        
        # Free up memory
        del chunks
        gc.collect()
        
        # Handle missing values - convert to empty string only for string columns
        for col in df.select_dtypes(include=['object']):
            df[col] = df[col].fillna('')
        
        # Log column information
        logger.info(f"Available columns: {df.columns.tolist()}")
        
        # Create content field based on available columns
        if 'title' in df.columns and 'author' in df.columns:
            logger.info("Creating content from author and title")
            df['content'] = df['author'].astype(str) + ': ' + df['title'].astype(str)
        elif 'title' in df.columns:
            logger.info("Using title as content")
            df['content'] = df['title'].astype(str)
        else:
            raise ValueError("Could not find required text columns in the dataset")
        
        # Ensure label column exists and is numeric
        if 'label' not in df.columns:
            raise ValueError("Label column not found in dataset")
        df['label'] = pd.to_numeric(df['label'], errors='raise')
        
        # Keep only needed columns
        columns_to_keep = ['content', 'label']
        df = df[columns_to_keep]
        gc.collect()
        
        # Shuffle the dataset
        df = shuffle(df, random_state=42)
        
        logger.info(f"Final dataset shape: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise

def train_model():
    """Train the fake news detection model using SVM"""
    try:
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        news_dataset = load_and_preprocess_data()
        logger.info(f"Dataset shape: {news_dataset.shape}")
        
        # Use smaller chunks for vectorization to manage memory
        batch_size = 5000
        total_batches = (len(news_dataset) + batch_size - 1) // batch_size
        
        logger.info("Processing text data in batches...")
        for i in range(0, len(news_dataset), batch_size):
            batch = news_dataset.iloc[i:i+batch_size]
            batch_num = (i // batch_size) + 1
            logger.info(f"Processing batch {batch_num}/{total_batches}")
            
            batch['content'] = apply_stemming_batch(batch['content'].values, batch_size=1000)
            
            # Update the original dataset
            news_dataset.iloc[i:i+batch_size, news_dataset.columns.get_loc('content')] = batch['content']
            
            # Force garbage collection
            del batch
            gc.collect()
        
        # Separate features and target
        X = news_dataset['content']
        Y = news_dataset['label']
        
        # Free up memory
        del news_dataset
        gc.collect()
        
        logger.info("Creating TF-IDF vectors...")
        vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1,2))
        
        # Process TF-IDF in batches
        X_transformed_batches = []
        for i in range(0, len(X), batch_size):
            batch = X.iloc[i:i+batch_size]
            if i == 0:
                # First batch: fit and transform
                X_batch = vectorizer.fit_transform(batch)
            else:
                # Subsequent batches: transform only
                X_batch = vectorizer.transform(batch)
            X_transformed_batches.append(X_batch)
            logger.info(f"Vectorized batch {(i // batch_size) + 1}/{total_batches}")
        
        # Combine all transformed batches
        X = sparse.vstack(X_transformed_batches)
        logger.info(f"TF-IDF matrix shape: {X.shape}")
        
        # Free memory
        del X_transformed_batches
        gc.collect()
        
        # Split the dataset
        logger.info("Splitting dataset into train and test sets...")
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, stratify=Y, random_state=42
        )
        
        # Free up memory
        del X
        gc.collect()
        
        # Train the SVM model
        logger.info("Training the SVM model...")
        model = LinearSVC(random_state=42, max_iter=10000)
        model.fit(X_train, Y_train)
        
        # Evaluate the model
        # Training data performance
        X_train_prediction = model.predict(X_train)
        training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
        logger.info(f'Training Data Accuracy: {training_data_accuracy:.2f}')
        
        # Free up memory
        del X_train, X_train_prediction
        gc.collect()
        
        # Test data performance
        Y_pred = model.predict(X_test)
        test_accuracy = accuracy_score(Y_test, Y_pred)
        logger.info(f'Test Data Accuracy: {test_accuracy:.2f}')
        
        # Print detailed classification report
        logger.info("\nClassification Report:")
        logger.info(classification_report(Y_test, Y_pred))
        
        # Create models directory
        create_models_directory()
        
        # Save the model and vectorizer
        logger.info("Saving model and vectorizer...")
        with open('models/fake_news_model.pkl', 'wb') as file:
            pickle.dump(model, file)
        
        with open('models/vectorizer.pkl', 'wb') as file:
            pickle.dump(vectorizer, file)
            
        logger.info("Model and vectorizer saved successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("Starting model training process...")
    success = train_model()
    if success:
        logger.info("Model training completed successfully!")
    else:
        logger.error("Model training failed!")