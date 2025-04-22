import pandas as pd
import os

def analyze_dataset(file_path, nrows=5):
    try:
        print(f"\nAnalyzing {file_path}:")
        # Try different encodings
        encodings = ['utf-8', 'latin1', 'cp1252']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, nrows=nrows, encoding=encoding)
                print(f"Successfully read with encoding: {encoding}")
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"Error with encoding {encoding}: {str(e)}")
                continue
                
        if df is not None:
            print("\nColumns:", df.columns.tolist())
            print("\nData Types:")
            print(df.dtypes)
            print("\nSample data:")
            print(df.head())
            print("\nShape of first chunk:", df.shape)
            
            # Get total number of rows
            try:
                total_rows = sum(1 for _ in open(file_path, 'r', encoding=encoding))
                print(f"\nTotal number of rows (including header): {total_rows}")
            except Exception as e:
                print(f"Could not count total rows: {str(e)}")
        else:
            print("Could not read file with any encoding")
            
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")

print("Analyzing datasets structure...")

# List of files to analyze
files = ['train.csv', 'test.csv', 'archive/fake.csv', 'archive/true.csv']

for file_path in files:
    if os.path.exists(file_path):
        analyze_dataset(file_path)
    else:
        print(f"\nFile not found: {file_path}")