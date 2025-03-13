import zipfile
import os

def extract_zip(zip_path, extract_to):
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Extracted {zip_path} to {extract_to}")
    else:
        print(f"Error: {zip_path} not found.")

# Example usage
extract_zip('data.zip', 'data/')
