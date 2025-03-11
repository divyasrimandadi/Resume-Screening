import re
import pickle
import nltk
import os
import docx
import PyPDF2

from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
nltk.download('stopwords')

# Load trained model and vectorizer
model = pickle.load(open("clf.pkl", "rb"))
tfidf_vectorizer = pickle.load(open("tfidf.pkl", "rb"))

def clean_resume(text):
    """Clean resume text by removing URLs, special characters, and extra spaces."""
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_text_from_file(file_path):
    """Extract text from TXT, DOCX, or PDF files."""
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    
    elif file_extension == ".docx":
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    
    elif file_extension == ".pdf":
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    
    else:
        return None  # Unsupported file type

def analyze_resume(file_path):
    """Reads a resume file, extracts text, cleans it, and predicts category."""
    try:
        resume_text = extract_text_from_file(file_path)
        if resume_text is None:
            return "Unsupported file format. Please upload TXT, DOCX, or PDF."

        cleaned_text = clean_resume(resume_text)
        vectorized_text = tfidf_vectorizer.transform([cleaned_text])
        category = model.predict(vectorized_text)
        
        return category[0]  # Return the predicted category
    except Exception as e:
        return f"Error processing file: {str(e)}"
