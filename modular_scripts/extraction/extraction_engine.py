import os
import json
import fitz  # PyMuPDF
import google.generativeai as genai
from dotenv import load_dotenv

# Load API Key (Streamlit Cloud secrets first, env var fallback for local)
load_dotenv()

api_key = None
try:
    import streamlit as st
    api_key = st.secrets.get("GEMINI_API_KEY")
except Exception:
    api_key = None

if not api_key:
    api_key = os.getenv("GEMINI_API_KEY")

if api_key:
    genai.configure(api_key=api_key)

def extract_text_from_pdf(pdf_path):
    """Extracts raw text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def analyze_with_gemini(raw_text):
    """Uses Gemini to convert raw text into structured JSON."""
    # Note: As of my last update, 'gemini-2.0-flash' is not a standard public model name.
    # The latest flash model is 'gemini-1.5-flash'. Using the name you provided,
    # but this may need to be adjusted if you encounter model not found errors.
    model = genai.GenerativeModel('gemini-1.5-flash') 
    
    prompt = f"""
    You are an expert Operations & Finance Analyst. 
    Extract the following data from the text below and return it strictly as a JSON object.
    
    Required Fields:
    - vendor_name (string)
    - total_amount (float)
    - currency (string)
    - date (YYYY-MM-DD)
    - category (e.g., Software, Rent, Utilities, Labor)
    - line_items (list of objects with 'description' and 'price')
    - cost_saving_insight (A brief note on if this looks like a duplicate or high-cost item)

    Text to analyze:
    {raw_text}
    """
    
    try:
        response = model.generate_content(prompt)
        # Cleaning the response to ensure only JSON is parsed
        json_text = response.text.replace('```json', '').replace('```', '').strip()
        return json.loads(json_text)
    except Exception as e:
        print(f"An error occurred during Gemini API call: {e}")
        # It's useful to see the raw response text for debugging
        print(f"Raw response text: {response.text if 'response' in locals() else 'No response'}")
        return {"error": "Failed to parse JSON from Gemini response."}


# Example Usage
if __name__ == "__main__":
    # Ensure you have a 'sample_invoice.pdf' in your data directory to test!
    sample_path = os.path.join("data", "sample_invoice.pdf")
    
    # Check if the sample invoice exists. If not, the old script created one,
    # so we'll just inform the user.
    if os.path.exists(sample_path):
        print(f"Analyzing '{sample_path}'...")
        if api_key:
            raw = extract_text_from_pdf(sample_path)
            if raw.strip():
                structured_data = analyze_with_gemini(raw)
                print("\nExtracted Data:")
                print(json.dumps(structured_data, indent=4))
            else:
                print("Could not extract text from the PDF. It might be an image-based PDF.")
                print("Consider using an OCR-based approach for such documents.")
        else:
            print("\nSkipping analysis because GEMINI_API_KEY is not set in Streamlit secrets or environment")
            print("Set Streamlit Cloud secret GEMINI_API_KEY or create a local .env with GEMINI_API_KEY='your_key_here'")
            
    else:
        print(f"Please add a PDF invoice to '{sample_path}' to test the extraction.")
        print("The dummy PDF created previously is compatible.")
