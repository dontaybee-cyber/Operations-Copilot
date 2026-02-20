
import os
import glob
import pandas as pd
from dotenv import load_dotenv

# It's good practice to load environment variables at the start
# in case any imported modules rely on them.
load_dotenv()

# Correctly import from the sibling 'extraction' module
from .extraction.extraction_engine import extract_text_from_pdf, analyze_with_gemini

# --- Constants ---
# Use os.path.join for cross-platform compatibility.
# Go up one level from 'modular_scripts' to the project root.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
LOG_FILE = os.path.join(DATA_DIR, 'master_ops_log.csv')

def get_processed_files(log_path: str) -> set:
    """Reads the master log and returns a set of already processed filenames."""
    if not os.path.exists(log_path):
        return set()
    try:
        df = pd.read_csv(log_path)
        if 'source_file' in df.columns:
            return set(df['source_file'].unique())
        return set()
    except pd.errors.EmptyDataError:
        return set() # File exists but is empty

def bulk_process_invoices():
    """
    Scans the data directory for new PDF invoices, processes them,
    and appends the results to a master CSV log.
    """
    print("Starting bulk processing engine...")
    
    # 1. Check for API Key first to avoid unnecessary processing
    api_key = None
    try:
        import streamlit as st
        api_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        api_key = None

    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("🔴 ERROR: GEMINI_API_KEY not found in Streamlit secrets or environment. Aborting.")
        return

    # 2. Duplicate Prevention: Get list of already processed files
    processed_files = get_processed_files(LOG_FILE)
    print(f"Found {len(processed_files)} previously processed files.")

    # 3. Directory Scanning: Find all PDFs in the data directory
    pdf_files = glob.glob(os.path.join(DATA_DIR, '*.pdf'))
    
    new_records = []
    
    # 4. Main Processing Loop
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        
        if filename in processed_files:
            continue # Skip already processed files
            
        print(f"\nProcessing new file: {filename}")
        
        # 5. Error Handling per file
        try:
            # 6. Integration: Use functions from extraction_engine
            raw_text = extract_text_from_pdf(pdf_path)
            
            if not raw_text or not raw_text.strip():
                print(f"⚠️ WARNING: Could not extract text from {filename}. It might be an image-only PDF.")
                continue

            structured_data = analyze_with_gemini(raw_text)
            
            # Check if Gemini returned an error
            if 'error' in structured_data:
                print(f"🔴 ERROR from Gemini for {filename}: {structured_data.get('error')}")
                continue

            # 7. Data Aggregation: Prepare the record for the CSV
            structured_data['source_file'] = filename # Add the source filename
            new_records.append(structured_data)
            print(f"✅ Successfully processed and staged for saving: {filename}")

        except Exception as e:
            print(f"🔴 CRITICAL ERROR processing {filename}: {e}")
            # This ensures one bad PDF doesn't stop the whole batch

    # 8. Save new records to CSV
    if new_records:
        print(f"\nFound {len(new_records)} new records to save.")
        new_df = pd.DataFrame(new_records)
        
        # Reorder columns to have source_file first, if it exists
        if 'source_file' in new_df.columns:
            cols = ['source_file'] + [col for col in new_df.columns if col != 'source_file']
            new_df = new_df[cols]
        
        # Append to CSV. Create header only if file is new.
        header = not os.path.exists(LOG_FILE)
        new_df.to_csv(LOG_FILE, mode='a', header=header, index=False)
        print(f"Successfully appended new data to {LOG_FILE}")
    else:
        print("\nNo new invoice files to process.")
        
    print("Bulk processing complete.")


if __name__ == '__main__':
    # To run this script directly from the root for testing:
    # python -m modular_scripts.bulk_processor
    bulk_process_invoices()
