import os
import json
import pandas as pd
import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv

# --- Configuration & Setup ---
# Load environment variables from the root .env file
load_dotenv()

# Define file paths relative to the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
LOG_FILE = os.path.join(DATA_DIR, 'master_ops_log.csv')
REPORT_FILE = os.path.join(DATA_DIR, 'anomalies_report.json')

def configure_genai():
    """Configures the Google Generative AI model, checking for API key.

    Cloud: uses Streamlit secrets (st.secrets["GEMINI_API_KEY"])
    Local: falls back to environment variable (os.getenv("GEMINI_API_KEY"))
    """
    api_key = None

    # Streamlit Cloud / deployed environments
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        api_key = None

    # Local dev fallback
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("🔴 ERROR: GEMINI_API_KEY not found in Streamlit secrets or environment. Cannot generate insights.")
        return False

    genai.configure(api_key=api_key)
    return True

def get_insights_from_gemini(anomalies_df: pd.DataFrame) -> str:
    """
    Sends a DataFrame of anomalies to Gemini and asks for a professional summary.
    """
    if not configure_genai():
        return "Could not generate AI summary because GEMINI_API_KEY is not configured."
        
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Convert DataFrame to a more readable string format for the prompt
    anomalies_str = anomalies_df.to_string(index=False)
    
    prompt = f"""
    You are an expert Chief Operating Officer (COO) reviewing a financial report.
    The following items have been flagged as potential anomalies from our invoice processing system.
    Please write a brief, professional summary explaining WHY these items are concerning and what the likely next steps should be (e.g., "investigate duplicate," "inquire about price change").

    Flagged Anomalies:
    {anomalies_str}
    """
    
    # FIX: Added try-except block for robust API calls
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"An error occurred while communicating with the AI model: {e}"


def detect_financial_anomalies(spike_threshold: float = 0.20):
    """
    Analyzes the master operations log for duplicate billings and price spikes.
    """
    print("Starting anomaly detection engine...")

    # 1. Data Ingestion
    if not os.path.exists(LOG_FILE):
        print(f"🔴 ERROR: Master log file not found at '{LOG_FILE}'. Please run the bulk processor first.")
        return

    try:
        df = pd.read_csv(LOG_FILE)
        # Ensure 'total_amount' is a numeric type for calculations
        df['total_amount'] = pd.to_numeric(df['total_amount'], errors='coerce')
        df.dropna(subset=['total_amount', 'vendor_name'], inplace=True)
    except pd.errors.EmptyDataError:
        print("✅ Master log is empty. No anomalies to report.")
        return
    except Exception as e:
        print(f"🔴 ERROR: Could not read or parse log file: {e}")
        return
        
    all_anomalies = []

    # 2. Duplicate Detection
    duplicates = df[df.duplicated(subset=['vendor_name', 'total_amount'], keep=False)].copy()
    if not duplicates.empty:
        duplicates['anomaly_type'] = 'Potential Duplicate Billing'
        all_anomalies.append(duplicates)
        print(f"🔍 Found {len(duplicates)} potential duplicate billings.")

    # 3. Price Spike Analysis
    spikes = []
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df.sort_values(by=['vendor_name', 'date'], inplace=True)

    for vendor, group in df.groupby('vendor_name'):
        if len(group) > 1:
            avg_cost = group['total_amount'].iloc[:-1].mean()
            latest_entry = group.iloc[-1]
            
            if latest_entry['total_amount'] > avg_cost * (1 + spike_threshold):
                spike_entry = latest_entry.to_dict()
                spike_entry['anomaly_type'] = f'Price Spike (>{spike_threshold:.0%})'
                spike_entry['details'] = f"Amount {latest_entry['total_amount']:.2f} vs. Avg {avg_cost:.2f}"
                spikes.append(spike_entry)
    
    if spikes:
        spike_df = pd.DataFrame(spikes)
        all_anomalies.append(spike_df)
        print(f"🔍 Found {len(spikes)} significant price spikes.")

    # 4. Consolidate and Output
    if not all_anomalies:
        print("✅ No financial anomalies detected.")
        return

    final_anomalies_df = pd.concat(all_anomalies, ignore_index=True)

    # 5. Natural Language Insights
    print("\n--- Generating COO Summary ---")
    summary = get_insights_from_gemini(final_anomalies_df)
    print(summary)

    # 6. Final Output
    print("\n--- Anomaly Details ---")
    print(final_anomalies_df.to_string())
    
    final_anomalies_df.to_json(REPORT_FILE, orient='records', indent=4)
    # FIX: Corrected the unterminated f-string literal on the next line
    print(f"\n✅ Detailed report saved to {REPORT_FILE}")

if __name__ == '__main__':
    detect_financial_anomalies()
