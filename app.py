import re
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import sqlite3
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
import os
import logging
from werkzeug.security import generate_password_hash, check_password_hash
import csv

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- User Auth & DB Setup ---
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'a-super-secret-key-that-is-long-and-secure')
jwt = JWTManager(app)

def get_db():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with app.app_context():
        db = get_db()
        with open('schema.sql', 'r') as f:
            db.cursor().executescript(f.read())
        db.commit()

# --- ML Model Persistence & Logic ---
CATEGORY_MODEL_PATH = 'category_model.joblib'
GL_MODEL_PATH = 'gl_model.joblib'
VECTORIZER_PATH = 'tfidf_vectorizer.joblib'

CATEGORY_TO_GL_MAP = {
    "Maintenance": "6100 - Repairs & Maintenance", "Junk Removal": "6150 - Waste Disposal",
    "Utilities": "6200 - Utilities Expense", "Supplies": "6300 - Office Supplies",
    "Services": "6400 - Professional Services", "Insurance": "6500 - Insurance Expense",
    "Taxes & Fees": "6600 - Taxes & Licenses", "Rent & Leasing": "6700 - Rent Expense",
    "Travel & Transport": "6800 - Travel Expense", "Other Expenses": "6900 - Miscellaneous Expenses",
    "Uncategorized": "9999 - Uncategorized"
}

def train_and_save_models():
    logger.info("Training and saving models...")
    # This function would contain your full training logic with feedback data
    # For now, it uses a predefined set of examples.
    initial_category_data = [
        ("yard maintenance", "Maintenance"), ("lawn care", "Maintenance"), ("junk removal", "Junk Removal"),
        ("electric bill", "Utilities"), ("office supplies", "Supplies"), ("consultation", "Services"),
        ("insurance premium", "Insurance"), ("tax payment", "Taxes & Fees"), ("rent", "Rent & Leasing"),
        ("airfare", "Travel & Transport")
    ]
    df_initial = pd.DataFrame(initial_category_data, columns=['description', 'correct_category'])
    df_initial['correct_gl_code'] = df_initial['correct_category'].map(CATEGORY_TO_GL_MAP)
    
    vectorizer_obj = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
    
    # Category Model
    X_category = vectorizer_obj.fit_transform(df_initial['description'])
    clf_category = LogisticRegression(max_iter=1000).fit(X_category, df_initial['correct_category'])
    
    # GL Code Model
    gl_texts = df_initial['description'] + " " + df_initial['correct_category']
    X_gl = vectorizer_obj.transform(gl_texts)
    clf_gl = LogisticRegression(max_iter=1000).fit(X_gl, df_initial['correct_gl_code'])
    
    joblib.dump(vectorizer_obj, VECTORIZER_PATH)
    joblib.dump(clf_category, CATEGORY_MODEL_PATH)
    joblib.dump(clf_gl, GL_MODEL_PATH)
    logger.info("Models saved.")
    return vectorizer_obj, clf_category, clf_gl

def load_or_train_models():
    if os.path.exists(VECTORIZER_PATH) and os.path.exists(CATEGORY_MODEL_PATH) and os.path.exists(GL_MODEL_PATH):
        logger.info("Loading existing models...")
        return joblib.load(VECTORIZER_PATH), joblib.load(CATEGORY_MODEL_PATH), joblib.load(GL_MODEL_PATH)
    return train_and_save_models()

vectorizer, category_model, gl_model = load_or_train_models()

def predict_category_and_gl_code(text):
    if not isinstance(text, str) or not text.strip():
        return 'Uncategorized', CATEGORY_TO_GL_MAP['Uncategorized']
    try:
        clean_text = text.lower().strip()
        X_vec = vectorizer.transform([clean_text])
        category = category_model.predict(X_vec)[0]
        gl_feature = f"{clean_text} {category.lower()}"
        X_gl_vec = vectorizer.transform([gl_feature])
        gl_code = gl_model.predict(X_gl_vec)[0]
        return category, gl_code
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return 'Uncategorized', CATEGORY_TO_GL_MAP['Uncategorized']

# --- Core Parsing and Anomaly Detection ---
def extract_date_from_text(text):
    patterns = [r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', r'([A-Za-z]{3,9}\s+\d{1,2},\s+\d{4})']
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return pd.to_datetime(match.group(1)).strftime('%Y-%m-%d')
            except (ValueError, TypeError): continue
    return datetime.today().strftime('%Y-%m-%d')

def extract_receipt_details_from_text(text, transaction_date):
    logger.info("Parsing OCR text for receipt details...")
    extracted_items = []
    amount_pattern = re.compile(r'(\$?(\d{1,3}(?:,\d{3})*\.\d{2}))$')
    
    for line in text.splitlines():
        line = line.strip()
        if not line or len(line) < 3 or re.search(r'subtotal|tax|total|payment|visa|mastercard', line, re.IGNORECASE):
            continue
        
        match = amount_pattern.search(line)
        if match:
            total = match.group(2).replace(',', '')
            description = line[:match.start()].strip()
            
            if description:
                category, gl_code = predict_category_and_gl_code(description)
                extracted_items.append({
                    'Transaction Date': transaction_date,
                    'Description': description,
                    'Quantity': '1',
                    'Unit Price': total,
                    'Total': total,
                    'category': category,
                    'GLCode': gl_code,
                    'original_text': line
                })
    
    if not extracted_items:
        logger.warning("No line items extracted. Creating a single entry for the document.")
        category, gl_code = predict_category_and_gl_code(text[:500]) # Use first 500 chars for context
        extracted_items.append({
            'Transaction Date': transaction_date,
            'Description': "Summary of Scanned Document",
            'Quantity': '1', 'Unit Price': '0.00', 'Total': '0.00',
            'category': category, 'GLCode': gl_code,
            'original_text': "See extracted text for details."
        })

    return pd.DataFrame(extracted_items)

def detect_anomalies(df):
    if df.empty: return df
    df['IsAnomaly'] = 'False'
    df['AnomalyReason'] = ''
    df['NumericTotal'] = pd.to_numeric(df['Total'], errors='coerce')
    df_numeric = df.dropna(subset=['NumericTotal']).copy()

    if not df_numeric.empty:
        for category in df_numeric['category'].unique():
            cat_df = df_numeric[df_numeric['category'] == category]
            if len(cat_df) >= 3:
                q1, q3 = cat_df['NumericTotal'].quantile(0.25), cat_df['NumericTotal'].quantile(0.75)
                iqr = q3 - q1
                upper_bound = q3 + 1.5 * iqr
                high_indices = df[(df['category'] == category) & (df['NumericTotal'] > upper_bound)].index
                df.loc[high_indices, 'IsAnomaly'] = 'True'
                df.loc[high_indices, 'AnomalyReason'] += 'High Amount; '
    
    dup_cols = ['Transaction Date', 'Description', 'Total']
    if all(c in df.columns for c in dup_cols):
        duplicates = df[df.duplicated(subset=dup_cols, keep=False)]
        if not duplicates.empty:
            df.loc[duplicates.index, 'IsAnomaly'] = 'True'
            df.loc[duplicates.index, 'AnomalyReason'] += 'Duplicate Entry; '
            
    df['AnomalyReason'] = df['AnomalyReason'].str.strip('; ')
    return df.drop(columns=['NumericTotal'])

# --- Flask Routes ---
@app.route('/parse', methods=['POST'])
def parse():
    logger.info("--- Starting /parse endpoint ---")
    try:
        if 'pdf' not in request.files:
            return jsonify({'error': 'Missing PDF or text file.'}), 400
        
        pdf_file = request.files['pdf']
        ocr_language = request.form.get('ocr_language', 'eng')
        
        pdf_bytes = pdf_file.read()
        extracted_text = ""
        
        logger.info("Processing with OCR...")
        images = convert_from_bytes(pdf_bytes, dpi=300)
        for img in images:
            extracted_text += pytesseract.image_to_string(img, lang=ocr_language, config='--psm 3') + "\n"

        if not extracted_text.strip():
            logger.warning("OCR resulted in empty text.")
            return jsonify({'error': 'No text could be extracted from the document.'}), 400

        date = extract_date_from_text(extracted_text)
        final_table = extract_receipt_details_from_text(extracted_text, date)

        if final_table.empty:
            logger.warning("Parsing logic failed to extract any table rows.")
            return jsonify({'error': 'Could not extract any meaningful table data from the document.'}), 400

        final_table = detect_anomalies(final_table)
        final_table = final_table.astype(str) # Ensure all data is string for JSON

        logger.info(f"Successfully processed and returning {len(final_table)} rows.")
        return jsonify({
            'headers': list(final_table.columns),
            'rows': final_table.to_dict(orient='records'),
            'extracted_text': extracted_text
        })
    except Exception as e:
        logger.exception("A critical error occurred in /parse")
        return jsonify({'error': f'An unexpected server error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
