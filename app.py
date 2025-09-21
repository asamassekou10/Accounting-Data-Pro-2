import re
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
import tabula
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import sqlite3
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from werkzeug.security import generate_password_hash, check_password_hash
import os

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# --- User Auth & DB Setup ---
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'a-super-secret-key-that-is-long-and-secure')
jwt = JWTManager(app)

def get_db():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_user_db():
    with get_db() as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )''')
        conn.commit()

init_user_db()

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    if not username or not password:
        return jsonify({'error': 'Username and password required.'}), 400
    password_hash = generate_password_hash(password)
    try:
        with get_db() as conn:
            conn.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)', (username, password_hash))
            conn.commit()
        return jsonify({'msg': 'User registered successfully.'}), 201
    except sqlite3.IntegrityError:
        return jsonify({'error': 'Username already exists.'}), 409

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    if not username or not password:
        return jsonify({'error': 'Username and password required.'}), 400
    with get_db() as conn:
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
    if user and check_password_hash(user['password_hash'], password):
        access_token = create_access_token(identity=username)
        return jsonify(access_token=access_token), 200
    return jsonify({'error': 'Invalid username or password.'}), 401

# --- Category Prediction Model ---
def load_category_model():
    data = [
        ("yard maintenance", "Maintenance"), ("lawn care", "Maintenance"), ("gardening", "Maintenance"),
        ("junk removal", "Junk Removal"), ("waste disposal", "Junk Removal"), ("trash pickup", "Junk Removal"),
        ("electric bill", "Utilities"), ("water bill", "Utilities"), ("internet", "Utilities"),
        ("office supplies", "Supplies"), ("materials", "Supplies"), ("ink", "Supplies"), ("paper", "Supplies"),
        ("consultation", "Services"), ("professional", "Services"), ("labor", "Services"),
        ("insurance premium", "Insurance"), ("policy", "Insurance"), ("tax payment", "Taxes & Fees"),
        ("license fee", "Taxes & Fees"), ("rent", "Rent & Leasing"), ("lease", "Rent & Leasing"),
        ("airfare", "Travel & Transport"), ("hotel", "Travel & Transport"), ("fuel", "Travel & Transport")
    ]
    texts, labels = zip(*data)
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)
    clf = LogisticRegression(max_iter=1000).fit(X, labels)
    return vectorizer, clf

vectorizer, category_model = load_category_model()

def predict_category(text):
    try:
        if not isinstance(text, str) or not text.strip():
            return 'Uncategorized'
        X = vectorizer.transform([text.lower()])
        return category_model.predict(X)[0]
    except Exception:
        return 'Uncategorized'

# --- Core Parsing Logic ---
def extract_date_from_text(text):
    patterns = [
        r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        r'([A-Za-z]{3,9}\s+\d{1,2},\s+\d{4})'
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                date_str = match.group(1)
                # Attempt to parse the date, handling various common formats
                return pd.to_datetime(date_str).strftime('%Y-%m-%d')
            except (ValueError, TypeError):
                continue
    return datetime.today().strftime('%Y-%m-%d')

def parse_ocr_text_to_df(text, date):
    rows = []
    money_pattern = re.compile(r'(\d{1,3}(?:,\d{3})*\.\d{2})$')
    for line in text.splitlines():
        line = line.strip()
        if not line or re.search(r'subtotal|tax|total|payment|visa|mastercard', line, re.IGNORECASE):
            continue
        match = money_pattern.search(line)
        if match:
            amount = match.group(1)
            description = line[:match.start()].strip()
            if description and len(description) > 2:
                rows.append({
                    'Date': date,
                    'Description': description,
                    'Category': predict_category(description),
                    'TotalAmount': amount
                })
    return pd.DataFrame(rows)

def clean_tabula_df(df, date):
    df.dropna(how='all', inplace=True)
    if df.empty:
        return pd.DataFrame()
    
    desc_col, amt_col = None, None
    for col in df.columns:
        col_lower = str(col).lower()
        if 'description' in col_lower or 'item' in col_lower: desc_col = col
        if 'total' in col_lower or 'amount' in col_lower: amt_col = col
    
    if not desc_col:
        for col in df.columns:
            if df[col].dtype == 'object': desc_col = col; break
    if not amt_col:
        for col in reversed(df.columns):
            if df[col].astype(str).str.contains(r'\d+\.\d{2}', na=False).any(): amt_col = col; break

    if not desc_col or not amt_col: return pd.DataFrame()

    result_df = pd.DataFrame()
    result_df['Description'] = df[desc_col].astype(str).str.strip()
    result_df['TotalAmount'] = df[amt_col].astype(str).str.replace(r'[$,]', '', regex=True).str.strip()
    result_df = result_df[~result_df['Description'].str.contains('subtotal|tax|total', case=False, na=False)]
    result_df.dropna(subset=['Description', 'TotalAmount'], inplace=True)
    result_df['Date'] = date
    result_df['Category'] = result_df['Description'].apply(predict_category)
    return result_df[['Date', 'Description', 'Category', 'TotalAmount']]


@app.route('/parse', methods=['POST'])
def parse():
    try:
        excel_file = request.files.get('excel')
        pdf_file = request.files.get('pdf')

        if not excel_file or not pdf_file:
            return jsonify({'error': 'Missing Excel/CSV or PDF file.'}), 400

        pdf_bytes = pdf_file.read()
        extracted_text = ""
        final_table = pd.DataFrame()

        # 1. Try Tabula first for structured PDFs
        try:
            tables = tabula.read_pdf(io.BytesIO(pdf_bytes), pages='all', multiple_tables=True, lattice=True)
            if tables:
                # Use OCR on the first page just to get the best date
                images = convert_from_bytes(pdf_bytes, last_page=1, dpi=200)
                date_text = pytesseract.image_to_string(images[0]) if images else ""
                date = extract_date_from_text(date_text)
                
                processed_tables = [clean_tabula_df(t, date) for t in tables if not t.empty]
                if processed_tables:
                    final_table = pd.concat(processed_tables, ignore_index=True)
        except Exception as e:
            print(f"Tabula failed, falling back to OCR. Reason: {e}")

        # 2. If Tabula fails or finds nothing, use OCR as a fallback
        if final_table.empty:
            images = convert_from_bytes(pdf_bytes, dpi=300)
            for img in images:
                extracted_text += pytesseract.image_to_string(img) + "\n"
            
            date = extract_date_from_text(extracted_text)
            final_table = parse_ocr_text_to_df(extracted_text, date)

        if final_table.empty:
            return jsonify({'error': 'Could not extract any meaningful table data from the document.'}), 400

        return jsonify({
            'headers': list(final_table.columns),
            'rows': final_table.to_dict(orient='records'),
            'extracted_text': extracted_text
        })

    except Exception as e:
        print(f"A critical error occurred in /parse: {e}")
        return jsonify({'error': f'An unexpected server error occurred: {e}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
