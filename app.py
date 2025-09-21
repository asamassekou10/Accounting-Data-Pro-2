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
import os
import logging
from werkzeug.security import generate_password_hash, check_password_hash
import csv # Added import for csv module

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# --- Configure Logging (Highly Recommended) ---
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'app.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info("Application starting up.")

# --- User Auth Setup ---
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'super-secret-key')
jwt = JWTManager(app)

# --- User DB Helper ---
def get_db():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_user_db():
    conn = get_db()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL
    )''')
    # Add training_data table for ML feedback, now including correct_gl_code
    c.execute('''CREATE TABLE IF NOT EXISTS training_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        description TEXT NOT NULL,
        correct_category TEXT NOT NULL,
        correct_gl_code TEXT, -- New column for GL code feedback
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    # New: Add parsed_transactions table for storing parsed and edited data, now including gl_code
    c.execute('''CREATE TABLE IF NOT EXISTS parsed_transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        transaction_date TEXT,
        description TEXT,
        quantity TEXT,
        unit_price TEXT,
        total TEXT,
        category TEXT,
        gl_code TEXT, -- New column for predicted/edited GL code
        original_text TEXT,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )''')
    conn.commit()
    conn.close()
    logger.info("User database initialized.")

init_user_db()

# --- ML Model Persistence ---
CATEGORY_MODEL_PATH = 'category_model.joblib'
GL_MODEL_PATH = 'gl_model.joblib' # New path for GL code model
VECTORIZER_PATH = 'tfidf_vectorizer.joblib'

# Define a mapping from categories to default GL codes for initial training and fallback
# In a real application, this would be customizable by the company
CATEGORY_TO_GL_MAP = {
    "Maintenance": "6100 - Repairs & Maintenance",
    "Junk Removal": "6150 - Waste Disposal",
    "Utilities": "6200 - Utilities Expense",
    "Supplies": "6300 - Office Supplies",
    "Services": "6400 - Professional Services",
    "Insurance": "6500 - Insurance Expense",
    "Taxes & Fees": "6600 - Taxes & Licenses",
    "Rent & Leasing": "6700 - Rent Expense",
    "Travel & Transport": "6800 - Travel Expense",
    "Other Expenses": "6900 - Miscellaneous Expenses",
    "Uncategorized": "9999 - Uncategorized" # Default for uncategorized
}

def train_and_save_models():
    logger.info("Training and saving models (Category and GL Code)...")
    conn = get_db()
    # Fetch feedback data for both category and GL code
    df_feedback = pd.read_sql_query("SELECT description, correct_category, correct_gl_code FROM training_data", conn)
    conn.close()

    # Initial training data for categories
    initial_category_data = [
        ("yard maintenance", "Maintenance"), ("lawn care", "Maintenance"), ("gardening", "Maintenance"),
        ("landscaping", "Maintenance"), ("cleaning", "Maintenance"), ("janitorial", "Maintenance"),
        ("repair", "Maintenance"), ("fix", "Maintenance"), ("plumbing", "Maintenance"),
        ("electrical repair", "Maintenance"), ("hvac", "Maintenance"), ("service call", "Maintenance"),
        ("maintenance fee", "Maintenance"), ("junk removal", "Junk Removal"), ("waste disposal", "Junk Removal"),
        ("trash pickup", "Junk Removal"), ("garbage", "Junk Removal"), ("bulk furniture removal", "Junk Removal"),
        ("debris removal", "Junk Removal"), ("haul away", "Junk Removal"), ("dump fee", "Junk Removal"),
        ("waste management", "Junk Removal"), ("electric bill", "Utilities"), ("water bill", "Utilities"),
        ("gas bill", "Utilities"), ("internet", "Utilities"), ("phone service", "Utilities"),
        ("cable", "Utilities"), ("utility", "Utilities"), ("office supplies", "Supplies"),
        ("materials", "Supplies"), ("ink", "Supplies"), ("paper", "Supplies"),
        ("toner", "Supplies"), ("equipment", "Supplies"), ("consultation", "Services"),
        ("professional", "Services"), ("service fee", "Services"), ("labor", "Services"),
        ("installation", "Services"), ("commission", "Services"), ("contractor", "Services"),
        ("insurance premium", "Insurance"), ("policy", "Insurance"), ("coverage", "Insurance"),
        ("liability", "Insurance"), ("tax payment", "Taxes & Fees"), ("license fee", "Taxes & Fees"),
        ("permit", "Taxes & Fees"), ("registration", "Taxes & Fees"), ("filing fee", "Taxes & Fees"),
        ("rent", "Rent & Leasing"), ("lease", "Rent & Leasing"), ("space rental", "Rent & Leasing"),
        ("office rent", "Rent & Leasing"), ("equipment rental", "Rent & Leasing"), ("airfare", "Travel & Transport"),
        ("hotel", "Travel & Transport"), ("mileage", "Travel & Transport"), ("transportation", "Travel & Transport"),
        ("parking", "Travel & Transport"), ("fuel", "Travel & Transport"), ("gas", "Travel & Transport")
    ]
    df_initial_category = pd.DataFrame(initial_category_data, columns=['description', 'correct_category'])
    
    # Concatenate initial category data with feedback, remove duplicates
    df_combined_category = pd.concat([df_initial_category, df_feedback[['description', 'correct_category']]]).drop_duplicates(subset=['description'], keep='last')

    # Prepare data for category model
    if df_combined_category.empty:
        logger.warning("No category training data available. Using a minimal default category model.")
        category_texts = ["default", "other"]
        category_labels = ["Other Expenses", "Other Expenses"]
    else:
        category_texts = df_combined_category['description'].tolist()
        category_labels = df_combined_category['correct_category'].tolist()

    # Train and save Category Model
    vectorizer_obj = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), max_features=1000, min_df=1, stop_words='english')
    X_category = vectorizer_obj.fit_transform(category_texts)
    clf_category_obj = LogisticRegression(C=10.0, class_weight='balanced', max_iter=1000, solver='liblinear', multi_class='ovr').fit(X_category, category_labels)
    joblib.dump(clf_category_obj, CATEGORY_MODEL_PATH)
    joblib.dump(vectorizer_obj, VECTORIZER_PATH) # Vectorizer is shared

    # Prepare data for GL Code model
    # For GL code prediction, we can use description and predicted category as features
    # For initial training, we'll use the initial category data and map it to default GL codes
    # For feedback, we'll use the provided correct_gl_code
    
    # Create initial GL data from initial category data
    initial_gl_data = []
    for desc, cat in initial_category_data:
        initial_gl_data.append({'description': desc, 'category': cat, 'correct_gl_code': CATEGORY_TO_GL_MAP.get(cat, "9999 - Uncategorized")})
    df_initial_gl = pd.DataFrame(initial_gl_data)

    # Filter feedback data to only include rows with a non-null correct_gl_code
    df_feedback_gl = df_feedback.dropna(subset=['correct_gl_code']).copy()
    # Ensure 'category' column exists in feedback_gl for combining features
    if 'category' not in df_feedback_gl.columns:
        df_feedback_gl['category'] = df_feedback_gl['description'].apply(lambda x: clf_category_obj.predict(vectorizer_obj.transform([x]))[0]) # Predict category if not present

    # Combine initial GL data with feedback, prioritizing feedback
    df_combined_gl = pd.concat([df_initial_gl, df_feedback_gl[['description', 'category', 'correct_gl_code']]]).drop_duplicates(subset=['description', 'category'], keep='last')

    if df_combined_gl.empty:
        logger.warning("No GL code training data available. Using a minimal default GL code model.")
        gl_texts_features = ["default category", "other category"]
        gl_labels = ["9999 - Uncategorized", "9999 - Uncategorized"]
    else:
        # Combine description and category for GL code prediction features
        gl_texts_features = (df_combined_gl['description'] + " " + df_combined_gl['category']).tolist()
        gl_labels = df_combined_gl['correct_gl_code'].tolist()

    # Train and save GL Code Model
    X_gl = vectorizer_obj.transform(gl_texts_features) # Use the same vectorizer
    clf_gl_obj = LogisticRegression(C=10.0, class_weight='balanced', max_iter=1000, solver='liblinear', multi_class='ovr').fit(X_gl, gl_labels)
    joblib.dump(clf_gl_obj, GL_MODEL_PATH)
    
    logger.info("Models and vectorizer saved.")
    return vectorizer_obj, clf_category_obj, clf_gl_obj

def load_or_train_models():
    # Check if all models exist
    if os.path.exists(CATEGORY_MODEL_PATH) and os.path.exists(GL_MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        logger.info("Loading existing models and vectorizer...")
        return joblib.load(VECTORIZER_PATH), joblib.load(CATEGORY_MODEL_PATH), joblib.load(GL_MODEL_PATH)
    else:
        return train_and_save_models()

vectorizer, category_model, gl_model = load_or_train_models()

# --- Retrain Model Endpoint ---
@app.route('/retrain_model', methods=['POST'])
@jwt_required()
def retrain_model_endpoint():
    global vectorizer, category_model, gl_model
    try:
        vectorizer, category_model, gl_model = train_and_save_models()
        return jsonify({'msg': 'AI models retrained successfully!'}), 200
    except Exception as e:
        logger.exception("Failed to retrain models.")
        return jsonify({'error': f'Failed to retrain models: {str(e)}'}), 500

# --- Submit Feedback Endpoint ---
@app.route('/submit_feedback', methods=['POST'])
@jwt_required()
def submit_feedback():
    data = request.get_json()
    description = data.get('description')
    correct_category = data.get('correct_category')
    correct_gl_code = data.get('correct_gl_code') # New: for GL code feedback

    if not description or not correct_category:
        logger.warning("Feedback submission failed: Missing description or category.")
        return jsonify({'error': 'Description and correct_category are required.'}), 400

    conn = get_db()
    c = conn.cursor()
    try:
        # Insert or update feedback, including GL code
        c.execute('INSERT INTO training_data (description, correct_category, correct_gl_code) VALUES (?, ?, ?)',
                    (description, correct_category, correct_gl_code))
        conn.commit()
        logger.info(f"Feedback submitted: '{description}' -> Category: '{correct_category}', GL Code: '{correct_gl_code}'")
        return jsonify({'msg': 'Feedback submitted successfully!'}), 200
    except Exception as e:
        logger.exception("Failed to submit feedback.")
        return jsonify({'error': f'Failed to submit feedback: {str(e)}'}), 500
    finally:
        conn.close()

# --- Register Endpoint ---
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    if not username or not password:
        logger.warning("Registration failed: Missing username or password.")
        return jsonify({'error': 'Username and password required.'}), 400
    password_hash = generate_password_hash(password)
    try:
        conn = get_db()
        c = conn.cursor()
        c.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)', (username, password_hash))
        conn.commit()
        conn.close()
        logger.info(f"User '{username}' registered successfully.")
        return jsonify({'msg': 'User registered successfully.'}), 201
    except sqlite3.IntegrityError:
        logger.warning(f"Registration failed: Username '{username}' already exists.")
        return jsonify({'error': 'Username already exists.'}), 409
    except Exception as e:
        logger.exception("An unexpected error occurred during registration.")
        return jsonify({'error': f'An internal error occurred: {str(e)}'}), 500

# --- Login Endpoint ---
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    if not username or not password:
        logger.warning("Login failed: Missing username or password.")
        return jsonify({'error': 'Username and password required.'}), 400
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username = ?', (username,))
    user = c.fetchone()
    conn.close()
    if user and check_password_hash(user['password_hash'], password):
        access_token = create_access_token(identity=username)
        logger.info(f"User '{username}' logged in successfully.")
        return jsonify({'access_token': access_token}), 200
    else:
        logger.warning(f"Login failed: Invalid credentials for username '{username}'.")
        return jsonify({'error': 'Invalid username or password.'}), 401

# --- Category and GL Code Prediction Logic ---
def predict_category_and_gl_code(text):
    try:
        clean_text = text.lower().strip()
        if not clean_text:
            return 'Uncategorized', CATEGORY_TO_GL_MAP.get('Uncategorized')

        # Predict Category
        X_category = vectorizer.transform([clean_text])
        category_prediction = category_model.predict(X_category)[0]
        category_probs = category_model.predict_proba(X_category)[0]
        max_category_prob = max(category_probs)

        # Apply confidence threshold for category
        if max_category_prob < 0.4:
            if any(word in clean_text for word in ['service', 'labor', 'work', 'job']):
                category_prediction = 'Services'
            elif any(word in clean_text for word in ['buy', 'purchase', 'item', 'product']):
                category_prediction = 'Supplies'
            else:
                category_prediction = 'Other Expenses'
        
        # Predict GL Code based on description and predicted category
        gl_feature_text = clean_text + " " + category_prediction.lower() # Combine description and category
        X_gl = vectorizer.transform([gl_feature_text])
        gl_prediction = gl_model.predict(X_gl)[0]
        gl_probs = gl_model.predict_proba(X_gl)[0]
        max_gl_prob = max(gl_probs)

        # Apply confidence threshold for GL code, or fallback to default based on category
        if max_gl_prob < 0.4:
            gl_prediction = CATEGORY_TO_GL_MAP.get(category_prediction, "9999 - Uncategorized")

        return category_prediction, gl_prediction
    except Exception as e:
        logger.error(f"Prediction error (category/GL code): {str(e)}", exc_info=True)
        return 'Services', CATEGORY_TO_GL_MAP.get('Services') # Default in case of prediction error

# --- Date Extraction ---
def extract_date(text):
    patterns = [
        r'Date[:\s]*([A-Za-z]{3,9}\s+\d{1,2},\s+\d{4})',
        r'(?:Invoice|Transaction|Receipt|Order|Issued|Payment)\s*(?:Date|Time|On)?:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        r'(?:Invoice|Transaction|Receipt|Order|Issued|Payment)\s*(?:Date|Time|On)?:?\s*(\d{2,4}[/-]\d{1,2}[/-]\d{1,2})',
        r'(?:Invoice|Transaction|Receipt|Order|Issued|Payment)\s*(?:Date|Time|On)?:?\s*(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{2,4})',
        r'(?:Invoice|Transaction|Receipt|Order|Issued|Payment)\s*(?:Date|Time|On)?:?\s*((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{2,4})',
        r'(\d{1,2}[/-]\d{1,2}[/-]\d{4})',
        r'(\d{4}[/-]\d{1,2}[/-]\d{1,2})',
        r'(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{2,4})',
        r'((?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{2,4})',
        r'(\d{2}[/-]\d{2}[/-]\d{2})'
    ]
    formats = ['%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d', '%m-%d-%Y', '%d-%m-%Y', '%Y-%m-%d', '%d %B %Y', '%d %b %Y', '%B %d, %Y', '%b %d, %Y', '%B %d %Y', '%b %d %Y', '%m/%d/%y', '%d/%m/%y', '%y/%m/%d', '%m-%d-%y', '%d-%m-%y', '%y-%m-%y']
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            date_str = match.group(1).strip()
            logger.debug(f"Matched date string: {date_str}")
            for fmt in formats:
                try:
                    parsed_date = datetime.strptime(date_str, fmt)
                    if parsed_date.year < 100:
                        if parsed_date.year < 50:
                            parsed_date = parsed_date.replace(year=parsed_date.year + 2000)
                        else:
                            parsed_date = parsed_date.replace(year=parsed_date.year + 1900)
                    if parsed_date <= datetime.now():
                        return parsed_date.strftime('%Y-%m-%d')
                except ValueError:
                    continue
    logger.warning("No valid date found in text, using current date")
    return datetime.today().strftime('%Y-%m-%d')

# --- Robust Text Parsing for Receipts (New/Improved Logic) ---
def extract_receipt_details_from_text(text, transaction_date):
    logger.info("Attempting robust text parsing for receipt details.")
    
    extracted_items = []
    
    # Regex patterns for common receipt elements (used in fallback and total finding)
    amount_pattern_flexible = r'\$?\s*\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})' # More flexible for matching

    total_patterns = [
        r'(?:TOTAL|AMOUNT DUE|BALANCE|GRAND TOTAL|TOTAL DUE|TOTAL AMOUNT|SUM|ALL TOTAL)\s*[:=]?\s*(' + amount_pattern_flexible + ')',
        r'(' + amount_pattern_flexible + r')\s*(?:TOTAL|AMOUNT DUE|BALANCE|GRAND TOTAL|TOTAL DUE|TOTAL AMOUNT|SUM|ALL TOTAL)'
    ]

    # Find global total first (can be on a separate line)
    global_total = ''
    for pattern in total_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            global_total = match.group(1).replace('$', '').replace(',', '').strip()
            logger.debug(f"Global total found: {global_total}")
            break

    # --- NEW: Pre-process the text to ensure each item is on its own line ---
    # This regex looks for a pattern where a quoted field ends, and then immediately
    # another quoted field begins, possibly with some spaces/newlines in between,
    # and inserts a definitive newline character. This is crucial for the csv.reader.
    # It specifically targets the transition from one item's "Total" to the next item's "Description".
    # Example: "49.99"\n"Description\n","Quantity" -> "49.99"\n"Description","Quantity"
    # Example: "39.98"\n"Leather Wallet" -> "39.98"\n"Leather Wallet"
    # The key is to insert a newline *before* the start of a new item's description,
    # which is often a quoted string.
    processed_text_for_csv = re.sub(r'"\s*[\r\n]*"\s*(?=[A-Za-z0-9])', r'"\n"', text)
    logger.debug(f"Processed text for CSV after newline insertion: {processed_text_for_csv}")

    # Split text into individual lines based on the *actual* newlines now present
    lines = processed_text_for_csv.split('\n')
    logger.info(f"Split processed text into {len(lines)} lines.")

    # --- Attempt CSV-like parsing first with refined pre-processing ---
    # Find the header line to identify the start of the CSV-like data block
    # This pattern is made more robust to handle potential newlines within the header itself
    csv_header_pattern = re.compile(r'"Description\s*[\r\n]*",?"Quantity\s*[\r\n]*",?"Unit Price \(USD\)\s*[\r\n]*",?"Total \(USD\)\s*[\r\n]*"')
    
    csv_data_block_lines = []
    header_found = False
    for line in lines:
        if csv_header_pattern.search(line):
            header_found = True
        if header_found:
            csv_data_block_lines.append(line)
    
    if csv_data_block_lines:
        logger.info("CSV-like header detected. Attempting direct CSV parsing of the relevant block.")
        csv_text_block = "\n".join(csv_data_block_lines) # Re-join the relevant lines
        
        try:
            csv_data_io = io.StringIO(csv_text_block)
            reader = csv.reader(csv_data_io)
            
            # Read the header row
            header_row_raw = next(reader, None)
            if header_row_raw:
                # Clean up header cells: remove quotes and internal newlines
                potential_headers = [cell.strip().replace('"', '').replace('\n', ' ').strip() for cell in header_row_raw]
                logger.debug(f"CSV-like header row parsed: {potential_headers}")
            else:
                logger.warning("No header row found in CSV-like block after pre-processing. Skipping CSV parsing.")
                potential_headers = [] # Reset to empty to fall back to regex

            if potential_headers:
                for row_list in reader:
                    # Clean up data cells: remove quotes and internal newlines
                    cleaned_row = [cell.strip().replace('"', '').replace('\n', ' ').strip() for cell in row_list]
                    
                    # Skip empty rows or rows that are clearly just "Total:" lines
                    if not any(cleaned_row) or any(re.search(r'(?:total|subtotal|tax)', cell, re.IGNORECASE) for cell in cleaned_row):
                        continue

                    row_data = {
                        'Transaction Date': transaction_date,
                        'Description': '',
                        'Quantity': '',
                        'Unit Price': '',
                        'Total': '',
                        'original_text': ','.join(row_list) # Store original CSV line
                    }
                    
                    desc_val, qty_val, unit_val, total_val = '', '', '', ''

                    for idx, header_cell in enumerate(potential_headers):
                        if idx < len(cleaned_row):
                            if 'description' in header_cell.lower():
                                desc_val = cleaned_row[idx]
                            elif 'quantity' in header_cell.lower() or 'qty' in header_cell.lower():
                                qty_val = cleaned_row[idx]
                            elif 'unit price' in header_cell.lower():
                                unit_val = cleaned_row[idx]
                            elif 'total' in header_cell.lower() or 'amount' in header_cell.lower():
                                total_val = cleaned_row[idx]
                    
                    # Refine values if not found by header or to ensure all are captured
                    amounts_in_row = [re.sub(r'[^\d.]', '', cell) for cell in cleaned_row if re.search(amount_pattern_flexible, cell)]
                    
                    if amounts_in_row:
                        if not total_val:
                            total_val = amounts_in_row[-1]
                        if not unit_val and len(amounts_in_row) > 1:
                            unit_val = amounts_in_row[-2]
                    
                    if not desc_val:
                        # Try to find a description from remaining cells
                        for cell in cleaned_row:
                            if not re.search(amount_pattern_flexible, cell) and cell.strip() and not any(keyword in cell.lower() for keyword in ['tax', 'subtotal', 'total']):
                                desc_val = cell.strip()
                                break
                    
                    if not qty_val and desc_val and total_val:
                        # Attempt to extract quantity from description if it's a leading number
                        qty_match = re.match(r'^(\d+)\s+(.*)', desc_val)
                        if qty_match:
                            qty_val = qty_match.group(1)
                            desc_val = qty_match.group(2).strip()
                        else:
                            qty_val = '1' # Default to 1 if no quantity found

                    row_data['Description'] = desc_val
                    row_data['Quantity'] = qty_val
                    row_data['Unit Price'] = unit_val.replace('$', '').replace(',', '').strip()
                    row_data['Total'] = total_val.replace('$', '').replace(',', '').strip()

                    if row_data['Description'] or row_data['Total']:
                        category_pred, gl_code_pred = predict_category_and_gl_code(row_data['Description'])
                        row_data['category'] = category_pred
                        row_data['GLCode'] = gl_code_pred
                        extracted_items.append(row_data)
                        logger.debug(f"CSV-like item extracted: {row_data}")
            
            if extracted_items:
                logger.info(f"Successfully extracted {len(extracted_items)} items using CSV-like parsing.")
                df = pd.DataFrame(extracted_items)
                for col in ['Description', 'Quantity', 'Unit Price', 'Total', 'category', 'GLCode', 'Transaction Date']:
                    if col not in df.columns:
                        df[col] = ''
                return df

        except csv.Error as e:
            logger.warning(f"CSV parsing failed: {e}. Falling back to regex parsing.")
        except Exception as e:
            logger.warning(f"General error during CSV-like parsing: {e}. Falling back to regex parsing.")

    # --- END NEW CSV-like parsing ---


    # Original approach (fallback): Iterate through lines, looking for amounts to anchor items
    # This will be the fallback if CSV-like parsing fails or is not applicable
    logger.info("Falling back to line-by-line regex parsing.")
    # Re-split lines from the original text (not the pre-processed one for CSV)
    lines = [line.strip() for line in text.splitlines() if line.strip()] 
    
    i = 0
    while i < len(lines):
        line = lines[i]
        line_data = {
            'Transaction Date': transaction_date,
            'Description': '',
            'Quantity': '',
            'Unit Price': '',
            'Total': '',
            'original_text': line # Store original line for context
        }

        # Try to match specific line item patterns first
        # These patterns are now more flexible about quantity
        line_item_pattern_1 = re.compile(r'(\d+)\s+(.+?)\s+(' + amount_pattern_flexible + r')\s+(' + amount_pattern_flexible + r')') # Qty Desc UnitPrice Total
        line_item_pattern_2 = re.compile(r'(.+?)\s+(' + amount_pattern_flexible + r')\s+(' + amount_pattern_flexible + r')') # Desc UnitPrice Total (assumes Qty 1)
        line_item_pattern_3 = re.compile(r'(.+?)\s+(' + amount_pattern_flexible + r')') # Desc Total (assumes Qty 1, UnitPrice = Total)
        line_item_pattern_4 = re.compile(r'^\s*(\S+)\s+(.+?)\s+(' + amount_pattern_flexible + r')\s*$') # Item code/number Desc Total

        match1 = line_item_pattern_1.search(line)
        match2 = line_item_pattern_2.search(line)
        match3 = line_item_pattern_3.search(line)
        match4 = line_item_pattern_4.search(line)

        if match1:
            line_data['Quantity'] = match1.group(1).strip()
            line_data['Description'] = match1.group(2).strip()
            line_data['Unit Price'] = match1.group(3).replace('$', '').replace(',', '').strip()
            line_data['Total'] = match1.group(4).replace('$', '').replace(',', '').strip()
            logger.debug(f"Matched pattern 1: {line_data}")
            extracted_items.append(line_data)
            i += 1 # Move to next line
            continue
        elif match2:
            line_data['Description'] = match2.group(1).strip()
            line_data['Unit Price'] = match2.group(2).replace('$', '').replace(',', '').strip()
            line_data['Total'] = match2.group(3).replace('$', '').replace(',', '').strip()
            line_data['Quantity'] = '1' # Assume quantity 1 if not specified
            logger.debug(f"Matched pattern 2: {line_data}")
            extracted_items.append(line_data)
            i += 1
            continue
        elif match3:
            line_data['Description'] = match3.group(1).strip()
            line_data['Total'] = match3.group(2).replace('$', '').replace(',', '').strip()
            line_data['Quantity'] = '1' # Assume quantity 1 if not specified
            line_data['Unit Price'] = line_data['Total'] # Assume unit price is total if only one amount
            logger.debug(f"Matched pattern 3: {line_data}")
            extracted_items.append(line_data)
            i += 1
            continue
        elif match4:
            line_data['Description'] = f"{match4.group(1).strip()} {match4.group(2).strip()}"
            line_data['Total'] = match4.group(3).replace('$', '').replace(',', '').strip()
            line_data['Quantity'] = '1' # Assume quantity 1
            line_data['Unit Price'] = line_data['Total'] # Assume unit price is total
            logger.debug(f"Matched pattern 4: {line_data}")
            extracted_items.append(line_data)
            i += 1
            continue
        
        # Fallback for less structured lines: look for an amount to anchor the item
        amount_matches = list(re.finditer(amount_pattern_flexible, line))
        if amount_matches:
            # Take the last amount found as the total for this line
            last_amount_match = amount_matches[-1]
            line_data['Total'] = last_amount_match.group(0).replace('$', '').replace(',', '').strip()
            
            # The description is everything before the last amount on this line
            description_part = line[:last_amount_match.start()].strip()
            
            # Look at the previous line for description if current line's description is short or empty
            # and previous line doesn't contain an amount itself
            if (not description_part or len(description_part.split()) < 3) and i > 0:
                prev_line = lines[i-1]
                if not re.search(amount_pattern_flexible, prev_line) and not re.search(r'(?:tax|subtotal|total|amount|balance|due)', prev_line, re.IGNORECASE):
                    line_data['Description'] = f"{prev_line.strip()} {description_part}".strip()
                    line_data['original_text'] = f"{prev_line.strip()}\n{line.strip()}" # Capture both lines
                    logger.debug(f"Combined with previous line for description: {line_data['Description']}")
                else:
                    line_data['Description'] = description_part
            else:
                line_data['Description'] = description_part

            if not line_data['Description']: # If still no description, use the whole line (excluding the amount)
                line_data['Description'] = re.sub(amount_pattern_flexible, '', line).strip()
                if not line_data['Description']: # If still empty, use the full line
                    line_data['Description'] = line.strip()

            # Attempt to extract quantity from the description part
            qty_match_desc = re.match(r'^(\d+)\s+(.*)', line_data['Description'])
            if qty_match_desc:
                line_data['Quantity'] = qty_match_desc.group(1)
                line_data['Description'] = qty_match_desc.group(2).strip()
            else:
                line_data['Quantity'] = '1' # Default quantity if not found

            line_data['Unit Price'] = line_data['Total'] # Default unit price
            logger.debug(f"Matched general amount and inferred description: {line_data}")
            extracted_items.append(line_data)
            i += 1 # Move to next line
            continue
        
        logger.debug(f"Skipping line (no amount, no specific pattern match): '{line}'")
        i += 1 # Move to next line

    # If no line items were extracted but a global total was found, create a single entry
    if not extracted_items and global_total:
        logger.info("No line items extracted, but a global total was found. Creating a single entry.")
        description = "Various items" # Default description if no specific items found
        category_pred, gl_code_pred = predict_category_and_gl_code(description)
        extracted_items.append({
            'Transaction Date': transaction_date,
            'Description': description,
            'Quantity': '1',
            'Unit Price': global_total,
            'Total': global_total,
            'category': category_pred,
            'GLCode': gl_code_pred,
            'original_text': text # Store full text for context if only one item
        })
    elif not extracted_items:
        logger.warning("No meaningful data extracted from text.")

    df = pd.DataFrame(extracted_items)
    # Ensure standard columns are present, even if empty
    for col in ['Description', 'Quantity', 'Unit Price', 'Total', 'category', 'GLCode', 'Transaction Date', 'original_text']:
        if col not in df.columns:
            df[col] = ''
            
    return df

# --- Anomaly Detection Logic ---
def detect_anomalies(df):
    logger.info("Starting anomaly detection...")
    df['IsAnomaly'] = False
    df['AnomalyReason'] = ''

    if df.empty:
        logger.info("DataFrame is empty, no anomalies to detect.")
        return df

    # Convert 'Total' to numeric for calculations, coercing errors to NaN
    # Ensure 'Total' column exists before attempting conversion
    if 'Total' in df.columns:
        df['NumericTotal'] = pd.to_numeric(df['Total'], errors='coerce')
    else:
        df['NumericTotal'] = pd.NA # Create a column of missing values if 'Total' is absent
        logger.warning("No 'Total' column found for numeric anomaly detection.")


    # Remove rows where 'NumericTotal' is NaN for anomaly detection, but keep them in the original df
    df_numeric = df.dropna(subset=['NumericTotal']).copy()

    if df_numeric.empty:
        logger.warning("No numeric 'Total' values found for anomaly detection based on amounts.")
        return df.drop(columns=['NumericTotal']) # Drop temp column before returning

    # Anomaly 1: Unusually High Amounts (within categories)
    # Using IQR method for outlier detection
    for category in df_numeric['category'].unique():
        category_df = df_numeric[df_numeric['category'] == category]
        if len(category_df) < 3: # Need at least 3 data points for meaningful quartiles
            logger.debug(f"Skipping high amount anomaly for category '{category}': too few data points ({len(category_df)}).")
            continue

        Q1 = category_df['NumericTotal'].quantile(0.25)
        Q3 = category_df['NumericTotal'].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR # 1.5 * IQR is a common multiplier for outliers

        # Flag transactions in the original df that are outliers in this category
        high_amount_indices = df[(df['category'] == category) & (df['NumericTotal'] > upper_bound)].index
        for idx in high_amount_indices:
            df.loc[idx, 'IsAnomaly'] = True
            if 'High Amount' not in df.loc[idx, 'AnomalyReason']:
                df.loc[idx, 'AnomalyReason'] = df.loc[idx, 'AnomalyReason'] + 'High Amount; '

    # Anomaly 2: Duplicate Entries
    # Check for duplicates based on Description, Total, and Transaction Date
    duplicate_columns = ['Description', 'Total', 'Transaction Date']
    # Ensure these columns exist before checking for duplicates
    existing_duplicate_columns = [col for col in duplicate_columns if col in df.columns]

    if len(existing_duplicate_columns) == len(duplicate_columns):
        duplicates = df[df.duplicated(subset=existing_duplicate_columns, keep=False)] # keep=False flags all occurrences of duplicates
        if not duplicates.empty:
            duplicate_indices = duplicates.index
            for idx in duplicate_indices:
                df.loc[idx, 'IsAnomaly'] = True
                if 'Duplicate Entry' not in df.loc[idx, 'AnomalyReason']:
                    df.loc[idx, 'AnomalyReason'] = df.loc[idx, 'AnomalyReason'] + 'Duplicate Entry; '
    else:
        logger.warning(f"Could not check for duplicates based on {duplicate_columns} as some columns are missing.")

    # Clean up AnomalyReason by removing trailing '; '
    df['AnomalyReason'] = df['AnomalyReason'].str.strip('; ')
    
    logger.info(f"Anomaly detection complete. Found {df['IsAnomaly'].sum()} anomalous transactions.")
    return df.drop(columns=['NumericTotal']) # Drop the temporary numeric column

# --- Main Parse Endpoint ---
@app.route('/parse', methods=['POST'])
# Removed @jwt_required() to allow public access to file processing
def parse():
    logger.info("---Starting parse function---")
    try:
        excel_file = request.files.get('excel')
        pdf_file = request.files.get('pdf')
        # Get OCR language from form data, default to 'eng' (English)
        ocr_language = request.form.get('ocr_language', 'eng')
        logger.info(f"OCR Language selected: {ocr_language}")

        if not excel_file or not pdf_file:
            logger.warning("Error: Missing files")
            return jsonify({'error': 'Missing Excel/CSV or PDF file.'}), 400

        # --- Excel/CSV ---
        ext = excel_file.filename.split('.')[-1].lower()
        logger.info(f"Excel file extension: {ext}")
        if ext == 'csv':
            try:
                excel_data = pd.read_csv(io.BytesIO(excel_file.read()))
            except Exception as e:
                logger.error(f"Error reading CSV: {str(e)}", exc_info=True)
                return jsonify({'error': f'Error reading CSV: {str(e)}'}), 400
        elif ext in ['xlsx', 'xls']:
            try:
                excel_data = pd.read_excel(io.BytesIO(excel_file.read()))
            except Exception as e:
                logger.error(f"Error reading Excel: {str(e)}", exc_info=True)
                return jsonify({'error': f'Error reading Excel: {str(e)}'}), 400
        else:
            logger.warning("Error: Unsupported Excel format")
            return jsonify({'error': 'Unsupported Excel format.'}), 400

        logger.info("Excel file read successfully")

        # --- PDF ---
        tabula_tables = None
        extracted_text = ""
        if pdf_file.filename.endswith('.txt'):
            extracted_text = pdf_file.read().decode('utf-8')
            logger.info("PDF file is a text file")
        elif pdf_file.filename.endswith('.pdf'):
            logger.info("PDF file is not a text file, processing...")
            pdf_bytes = pdf_file.read()
            try:
                logger.debug("Attempting Tabula extraction")
                # Adding stream=True for better table detection
                tables = tabula.read_pdf(io.BytesIO(pdf_bytes), pages='all', multiple_tables=True, lattice=True, stream=True)
                
                if tables and any(not t.empty for t in tables):
                    tabula_tables = [t for t in tables if not t.empty] # Filter out empty tables
                    logger.info("Tabula extraction successful, tables found.")
                else:
                    logger.info("Tabula found no usable tables. Falling back to OCR.")

                if not tabula_tables: # If Tabula didn't find tables, proceed with OCR for text
                    images = convert_from_bytes(pdf_bytes, dpi=300)
                    ocr_text = ""
                    # --- IMPORTANT CHANGE HERE: Changed PSM from 6 to 3 ---
                    # PSM 3: Fully automatic page segmentation, but no OSD.
                    # This should preserve line breaks better for tabular data.
                    for img in images:
                        ocr_text += pytesseract.image_to_string(img, lang=ocr_language, config='--psm 3') + "\n"
                    extracted_text = ocr_text
                    logger.info("OCR extraction complete with PSM 3")
                else: # If Tabula found tables, still do OCR for date extraction from general text
                    images = convert_from_bytes(pdf_bytes, dpi=300)
                    ocr_text = ""
                    # Pass the selected OCR language to pytesseract
                    for img in images:
                        ocr_text += pytesseract.image_to_string(img, lang=ocr_language, config='--psm 3') + "\n" # Also use PSM 3 here
                    extracted_text = ocr_text # Use OCR text for date and general text parsing
                    
            except Exception as e:
                logger.error(f"Error processing PDF: {str(e)}", exc_info=True)
                return jsonify({'error': f'Error processing PDF: {str(e)}'}), 500
        else:
            logger.warning("Error: Unsupported file type for PDF processing.")
            return jsonify({'error': 'Unsupported file type for PDF processing.'}), 400

        logger.info(f"Extracted text length: {len(extracted_text)}")
        
        # New: If no text was extracted at all, return an error
        if not extracted_text.strip():
            logger.warning("No text extracted from PDF/OCR file.")
            return jsonify({'error': 'No text could be extracted from the PDF/OCR file. Please ensure it contains readable text.'}), 400

        date = extract_date(extracted_text) # Extract date from full OCR text
        logger.info(f"Extracted date: {date}")
        
        final_table = pd.DataFrame() # Initialize empty DataFrame

        # Decide which table to use: Tabula or Text Parsed
        # If Tabula found tables AND extracted multiple rows, prioritize Tabula
        if tabula_tables and any(len(t) > 1 for t in tabula_tables):
            logger.info("Tabula found multiple rows. Prioritizing Tabula output.")
            table = pd.concat(tabula_tables, ignore_index=True)
            table['Transaction Date'] = date # Assign extracted date
            
            # --- NEW: Robust column mapping for Tabula output ---
            mapped_data = []
            
            # Identify columns based on keywords (case-insensitive, flexible)
            desc_col_name, qty_col_name, unit_price_col_name, total_col_name = None, None, None, None

            for col in table.columns:
                col_lower = str(col).lower() # Ensure column name is string and lowercase
                if 'description' in col_lower or 'item' in col_lower or 'product' in col_lower or 'service' in col_lower:
                    desc_col_name = col
                elif 'quantity' in col_lower or 'qty' in col_lower:
                    qty_col_name = col
                elif 'unit price' in col_lower or 'price' in col_lower: # 'price' might be ambiguous, prioritize 'unit price'
                    if 'unit price' in col_lower:
                        unit_price_col_name = col
                    elif unit_price_col_name is None: # Only assign if a more specific match wasn't found
                        unit_price_col_name = col
                elif 'total' in col_lower or 'amount' in col_lower:
                    total_col_name = col

            # Process each row from the Tabula output
            for index, row in table.iterrows():
                description = str(row[desc_col_name]).strip() if desc_col_name and pd.notna(row[desc_col_name]) else ''
                quantity = str(row[qty_col_name]).strip() if qty_col_name and pd.notna(row[qty_col_name]) else '1' # Default to '1'
                unit_price = str(row[unit_price_col_name]).replace('$', '').replace(',', '').strip() if unit_price_col_name and pd.notna(row[unit_price_col_name]) else ''
                total = str(row[total_col_name]).replace('$', '').replace(',', '').strip() if total_col_name and pd.notna(row[total_col_name]) else ''

                # Fallback for unit_price if only total is available
                if not unit_price and total:
                    unit_price = total

                # Predict category and GL code
                category_pred, gl_code_pred = predict_category_and_gl_code(description)

                mapped_data.append({
                    'Transaction Date': date,
                    'Description': description,
                    'Quantity': quantity,
                    'Unit Price': unit_price,
                    'Total': total,
                    'category': category_pred,
                    'GLCode': gl_code_pred,
                    'original_text': row.to_json() # Store original row as JSON string for context
                })
            
            if mapped_data:
                final_table = pd.DataFrame(mapped_data)
                logger.info("Successfully mapped Tabula output to final table.")
            else:
                logger.warning("Tabula output could not be mapped to meaningful data. Falling back to text parsed table.")
                final_table = extract_receipt_details_from_text(extracted_text, date) # Fallback if Tabula mapping fails or results in empty data

        else:
            logger.info("Tabula found no usable tables or only a single row. Using text parsed output.")
            final_table = extract_receipt_details_from_text(extracted_text, date)

        if final_table.empty:
            logger.warning("Error: No rows extracted after all parsing attempts.")
            return jsonify({'error': 'No meaningful data could be extracted from the receipt. Please ensure it contains clear transaction details.'}), 400

        # Run anomaly detection
        final_table = detect_anomalies(final_table)

        # Ensure all columns are strings for consistent frontend handling (important for editable fields)
        for col in final_table.columns:
            final_table[col] = final_table[col].astype(str)

        logger.info("Final table extraction successful")
        return jsonify({
            'headers': list(final_table.columns),
            'rows': final_table.to_dict(orient='records'),
            'extracted_text': extracted_text
        })

    except Exception as e:
        logger.exception(f"General error in parse endpoint: {str(e)}")
        return jsonify({'error': f'Internal error: {str(e)}'}), 500

# New: Save parsed and potentially edited data endpoint
@app.route('/save_parsed_data', methods=['POST'])
@jwt_required()
def save_parsed_data():
    current_user = get_jwt_identity()
    logger.info(f"Received request to save parsed data for user: {current_user}")
    data_rows = request.get_json()

    if not data_rows:
        logger.warning("No data rows provided for saving.")
        return jsonify({'error': 'No data provided to save.'}), 400

    conn = get_db()
    c = conn.cursor()
    try:
        # Get user_id
        c.execute('SELECT id FROM users WHERE username = ?', (current_user,))
        user = c.fetchone()
        if not user:
            logger.error(f"User '{current_user}' not found in database.")
            return jsonify({'error': 'User not found.'}), 404
        user_id = user['id']

        # Clear existing data for this user to avoid duplicates if this is a re-save
        c.execute('DELETE FROM parsed_transactions WHERE user_id = ?', (user_id,))
        logger.info(f"Cleared existing parsed data for user_id: {user_id}")

        # Insert new/updated data
        for row in data_rows:
            # Ensure all expected columns are present, fill with None or default if missing
            description = row.get('Description', '')
            transaction_date = row.get('Transaction Date', datetime.today().strftime('%Y-%m-%d'))
            quantity = row.get('Quantity', '')
            unit_price = row.get('Unit Price', '')
            total = row.get('Total', '')
            category = row.get('category', 'Uncategorized')
            gl_code = row.get('GLCode', CATEGORY_TO_GL_MAP.get(category, '9999 - Uncategorized')) # Get GL code, with fallback
            original_text = row.get('original_text', '') # Use 'original_text' from the extracted row

            c.execute('''INSERT INTO parsed_transactions 
                         (user_id, transaction_date, description, quantity, unit_price, total, category, gl_code, original_text)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                      (user_id, transaction_date, description, quantity, unit_price, total, category, gl_code, original_text))
        
        conn.commit()
        logger.info(f"Successfully saved {len(data_rows)} parsed transactions for user_id: {user_id}")
        return jsonify({'msg': f'Successfully saved {len(data_rows)} transactions!'}), 200

    except Exception as e:
        conn.rollback()
        logger.exception(f"Error saving parsed data for user {current_user}: {str(e)}")
        return jsonify({'error': f'Failed to save data: {str(e)}'}), 500
    finally:
        conn.close()

if __name__ == '__main__':
    # When running directly, ensure the model is loaded/trained
    vectorizer, category_model, gl_model = load_or_train_models()
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=5000)
