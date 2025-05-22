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
    conn.commit()
    conn.close()

init_user_db()

# --- Register Endpoint ---
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    if not username or not password:
        return jsonify({'error': 'Username and password required.'}), 400
    password_hash = generate_password_hash(password)
    try:
        conn = get_db()
        c = conn.cursor()
        c.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)', (username, password_hash))
        conn.commit()
        conn.close()
        return jsonify({'msg': 'User registered successfully.'}), 201
    except sqlite3.IntegrityError:
        return jsonify({'error': 'Username already exists.'}), 409

# --- Login Endpoint ---
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    if not username or not password:
        return jsonify({'error': 'Username and password required.'}), 400
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username = ?', (username,))
    user = c.fetchone()
    conn.close()
    if user and check_password_hash(user['password_hash'], password):
        access_token = create_access_token(identity=username)
        return jsonify({'access_token': access_token}), 200
    else:
        return jsonify({'error': 'Invalid username or password.'}), 401

# --- Category Prediction Model (as provided previously) ---
def load_category_model():
    """
    Creates a more comprehensive category prediction model with
    expanded training data and improved features.
    """
    data = [
        ("yard maintenance", "Maintenance"),
        ("lawn care", "Maintenance"),
        ("gardening", "Maintenance"),
        ("landscaping", "Maintenance"),
        ("cleaning", "Maintenance"),
        ("janitorial", "Maintenance"),
        ("repair", "Maintenance"),
        ("fix", "Maintenance"),
        ("plumbing", "Maintenance"),
        ("electrical repair", "Maintenance"),
        ("hvac", "Maintenance"),
        ("service call", "Maintenance"),
        ("maintenance fee", "Maintenance"),
        ("junk removal", "Junk Removal"),
        ("waste disposal", "Junk Removal"),
        ("trash pickup", "Junk Removal"),
        ("garbage", "Junk Removal"),
        ("bulk furniture removal", "Junk Removal"),
        ("debris removal", "Junk Removal"),
        ("haul away", "Junk Removal"),
        ("dump fee", "Junk Removal"),
        ("waste management", "Junk Removal"),
        ("electric bill", "Utilities"),
        ("water bill", "Utilities"),
        ("gas bill", "Utilities"),
        ("internet", "Utilities"),
        ("phone service", "Utilities"),
        ("cable", "Utilities"),
        ("utility", "Utilities"),
        ("office supplies", "Supplies"),
        ("materials", "Supplies"),
        ("ink", "Supplies"),
        ("paper", "Supplies"),
        ("toner", "Supplies"),
        ("equipment", "Supplies"),
        ("consultation", "Services"),
        ("professional", "Services"),
        ("service fee", "Services"),
        ("labor", "Services"),
        ("installation", "Services"),
        ("commission", "Services"),
        ("contractor", "Services"),
        ("insurance premium", "Insurance"),
        ("policy", "Insurance"),
        ("coverage", "Insurance"),
        ("liability", "Insurance"),
        ("tax payment", "Taxes & Fees"),
        ("license fee", "Taxes & Fees"),
        ("permit", "Taxes & Fees"),
        ("registration", "Taxes & Fees"),
        ("filing fee", "Taxes & Fees"),
        ("rent", "Rent & Leasing"),
        ("lease", "Rent & Leasing"),
        ("space rental", "Rent & Leasing"),
        ("office rent", "Rent & Leasing"),
        ("equipment rental", "Rent & Leasing"),
        ("airfare", "Travel & Transport"),
        ("hotel", "Travel & Transport"),
        ("mileage", "Travel & Transport"),
        ("transportation", "Travel & Transport"),
        ("parking", "Travel & Transport"),
        ("fuel", "Travel & Transport"),
        ("gas", "Travel & Transport")
    ]
    texts, labels = zip(*data)
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), max_features=1000, min_df=1, stop_words='english')
    X = vectorizer.fit_transform(texts)
    clf = LogisticRegression(C=10.0, class_weight='balanced', max_iter=1000, solver='liblinear', multi_class='ovr').fit(X, labels)
    return vectorizer, clf

vectorizer, category_model = load_category_model()

def predict_category(text):
    try:
        clean_text = text.lower().strip()
        if not clean_text:
            return 'Uncategorized'
        X = vectorizer.transform([clean_text])
        prediction = category_model.predict(X)[0]
        probs = category_model.predict_proba(X)[0]
        max_prob = max(probs)
        if max_prob < 0.4:
            if any(word in clean_text for word in ['service', 'labor', 'work', 'job']):
                return 'Services'
            elif any(word in clean_text for word in ['buy', 'purchase', 'item', 'product']):
                return 'Supplies'
            else:
                return 'Other Expenses'
        return prediction
    except Exception as e:
        print(f"Category prediction error: {str(e)}")
        return 'Services'

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
            print(f"Matched date string: {date_str}")
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
    print("No valid date found in text, using current date")
    return datetime.today().strftime('%Y-%m-%d')

def parse_text_to_table(text, date):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    data = []
    headers = []

    # First try to find headers in the text
    header_pattern = r'(?:Item|Description|Quantity|Unit Price|Total|Amount|Price|Qty|#|No\.|Number|Product|Service|Description|Details|Particulars|Item Description|Item No\.|Item Number|Item #|Item Name|Product Name|Product Description|Service Description|Service Name|Service Details|Service Particulars|Service Item|Service Product|Service Item Description|Service Item Name|Service Item Number|Service Item #|Service Item No\.|Service Product Name|Service Product Description|Service Product Number|Service Product #|Service Product No\.)'
    
    for i, line in enumerate(lines):
        if re.search(header_pattern, line, re.IGNORECASE):
            # Try to split the line into potential headers
            potential_headers = re.split(r'\s{2,}|\t|,', line)
            potential_headers = [h.strip() for h in potential_headers if h.strip()]
            
            # Validate headers
            if len(potential_headers) >= 2:  # At least 2 columns
                headers = potential_headers
                lines = lines[i+1:]  # Remove header line and everything before it
                break

    # If no headers found, use default headers
    if not headers:
        headers = ['Description', 'Quantity', 'Unit Price', 'Total']

    # Process each line
    for line in lines:
        if not line.strip():
            continue

        # Split line into values
        values = re.split(r'\s{2,}|\t|,', line)
        values = [v.strip() for v in values if v.strip()]

        if not values:
            continue

        row = {'Transaction Date': date}
        
        # Map values to headers
        for i, header in enumerate(headers):
            if i < len(values):
                value = values[i]
                # Clean up the value
                if re.match(r'^\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})', value):
                    value = value.replace('$', '').replace(',', '')
                row[header] = value

        # Ensure required fields exist
        if 'Description' not in row:
            row['Description'] = ' '.join(values)
        if 'Total' not in row:
            # Try to find total in the values
            for value in values:
                if re.match(r'^\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})', value):
                    row['Total'] = value.replace('$', '').replace(',', '')
                    break

        row['category'] = predict_category(row.get('Description', ''))
        data.append(row)

    return pd.DataFrame(data)

@app.route('/parse', methods=['POST'])
def parse():
    print("---Starting parse function---")
    try:
        excel_file = request.files.get('excel')
        pdf_file = request.files.get('pdf')

        if not excel_file or not pdf_file:
            print("Error: Missing files")
            return jsonify({'error': 'Missing Excel/CSV or PDF file.'}), 400

        # --- Excel/CSV ---
        ext = excel_file.filename.split('.')[-1].lower()
        print(f"Excel file extension: {ext}")
        if ext == 'csv':
            try:
                excel_data = pd.read_csv(excel_file)
            except Exception as e:
                print(f"Error reading CSV: {str(e)}")
                return jsonify({'error': f'Error reading CSV: {str(e)}'}), 400
        elif ext in ['xlsx', 'xls']:
            try:
                excel_data = pd.read_excel(excel_file)
            except Exception as e:
                print(f"Error reading Excel: {str(e)}")
                return jsonify({'error': f'Error reading Excel: {str(e)}'}), 400
        else:
            print("Error: Unsupported Excel format")
            return jsonify({'error': 'Unsupported Excel format.'}), 400

        print("Excel file read successfully")

        # --- PDF ---
        tabula_tables = None
        if pdf_file.filename.endswith('.txt'):
            extracted_text = pdf_file.read().decode('utf-8')
            print("PDF file is a text file")
        elif pdf_file.filename.endswith('.pdf'):
            print("PDF file is not a text file, processing...")
            pdf_bytes = pdf_file.read()
            try:
                print("Attempting Tabula extraction")
                tables = tabula.read_pdf(io.BytesIO(pdf_bytes), pages='all', multiple_tables=True, lattice=True)
                extracted_text_tabula = ""
                extracted_headers = []
                tabula_tables = None
                if tables and any(not t.empty for t in tables):
                    tabula_tables = tables
                    if not tables[0].empty:
                        extracted_headers = tables[0].columns.tolist()
                        print(f"Extracted headers: {extracted_headers}")
                    for table in tables:
                        if not table.empty:
                            table_str = table.to_string(index=False)
                            extracted_text_tabula += table_str + "\n\n"
                    print("Tabula extraction successful")
                else:
                    print("Tabula found no usable tables.")

                if extracted_text_tabula:
                    extracted_text = extracted_text_tabula
                else:
                    print("Falling back to OCR for PDF")
                    images = convert_from_bytes(pdf_bytes, dpi=300)
                    ocr_text = ""
                    for img in images:
                        ocr_text += pytesseract.image_to_string(img, config='--psm 6') + "\n"
                    extracted_text = ocr_text
                    print("OCR extraction complete")

            except Exception as e:
                print(f"Error processing PDF: {str(e)}")
                return jsonify({'error': f'Error processing PDF: {str(e)}'}), 500
        else:
            print("Error: Unsupported file type for PDF processing.")
            return jsonify({'error': 'Unsupported file type for PDF processing.'}), 400

        print("Extracted text length:", len(extracted_text))
        date = extract_date(extracted_text)
        print(f"Extracted date: {date}")
        
        # Use Tabula DataFrame(s) directly if available
        if tabula_tables:
            print("Tabula tables found, concatenating...")
            table = pd.concat(tabula_tables, ignore_index=True)
            print(f"Table shape after concat: {table.shape}")
            # OCR for date extraction
            print("Running OCR for date extraction...")
            images = convert_from_bytes(pdf_bytes, dpi=300)
            ocr_text = ""
            for img in images:
                ocr_text += pytesseract.image_to_string(img, config='--psm 6') + "\n"
            date = extract_date(ocr_text)
            print(f"Extracted date from OCR: {date}")
            table['Transaction Date'] = date
            # Try to use the most likely description column for category
            desc_col = None
            for col in table.columns:
                if col.lower() in ['item', 'description', 'product', 'service']:
                    desc_col = col
                    break
            if desc_col:
                table['category'] = table[desc_col].apply(predict_category)
            else:
                table['category'] = ''
        else:
            print("No Tabula tables found, using text parsing fallback.")
            table = parse_text_to_table(extracted_text, date)
            print(f"Table shape after text parsing: {table.shape}")

        if table.empty:
            print("Error: No rows extracted")
            return jsonify({'error': 'No rows extracted from receipt.'}), 400

        print("Table extraction successful")
        return jsonify({
            'headers': list(table.columns),
            'rows': table.to_dict(orient='records'),
            'extracted_text': extracted_text
        })

    except Exception as e:
        print(f"General error: {str(e)}")
        return jsonify({'error': f'Internal error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)