# Use a full Debian Bullseye image as a base for better package compatibility
FROM python:3.9-bullseye

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for the app
# - tesseract-ocr for pytesseract
# - poppler-utils for pdf2image
# - default-jre for tabula-py
# - libgl1-mesa-glx is a common graphics dependency
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    default-jre \
    libgl1-mesa-glx \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port the app runs on
EXPOSE 5000

# Define the command to run the application with an increased timeout
CMD ["gunicorn", "--workers", "2", "--timeout", "120", "--bind", "0.0.0.0:5000", "app:app"]
