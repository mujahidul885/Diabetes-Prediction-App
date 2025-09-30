# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install Python dependencies first
RUN pip install --no-cache-dir -r requirements.txt

# Downgrade scikit-learn to match model version (if your model was trained with 1.6.1)
RUN pip install --no-cache-dir scikit-learn==1.6.1

# Expose Streamlit default port
EXPOSE 8501

# Run the Streamlit app from project root (app.py must be at project root)
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
