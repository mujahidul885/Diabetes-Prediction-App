FROM python:3.11-slim
WORKDIR /app

# Install dependencies first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# lock sklearn if needed
RUN pip install --no-cache-dir scikit-learn==1.6.1

# Copy only what we need to predictable locations
COPY app/app.py /app/app.py
COPY models/ /app/models/
COPY README.md /app/README.md

EXPOSE 8501

ENV MODEL_PATH=models/diabeties_model_rf.pkl
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
