FROM python:3.11-slim
WORKDIR /app
# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gradio>=4.0.0 matplotlib>=3.7.0
# Copy application code
COPY . .
# Create __init__ files
RUN touch env/__init__.py tasks/__init__.py graders/__init__.py
# HuggingFace Spaces runs on port 7860
EXPOSE 7860
# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/')"
# Start Gradio demo directly
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]