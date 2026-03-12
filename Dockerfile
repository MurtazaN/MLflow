FROM python:3.11-slim

WORKDIR /app

# Install dependencies (cached layer — only rebuilds when requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY pyproject.toml .
COPY src/ src/
COPY data/ data/

# Install the package
RUN pip install -e .

EXPOSE 5000 5001
