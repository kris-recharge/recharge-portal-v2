FROM python:3.11-slim

WORKDIR /app

# Common libs (psycopg / ssl / etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    libpq5 \
  && rm -rf /var/lib/apt/lists/*

# Install python deps
# (Assumes App_v2/requirements.txt exists)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the Streamlit app
COPY . /app

EXPOSE 8501

CMD ["streamlit", "run", "app_v2.py", "--server.address=0.0.0.0", "--server.port=8501"]