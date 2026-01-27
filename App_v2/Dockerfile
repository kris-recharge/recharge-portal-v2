FROM python:3.11-slim

WORKDIR /app

# Common libs (psycopg / ssl / etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    libpq5 \
  && rm -rf /var/lib/apt/lists/*

# Install python deps (requirements.txt lives at repo root)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy ONLY the Streamlit app code into the container workdir
COPY App_v2/ /app/

EXPOSE 8501

CMD ["streamlit", "run", "app_v2.py", "--server.address=0.0.0.0", "--server.port=8501"]