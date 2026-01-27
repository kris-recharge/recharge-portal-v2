# ReCharge Alaska Portal v2 - Streamlit (App_v2)
# NOTE: This Dockerfile is intended to be built from the REPO ROOT as build context,
# so it can copy the root-level requirements.txt and the App_v2/ source tree.

FROM python:3.11-slim

# Prevents Python from buffering stdout/stderr (useful for docker logs)
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (psycopg may need libpq at runtime; build tools help with any compiled deps)
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
     build-essential \
     gcc \
     libpq-dev \
     curl \
  && rm -rf /var/lib/apt/lists/*

# Install python deps (expects requirements.txt at repo root)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy Streamlit app source
COPY App_v2/ /app/

EXPOSE 8501

CMD ["streamlit", "run", "app_v2.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false", "--server.baseUrlPath=st", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]