# Build context must be this directory (cloud/), which includes a copy of
# forecasting.py from the repository root.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
RUN pip install --no-cache-dir git+https://github.com/google-research/timesfm.git

COPY forecasting.py /app/forecasting.py
COPY app.py /app/app.py

EXPOSE 7860

CMD ["python", "app.py"]
