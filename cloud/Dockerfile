# Use a lightweight Python base image
FROM python:3.12-slim

# Install git so pip can clone the TimesFM repository
RUN apt-get update && apt-get install -y git

# Set the working directory inside the container
WORKDIR /app

# Copy your project files into the container
COPY . /app

# Install standard dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Google's V3 architecture directly from GitHub
RUN pip install --no-cache-dir git+https://github.com/google-research/timesfm.git

# Expose the mandatory Hugging Face port
EXPOSE 7860

# Boot up the server when the container starts
CMD ["python", "app.py"]