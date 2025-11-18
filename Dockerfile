# Use the official Python image as a base
FROM python:3.8-slim-buster

# Install necessary system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libcurl4-openssl-dev \
    build-essential \
    python3-dev \
    libssl-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Upgrade pip and install wheel for faster package installation
RUN pip install --upgrade pip wheel setuptools

# Copy only requirements first (for caching layers)
COPY requirements.txt /app/

# Install required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy all other project files to the working directory
COPY . /app/

# Expose port 8000 (or the port your Django app runs on)
EXPOSE 8000

# Run migrations and start the Django development server
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
