FROM python:3.11-slim

# System dependencies for scientific Python + HDF5
RUN apt-get update && apt-get install -y \
    build-essential \
    libhdf5-dev \
 && rm -rf /var/lib/apt/lists/*

# Workdir inside the container
WORKDIR /app

# Copy the entire repo into the image
COPY . /app

# Python dependencies
RUN pip install --no-cache-dir \
    numpy \
    scipy \
    matplotlib \
    h5py \
    astropy \
    SALib \
    xgboost \
    scikit-learn \
    Flask

# Ensure these directories exist
RUN mkdir -p data models logs plots

# Flask listens on port 5000
EXPOSE 5000

# Default command: start the API
CMD ["python", "app.py"]
