# Dockerfile
FROM python:3.11-slim

# System deps (HDF5 + basic build tools in case wheels need compiling)
RUN apt-get update && apt-get install -y \
    build-essential \
    libhdf5-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Workdir inside the container
WORKDIR /app

# Copy everything into the image
COPY . /app

# Python dependencies (keep it simple)
RUN pip install --no-cache-dir \
    numpy \
    scipy \
    matplotlib \
    h5py \
    astropy \
    SALib \
    xgboost \
    scikit-learn

# Make sure the expected directories exist (idempotent)
RUN mkdir -p data models logs plots

# When you "docker run", drop into a shell so you can run your usual pipeline
CMD ["bash"]
