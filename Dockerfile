FROM python:3.10-slim

# Install necessary system libraries
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    netcdf-bin \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for GDAL if necessary
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Set the working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model weights
COPY main.py .
COPY models/ ./models/
RUN gunzip models/deeplabv3plus_resnet50_oilspill.pth.gz

# Expose port 8000 for the FastAPI server
EXPOSE 8000

# Set entrypoint to run Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
