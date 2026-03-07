# Use an official Python slim runtime as a parent image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Install OS-level dependencies for OpenCV and librosa
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 libsndfile1 && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
# We assume there is or will be a requirements.txt file in backend/
COPY requirements.txt ./

# Install any dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend directory contents into the container at /app
# Note: Ensure sensitive files like .key/ folder or .env are excluded in .dockerignore!
COPY . .

# Expose port (default FastAPI runs on 8000)
EXPOSE 8000

# Command to run the Fast API server using Uvicorn
# Update the main app import path according to your actual folder structure.
# E.g., if you run FastAPI/main/Test.py as the entry point 
# CMD ["uvicorn", "FastAPI.main.Test:app", "--host", "0.0.0.0", "--port", "8000"]

# As a placeholder, we drop into bash if there isn't a definitive start script yet.
# When ready, replace the below CMD with the uvicorn command above.
CMD ["/bin/bash"]
