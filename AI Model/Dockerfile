# Use an official Python runtime as a parent image (preferably with PyTorch support)
# Here we use a slim version of Python 3.11, but you can also use pytorch official images
# e.g., FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (required for some ML libraries like OpenCV)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies
COPY requirements.txt ./
# (Optional) If you have a specific PyTorch index url, you can add it here
RUN pip install --no-cache-dir -r requirements.txt

# Copy the AI models
COPY AI_pth/ ./AI_pth/

# Copy the main serving script (e.g., FastAPI or inference api)
COPY . .

# Expose port for the API (e.g., 8000 for FastAPI)
EXPOSE 8000

# Command to run the inference server
# For example, if you use FastAPI to serve the model:
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

# If there isn't a serving script yet, the container will just exit,
# or you could run an interactive shell for now.
CMD ["/bin/bash"]
