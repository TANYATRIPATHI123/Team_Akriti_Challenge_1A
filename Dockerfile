# FROM --platform=linux/amd64 python:3.10

# WORKDIR /app

# # Copy the processing script
# COPY process_pdfs.py .

# # Run the script
# CMD ["python", "process_pdfs.py"] 

# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy the 1a folder contents into container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Default command
CMD ["python", "process_pdfs.py"]
