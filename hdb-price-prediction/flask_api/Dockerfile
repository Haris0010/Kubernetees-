# Use a base image (for Python, you can use a suitable version)
FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container
COPY app.py .
COPY requirements.txt .

# Create an uploads directory
RUN mkdir -p /app/uploads

# Install any dependencies (ensure you have a requirements.txt)
RUN pip install --no-cache-dir -r requirements.txt

# Expose the Flask API port
EXPOSE 5000

# Run the Flask API
CMD ["python", "app.py"]
