# Use a base image (for Python, you can use a suitable version)
FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container
COPY predict.py .
COPY requirements.txt .

# Install any dependencies (ensure you have a requirements.txt)
RUN pip install --no-cache-dir -r requirements.txt

# Command to run when the container starts
CMD ["python", "predict.py"]
