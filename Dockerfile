FROM python:3.9-slim

WORKDIR /app

# Copy the app folder into the container
COPY . .
 

# Install dependencies
RUN pip install --no-cache-dir -e .

# Expose the port
EXPOSE 5000

# Command to run your app
CMD ["python", "app.py"]
