# 1. Use the official lightweight Python image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install necessary system graphics libraries for OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

# 4. Copy the requirements file into the container
COPY requirements.txt .

# 5. Install all Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy your entire project into the container
COPY . .

# 7. Expose the port FastAPI uses
EXPOSE 8000

# 8. Command to start your FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]