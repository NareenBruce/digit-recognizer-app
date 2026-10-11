# 1. Use the exact Python version you used for training to prevent environment drift
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy only the requirements first (this leverages Docker caching to speed up future builds)
COPY requirements.txt .

# 4. Install dependencies (we add --no-cache-dir to keep the container size small)
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your backend application code and the model
COPY main.py .
COPY digit_recognition_model.keras .

# 6. Expose the port FastAPI runs on
EXPOSE 8000

# 7. Command to run the application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]