# ✍️ AI Handwritten Digit Recognition | End-to-End MLOps Pipeline

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![Render](https://img.shields.io/badge/Render-46E3B7?style=flat&logo=render&logoColor=white)](https://render.com/)
[![Vercel](https://img.shields.io/badge/Vercel-000000?style=flat&logo=vercel&logoColor=white)](https://vercel.com/)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF?logo=github-actions)](https://github.com/features/actions)

An end-to-end Machine Learning web application that allows users to draw a number (0-9) on a digital canvas and receive real-time AI predictions. 

This project goes beyond a standard Jupyter Notebook by implementing a complete **MLOps pipeline**, featuring a custom Convolutional Neural Network (CNN), a containerized FastAPI backend, automated testing via GitHub Actions, and a mobile-responsive frontend.

## 🚀 Live Demo
* **Frontend UI:** [Draw a Number Here](https://digit-recognizer-web-blond.vercel.app)
* **API Documentation (Swagger UI):** [API Docs](https://YOUR-RENDER-URL.onrender.com/docs) *(Replace with your actual Render URL)*

---

## 🏗️ System Architecture & MLOps Pipeline

1. **Model Training (Data Science):** A custom CNN model was built using TensorFlow/Keras and trained on the MNIST dataset.
2. **Model Serving (Backend):** The `.keras` model is loaded into a lightweight, asynchronous FastAPI server. Incoming base64 images from the web are preprocessed (grayscale conversion, inversion, normalization, and resizing to 28x28) before inference.
3. **Containerization (DevOps):** The backend is packaged into an isolated Docker container using a `python:3.11-slim` image to ensure environment parity and prevent dependency drift.
4. **CI/CD Pipeline (Automation):** GitHub Actions automatically triggers on every push to the `main` branch, running `pytest` unit tests and verifying the Docker build process.
5. **Cloud Deployment:** The Dockerized backend is deployed to **Render**, while the static HTML/JS frontend is hosted on **Vercel**.

---

## 🧠 Model Details
The underlying AI is a Convolutional Neural Network trained to recognize handwritten digits.
* **Architecture:** `Conv2D (32 filters)` → `MaxPooling2D` → `Flatten` → `Dense (128 units, ReLU)` → `Dense (10 units, Softmax)`
* **Training Performance:** Achieved **~99.8% accuracy** and a loss of **0.005** on the training set after 3 epochs, utilizing Early Stopping to prevent overfitting.
* **Parameters:** 591,786 trainable parameters.

---

## 💻 Tech Stack
* **Machine Learning:** Python 3.11, TensorFlow, Keras, NumPy, Pandas, Matplotlib
* **Backend API:** FastAPI, Uvicorn, Pydantic, Pillow (Image Processing)
* **Frontend:** HTML5, CSS3, Vanilla JavaScript, Canvas API
* **DevOps & Hosting:** Docker, GitHub Actions, Render, Vercel, Pytest

---

## 📂 Project Structure
```text
📦 CNN_Handwritten
 ┣ 📂 .github/workflows   # CI/CD pipeline configurations (ci.yml)
 ┣ 📂 frontend            # Vercel-hosted web interface
 ┃ ┗ 📜 index.html        # Mobile-responsive canvas and fetch logic
 ┣ 📜 .dockerignore       # Docker build exclusions
 ┣ 📜 .gitignore          # Git tracking exclusions
 ┣ 📜 Dockerfile          # Containerization instructions
 ┣ 📜 digit_recognition_model.keras # Exported CNN model
 ┣ 📜 main.py             # FastAPI backend application
 ┣ 📜 requirements.txt    # Python dependencies
 ┣ 📜 test_main.py        # Pytest unit tests for the API
 ┗ 📜 train.ipynb         # Model architecture and training notebook
