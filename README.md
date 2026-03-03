# Face Detection Attendance System

## Overview
This project automates attendance marking using real-time face recognition technology.

## Features
- Face data collection
- Face encoding and training
- Real-time recognition
- Automatic attendance logging with timestamp
- Duplicate attendance prevention

## Technologies Used
Python, OpenCV, face_recognition, NumPy, Pandas

## How to Run

1. Install dependencies:
pip install -r requirements.txt

2. Capture faces:
python src/capture_faces.py

3. Train model:
python src/train_model.py

4. Start recognition:
python src/recognize_faces.py
