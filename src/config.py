"""Paths used by 01_mediapipe_extract_landmarks.py."""
import os

# Raw data (inside the repository folder, never versioned)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VICON_CSV_DIR = os.path.join(BASE_DIR, "data", "raw", "vicon_csv")
VIDEO_DIR = os.path.join(BASE_DIR, "data", "raw", "videos")

# Generated files
INTERMEDIATE_DIR = os.path.join(BASE_DIR, "data", "intermediate")
os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
