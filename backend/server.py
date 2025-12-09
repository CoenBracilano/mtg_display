import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import easyocr
import requests
import argparse


app = FastAPI()

@app.get("/")
def home():
    return {"message": "Backend is running!"}

# Usage:
# python server.py --port port_number
# If no port number is provided, uses port 8000
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, port=args.port)