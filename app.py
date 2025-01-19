import json
import cv2
from ultralytics import YOLOv10
import numpy as np
import re
import os
import sqlite3
from datetime import datetime
from paddleocr import PaddleOCR
from threading import Thread
from queue import Queue

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Create a Video Capture Object
cap = cv2.VideoCapture("data/carLicence5.mp4")

# Initialize the YOLOv10 Model
model = YOLOv10("weights/best.pt")

# Class Names
className = ["License"]

# Initialize the Paddle OCR
ocr = PaddleOCR(use_angle_cls=True, use_gpu=False)

# Country-specific patterns
country_patterns = {
    'USA': r'^[A-Z0-9]{1,7}$',
    'EU': r'^[A-Z]{1,3}-[A-Z]{1,2}-\d{1,4}$',
    'IN': r'^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$'
}

# Ask the user to select the country
print("Select the country for license plate detection:")
print("1. USA\n2. EU\n3. India (IN)")
country_choice = input("Enter the number corresponding to the country: ")

country_map = {'1': 'USA', '2': 'EU', '3': 'IN'}
selected_country = country_map.get(country_choice, 'USA')

def validate_plate(plate, country):
    """Validate the license plate against the country's pattern."""
    pattern = re.compile(country_patterns[country])
    return bool(pattern.match(plate))

def paddle_ocr(frame, x1, y1, x2, y2):
    """Extract and return the recognized text from the given region."""
    cropped_frame = frame[y1:y2, x1:x2]
    result = ocr.ocr(cropped_frame, det=False, rec=True, cls=False)
    text = ""
    for r in result:
        scores = r[0][1]
        if np.isnan(scores):
            scores = 0
        else:
            scores = int(scores * 100)
        if scores > 60:
            text = r[0][0]
    text = re.sub('[\W]', '', text)
    text = text.replace("O", "0").replace("粤", "")
    return str(text)

def save_to_database(license_plates, start_time, end_time):
    """Save the detected plates and timestamps to the SQLite database."""
    conn = sqlite3.connect('licensePlatesDatabase.db')
    cursor = conn.cursor()
    for plate in license_plates:
        cursor.execute('''
            INSERT INTO LicensePlates(start_time, end_time, license_plate)
            VALUES (?, ?, ?)
        ''', (start_time.isoformat(), end_time.isoformat(), plate))
    conn.commit()
    conn.close()

# Queue for frames to be processed
frame_queue = Queue(maxsize=10)

license_plates = set()
startTime = datetime.now()
stop_flag = False  # Global flag to signal when to stop

def frame_producer():
    """Capture frames and put them into the queue."""
    global stop_flag
    while not stop_flag:
        ret, frame = cap.read()
        if not ret:
            frame_queue.put(None)  # Signal the end of frames
            break
        frame_queue.put(frame)

def frame_consumer():
    """Process frames from the queue for detection and OCR."""
    global startTime, stop_flag
    while True:
        frame = frame_queue.get()
        if frame is None:
            break  # No more frames to process

        currentTime = datetime.now()
        results = model.predict(frame, conf=0.45)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Perform OCR and validate
                label = paddle_ocr(frame, x1, y1, x2, y2)
                if label and validate_plate(label, selected_country):
                    license_plates.add(label)
                    # Draw the rectangle and display the text
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    textSize = cv2.getTextSize(label, 0, fontScale=0.5, thickness=2)[0]
                    c2 = x1 + textSize[0], y1 - textSize[1] - 3
                    cv2.rectangle(frame, (x1, y1), c2, (255, 0, 0), -1)
                    cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

        # Save data every 20 seconds
        if (currentTime - startTime).seconds >= 20:
            endTime = currentTime
            save_to_database(license_plates, startTime, endTime)
            startTime = currentTime
            license_plates.clear()

        # Display the frame
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('1'):
            stop_flag = True  # Signal the producer to stop
            break

# Start producer and consumer threads
producer_thread = Thread(target=frame_producer, daemon=True)
consumer_thread = Thread(target=frame_consumer, daemon=True)

producer_thread.start()
consumer_thread.start()

# Wait for threads to finish
producer_thread.join()
consumer_thread.join()

cap.release()
cv2.destroyAllWindows()
