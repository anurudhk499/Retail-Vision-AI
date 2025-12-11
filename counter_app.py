import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import * # Importing the file you just created

# 1. SETUP
# Load the Video
cap = cv2.VideoCapture("people.mp4") # Make sure this matches your video name!

# Load the AI Model
model = YOLO('yolov8n.pt')

# Load the Class Names (We only care about "person" which is ID 0)
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Initialize Tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# 2. DEFINE THE COUNTING LINE
# [x1, y1, x2, y2] coordinates
# You might need to adjust these numbers based on your specific video resolution!
limits = [400,350, 1000, 350] 

totalCount = []

while True:
    success, img = cap.read()
    
    # Loop video logic
    if not success:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # Make detections
    results = model(img, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Extract Coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # ONLY detect people with high confidence
            if currentClass == "person" and conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    # 3. UPDATE TRACKER
    resultsTracker = tracker.update(detections)

    # DRAW THE LINE
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        
        # Draw Box and ID
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=10)

        # 4. COUNTING LOGIC
        # Find center point of the person
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # Check if center point crosses the line
        # Logic: Is cx between line width? Is cy touching the line height?
        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            if totalCount.count(id) == 0:
                totalCount.append(id)
                # Flash the line Green when counted
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    # 5. DISPLAY COUNT
    cvzone.putTextRect(img, f' Count: {len(totalCount)}', (50, 50))

    cv2.imshow("AI Footfall Counter", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()